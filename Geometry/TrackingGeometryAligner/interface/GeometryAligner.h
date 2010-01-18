#ifndef Geometry_TrackingGeometryAligner_GeometryAligner_h
#define Geometry_TrackingGeometryAligner_GeometryAligner_h

#include <vector>
#include <algorithm>
#include <iterator>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonDetUnit/interface/DetPositioner.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/Alignment/interface/AlignTransformError.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"

class Alignments;


/// Class to update a given geometry with a set of alignments

class GeometryAligner : public DetPositioner { 

public:
  template<class C> 
  void applyAlignments( C* geometry,
			const Alignments* alignments,
			const AlignmentErrors* alignmentErrors,
			const AlignTransform& globalCoordinates );

  inline void removeGlobalTransform( const Alignments* alignments,
                                     const AlignmentErrors* alignmentErrors,
                                     const AlignTransform& globalCoordinates,
                                     Alignments* newAlignments,
                                     AlignmentErrors* newAlignmentErrors );
};


template<class C>
void GeometryAligner::applyAlignments( C* geometry,
				       const Alignments* alignments,
				       const AlignmentErrors* alignmentErrors,
				       const AlignTransform& globalCoordinates )
{

  edm::LogInfo("Alignment") << "@SUB=GeometryAligner::applyAlignments" 
			    << "Starting to apply alignments.";

  // Preliminary checks (or we can't loop!)
  if ( alignments->m_align.size() != geometry->theMap.size() )
	throw cms::Exception("GeometryMismatch") 
	  << "Size mismatch between geometry (size=" << geometry->theMap.size() 
	  << ") and alignments (size=" << alignments->m_align.size() << ")";
  if ( alignments->m_align.size() != alignmentErrors->m_alignError.size() )
	throw cms::Exception("GeometryMismatch") 
	  << "Size mismatch between geometry (size=" << geometry->theMap.size() 
	  << ") and alignment errors (size=" << alignmentErrors->m_alignError.size() << ")";

  const AlignTransform::Translation &globalShift = globalCoordinates.translation();
  const AlignTransform::Rotation globalRotation = globalCoordinates.rotation(); // by value!

  // Parallel loop on alignments, alignment errors and geomdets
  std::vector<AlignTransform>::const_iterator iAlign = alignments->m_align.begin();
  std::vector<AlignTransformError>::const_iterator 
	iAlignError = alignmentErrors->m_alignError.begin();
  //copy  geometry->theMap to a real map to order it....
  std::map<unsigned int, GeomDet*> theMap;
  std::copy(geometry->theMap.begin(), geometry->theMap.end(), std::inserter(theMap,theMap.begin()));
  unsigned int nAPE = 0;
  for ( std::map<unsigned int, GeomDet*>::const_iterator iPair = theMap.begin(); 
		iPair != theMap.end(); ++iPair, ++iAlign, ++iAlignError )
	{
	  // Check DetIds
	  if ( (*iPair).first != (*iAlign).rawId() )
	    throw cms::Exception("GeometryMismatch") 
	      << "DetId mismatch between geometry (rawId=" << (*iPair).first
	      << ") and alignments (rawId=" << (*iAlign).rawId();
	  
	  if ( (*iPair).first != (*iAlignError).rawId() )
	    throw cms::Exception("GeometryMismatch") 
	      << "DetId mismatch between geometry (rawId=" << (*iPair).first
	      << ") and alignment errors (rawId=" << (*iAlignError).rawId();

	  // Apply global correction
	  CLHEP::Hep3Vector positionHep = globalRotation * CLHEP::Hep3Vector( (*iAlign).translation() ) + globalShift;
	  CLHEP::HepRotation rotationHep = globalRotation * CLHEP::HepRotation( (*iAlign).rotation() );

	  // Define new position/rotation objects and apply
	  Surface::PositionType position( positionHep.x(), positionHep.y(), positionHep.z() );
	  Surface::RotationType rotation( rotationHep.xx(), rotationHep.xy(), rotationHep.xz(), 
					  rotationHep.yx(), rotationHep.yy(), rotationHep.yz(), 
					  rotationHep.zx(), rotationHep.zy(), rotationHep.zz() );
	  GeomDet* iGeomDet = (*iPair).second;
	  this->setGeomDetPosition( *iGeomDet, position, rotation );

	  // Alignment Position Error only if non-zero to save memory
	  GlobalError error( (*iAlignError).matrix() );
	  if ( error.cxx() || error.cyy() || error.czz() ||
	       error.cyx() || error.czx() || error.czy() ||
               iGeomDet->alignmentPositionError() ) {
	    // FIXME (AM): The check on the existence of a pointer to an AlignmentPositionError
	    // in iGoemDet is needed to make sure that a previously set APE is reset to all zeros
	    // in case the new APE is all zero. Ideally the checking of an all zero value APE
	    // should go into GeomDet::setAlignmentPositionError.
	    
	    // Apply transformation to APE
	    //
	    //AlgebraicSymMatrix as(3,0);
	    //as[0][0] = error.cxx();
	    //as[1][0] = error.cyx(); as[1][1] = error.cyy();
	    //as[2][0] = error.czx(); as[2][1] = error.czy(); as[2][2] = error.czz();
	    
	    //AlgebraicMatrix am(3,3);
	    //am[0][0] = globalRotation.xx(); am[0][1] = globalRotation.xy(); am[0][2] = globalRotation.xz();
	    //am[1][0] = globalRotation.yx(); am[1][1] = globalRotation.yy(); am[1][2] = globalRotation.yz();
	    //am[2][0] = globalRotation.zx(); am[2][1] = globalRotation.zy(); am[2][2] = globalRotation.zz();
	    //as = as.similarityT( am );
	    
	    //GlobalError newError( as );
	    //AlignmentPositionError ape( newError );

	    AlignmentPositionError ape( error );
	    this->setAlignmentPositionError( *iGeomDet, ape );
	    ++nAPE;
	  }
	}

  edm::LogInfo("Alignment") << "@SUB=GeometryAligner::applyAlignments" 
			    << "Finished to apply " << theMap.size() << " alignments with "
			    << nAPE << " non-zero APE.";
}

void GeometryAligner::removeGlobalTransform( const Alignments* alignments,
                                             const AlignmentErrors* alignmentErrors,
                                             const AlignTransform& globalCoordinates,
                                             Alignments* newAlignments,
                                             AlignmentErrors* newAlignmentErrors )
{
  edm::LogInfo("Alignment") << "@SUB=GeometryAligner::removeGlobalTransform" 
			    << "Starting to remove global position from alignments and errors";
  
  if ( alignments->m_align.size() != alignmentErrors->m_alignError.size() )
    throw cms::Exception("GeometryMismatch") 
      << "Size mismatch between alignments (size=" << alignments->m_align.size()
      << ") and alignment errors (size=" << alignmentErrors->m_alignError.size() << ")";
  
  const AlignTransform::Translation &globalShift = globalCoordinates.translation();
  const AlignTransform::Rotation globalRotation = globalCoordinates.rotation(); // by value!
  const AlignTransform::Rotation inverseGlobalRotation = globalRotation.inverse();
  
  AlignTransform::Translation newPosition;
  AlignTransform::Rotation newRotation;

  std::vector<AlignTransform>::const_iterator iAlign = alignments->m_align.begin();
  std::vector<AlignTransformError>::const_iterator iAlignError = alignmentErrors->m_alignError.begin();
  unsigned int nAPE = 0;
  for ( iAlign = alignments->m_align.begin();
        iAlign != alignments->m_align.end();
        ++iAlign, ++iAlignError ) {
    
    // Remove global position transformation from alignment
    newPosition = inverseGlobalRotation * ( (*iAlign).translation() - globalShift );
    newRotation = inverseGlobalRotation * (*iAlign).rotation();
    
    newAlignments->m_align.push_back( AlignTransform(newPosition,
                                                     newRotation,
                                                     (*iAlign).rawId()) );
    
    // Don't remove global position transformation from APE
    // as it wasn't applied. Just fill vector with original
    // values
    GlobalError error( (*iAlignError).matrix() );
    newAlignmentErrors->m_alignError.push_back( AlignTransformError( error.matrix(),
								     (*iAlignError).rawId() ) );
    if ( error.cxx() || error.cyy() || error.czz() ||
	 error.cyx() || error.czx() || error.czy() ) {
      ++nAPE;
    }

    // Code that removes the global postion transformation
    // from the APE.
    //     
    //AlgebraicSymMatrix as(3,0);
    //as[0][0] = error.cxx();
    //as[1][0] = error.cyx(); as[1][1] = error.cyy();
    //as[2][0] = error.czx(); as[2][1] = error.czy(); as[2][2] = error.czz();
    
    //AlgebraicMatrix am(3,3);
    //am[0][0] = inverseGlobalRotation.xx(); am[0][1] = inverseGlobalRotation.xy(); am[0][2] = inverseGlobalRotation.xz();
    //am[1][0] = inverseGlobalRotation.yx(); am[1][1] = inverseGlobalRotation.yy(); am[1][2] = inverseGlobalRotation.yz();
    //am[2][0] = inverseGlobalRotation.zx(); am[2][1] = inverseGlobalRotation.zy(); am[2][2] = inverseGlobalRotation.zz();
    //as = as.similarityT( am );
    
    //GlobalError newError( as );
    //newAlignmentErrors->m_alignError.push_back( AlignTransformError( newError.matrix(),
    //                                                                 (*iAlignError).rawId() ) );
    //++nAPE;
  }

  edm::LogInfo("Alignment") << "@SUB=GeometryAligner::removeGlobalTransform" 
			    << "Finished to remove global transformation from " 
			    << alignments->m_align.size() << " alignments with "
			    << nAPE << " non-zero APE.";
}

#endif
