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
  void applyAlignments( C* geometry, const Alignments* alignments, 
						const AlignmentErrors* alignmentErrors );

};


template<class C>
void GeometryAligner::applyAlignments( C* geometry, const Alignments* alignments,
									   const AlignmentErrors* alignmentErrors )
{

  edm::LogInfo("Starting") << "Starting to apply alignments";

  // Preliminary checks (or we can't loop!)
  if ( alignments->m_align.size() != geometry->theMap.size() )
	throw cms::Exception("GeometryMismatch") 
	  << "Size mismatch between geometry (size=" << geometry->theMap.size() 
	  << ") and alignments (size=" << alignments->m_align.size() << ")";
  if ( alignments->m_align.size() != alignmentErrors->m_alignError.size() )
	throw cms::Exception("GeometryMismatch") 
	  << "Size mismatch between geometry (size=" << geometry->theMap.size() 
	  << ") and alignment errors (size=" << alignmentErrors->m_alignError.size() << ")";

  // Parallel loop on alignments, alignment errors and geomdets
  std::vector<AlignTransform>::const_iterator iAlign = alignments->m_align.begin();
  std::vector<AlignTransformError>::const_iterator 
	iAlignError = alignmentErrors->m_alignError.begin();
  //copy  geometry->theMap to a real map to order it....
  std::map<unsigned int, GeomDet*> theMap;
  std::copy(geometry->theMap.begin(), geometry->theMap.end(), std::inserter(theMap,theMap.begin()));
  for ( std::map<unsigned int, GeomDet*>::const_iterator iPair = theMap.begin(); 
		iPair != theMap.end(); iPair++, iAlign++, iAlignError++ )
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

	  // Define new quantities from alignments
	  Surface::PositionType position( (*iAlign).translation().x(), 
					  (*iAlign).translation().y(), 
					  (*iAlign).translation().z() );
	  Surface::RotationType 
	    rotation( (*iAlign).rotation().xx(), (*iAlign).rotation().xy(), (*iAlign).rotation().xz(),
		      (*iAlign).rotation().yx(), (*iAlign).rotation().yy(), (*iAlign).rotation().yz(),
		      (*iAlign).rotation().zx(), (*iAlign).rotation().zy(), (*iAlign).rotation().zz() );

	  GlobalError error( (*iAlignError).matrix() );
	  AlignmentPositionError ape( error );

	  // Apply new quantities
	  GeomDet* iGeomDet = (*iPair).second;
	  DetPositioner::setGeomDetPosition( *iGeomDet, position, rotation );
	  DetPositioner::setAlignmentPositionError( *iGeomDet, ape );

	}

  edm::LogInfo("Done") << "Finished to apply alignments";
}



#endif
