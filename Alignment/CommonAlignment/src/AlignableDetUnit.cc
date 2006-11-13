#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"

#include <boost/cstdint.hpp>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/Alignment/interface/AlignTransformError.h"

//__________________________________________________________________________________________________
AlignableDetUnit::AlignableDetUnit( const GeomDetUnit* geomDetUnit ) :
  theOriginalSurface( geomDetUnit->surface().position(), geomDetUnit->surface().rotation() ),
  theSurface( geomDetUnit->surface().position(), geomDetUnit->surface().rotation() ),
  theSavedSurface( geomDetUnit->surface().position(), geomDetUnit->surface().rotation() ),
  theAlignmentPositionError(0)
{
  
  // Also store width and length of geomdet surface
  theWidth  = geomDetUnit->surface().bounds().width();
  theLength = geomDetUnit->surface().bounds().length();

  this->setDetId( geomDetUnit->geographicalId() );

}

//__________________________________________________________________________________________________
AlignableDetUnit::~AlignableDetUnit() {
  delete theAlignmentPositionError;
}

//__________________________________________________________________________________________________
void AlignableDetUnit::recursiveComponents(std::vector<Alignable*> &result) const
{
  // There are no subcomponents...
  return;
}

//__________________________________________________________________________________________________
void AlignableDetUnit::move( const GlobalVector& displacement) 
{
  
  if ( misalignmentActive() ) 
    {
      theSurface.move( displacement );
      this->addDisplacement( displacement );
    }
  else 
    edm::LogError("NoMisalignment")
      << "AlignableDetUnit: Misalignment currently deactivated"
      << " - no move done";

}


//__________________________________________________________________________________________________
/// Rotation intepreted such, that the orientation of the rotation
/// axis is w.r.t. to the global coordinate system. Rotation is
/// relative to current orientation
void AlignableDetUnit::rotateInGlobalFrame( const RotationType& rotation) 
{
  
  if ( misalignmentActive() ) 
    {
      theSurface.rotate( rotation );
      this->addRotation( rotation );
    }
  else 
    edm::LogError("NoMisalignment") 
      << "AlignableDetUnit: Misalignment currently deactivated"
      << " - no rotation done";
  
}


//__________________________________________________________________________________________________
void AlignableDetUnit::setAlignmentPositionError(const AlignmentPositionError& ape)
{

  if ( !theAlignmentPositionError ) 
    theAlignmentPositionError = new AlignmentPositionError( ape );
  else
    *theAlignmentPositionError = ape;

}


//__________________________________________________________________________________________________
void AlignableDetUnit::addAlignmentPositionError(const AlignmentPositionError& ape )
{

  if ( !theAlignmentPositionError )
    this->setAlignmentPositionError( ape );
  else 
    this->setAlignmentPositionError( *theAlignmentPositionError += ape );

}


//__________________________________________________________________________________________________
void AlignableDetUnit::addAlignmentPositionErrorFromRotation(const RotationType& rot ) 
{


  // average error calculated by movement of a local point at
  // (xWidth/2,yLength/2,0) caused by the rotation rot
  const GlobalVector localPositionVector = this->globalPosition()
    - this->surface().toGlobal( Local3DPoint(theWidth/2.0, theLength/2.0, 0.) );

  LocalVector::BasicVectorType lpvgf = localPositionVector.basicVector();
  GlobalVector gv( rot.multiplyInverse(lpvgf) - lpvgf );

  AlignmentPositionError  ape( gv.x(),gv.y(),gv.z() );
  this->addAlignmentPositionError( ape );

}


//__________________________________________________________________________________________________
void AlignableDetUnit::addAlignmentPositionErrorFromLocalRotation(const RotationType& rot )
{

  RotationType globalRot = globalRotation().multiplyInverse(rot*globalRotation());
  this->addAlignmentPositionErrorFromRotation(globalRot);

}


//__________________________________________________________________________________________________
void AlignableDetUnit::deactivateMisalignment ()
{

  // check status
  if ( !misalignmentActive() ) 
    {
      edm::LogError("AlreadyDone") << "Misalignment already inactive.";
      return;
    }

  // Set back to original surface
  theSavedSurface = theSurface;
  theSurface = theOriginalSurface;

  theMisalignmentActive = false;

}


//__________________________________________________________________________________________________
void AlignableDetUnit::reactivateMisalignment ()
{

  // check status
  if ( misalignmentActive() ) 
    {
      edm::LogError("AlreadyDone") << "Misalignment already active";
      return;
    }

  // Set to saved surface
  theSurface = theSavedSurface;

  theMisalignmentActive = true;


}


//__________________________________________________________________________________________________
void AlignableDetUnit::dump() const
{

  edm::LogInfo("AlignableDump") 
    << " AlignableDetUnit has position = " << this->globalPosition() 
    << ", orientation:" << std::endl << this->globalRotation() << std::endl
    << " total displacement and rotation: " << this->displacement() << std::endl
    << this->rotation();

}


//__________________________________________________________________________________________________
Alignments* AlignableDetUnit::alignments() const
{

  Alignments* m_alignments = new Alignments();
  RotationType rot( this->globalRotation() );
  
  // Get alignments (position, rotation, detId)
  Hep3Vector clhepVector( globalPosition().x(), globalPosition().y(), globalPosition().z() );
  HepRotation clhepRotation( HepRep3x3( rot.xx(), rot.xy(), rot.xz(),
										rot.yx(), rot.yy(), rot.yz(),
										rot.zx(), rot.zy(), rot.zz() ) );
  uint32_t detId = this->geomDetId().rawId();
  
  AlignTransform transform( clhepVector, clhepRotation, detId );

  // Add to alignments container
  m_alignments->m_align.push_back( transform );

  return m_alignments;

}


//__________________________________________________________________________________________________
AlignmentErrors* AlignableDetUnit::alignmentErrors() const
{
  
  AlignmentErrors* m_alignmentErrors = new AlignmentErrors();
  
  uint32_t detId = this->geomDetId().rawId();
 
  HepSymMatrix clhepSymMatrix(3,0);
  if ( theAlignmentPositionError ) // Might not be set
    clhepSymMatrix = theAlignmentPositionError->globalError().matrix();
  
  AlignTransformError transformError( clhepSymMatrix, detId );
  
  m_alignmentErrors->m_alignError.push_back( transformError );
  
  return m_alignmentErrors;
  

}
