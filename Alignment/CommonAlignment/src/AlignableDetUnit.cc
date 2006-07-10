#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"

#include <boost/cstdint.hpp>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/Alignment/interface/AlignTransformError.h"

//__________________________________________________________________________________________________
void AlignableDetUnit::move( const GlobalVector& displacement) 
{
  
  if ( misalignmentActive() ) 
    {
	  DetPositioner::moveGeomDet( *theGeomDetUnit, displacement );
      theDisplacement += displacement;
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
      DetPositioner::rotateGeomDet( *theGeomDetUnit, rotation );
      theRotation *= rotation;
    }
  else 
	edm::LogError("NoMisalignment") 
	  << "AlignableDetUnit: Misalignment currently deactivated"
	  << " - no rotation done";

}


//__________________________________________________________________________________________________
void AlignableDetUnit::setAlignmentPositionError(const AlignmentPositionError& ape)
{

  DetPositioner::setAlignmentPositionError( *theGeomDetUnit, ape );

}


//__________________________________________________________________________________________________
void AlignableDetUnit::addAlignmentPositionError(const AlignmentPositionError& ape )
{

  AlignmentPositionError* apePtr = theGeomDetUnit->alignmentPositionError();
  if ( apePtr != 0 )
    this->setAlignmentPositionError( (*apePtr) += ape );
  else 
    this->setAlignmentPositionError( ape );

}


//__________________________________________________________________________________________________
void AlignableDetUnit::addAlignmentPositionErrorFromRotation(const RotationType& rot ) 
{

  float xWidth = theGeomDetUnit->surface().bounds().width();
  float yLength = theGeomDetUnit->surface().bounds().length();

  // average error calculated by movement of a local point at
  // (xWidth/2,yLength/2,0) caused by the rotation rot
  const GlobalVector localPositionVector = this->globalPosition()
	- this->surface().toGlobal( Local3DPoint(xWidth/2.0, yLength/2.0, 0.) );

  LocalVector::BasicVectorType lpvgf = localPositionVector.basicVector();
  GlobalVector gv( rot.multiplyInverse(lpvgf) - lpvgf );

  AlignmentPositionError  ape( gv.x(),gv.y(),gv.z() );

  AlignmentPositionError* apePtr = theGeomDetUnit->alignmentPositionError();
  if ( apePtr != 0 )
    this->setAlignmentPositionError( (*apePtr) += ape );
  else 
    this->setAlignmentPositionError( ape );

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

  // create auxiliary data used for switching (if not yet done)
  if ( !theModifiedPosition ) 
    {
      theModifiedPosition = new GlobalPoint(theOriginalPosition);
      theModifiedRotation = new RotationType(theOriginalRotation);
      theReactivatedPosition = theModifiedPosition;
      theReactivatedRotation = theModifiedRotation;
    }

  //
  // save current position if change since construction or last reactivation
  // (otherwise keep "nominal" modified position / rotation to avoid adding
  // numerical errors). Could be removed if DetUnit::setPosition and
  // DetUnit::setRotation were available.
  //
  *theModifiedPosition = globalPosition();
  *theModifiedRotation = globalRotation();
  //
  // set position and rotation to original values
  //
  this->setGeomDetPosition( *theGeomDetUnit, theOriginalPosition, theOriginalRotation );

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

  theMisalignmentActive = true;

  // set position and rotation to modified values
  this->setGeomDetPosition(*theGeomDetUnit, *theModifiedPosition, *theModifiedRotation );

  // save position and rotation after reactivation (for check at next deactivation)
  *theReactivatedPosition = globalPosition();
  *theReactivatedRotation = globalRotation();

}



//__________________________________________________________________________________________________
void AlignableDetUnit::dump() const
{

    edm::LogInfo("AlignableDump") 
	<< " AlignableDetUnit has position = " << this->globalPosition() 
	<< ", orientation:" << std::endl << this->globalRotation() << std::endl
	<< " total displacement and rotation: " << theDisplacement << std::endl
	<< theRotation;

}

//__________________________________________________________________________________________________
Alignments* AlignableDetUnit::alignments() const
{

  Alignments* m_alignments = new Alignments();
  RotationType rot( theGeomDetUnit->rotation() );
  
  // Get alignments (position, rotation, detId)
  Hep3Vector clhepVector( globalPosition().x(), globalPosition().y(), globalPosition().z() );
  HepRotation clhepRotation( HepRep3x3( rot.xx(), rot.xy(), rot.xz(),
										rot.yx(), rot.yy(), rot.yz(),
										rot.zx(), rot.zy(), rot.zz() ) );
  uint32_t detId = this->geomDetUnit()->geographicalId().rawId();

  AlignTransform transform( clhepVector, clhepRotation, detId );

  // Add to alignments container
  m_alignments->m_align.push_back( transform );

  return m_alignments;

}


//__________________________________________________________________________________________________
AlignmentErrors* AlignableDetUnit::alignmentErrors() const
{
  
  AlignmentErrors* m_alignmentErrors = new AlignmentErrors();
  
  uint32_t detId = this->geomDetUnit()->geographicalId().rawId();
 
  HepSymMatrix clhepSymMatrix(3,0);
  if ( this->geomDetUnit()->alignmentPositionError() ) // Might not be set
	clhepSymMatrix = 
	  this->geomDetUnit()->alignmentPositionError()->globalError().matrix();
  
  AlignTransformError transformError( clhepSymMatrix, detId );
  
  m_alignmentErrors->m_alignError.push_back( transformError );
  
  return m_alignmentErrors;
  

}
