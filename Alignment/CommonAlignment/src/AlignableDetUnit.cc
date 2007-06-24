#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"

//__________________________________________________________________________________________________
AlignableDetUnit::AlignableDetUnit( const GeomDet* geomDet ) :
  Alignable(geomDet),
  theAlignmentPositionError(0)
{
  theDeepComponents.push_back(this);
}

//__________________________________________________________________________________________________
AlignableDetUnit::~AlignableDetUnit() {
  delete theAlignmentPositionError;
}

//__________________________________________________________________________________________________
void AlignableDetUnit::move( const GlobalVector& displacement) 
{
  
  theSurface.move( displacement );
  this->addDisplacement( displacement );

}


//__________________________________________________________________________________________________
void AlignableDetUnit::rotateInGlobalFrame( const RotationType& rotation) 
{
  
  theSurface.rotate( rotation );
  this->addRotation( rotation );
  
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
    - this->surface().toGlobal( Local3DPoint(.5 * surface().width(), .5 * surface().length(), 0.) );

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
