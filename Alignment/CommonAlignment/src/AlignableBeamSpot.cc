/** \file AlignableBeamSpot
 *
 *  Original author: Andreas Mussgiller, August 2010
 *
 *  $Date: 2010/10/26 19:53:53 $
 *  $Revision: 1.2 $
 *  (last update by $Author: flucke $)
 */

#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CLHEP/Vector/RotationInterfaces.h" 
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "Alignment/CommonAlignment/interface/AlignableBeamSpot.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//__________________________________________________________________________________________________
AlignableBeamSpot::AlignableBeamSpot() : 
  Alignable( AlignableBeamSpot::detId().rawId(), AlignableSurface() ), 
  theAlignmentPositionError(0),
  theInitializedFlag(false)
{

}

//__________________________________________________________________________________________________
AlignableBeamSpot::~AlignableBeamSpot()
{
  delete theAlignmentPositionError;
}

//__________________________________________________________________________________________________
void AlignableBeamSpot::move( const GlobalVector& displacement) 
{
  theSurface.move( displacement );
  this->addDisplacement( displacement );
}

//__________________________________________________________________________________________________
void AlignableBeamSpot::rotateInGlobalFrame( const RotationType& rotation) 
{
  theSurface.rotate( rotation );
  this->addRotation( rotation );
}

//__________________________________________________________________________________________________
void AlignableBeamSpot::setAlignmentPositionError(const AlignmentPositionError& ape, bool propagateDown)
{
  if ( !theAlignmentPositionError )
    theAlignmentPositionError = new AlignmentPositionError( ape );
  else 
    *theAlignmentPositionError = ape;
}

//__________________________________________________________________________________________________
void AlignableBeamSpot::addAlignmentPositionError(const AlignmentPositionError& ape, bool propagateDown)
{
  if ( !theAlignmentPositionError ) {
    theAlignmentPositionError = new AlignmentPositionError( ape );
  } else {
    *theAlignmentPositionError += ape;
  }
}

//__________________________________________________________________________________________________
void AlignableBeamSpot::addAlignmentPositionErrorFromRotation(const RotationType& rot,
							      bool propagateDown)
{

}

//__________________________________________________________________________________________________
/// Adds the AlignmentPositionError (in x,y,z coordinates) that would result
/// on the various components from a possible Rotation of a composite the 
/// rotation matrix is in interpreted in LOCAL  coordinates of the composite
void AlignableBeamSpot::addAlignmentPositionErrorFromLocalRotation(const RotationType& rot,
								   bool propagateDown )
{

}

//__________________________________________________________________________________________________
void AlignableBeamSpot::setSurfaceDeformation(const SurfaceDeformation*, bool)
{
  edm::LogInfo("Alignment") << "@SUB=AlignableBeamSpot::setSurfaceDeformation"
			    << "Useless method for beam spot, do nothing.";
}

//__________________________________________________________________________________________________
void AlignableBeamSpot::addSurfaceDeformation(const SurfaceDeformation*, bool)
{
  edm::LogInfo("Alignment") << "@SUB=AlignableBeamSpot::addSurfaceDeformation"
			    << "Useless method for beam spot, do nothing.";
}

//__________________________________________________________________________________________________
void AlignableBeamSpot::dump( void ) const
{
  // Dump this

  LocalVector lv(0.0, 0.0, 1.0);
  GlobalVector gv = theSurface.toGlobal(lv);

  edm::LogInfo("AlignableDump") 
    << " Alignable of type " << this->alignableObjectId() 
    << " has 0 components" << std::endl
    << " position = " << this->globalPosition() << ", orientation:" << std::endl << std::flush
    << this->globalRotation() << std::endl << std::flush
    << " dxdz = " << gv.x()/gv.z() << " dydz = " << gv.y()/gv.z() << std::endl;
}

//__________________________________________________________________________________________________
Alignments* AlignableBeamSpot::alignments() const
{
  Alignments* m_alignments = new Alignments();
  RotationType rot( this->globalRotation() );

  // Get position, rotation, detId
  CLHEP::Hep3Vector clhepVector( globalPosition().x(), globalPosition().y(), globalPosition().z() );
  CLHEP::HepRotation clhepRotation( CLHEP::HepRep3x3( rot.xx(), rot.xy(), rot.xz(),
						      rot.yx(), rot.yy(), rot.yz(),
						      rot.zx(), rot.zy(), rot.zz() ) );
  uint32_t detId = theId;
  
  AlignTransform transform( clhepVector, clhepRotation, detId );
  
  // Add to alignments container
  m_alignments->m_align.push_back( transform );

  return m_alignments;
}

//__________________________________________________________________________________________________
AlignmentErrorsExtended* AlignableBeamSpot::alignmentErrors( void ) const
{
  AlignmentErrorsExtended* m_alignmentErrors = new AlignmentErrorsExtended();

  // Add associated alignment position error
  uint32_t detId = theId;
  CLHEP::HepSymMatrix clhepSymMatrix(6,0);
  if ( theAlignmentPositionError ) // Might not be set
    clhepSymMatrix = asHepMatrix(theAlignmentPositionError->globalError().matrix());
  AlignTransformErrorExtended transformError( clhepSymMatrix, detId );
  m_alignmentErrors->m_alignError.push_back( transformError );
  
  return m_alignmentErrors;
}

void AlignableBeamSpot::initialize(double x, double y, double z,
				   double dxdz, double dydz)
{
  if (theInitializedFlag) return;

  GlobalVector gv(x, y, z);
  theSurface.move(gv);

  double angleY = std::atan(dxdz);
  double angleX = -std::atan(dydz);

  align::RotationType rotY( std::cos(angleY),  0., -std::sin(angleY), 
			    0.,                1.,  0.,
			    std::sin(angleY),  0.,  std::cos(angleY) );

  align::RotationType rotX( 1.,  0.,                0.,
			    0.,  std::cos(angleX),  std::sin(angleX),
			    0., -std::sin(angleX),  std::cos(angleX) );  

  theSurface.rotate(rotY * rotX);

  this->dump();
  
  theInitializedFlag = true;
}
