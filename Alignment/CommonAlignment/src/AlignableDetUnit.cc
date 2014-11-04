#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CLHEP/Vector/RotationInterfaces.h" 
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

//__________________________________________________________________________________________________
AlignableDetUnit::AlignableDetUnit(const GeomDetUnit *geomDetUnit) : // rely on non-NULL pointer!
  Alignable(geomDetUnit->geographicalId().rawId(), geomDetUnit->surface()),
  theAlignmentPositionError(0),
  theSurfaceDeformation(0),
  theCachedSurfaceDeformation(0)
{
  if (geomDetUnit->alignmentPositionError()) { // take over APE from geometry
    // 2nd argument w/o effect:
    this->setAlignmentPositionError(*(geomDetUnit->alignmentPositionError()), false);
  }

  if (geomDetUnit->surfaceDeformation()) { // take over surface modification
    // 2nd argument w/o effect:
    this->setSurfaceDeformation(geomDetUnit->surfaceDeformation(), false);
  }
  
  theDeepComponents.push_back(this);

}

//__________________________________________________________________________________________________
AlignableDetUnit::~AlignableDetUnit()
{
  delete theAlignmentPositionError;
  delete theSurfaceDeformation;
  delete theCachedSurfaceDeformation;
}

//__________________________________________________________________________________________________
void AlignableDetUnit::addComponent( Alignable* /*unused*/)
{
  throw cms::Exception("LogicError") 
    << "AlignableDetUnit cannot have components, but try to add one!";
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
void AlignableDetUnit::setAlignmentPositionError(const AlignmentPositionError& ape,
						 bool /*propagateDown*/)
{

  if ( !theAlignmentPositionError ) 
    theAlignmentPositionError = new AlignmentPositionError( ape );
  else
    *theAlignmentPositionError = ape;

}


//__________________________________________________________________________________________________
void AlignableDetUnit::addAlignmentPositionError(const AlignmentPositionError& ape,
						 bool propagateDown )
{

  if ( !theAlignmentPositionError )
    this->setAlignmentPositionError( ape, propagateDown ); // 2nd argument w/o effect
  else 
    *theAlignmentPositionError += ape;
}


//__________________________________________________________________________________________________
void AlignableDetUnit::addAlignmentPositionErrorFromRotation(const RotationType& rot,
							     bool propagateDown ) 
{

  // average error calculated by movement of a local point at
  // (xWidth/2,yLength/2,0) caused by the rotation rot
  GlobalVector localPositionVector = surface().toGlobal( LocalVector(.5 * surface().width(),
								     .5 * surface().length(),
								     0.) );

  LocalVector::BasicVectorType lpvgf = localPositionVector.basicVector();
  GlobalVector gv( rot.multiplyInverse(lpvgf) - lpvgf );

  AlignmentPositionError  ape( gv.x(),gv.y(),gv.z() );
  this->addAlignmentPositionError( ape, propagateDown ); // 2nd argument w/o effect

}


//__________________________________________________________________________________________________
void AlignableDetUnit::addAlignmentPositionErrorFromLocalRotation(const RotationType& rot,
								  bool propagateDown )
{

  RotationType globalRot = globalRotation().multiplyInverse(rot*globalRotation());
  this->addAlignmentPositionErrorFromRotation(globalRot, propagateDown); // 2nd argument w/o effect

}

//__________________________________________________________________________________________________
void AlignableDetUnit::setSurfaceDeformation(const SurfaceDeformation *deformation,
					      bool /* propagateDown */ )
{
  delete theSurfaceDeformation; // OK for zero pointers
  if (deformation) {
    theSurfaceDeformation = deformation->clone();
  } else {
    theSurfaceDeformation = 0;
  }
}

//__________________________________________________________________________________________________
void AlignableDetUnit::addSurfaceDeformation(const SurfaceDeformation *deformation,
					     bool propagateDown)
{
  if (!deformation) {
    // nothing to do
  } else if (!theSurfaceDeformation) {
    this->setSurfaceDeformation(deformation, propagateDown); // fine since no components
  } else if (!theSurfaceDeformation->add(*deformation)) {
    edm::LogError("Alignment") << "@SUB=AlignableDetUnit::addSurfaceDeformation"
			       << "Cannot add deformation type " << deformation->type()
			       << " to type " << theSurfaceDeformation->type()
			       << ", so erase deformation information.";
    delete theSurfaceDeformation;
    theSurfaceDeformation = 0;
  }
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
  CLHEP::Hep3Vector clhepVector( globalPosition().x(), globalPosition().y(), globalPosition().z() );
  CLHEP::HepRotation clhepRotation( CLHEP::HepRep3x3( rot.xx(), rot.xy(), rot.xz(),
										rot.yx(), rot.yy(), rot.yz(),
										rot.zx(), rot.zy(), rot.zz() ) );
  uint32_t detId = this->geomDetId().rawId();
  
  AlignTransform transform( clhepVector, clhepRotation, detId );

  // Add to alignments container
  m_alignments->m_align.push_back( transform );

  return m_alignments;

}


//__________________________________________________________________________________________________
AlignmentErrorsExtended* AlignableDetUnit::alignmentErrors() const
{
  
  AlignmentErrorsExtended* m_alignmentErrors = new AlignmentErrorsExtended();
  
  uint32_t detId = this->geomDetId().rawId();
 
  CLHEP::HepSymMatrix clhepSymMatrix(6,0);
  if ( theAlignmentPositionError ) // Might not be set
    clhepSymMatrix = asHepMatrix(theAlignmentPositionError->globalError().matrix());
  
  AlignTransformErrorExtended transformError( clhepSymMatrix, detId );
  
  m_alignmentErrors->m_alignError.push_back( transformError );
  
  return m_alignmentErrors;

}


//__________________________________________________________________________________________________
int AlignableDetUnit::surfaceDeformationIdPairs(std::vector<std::pair<int,SurfaceDeformation*> > & result) const
{
  if ( theSurfaceDeformation ) {
    result.push_back( std::pair<int,SurfaceDeformation*>(this->geomDetId().rawId(),theSurfaceDeformation) );
    return 1;
  }
  
  return 0;
}
 
//__________________________________________________________________________________________________
void AlignableDetUnit::cacheTransformation()
{
  theCachedSurface = theSurface;
  theCachedDisplacement = theDisplacement;
  theCachedRotation = theRotation;

  if (theCachedSurfaceDeformation) {
    delete theCachedSurfaceDeformation;
    theCachedSurfaceDeformation = 0;
  }

  if (theSurfaceDeformation)
    theCachedSurfaceDeformation = theSurfaceDeformation->clone();
}

//__________________________________________________________________________________________________
void AlignableDetUnit::restoreCachedTransformation()
{
  theSurface = theCachedSurface;
  theDisplacement = theCachedDisplacement;
  theRotation = theCachedRotation;

  if (theSurfaceDeformation) {
    delete theSurfaceDeformation;
    theSurfaceDeformation = 0;
  }

  if (theCachedSurfaceDeformation) {
    this->setSurfaceDeformation(theCachedSurfaceDeformation, false);
  }
}
