#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"


/// Default constructor
AlignableComposite::AlignableComposite() : 
  theGeomDet(0),
  theSurface( PositionType(0,0,0), RotationType()) {}


/// Constructor from GeomDet
AlignableComposite::AlignableComposite( GeomDet* geomDet ) : 
  theGeomDet( geomDet ),
  theSurface( geomDet->surface().position(), geomDet->surface().rotation() ) 
{}


/// Move components and associated geomdet (if available)
void AlignableComposite::move( const GlobalVector& displacement) 
{
  
  moveAlignableOnly( displacement );

  // Movement is done through the DetPositioner interface
  if ( this->geomDet() ) this->moveGeomDet( *theGeomDet, displacement );

}


/// Move components only 
void AlignableComposite::moveAlignableOnly( const GlobalVector& displacement) 
{

  std::vector<Alignable*> comp = this->components();
  for ( std::vector<Alignable*>::iterator i=comp.begin(); i!=comp.end(); i++ )
    (**i).move( displacement);

  this->addDisplacement( displacement );
  theSurface.move( displacement );

}


/// Local movement of given component
void 
AlignableComposite::moveComponentLocal( 
				       int i, 
				       const LocalVector& localDisplacement
				       )
{
  if (i >= size() ) 
    throw cms::Exception("LogicError")
      << "Al..Composite index (" << i << ") out of range";

  std::vector<Alignable*> comp = this->components();
  comp[i]->move( this->surface().toGlobal( localDisplacement ) ) ;

}


/// Global rotation of components and geomdet (if available)
void AlignableComposite::rotateInGlobalFrame( 
				 const RotationType& rotation) 
{

  rotateAlignableOnly( rotation );

  // Rotation is performed through the DetPostion interface
  if ( this->geomDet() )  this->rotateGeomDet( *theGeomDet, rotation );

}


/// Rotate components only
void AlignableComposite::rotateAlignableOnly( const RotationType& rotation )
{

  std::vector<Alignable*> comp = this->components();
  
  GlobalPoint  myPosition = this->globalPosition();
  
  for ( std::vector<Alignable*>::iterator i=comp.begin(); i!=comp.end(); i++ )
    {
    
      // because each component might have its own local frame oriented in 
      // whatever direction...I'd normally have to transfrom each rotation
      // into a rotation in the local frame of the component if I want to
      // apply the resulting rotation to the component. Same thing holds true
      // for the calculating the resulting movement. Here I do have to first
      // transform the rotation into the local frame IF I operate really with
      // local position, in the local frame. However...it's much simpler to
      // calculate the local position give in Koordinates of the GLOBAL frame
      // then just apply the rotation matrix given in the GLOBAL frame as well.
      // ONLY this is somewhat tricky...as Teddy's frames don't like this 
      // kind of mixing...
    
      // Rotations are defined for "Basic3DVector" types, without any FrameTAG,
      // because Rotations usually switch between different frames. You get
      // this by using the method .basicVector()
    
      // localPosition = globalPosition (Component) - globalPosition(Composite)
      // moveVector = rotated localPosition  - original localposition
      //LocalVector localPositionVector = (**i).globalPosition()-myPosition;
    
    
      // Local Position given in coordinates of the GLOBAL Frame
      const GlobalVector localPositionVector = (**i).globalPosition()-myPosition;
      GlobalVector::BasicVectorType lpvgf = localPositionVector.basicVector();

      // rotate with GLOBAL rotation matrix  and subtract => moveVector in 
      // global Coordinates
      // apparently... you have to use the inverse of the rotation here
      // (rotate the VECTOR rather than the frame) 
      GlobalVector moveVector(rotation.multiplyInverse(lpvgf) - lpvgf);
    
    
      (**i).move( moveVector );
      (**i).rotateInGlobalFrame( rotation );

    }

  this->addRotation( rotation );
  
  theSurface.rotate( rotation );

}   


/// Set the alignment position error of all components to given error
void 
AlignableComposite::setAlignmentPositionError( const 
					       AlignmentPositionError& ape
					       )
{

  std::vector<Alignable*> comp = this->components();
  for ( std::vector<Alignable*>::const_iterator i=comp.begin(); i!=comp.end(); i++) 
    {
      (*i)->setAlignmentPositionError(ape);
    }

}


/// Add given error to alignment position error of all components
void 
AlignableComposite::addAlignmentPositionError( const 
					       AlignmentPositionError& ape 
					       )
{
  std::vector<Alignable*> comp = this->components();
  for ( std::vector<Alignable*>::const_iterator i=comp.begin(); 
	i!=comp.end(); i++) 
    (*i)->addAlignmentPositionError(ape);

}


/// Add position error to all components as resulting from given global rotation
void 
AlignableComposite::addAlignmentPositionErrorFromRotation( const
							   RotationType& rotation
							   )
{

  std::vector<Alignable*> comp = this->components();

  GlobalPoint  myPosition=this->globalPosition();

  for ( std::vector<Alignable*>::const_iterator i=comp.begin(); 
	i!=comp.end(); i++) 
    {

      // It is just similar to to the "movement" that results to the components
      // when the composite is rotated. 
      // Local Position given in coordinates of the GLOBAL Frame
      const GlobalVector localPositionVector = (**i).globalPosition()-myPosition;
      //    Basic3DVector<float> lpvgf = localPositionVector.basicVector();
      GlobalVector::BasicVectorType lpvgf = localPositionVector.basicVector();

      // rotate with GLOBAL rotation matrix  and subtract => moveVector in 
      // global Koordinates
      // apparently... you have to use the inverse of the rotation here
      // (rotate the VECTOR rather than the frame) 
      GlobalVector moveVector( rotation.multiplyInverse(lpvgf) - lpvgf );    
      
      AlignmentPositionError ape( moveVector.x(),moveVector.y(),
				  moveVector.z() );
      (*i)->addAlignmentPositionError( ape );
      (*i)->addAlignmentPositionErrorFromRotation( rotation );
    }

}


/// Add position error to all components as resulting from given local rotation
void 
AlignableComposite::addAlignmentPositionErrorFromLocalRotation( const
							        RotationType& rot
								)
{
  RotationType globalRot = globalRotation().multiplyInverse(rot*globalRotation());
  this->addAlignmentPositionErrorFromRotation(globalRot);
}


/// Unset misalignment
void AlignableComposite::deactivateMisalignment ()
{

  // Check status
  if ( !misalignmentActive() ) 
    {
      edm::LogError("AlreadyDone") 
	<< __FILE__ ": "
	<< "Attempt to deactivate misalignment a second time.";
      return;
    }
  
  // Forward to components
  std::vector<Alignable*> components(components());
  for ( std::vector<Alignable*>::iterator i=components.begin();
	i!=components.end(); i++ ) 
      (**i).deactivateMisalignment();

  theMisalignmentActive = false;

}



/// Re-set misalignment
void AlignableComposite::reactivateMisalignment ()
{

  // Check status
  if ( misalignmentActive() ) {
    edm::LogError("AlreadyDone")
      << __FILE__ ": "
      << "Attempt to reactivate misalignment a second time";
    return;
  }

  // Forward to components
  std::vector<Alignable*> components(components());
  for ( std::vector<Alignable*>::iterator i=components.begin();
	i!=components.end(); i++ )  
    (**i).reactivateMisalignment();

  theMisalignmentActive = true;

}

