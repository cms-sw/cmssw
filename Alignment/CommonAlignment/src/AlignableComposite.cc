// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/Alignment/interface/AlignTransform.h" 
#include "CondFormats/Alignment/interface/AlignTransformError.h" 

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"



//__________________________________________________________________________________________________
AlignableComposite::AlignableComposite() : 
  theSurface( PositionType(0,0,0), RotationType()) {}


//__________________________________________________________________________________________________
AlignableComposite::AlignableComposite( const GeomDet* geomDet ) : 
  theSurface( geomDet->surface().position(), geomDet->surface().rotation() ) 
{
  
  this->setDetId( geomDet->geographicalId() );

}

//__________________________________________________________________________________________________
void AlignableComposite::recursiveComponents(std::vector<Alignable*> &result) const
{

  std::vector<Alignable*> components = this->components();
  if (components.size() <= 1) return; // Non-glued AlignableDets contain themselves

  for (std::vector<Alignable*>::const_iterator iter = components.begin();
       iter != components.end(); ++iter) {
    result.push_back(*iter); // could use std::copy(..), but here we build a real hierarchy
    (*iter)->recursiveComponents(result);
  }
}

//__________________________________________________________________________________________________
void AlignableComposite::move( const GlobalVector& displacement ) 
{
  
  // Move components
  std::vector<Alignable*> comp = this->components();
  for ( std::vector<Alignable*>::iterator i=comp.begin(); i!=comp.end(); i++ )
    (**i).move( displacement);

  // Move surface
  this->addDisplacement( displacement );
  theSurface.move( displacement );

}


//__________________________________________________________________________________________________
void AlignableComposite::moveComponentsLocal( const LocalVector& localDisplacement )
{

  this->move( this->surface().toGlobal(localDisplacement) );

}

//__________________________________________________________________________________________________
void AlignableComposite::moveComponentLocal( const int i, const LocalVector& localDisplacement )
{

  if (i >= size() ) 
    throw cms::Exception("LogicError")
      << "AlignableComposite index (" << i << ") out of range";

  std::vector<Alignable*> comp = this->components();
  comp[i]->move( this->surface().toGlobal( localDisplacement ) );

}


//__________________________________________________________________________________________________
/// Rotation intepreted such, that the orientation of the rotation
/// axis is w.r.t. to the global coordinate system. This, however, does NOT
/// mean the center of the rotation. This is simply taken as the center of
/// the Alignable-object 
void AlignableComposite::rotateInGlobalFrame( const RotationType& rotation )
{
  
  std::vector<Alignable*> comp = this->components();
  
  GlobalPoint  myPosition = this->globalPosition();
  
  for ( std::vector<Alignable*>::iterator i=comp.begin(); i!=comp.end(); i++ )
    {
      
      // It is much simpler to calculate the local position given in coordinates 
      // of the GLOBAL frame and then just apply the rotation matrix given in the 
      // GLOBAL frame as well. ONLY this is somewhat tricky... as Teddy's frames 
      // don't like this kind of mixing...
      
      // Rotations are defined for "Basic3DVector" types, without any FrameTAG,
      // because Rotations usually switch between different frames. You get
      // this by using the method .basicVector()
    
      // localPosition = globalPosition (Component) - globalPosition(Composite)
      // moveVector = rotated localPosition  - original localposition
      // LocalVector localPositionVector = (**i).globalPosition()-myPosition;
    
    
      // Local Position given in coordinates of the GLOBAL Frame
      const GlobalVector localPositionVector = (**i).globalPosition() - myPosition;
      GlobalVector::BasicVectorType lpvgf = localPositionVector.basicVector();

      // rotate with GLOBAL rotation matrix  and subtract => moveVector in 
      // global Coordinates
      // apparently... you have to use the inverse of the rotation here
      // (rotate the VECTOR rather than the frame) 
      GlobalVector moveVector( rotation.multiplyInverse(lpvgf) - lpvgf );
    
    
      (**i).move( moveVector );
      (**i).rotateInGlobalFrame( rotation );

    }

  this->addRotation( rotation );
  
  theSurface.rotate( rotation );

}



//__________________________________________________________________________________________________
/// Set the alignment position error of all components to given error
void AlignableComposite::setAlignmentPositionError( const AlignmentPositionError& ape )
{

  // Since no geomDet is attached, alignable composites do not have an APE
  // The APE is, therefore, just propagated down
  std::vector<Alignable*> comp = this->components();
  for ( std::vector<Alignable*>::const_iterator i=comp.begin(); i!=comp.end(); i++) 
    {
      (*i)->setAlignmentPositionError(ape);
    }

}


//__________________________________________________________________________________________________
void 
AlignableComposite::addAlignmentPositionError( const AlignmentPositionError& ape )
{

  std::vector<Alignable*> comp = this->components();
  for ( std::vector<Alignable*>::const_iterator i=comp.begin(); i!=comp.end(); i++) 
    (*i)->addAlignmentPositionError(ape);

}


//__________________________________________________________________________________________________
/// Adds the AlignmentPositionError (in x,y,z coordinates) that would result
/// on the various components from a possible Rotation of a composite the 
/// rotation matrix is in interpreted in GLOBAL coordinates
void AlignableComposite::addAlignmentPositionErrorFromRotation( const RotationType& rotation )
{

  std::vector<Alignable*> comp = this->components();

  GlobalPoint  myPosition=this->globalPosition();

  for ( std::vector<Alignable*>::const_iterator i=comp.begin(); i!=comp.end(); i++ )
    {

      // It is just similar to to the "movement" that results to the components
      // when the composite is rotated. 
      // Local Position given in coordinates of the GLOBAL Frame
      const GlobalVector localPositionVector = (**i).globalPosition()-myPosition;
      GlobalVector::BasicVectorType lpvgf = localPositionVector.basicVector();

      // rotate with GLOBAL rotation matrix  and subtract => moveVector in global coordinates
      // apparently... you have to use the inverse of the rotation here
      // (rotate the VECTOR rather than the frame) 
      GlobalVector moveVector( rotation.multiplyInverse(lpvgf) - lpvgf );    
      
      AlignmentPositionError ape( moveVector.x(), moveVector.y(), moveVector.z() );
      (*i)->addAlignmentPositionError( ape );
      (*i)->addAlignmentPositionErrorFromRotation( rotation );
	  
    }

}


//__________________________________________________________________________________________________
/// Adds the AlignmentPositionError (in x,y,z coordinates) that would result
/// on the various components from a possible Rotation of a composite the 
/// rotation matrix is in interpreted in LOCAL  coordinates of the composite
void AlignableComposite::addAlignmentPositionErrorFromLocalRotation( const RotationType& rot )
{

  RotationType globalRot = globalRotation().multiplyInverse(rot*globalRotation());
  this->addAlignmentPositionErrorFromRotation(globalRot);

}


//__________________________________________________________________________________________________
void AlignableComposite::deactivateMisalignment ()
{

  // Check status
  if ( !misalignmentActive() ) 
    {
      edm::LogError("AlreadyDone") << "Attempt to deactivate misalignment again.";
      return;
    }
  
  // Forward to components
  std::vector<Alignable*> components = this->components();
  for ( std::vector<Alignable*>::iterator i=components.begin(); i!=components.end(); i++ ) 
    (**i).deactivateMisalignment();
  
  theMisalignmentActive = false;
  
}



//__________________________________________________________________________________________________
void AlignableComposite::reactivateMisalignment ()
{
  
  // Check status
  if ( misalignmentActive() ) {
    edm::LogError("AlreadyDone") << "Attempt to reactivate misalignment again";
    return;
  }
  
  // Forward to components
  std::vector<Alignable*> components = this->components();
  for ( std::vector<Alignable*>::iterator i=components.begin(); i!=components.end(); i++ )  
    (**i).reactivateMisalignment();

  theMisalignmentActive = true;

}

//__________________________________________________________________________________________________
void AlignableComposite::dump( void ) const
{

  // A simple printout method. Could be specialized in the implementation classes.

  std::vector<Alignable*> comp = this->components();

  // Dump this
  edm::LogInfo("AlignableDump") 
    << " Alignable of type " << this->alignableObjectId() 
    << " has " << comp.size() << " components" << std::endl
    << " position = " << this->globalPosition() << ", orientation:" << std::endl
    << this->globalRotation();

  // Dump components
  for ( std::vector<Alignable*>::iterator i=comp.begin(); i!=comp.end(); i++ )
    (*i)->dump();

}



//__________________________________________________________________________________________________
Alignments* AlignableComposite::alignments( void ) const
{

  // Recursively call alignments, until we get to an AlignableDetUnit
  std::vector<Alignable*> comp = this->components();

  Alignments* m_alignments = new Alignments();

  // Add components recursively
  for ( std::vector<Alignable*>::iterator i=comp.begin(); i!=comp.end(); i++ )
    {
      Alignments* tmpAlignments = (*i)->alignments();
      std::copy( tmpAlignments->m_align.begin(), tmpAlignments->m_align.end(), 
		 std::back_inserter(m_alignments->m_align) );
	  delete tmpAlignments;
    }

  
  return m_alignments;

}


//__________________________________________________________________________________________________
AlignmentErrors* AlignableComposite::alignmentErrors( void ) const
{

  // Recursively call alignmentsErrors, until we get to an AlignableDetUnit
  std::vector<Alignable*> comp = this->components();

  AlignmentErrors* m_alignmentErrors = new AlignmentErrors();

  // Add components recursively
  for ( std::vector<Alignable*>::iterator i=comp.begin(); i!=comp.end(); i++ )
    {
      AlignmentErrors* tmpAlignmentErrors = (*i)->alignmentErrors();
      std::copy( tmpAlignmentErrors->m_alignError.begin(), tmpAlignmentErrors->m_alignError.end(), 
		 std::back_inserter(m_alignmentErrors->m_alignError) );
	  delete tmpAlignmentErrors;
    }

  
  return m_alignmentErrors;

}
