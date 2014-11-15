// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"


//__________________________________________________________________________________________________
AlignableComposite::AlignableComposite( const GeomDet* geomDet ) : 
  Alignable( geomDet->geographicalId().rawId(), geomDet->surface() ),
  theStructureType(align::AlignableDet)
{
}

AlignableComposite::AlignableComposite(align::ID id,
				       StructureType type,
				       const RotationType& rot):
  Alignable(id, rot),
  theStructureType(type)
{
}

AlignableComposite::~AlignableComposite()
{
  for (unsigned int i = 0; i < theComponents.size(); ++i) delete theComponents[i];
}

void AlignableComposite::addComponent(Alignable* ali)
{
  const Alignables& newComps = ali->deepComponents();

  theDeepComponents.insert( theDeepComponents.end(), newComps.begin(), newComps.end() );

  Scalar k = static_cast<Scalar>( newComps.size() ) / theDeepComponents.size();

  theSurface.move( ( ali->globalPosition() - globalPosition() ) * k );

  ali->setMother(this);
  theComponents.push_back(ali);
}

//__________________________________________________________________________________________________
void AlignableComposite::recursiveComponents(Alignables &result) const
{

  Alignables components = this->components();
  if (this->alignableObjectId() == align::AlignableDet 
      && components.size() <= 1) { // Non-glued AlignableDets (still) contain themselves
    return; // (would be better to implement AlignableDet::recursiveComponents!)
  }
  for (Alignables::const_iterator iter = components.begin();
       iter != components.end(); ++iter) {
    result.push_back(*iter); // could use std::copy(..), but here we build a real hierarchy
    (*iter)->recursiveComponents(result);
  }
}

//__________________________________________________________________________________________________
void AlignableComposite::move( const GlobalVector& displacement ) 
{
  
  // Move components
  Alignables comp = this->components();
  for ( Alignables::iterator i=comp.begin(); i!=comp.end(); i++ )
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

  Alignables comp = this->components();
  comp[i]->move( this->surface().toGlobal( localDisplacement ) );

}


//__________________________________________________________________________________________________
/// Rotation intepreted such, that the orientation of the rotation
/// axis is w.r.t. to the global coordinate system. This, however, does NOT
/// mean the center of the rotation. This is simply taken as the center of
/// the Alignable-object 
void AlignableComposite::rotateInGlobalFrame( const RotationType& rotation )
{
  
  Alignables comp = this->components();
  
  PositionType myPosition = this->globalPosition();
  
  for ( Alignables::iterator i=comp.begin(); i!=comp.end(); i++ )
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
void AlignableComposite::setAlignmentPositionError( const AlignmentPositionError& ape,
						    bool propagateDown )
{

  // Since no geomDet is attached, alignable composites do not have an APE
  // The APE is, therefore, just propagated down
  if (!propagateDown) return;

  Alignables comp = this->components();
  for (Alignables::const_iterator i = comp.begin(); i != comp.end(); ++i) {
    (*i)->setAlignmentPositionError(ape, propagateDown);
  }
}


//__________________________________________________________________________________________________
void 
AlignableComposite::addAlignmentPositionError( const AlignmentPositionError& ape,
					       bool propagateDown )
{

  // Since no geomDet is attached, alignable composites do not have an APE
  // The APE is, therefore, just propagated down
  if (!propagateDown) return;

  Alignables comp = this->components();
  for (Alignables::const_iterator i = comp.begin(); i != comp.end(); ++i) {
    (*i)->addAlignmentPositionError(ape, propagateDown);
  }
  
}


//__________________________________________________________________________________________________
/// Adds the AlignmentPositionError (in x,y,z coordinates) that would result
/// on the various components from a possible Rotation of a composite the 
/// rotation matrix is in interpreted in GLOBAL coordinates
void AlignableComposite::addAlignmentPositionErrorFromRotation( const RotationType& rotation,
								bool propagateDown )
{

  if (!propagateDown) return;

  Alignables comp = this->components();
  PositionType myPosition=this->globalPosition();

  for ( Alignables::const_iterator i=comp.begin(); i!=comp.end(); i++ )
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
      (*i)->addAlignmentPositionError( ape, propagateDown );
      (*i)->addAlignmentPositionErrorFromRotation( rotation, propagateDown );
	  
    }

}


//__________________________________________________________________________________________________
/// Adds the AlignmentPositionError (in x,y,z coordinates) that would result
/// on the various components from a possible Rotation of a composite the 
/// rotation matrix is in interpreted in LOCAL  coordinates of the composite
void AlignableComposite::addAlignmentPositionErrorFromLocalRotation( const RotationType& rot,
								     bool propagateDown )
{
  // if (!propagateDown) return; // No! Cannot yet jump out since
  // addAlignmentPositionErrorFromRotation(..) below might be overwritten in derived
  // classes to do something on 'this' (and in fact does so in AlignableDet).

  RotationType globalRot = globalRotation().multiplyInverse(rot*globalRotation());
  this->addAlignmentPositionErrorFromRotation(globalRot, propagateDown);

}

//__________________________________________________________________________________________________
void AlignableComposite::setSurfaceDeformation(const SurfaceDeformation *deformation,
					       bool propagateDown)
{
  // Only DetUnits have surface deformations.
  // The parameters are, therefore, just propagated down.
  if (!propagateDown) return;

  Alignables comp(this->components());
  for (Alignables::const_iterator i = comp.begin(); i != comp.end(); ++i) {
    (*i)->setSurfaceDeformation(deformation, propagateDown);
  }
}

//__________________________________________________________________________________________________
void AlignableComposite::addSurfaceDeformation(const SurfaceDeformation *deformation,
					       bool propagateDown)
{
  // Only DetUnits have surface deformations.
  // The parameters are, therefore, just propagated down.
  if (!propagateDown) return;

  Alignables comp(this->components());
  for (Alignables::const_iterator i = comp.begin(); i != comp.end(); ++i) {
    (*i)->addSurfaceDeformation(deformation, propagateDown);
  }
}

//__________________________________________________________________________________________________
void AlignableComposite::dump( void ) const
{

  // A simple printout method. Could be specialized in the implementation classes.

  Alignables comp = this->components();

  // Dump this
  edm::LogInfo("AlignableDump") 
    << " Alignable of type " << this->alignableObjectId() 
    << " has " << comp.size() << " components" << std::endl
    << " position = " << this->globalPosition() << ", orientation:" << std::endl
    << this->globalRotation();

  // Dump components
  for ( Alignables::iterator i=comp.begin(); i!=comp.end(); i++ )
    (*i)->dump();

}



//__________________________________________________________________________________________________
Alignments* AlignableComposite::alignments( void ) const
{

  // Recursively call alignments, until we get to an AlignableDetUnit
  Alignables comp = this->components();

  Alignments* m_alignments = new Alignments();

  // Add components recursively
  for ( Alignables::iterator i=comp.begin(); i!=comp.end(); i++ )
    {
      Alignments* tmpAlignments = (*i)->alignments();
      std::copy( tmpAlignments->m_align.begin(), tmpAlignments->m_align.end(), 
		 std::back_inserter(m_alignments->m_align) );
	  delete tmpAlignments;
    }

  
  return m_alignments;

}


//__________________________________________________________________________________________________
AlignmentErrorsExtended* AlignableComposite::alignmentErrors( void ) const
{

  // Recursively call alignmentsErrors, until we get to an AlignableDetUnit
  Alignables comp = this->components();

  AlignmentErrorsExtended* m_alignmentErrors = new AlignmentErrorsExtended();

  // Add components recursively
  for ( Alignables::iterator i=comp.begin(); i!=comp.end(); i++ )
    {
      AlignmentErrorsExtended* tmpAlignmentErrorsExtended = (*i)->alignmentErrors();
      std::copy( tmpAlignmentErrorsExtended->m_alignError.begin(), tmpAlignmentErrorsExtended->m_alignError.end(), 
		 std::back_inserter(m_alignmentErrors->m_alignError) );
	  delete tmpAlignmentErrorsExtended;
    }

  
  return m_alignmentErrors;

}


//__________________________________________________________________________________________________
int AlignableComposite::surfaceDeformationIdPairs(std::vector<std::pair<int,SurfaceDeformation*> > & result) const
{

  Alignables comp = this->components();

  int count = 0;

  // Add components recursively
  for ( Alignables::iterator i=comp.begin(); i!=comp.end(); ++i) {
    count += (*i)->surfaceDeformationIdPairs(result);
  }
  
  return count;

}
