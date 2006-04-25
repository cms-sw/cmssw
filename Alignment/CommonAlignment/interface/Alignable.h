#ifndef Alignment_CommonAlignment_Alignable_H
#define Alignment_CommonAlignment_Alignable_H

#include <iostream>

#include "Geometry/Surface/interface/Surface.h"
#include "Geometry/Vector/interface/LocalVector.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "Geometry/CommonDetUnit/interface/DetPositioner.h"

// Headers in the same directory
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

///
///  Abstract base class for alignable entities.
///  Any Alignable object can be moved and rotated.
///  Also an alignment uncertainty can be set.
///  The class derives from DetPositioner, a friend class of
///  GeomDet, thus allowing to move the GeomDet.

class Alignable : public DetPositioner
{  
  
 public:
  
  typedef Surface::RotationType    RotationType;
  typedef Surface::PositionType    PositionType;
  typedef AlignableObjectId::AlignableObjectIdType AlignableObjectIdType;
  
  /// Default constructor
  Alignable() : theMisalignmentActive(true) {}
  
//FR   /// Methods to get/set AlignmentParameters
//FR   void setAlignmentParameters(AlignmentParameters* dap) 
//FR  {
//FR     delete theAlignmentParameters;
//FR     theAlignmentParameters = dap;
//FR   }
  
  virtual ~Alignable() { 
    //FR delete theAlignmentParameters; 
  }

  /// Return vector of all direct components
  virtual std::vector<Alignable*> components() const = 0;
  inline int size() const { return components().size(); }


  /// movement with respect to the GLOBAL CMS reference frame
  virtual void move( const GlobalVector& displacement) = 0;
  
  /// rotation intepreted such, that the orientation of the rotation
  /// axis is w.r.t. to the global coordinate system, however, the
  /// NOT the center of the rotation is simply taken as the center of
  /// the Alignable-object 
  ///
  virtual void rotateInGlobalFrame( const RotationType& rotation) = 0;
  
  /// rotation intepreted with respect to its local coordinate system
  virtual void rotateInLocalFrame( const RotationType& rotation) 
  {
    // this is done by simply transforming the rotation from the
    // local system O to the global one  O^-1 * Rot * O
    // and then applying the global rotation  O * Rot
    
    rotateInGlobalFrame(
	globalRotation().multiplyInverse(rotation*globalRotation()));
  }   
  

  /// rotate around arbitratry global axis by deltaPhi 
  virtual void rotateAroundGlobalAxis( 
				      const GlobalVector axis, 
				      float deltaPhi
				      )  
  {
    rotateInGlobalFrame(RotationType(axis.basicVector(), deltaPhi));
  }

  /// rotate around arbitratry local axis by deltaPhi
  virtual void rotateAroundLocalAxis( const LocalVector axis,
				      float deltaPhi )
  {
    rotateInLocalFrame(RotationType(axis.basicVector(), deltaPhi));
  }

  ///  rotate around global x-axis by deltaPhi 
  virtual void rotateAroundGlobalX( float deltaPhi)  
  {
    RotationType rot( 1., 0., 0. ,
		      0., cos(deltaPhi),  sin(deltaPhi),
		      0., -sin(deltaPhi),  cos(deltaPhi) );
    rotateInGlobalFrame(rot);
  }

  /// rotate around local x-axis by deltaPhi
  virtual void rotateAroundLocalX( float deltaPhi)  
  {
    RotationType rot( 1., 0., 0. ,
		      0., cos(deltaPhi),  sin(deltaPhi),
		      0., -sin(deltaPhi),  cos(deltaPhi) );
    rotateInLocalFrame(rot);
  }

  /// rotate around global y-axis by deltaPhi 
  virtual void rotateAroundGlobalY( float deltaPhi)  
  {
    RotationType rot( cos(deltaPhi),  0., -sin(deltaPhi), 
		      0.,              1.,             0.,
		      sin(deltaPhi), 0.,  cos(deltaPhi) );
    rotateInGlobalFrame(rot);
  }

  /// rotate around local y-axis by deltaPhi
  virtual void rotateAroundLocalY( float deltaPhi)  
  {
    RotationType rot( cos(deltaPhi),  0., -sin(deltaPhi), 
		      0.,             1.,             0.,
		      sin(deltaPhi),  0.,  cos(deltaPhi) );
    
    rotateInLocalFrame(rot);
  }



  /// rotate around global z-axis by deltaPhi
  virtual void rotateAroundGlobalZ( float deltaPhi)  
  {
    RotationType rot( cos(deltaPhi),  sin(deltaPhi),  0. ,
		      -sin(deltaPhi),  cos(deltaPhi), 0. ,
		      0.,              0.,            1. );
    rotateInGlobalFrame(rot);
  }

  /// rotate around local z-axis by deltaPhi
  virtual void rotateAroundLocalZ( float deltaPhi)  
  {
    RotationType rot( cos(deltaPhi),   sin(deltaPhi), 0. ,
		      -sin(deltaPhi),  cos(deltaPhi), 0. ,
		      0.,              0.,            1. );
    rotateInLocalFrame(rot);
  }


  /// the global position of the object
  virtual GlobalPoint globalPosition () const 
  {
    return surface().position();
  }

  /// the global orientation of the object
  virtual RotationType globalRotation () const 
  {
    return surface().rotation();
  }

  /// the global Position+Orientation (Surface) of the object
  virtual const AlignableSurface& surface () const = 0;

  /// change of the global position since the creation of the object
  virtual GlobalVector displacement() const { return theDisplacement; }


  /// global rotation since the creation of the object 
  virtual RotationType rotation() const { return theRotation; }

  /// set the alignment position error
  virtual void 
  setAlignmentPositionError(const AlignmentPositionError& ape) = 0;

  /// add (or set if not already presnt) the AlignmentPositionError
  virtual void 
  addAlignmentPositionError(const AlignmentPositionError& ape) = 0;

  /// add (or set if not already present) the AlignmentPositionError 
  /// which would in x,y,z result from a rotation (given in the GLOBAL frame
  /// of CMS)  of the alignable object, rather than its movement
  ///
  virtual void 
  addAlignmentPositionErrorFromRotation(const RotationType& rotation) = 0;

  /// add (or set if not already present) the AlignmentPositionError 
  /// which would in x,y,z result from a rotation (given in the LOCAL frame
  /// of the Alignable)  of the alignable object, rather than it's movement
  ///
  virtual void 
  addAlignmentPositionErrorFromLocalRotation(const RotationType& rotation) = 0;

  /// Restore original position
  virtual void deactivateMisalignment () = 0;

  /// Redo misalignment
  virtual void reactivateMisalignment () = 0;

  /// Status
  bool misalignmentActive () const { return theMisalignmentActive; }

  /// Object ID
  virtual int alignableObjectId () const =0;

  
 protected:
  virtual void addDisplacement(const GlobalVector& displacement) 
  {
    theDisplacement += displacement;
  }
  
  virtual void addRotation(const RotationType& rotation) 
  {
    theRotation *= rotation;
  }

 protected:
  bool theMisalignmentActive;           ///< (de)activation flag

 private:

  GlobalVector theDisplacement;
  RotationType theRotation;

  //FR AlignmentParameters* theAlignmentParameters;

};

std::ostream& operator << (std::ostream& os, const Alignable & a);


#endif







