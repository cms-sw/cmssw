#ifndef Alignment_CommonAlignment_Alignable_H
#define Alignment_CommonAlignment_Alignable_H

#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h" // fixme: should forward declare
#include "Geometry/CommonDetUnit/interface/DetPositioner.h"
#include "CondFormats/Alignment/interface/Alignments.h" // fixme: should forward declare
#include "CondFormats/Alignment/interface/AlignmentErrors.h" // fixme: should forward declare

// Headers in the same package
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h" // fixme: should forward declare
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"
#include "DataFormats/DetId/interface/DetId.h"


/** \class Alignable
 *
 * Abstract base class for alignable entities.
 * Any Alignable object can be moved and rotated.
 * Also an alignment uncertainty can be set.
 * The class derives from DetPositioner, a friend class of
 * GeomDet, which allows to move the GeomDet. 
 *
 *  $Date: 2007/06/21 16:18:27 $
 *  $Revision: 1.24 $
 *  (last update by $Author: flucke $)
 */

class AlignmentParameters;
class GeomDet;
class SurveyDet;

class Alignable : public DetPositioner
{  
  
public:

  typedef align::Scalar       Scalar;
  typedef align::PositionType PositionType;
  typedef align::RotationType RotationType;
  typedef align::GlobalVector GlobalVector;
  typedef align::LocalVector  LocalVector;
  typedef AlignableObjectId::AlignableObjectIdType AlignableObjectIdType; // fixme: put in namespace

  typedef std::vector<Alignable*> Alignables;

  /// Constructor from GeomDet
  Alignable( const GeomDet* );

  /// Constructor for a composite with given rotation.
  /// Position is found (later) from average of daughters' positions.
  Alignable( const DetId&, const RotationType& );

  /// Destructor
  virtual ~Alignable();

  /// Set the AlignmentParameters
  void setAlignmentParameters( AlignmentParameters* dap );

  /// Get the AlignmentParameters
  AlignmentParameters* alignmentParameters() const { return theAlignmentParameters; }

  /// Add a component to alignable
  virtual void addComponent( Alignable* ) = 0;

  /// Return vector of all direct components
  virtual Alignables components() const = 0;

  /// Return number of direct components
  const int size() const { return components().size(); }

  /// Return the list of lowest daughters (non-composites) of Alignable
  const Alignables& deepComponents() const { return theDeepComponents; }

  /// Provide all components, subcomponents, subsub... etc. of Alignable
  /// down to AlignableDetUnit, except for 'single childs' like e.g.
  /// AlignableDetUnits of AlignableDets representing single sided SiStrip
  /// modules. (for performance reason by adding to argument) 
  virtual void recursiveComponents(Alignables &result) const = 0;

  /// Steps down hierarchy until components with AlignmentParameters are found 
  /// and adds them to argument. True either if no such components are found
  /// or if all branches of components end with such components (i.e. 'consistent').
  bool firstCompsWithParams(Alignables &paramComps) const;

  /// Return pointer to container alignable (if any)
  Alignable* mother() const { return theMother; }

  /// Assign mother to alignable
  void setMother( Alignable* mother ) { theMother = mother; }

  /// Movement with respect to the global reference frame
  virtual void move( const GlobalVector& displacement) = 0;

  /// Rotation intepreted such that the orientation of the rotation
  /// axis is w.r.t. to the global coordinate system. Rotation is
  /// relative to current orientation
  virtual void rotateInGlobalFrame( const RotationType& rotation) = 0;
  
  /// Rotation intepreted in the local reference frame
  virtual void rotateInLocalFrame( const RotationType& rotation);
  
  /// Rotation around arbitratry global axis
  virtual void rotateAroundGlobalAxis( const GlobalVector& axis, Scalar radians );

  /// Rotation around arbitratry local axis
  virtual void rotateAroundLocalAxis( const LocalVector& axis, Scalar radians );

  /// Rotation around global x-axis
  virtual void rotateAroundGlobalX( Scalar radians );

  /// Rotation around local x-axis
  virtual void rotateAroundLocalX( Scalar radians );

  /// Rotation around global y-axis
  virtual void rotateAroundGlobalY( Scalar radians );

  /// Rotation around local y-axis
  virtual void rotateAroundLocalY( Scalar radians ); 

  /// Rotation around global z-axis
  virtual void rotateAroundGlobalZ( Scalar radians );

  /// Rotation around local z-axis
  virtual void rotateAroundLocalZ( Scalar radians);

  /// Return the Surface (global position and orientation) of the object 
  const AlignableSurface& surface() const { return theSurface; }

    /// Return the global position of the object 
  const PositionType& globalPosition() const { return surface().position(); }
  
  /// Return the global orientation of the object 
  const RotationType& globalRotation() const { return surface().rotation(); }

  /// Return change of the global position since the creation of the object
  const GlobalVector& displacement() const { return theDisplacement; }

  /// Return change of orientation since the creation of the object 
  const RotationType& rotation() const { return theRotation; }

  /// Set the alignment position error
  virtual void 
  setAlignmentPositionError( const AlignmentPositionError& ape ) = 0;

  /// Add (or set if not already present) the AlignmentPositionError
  virtual void 
  addAlignmentPositionError( const AlignmentPositionError& ape ) = 0;

  /// add (or set if not already present) the AlignmentPositionError 
  /// which would result from a rotation (given in the GLOBAL frame
  /// of CMS) of the alignable object
  virtual void 
  addAlignmentPositionErrorFromRotation( const RotationType& rotation ) = 0;

  /// add (or set if not already present) the AlignmentPositionError 
  /// which would result from a rotation (given in the LOCAL frame
  /// of the Alignable)  of the alignable object
  virtual void 
  addAlignmentPositionErrorFromLocalRotation( const RotationType& rotation ) = 0;

  /// Return the alignable type identifier
  virtual int alignableObjectId() const = 0;

  /// Return the DetId of the associated GeomDet (0 by default)
  const DetId& geomDetId() const { return theDetId; }

  /// Recursive printout of alignable information
  virtual void dump() const = 0;

  /// Return vector of alignment data
  virtual Alignments* alignments() const = 0;
  
  /// Return vector of alignment errors
  virtual AlignmentErrors* alignmentErrors() const = 0;

  /// Return survey info
  const SurveyDet* survey() const { return theSurvey; }

  /// Set survey info
  void setSurvey( const SurveyDet* );

protected:

  void addDisplacement( const GlobalVector& displacement );
  void addRotation( const RotationType& rotation );

protected:

  DetId theDetId;

  AlignableSurface theSurface; // Global position and orientation of surface

  GlobalVector theDisplacement; // total linear displacement
  RotationType theRotation;     // total angular displacement

  Alignables theDeepComponents; // list of lowest daughters
                                // contain itself if Alignable is a unit

private:

  AlignmentParameters* theAlignmentParameters;

  Alignable* theMother;       // Pointer to container

  const SurveyDet* theSurvey; // Pointer to survey info; owned by class

};

#endif
