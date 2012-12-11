#ifndef Alignment_CommonAlignment_Alignable_H
#define Alignment_CommonAlignment_Alignable_H

#include "Alignment/CommonAlignment/interface/AlignableSurface.h"
#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "DataFormats/DetId/interface/DetId.h"

class AlignmentErrors;
class AlignmentParameters;
class AlignmentPositionError;
class Alignments;
class AlignmentSurfaceDeformations;
class SurfaceDeformation;

/** \class Alignable
 *
 * Abstract base class for alignable entities.
 * Any Alignable object can be moved and rotated.
 * Also an alignment uncertainty can be set.
 *
 *  $Date: 2011/09/19 11:42:35 $
 *  $Revision: 1.36 $
 *  (last update by $Author: mussgill $)
 */

class AlignmentParameters;
class SurveyDet;

class Alignable
{
  
public:

  typedef align::Scalar       Scalar;
  typedef align::PositionType PositionType;
  typedef align::RotationType RotationType;
  typedef align::GlobalVector GlobalVector;
  typedef align::LocalVector  LocalVector;
  typedef align::Alignables   Alignables;
  typedef align::StructureType StructureType;

  /// Constructor from id and surface, setting also geomDetId
  /// (AlignableNavigator relies on the fact that only AlignableDet/DetUnit have geomDetId!)
  Alignable( align::ID, const AlignableSurface& );

  /// Constructor for a composite with given rotation.
  /// Position is found (later) from average of daughters' positions.
  Alignable( align::ID, const RotationType& );

  /// Destructor
  virtual ~Alignable();

  /// Set the AlignmentParameters
  void setAlignmentParameters( AlignmentParameters* dap );

  /// Get the AlignmentParameters
  AlignmentParameters* alignmentParameters() const { return theAlignmentParameters; }

  /// Add a component to alignable
  /// (GF: Should be interface in Composite, but needed in AlignableBuilder::build)
  virtual void addComponent( Alignable* ) = 0;

  /// Return vector of all direct components
  virtual Alignables components() const = 0;

  /// Return number of direct components
  int size() const { return components().size(); }

  /// Return the list of lowest daughters (non-composites) of Alignable.
  /// Contain itself if Alignable is a unit.
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

  /// Set the alignment position error - if (!propagateDown) do not affect daughters
  virtual void 
  setAlignmentPositionError( const AlignmentPositionError& ape, bool propagateDown) = 0;

  /// Add (or set if not already present) the AlignmentPositionError,
  /// but if (!propagateDown) do not affect daughters
  virtual void 
  addAlignmentPositionError( const AlignmentPositionError& ape, bool propagateDown ) = 0;

  /// add (or set if not already present) the AlignmentPositionError 
  /// which would result from a rotation (given in the GLOBAL frame
  /// of CMS) of the alignable object,
  /// but if (!propagateDown) do not affect daughters
  virtual void 
  addAlignmentPositionErrorFromRotation( const RotationType& rotation, bool propagateDown ) = 0;

  /// add (or set if not already present) the AlignmentPositionError 
  /// which would result from a rotation (given in the LOCAL frame
  /// of the Alignable)  of the alignable object,
  /// but if (!propagateDown) do not affect daughters
  virtual void 
  addAlignmentPositionErrorFromLocalRotation( const RotationType& rotation, bool propagateDown ) = 0;

  /// Set the surface deformation parameters - if (!propagateDown) do not affect daughters
  virtual void
  setSurfaceDeformation(const SurfaceDeformation *deformation, bool propagateDown) = 0;

  /// Add the surface deformation parameters to the existing ones,
  /// if (!propagateDown) do not affect daughters.
  virtual void
  addSurfaceDeformation(const SurfaceDeformation *deformation, bool propagateDown) = 0;
  
  /// Return the alignable type identifier
  virtual StructureType alignableObjectId() const = 0;

  /// Return the DetId of the associated GeomDet (0 by default)
  /// This should be removed. Ultimately we need only one ID.
  const DetId& geomDetId() const { return theDetId; }

  /// Return the ID of Alignable, i.e. DetId of 'first' component GeomDet(Unit). 
  align::ID id() const { return theId; } 

  /// Recursive printout of alignable information
  virtual void dump() const = 0;

  /// Return vector of alignment data
  virtual Alignments* alignments() const = 0;
  
  /// Return vector of alignment errors
  virtual AlignmentErrors* alignmentErrors() const = 0;

  /// Return surface deformations, sorted by DetId
  AlignmentSurfaceDeformations* surfaceDeformations() const;

  /// Return surface deformations as a vector of pairs of raw DetId
  /// and pointers to surface deformations
  virtual int surfaceDeformationIdPairs(std::vector<std::pair<int,SurfaceDeformation*> > &) const = 0;

  /// cache the current position, rotation and other parameters (e.g. surface deformations)
  virtual void cacheTransformation();

  /// restore the previously cached transformation, also for possible components
  virtual void restoreCachedTransformation();

  /// Return survey info
  const SurveyDet* survey() const { return theSurvey; }

  /// Set survey info
  void setSurvey( const SurveyDet* );

protected:

  void addDisplacement( const GlobalVector& displacement );
  void addRotation( const RotationType& rotation );

protected:

  DetId theDetId; // used to check if Alignable is associated to a GeomDet 
                  // ugly way to keep AlignableNavigator happy for now 

  align::ID theId; // real ID as int, above DetId should be removed

  AlignableSurface theSurface; // Global position and orientation of surface

  GlobalVector theDisplacement; // total linear displacement
  RotationType theRotation;     // total angular displacement

  AlignableSurface theCachedSurface;
  GlobalVector theCachedDisplacement;
  RotationType theCachedRotation;

  Alignables theDeepComponents; // list of lowest daughters
                                // contain itself if Alignable is a unit

private:
  /// private default ctr. to enforce usage of the specialised ones
  Alignable() {};

  AlignmentParameters* theAlignmentParameters;

  Alignable* theMother;       // Pointer to container

  const SurveyDet* theSurvey; // Pointer to survey info; owned by class

};

#endif
