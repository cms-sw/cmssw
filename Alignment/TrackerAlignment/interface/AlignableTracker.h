#ifndef Alignment_TrackerAlignment_AlignableTracker_H
#define Alignment_TrackerAlignment_AlignableTracker_H

/// AlignableTracker contains all the elements in the Tracker in the 
/// highest hierarchy level.
/// 

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"

#include <vector>

// Classes that will be used to construct the tracker
class AlignableTrackerHalfBarrel;
class AlignablePxHalfBarrel;
class AlignableTrackerEndcap;
class AlignableTID;


static const float MIN_ANGLE = 1.0e-5; // Minimum phi angle, including rounding errors

/// AlignableTracker composites out of all the elements in the Tracker in 
/// the first hierachy, i.e. we want to be able to move
///      Forward/Backward OuterBarrel
///      Forward/Backward InnerBarrel
///      Pixels: Positive-/Negative-x HalfBarrel
///      Forward/Backward EndCaps
///
/// This class really implements the alignment structures.
class AlignableTracker: public AlignableComposite 
{
private:
  
  // private constructor to avoid multiple instances
  AlignableTracker( const edm::EventSetup& iSetup ); 

public:

  // Some typdefs to simplify notation
  typedef GlobalPoint           _PositionType;
  typedef TkRotation<float>     _RotationType;
  typedef GeometricDet::ConstGeometricDetContainer _DetContainer;

  
  ~AlignableTracker();

  /// public access to the unique instance of the alignable tracker
  /// The EventSetup is required to access the geometry.
  static AlignableTracker* instance( const edm::EventSetup& iSetup );

  /// Print out the full tracker structure
  void dump( void );


  virtual std::vector<Alignable*> components() const 
  {
    return theTrackerComponents;
  }

  AlignableTrackerHalfBarrel &outerHalfBarrel(unsigned int i);
  AlignableTrackerHalfBarrel &innerHalfBarrel(unsigned int i);
  AlignablePxHalfBarrel &pixelHalfBarrel(unsigned int i);
  AlignableTrackerEndcap &endCap(unsigned int i);
  AlignableTID &TID(unsigned int i);

  /// inner and outer barrel Dets together 
  std::vector<Alignable*> barrelGeomDets(); 
  /// inner barrel Dets 
  std::vector<Alignable*> innerBarrelGeomDets();
  /// outer barrel Dets
  std::vector<Alignable*> outerBarrelGeomDets();

  /// pixel barrel Dets
  std::vector<Alignable*> pixelHalfBarrelGeomDets();

  /// endcap  Dets  (just the 2x9 large discs)
  std::vector<Alignable*> endcapGeomDets();
  ///  TID  Dets  
  std::vector<Alignable*> TIDGeomDets();
  /// pixel endcap  Dets
  std::vector<Alignable*> pixelEndcapGeomDets();
		     
  std::vector<Alignable*> barrelRods();             // inner & outer barrel rods
  std::vector<Alignable*> innerBarrelRods();
  std::vector<Alignable*> outerBarrelRods();
  std::vector<Alignable*> pixelHalfBarrelLadders(); // though AlignableRods
  std::vector<Alignable*> endcapPetals();
  std::vector<Alignable*> TIDRings();
  std::vector<Alignable*> pixelEndcapPetals();
		     
  std::vector<Alignable*> barrelLayers(); 
  std::vector<Alignable*> innerBarrelLayers();
  std::vector<Alignable*> outerBarrelLayers();
  std::vector<Alignable*> pixelHalfBarrelLayers();
  std::vector<Alignable*> endcapLayers();
  std::vector<Alignable*> TIDLayers();
  std::vector<Alignable*> pixelEndcapLayers();
  
  /// Alignable object identifier
  virtual int alignableObjectId () const 
  {
    return AlignableObjectId::AlignableTracker;
  }

 private:
  
  PositionType computePosition(); 
  // get the global orientation
  RotationType computeOrientation(); // see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface();

  /// sub-structure builders (driven by the sub-components of GeometricDet)
  void buildTOB( const GeometricDet* navigator );   // Tracker Outer Barrel
  void buildTIB( const GeometricDet* navigator );   // Tracker Inner Barrel
  void buildTID( const GeometricDet* navigator );   // Tracker Inner Disks
  void buildTEC( const GeometricDet* navigator );   // Tracker End Cap
  void buildTPB( const GeometricDet* navigator );   // Pixel Barrel
  void buildTPE( const GeometricDet* navigator );   // Pixel EndCap

  /// Return all components of a given type
  std::vector<const GeometricDet*> getAllComponents( const GeometricDet* Det,
													 const GeometricDet::GDEnumType type ) const;  

 private:

  edm::ESHandle<GeometricDet>     theGeometricTracker;  // To get tracker geometry
  edm::ESHandle<TrackerGeometry>  theTrackingGeometry;  // To convert DetIds to GeomDets

  
  std::vector<Alignable*> theTrackerComponents;

  // Well the tracker is more complicated than the other parts which all 
  // have only ONE type of components. Therefore for the Al..Tracker I store
  // ALSO seperately to "Outer/InnerBarrel", "PixelHalfBarrel" 
  // and Endcap" in order to be able to return them as such..

  std::vector<AlignableTrackerHalfBarrel*>   theOuterHalfBarrels;
  std::vector<AlignableTrackerHalfBarrel*>   theInnerHalfBarrels;
  std::vector<AlignablePxHalfBarrel*>        thePixelHalfBarrels;
  std::vector<AlignableTrackerEndcap*>       theEndcaps;
  std::vector<AlignableTrackerEndcap*>       thePixelEndcaps;
  std::vector<AlignableTID*>                 theTIDs;


};

#endif //AlignableTracker_H






