#ifndef Alignment_TrackerAlignment_AlignableTracker_H
#define Alignment_TrackerAlignment_AlignableTracker_H

#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"

// Classes that will be used to construct the tracker
class AlignableTrackerHalfBarrel;
class AlignablePixelHalfBarrel;
class AlignableTrackerEndcap;
class AlignableTID;

/// Constructor of the full tracker geometry.
/// This object is stored to the EventSetup for further retrieval

class AlignableTracker: public AlignableComposite 
{

public:
  
  /// Constructor (builds the full hierarchy)
  AlignableTracker( const GeometricDet* geometricDet, const TrackerGeometry* trackerGeometry ); 

  /// Destructor
  ~AlignableTracker();
  
public:

  // Some typdefs to simplify notation
  typedef GlobalPoint           _PositionType;
  typedef TkRotation<float>     _RotationType;
  typedef GeometricDet::ConstGeometricDetContainer _DetContainer;

  /// Recursive printout of the tracker structure
  void dump( void ) const;
  
  /// Return all components
  virtual std::vector<Alignable*> components() const { return theTrackerComponents; }

  /// Alignable tracker has no mother
  virtual Alignable* mother() { return 0; }

  /// Return TOB half barrels
  std::vector<Alignable*> outerHalfBarrels();
  /// Return TIB half barrels
  std::vector<Alignable*> innerHalfBarrels();
  /// Return Pixel half barrels
  std::vector<Alignable*> pixelHalfBarrels();
  /// Return TECs
  std::vector<Alignable*> endCaps();
  /// Return TPEs
  std::vector<Alignable*> pixelEndCaps();
  /// Return TIDs
  std::vector<Alignable*> TIDs();

  /// Return TOB half barrel at given index
  AlignableTrackerHalfBarrel &outerHalfBarrel(unsigned int i);
  /// Return TIB half barrel at given index
  AlignableTrackerHalfBarrel &innerHalfBarrel(unsigned int i);
  /// Return Pixel half barrel at given index
  AlignablePixelHalfBarrel &pixelHalfBarrel(unsigned int i);
  /// Return endcap at given index
  AlignableTrackerEndcap &endCap(unsigned int i);
  /// Return Pixel endcap at given index
  AlignableTrackerEndcap &pixelEndCap(unsigned int i);
  /// Return TID at given index
  AlignableTID &TID(unsigned int i);

  /// Return inner and outer barrel GeomDets together 
  std::vector<Alignable*> barrelGeomDets(); 
  /// Return inner barrel GeomDets 
  std::vector<Alignable*> innerBarrelGeomDets();
  /// Return outer barrel GeomDets
  std::vector<Alignable*> outerBarrelGeomDets();
  /// Return pixel barrel GeomDets
  std::vector<Alignable*> pixelHalfBarrelGeomDets();
  /// Return endcap  GeomDets
  std::vector<Alignable*> endcapGeomDets();
  /// Return TID  GeomDets  
  std::vector<Alignable*> TIDGeomDets();
  /// Return pixel endcap GeomDets
  std::vector<Alignable*> pixelEndcapGeomDets();
  
  /// Return inner and outer barrel rods
  std::vector<Alignable*> barrelRods();
  /// Return inner barrel rods
  std::vector<Alignable*> innerBarrelRods();
  /// Return outer barrel rods
  std::vector<Alignable*> outerBarrelRods();
  /// Return pixel half barrel ladders (implemented as AlignableRods)
  std::vector<Alignable*> pixelHalfBarrelLadders(); // though AlignableRods
  /// Return encap petals
  std::vector<Alignable*> endcapPetals();
  /// Return TID rings
  std::vector<Alignable*> TIDRings();
  /// Return pixel endcap petals
  std::vector<Alignable*> pixelEndcapPetals();
		     
  /// Return inner and outer barrel layers
  std::vector<Alignable*> barrelLayers(); 
  /// Return inner barrel layers
  std::vector<Alignable*> innerBarrelLayers();
  /// Return outer barrel layers
  std::vector<Alignable*> outerBarrelLayers();
  /// Return pixel half barrel layers
  std::vector<Alignable*> pixelHalfBarrelLayers();
  /// Return endcap layers
  std::vector<Alignable*> endcapLayers();
  /// Return TID layers
  std::vector<Alignable*> TIDLayers();
  /// Return pixel endcap layers
  std::vector<Alignable*> pixelEndcapLayers();
  
  /// Return alignable object identifier 
  virtual int alignableObjectId() const { return AlignableObjectId::AlignableTracker; }

private:
  
  /// Get the position (centered at 0 by default)
  PositionType computePosition(); 
  /// Get the global orientation (no rotation by default)
  RotationType computeOrientation();
  /// Get the Surface
  AlignableSurface computeSurface();

  // Sub-structure builders (driven by the sub-components of GeometricDet)
  void buildTOB( const GeometricDet* navigator );   /// Build the tracker outer barrel
  void buildTIB( const GeometricDet* navigator );   /// Build the tracker inner barrel
  void buildTID( const GeometricDet* navigator );   /// Build the tracker inner disks
  void buildTEC( const GeometricDet* navigator );   /// Build the tracker endcap
  void buildTPB( const GeometricDet* navigator );   /// Build the pixel barrel
  void buildTPE( const GeometricDet* navigator );   /// Build the pixel endcap

  /// Return all components of a given type
  std::vector<const GeometricDet*> getAllComponents( const GeometricDet* Det,
													 const GeometricDet::GDEnumType type ) const;  

  /// Set mothers recursively
  void recursiveSetMothers( Alignable* alignable );

private:

  const TrackerGeometry* theTrackerGeometry;   // To convert DetIds to GeomDets
  
  // Container of all components
  std::vector<Alignable*> theTrackerComponents;

  // Containers of separate components
  std::vector<AlignableTrackerHalfBarrel*>   theOuterHalfBarrels;
  std::vector<AlignableTrackerHalfBarrel*>   theInnerHalfBarrels;
  std::vector<AlignablePixelHalfBarrel*>     thePixelHalfBarrels;
  std::vector<AlignableTrackerEndcap*>       theEndcaps;
  std::vector<AlignableTrackerEndcap*>       thePixelEndcaps;
  std::vector<AlignableTID*>                 theTIDs;


};

#endif //AlignableTracker_H




