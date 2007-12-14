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

  // Some typdefs to simplify notation
  typedef GeometricDet::ConstGeometricDetContainer _DetContainer;

public:
  
  /// Constructor (builds the full hierarchy)
  AlignableTracker( const GeometricDet* geometricDet, const TrackerGeometry* trackerGeometry ); 
  
  /// Recursive printout of the tracker structure
  void dump( void ) const;

  /// Return TOB half barrels
  Alignables outerHalfBarrels();
  /// Return TIB half barrels
  Alignables innerHalfBarrels();
  /// Return Pixel half barrels
  Alignables pixelHalfBarrels();
  /// Return TECs
  Alignables endCaps();
  /// Return TPEs
  const Alignables& pixelEndCaps() const { return getAlignables("pixelEndcaps"); }
  /// Return TIDs
  Alignables TIDs();

  /// Return TOB half barrel at given index
  AlignableTrackerHalfBarrel &outerHalfBarrel(unsigned int i);
  /// Return TIB half barrel at given index
  AlignableTrackerHalfBarrel &innerHalfBarrel(unsigned int i);
  /// Return Pixel half barrel at given index
  AlignablePixelHalfBarrel &pixelHalfBarrel(unsigned int i);
  /// Return endcap at given index
  AlignableTrackerEndcap &endCap(unsigned int i);
  /// Return Pixel endcap at given index
//   AlignableTrackerEndcap &pixelEndCap(unsigned int i);
  /// Return TID at given index
  AlignableTID &TID(unsigned int i);

  /// Return inner and outer barrel GeomDets together 
  Alignables barrelGeomDets(); 
  /// Return inner barrel and TID GeomDets together 
  Alignables TIBTIDGeomDets(); 
  /// Return inner barrel GeomDets 
  Alignables innerBarrelGeomDets();
  /// Return outer barrel GeomDets
  Alignables outerBarrelGeomDets();
  /// Return pixel barrel GeomDets
  Alignables pixelHalfBarrelGeomDets();
  /// Return endcap  GeomDets
  Alignables endcapGeomDets();
  /// Return TID  GeomDets  
  Alignables TIDGeomDets();
  /// Return pixel endcap GeomDets
  const Alignables& pixelEndcapGeomDets() const { return getAlignables("pixelEndcapSensors"); }
  
  /// Return inner and outer barrel rods
  Alignables barrelRods();
  /// Return inner barrel rods
  Alignables innerBarrelRods();
  /// Return outer barrel rods
  Alignables outerBarrelRods();
  /// Return pixel half barrel ladders (implemented as AlignableRods)
  Alignables pixelHalfBarrelLadders(); // though AlignableRods
  /// Return encap petals
  Alignables endcapPetals();
  /// Return TID rings
  Alignables TIDRings();
  /// Return pixel endcap petals
  const Alignables& pixelEndcapPetals() const { return getAlignables("pixelEndcapPanels"); }
		     
  /// Return inner and outer barrel layers
  Alignables barrelLayers(); 
  /// Return inner barrel layers
  Alignables innerBarrelLayers();
  /// Return outer barrel layers
  Alignables outerBarrelLayers();
  /// Return pixel half barrel layers
  Alignables pixelHalfBarrelLayers();
  /// Return endcap layers
  Alignables endcapLayers();
  /// Return TID layers
  Alignables TIDLayers();
  /// Return pixel endcap layers
  const Alignables& pixelEndcapLayers() const { return getAlignables("pixelEndcapHalfDisks"); }
  
  /// Return alignable object identifier 
  virtual int alignableObjectId() const { return AlignableObjectId::AlignableTracker; }

  /// Return alignments, sorted by DetId
  Alignments* alignments() const;

  /// Return alignment errors, sorted by DetId
  AlignmentErrors* alignmentErrors() const;

private:

  // Sub-structure builders (driven by the sub-components of GeometricDet)
  void buildTOB( const GeometricDet* navigator );   /// Build the tracker outer barrel
  void buildTIB( const GeometricDet* navigator );   /// Build the tracker inner barrel
  void buildTID( const GeometricDet* navigator );   /// Build the tracker inner disks
  void buildTEC( const GeometricDet* navigator );   /// Build the tracker endcap
  void buildTPB( const GeometricDet* navigator );   /// Build the pixel barrel

  void buildTPE( const TrackerGeometry::DetContainer& ); // Build the pixel endcaps

  /// Return all components of a given type
  std::vector<const GeometricDet*> getAllComponents( const GeometricDet* Det,
													 const GeometricDet::GDEnumType type ) const;  

  /// Set mothers recursively
  void recursiveSetMothers( Alignable* alignable );

  /// Helper function to fetch alignables from theTracker for a given structure
  const Alignables& getAlignables( const std::string& structure ) const;

private:

  const TrackerGeometry* theTrackerGeometry;   // To convert DetIds to GeomDets

  // Containers of separate components
  std::vector<AlignableTrackerHalfBarrel*>   theOuterHalfBarrels;
  std::vector<AlignableTrackerHalfBarrel*>   theInnerHalfBarrels;
  std::vector<AlignablePixelHalfBarrel*>     thePixelHalfBarrels;
  std::vector<AlignableTrackerEndcap*>       theEndcaps;
  std::vector<AlignableTID*>                 theTIDs;

  std::map<std::string, Alignables> theTracker; // container to hold the alignables for each level in the hierarchy
};

#endif //AlignableTracker_H
