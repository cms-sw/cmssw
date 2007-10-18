#ifndef Alignment_TrackerAlignment_AlignableTracker_H
#define Alignment_TrackerAlignment_AlignableTracker_H

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignSetup.h"
#include "Alignment/TrackerAlignment/interface/TrackerCounters.h"

class GeometricDet;
class TrackerGeometry;

class AlignableTracker: public AlignableComposite 
{

public:
  
  /// Constructor (builds the full hierarchy)
  AlignableTracker( const TrackerGeometry* tracker ); 

  /// Return TOB half barrels
  Alignables& outerHalfBarrels() { return alignableLists_.find("TOBHalfBarrel"); }
  /// Return TIB half barrels
  Alignables& innerHalfBarrels() { return alignableLists_.find("TIBHalfBarrel"); }
  /// Return Pixel half barrels
  Alignables& pixelHalfBarrels() { return alignableLists_.find("TPBHalfBarrel"); }
  /// Return TECs
  Alignables& endCaps() { return alignableLists_.find("TECEndcap"); }
  /// Return TPEs
  Alignables& pixelEndCaps() { return alignableLists_.find("TPEEndcap"); }
  /// Return TIDs
  Alignables& TIDs() { return alignableLists_.find("TIDEndcap"); }

  /// Return inner and outer barrel GeomDets together 
  Alignables barrelGeomDets() { return merge( innerBarrelGeomDets(), outerBarrelGeomDets() ); }
  /// Return inner barrel and TID GeomDets together 
  Alignables TIBTIDGeomDets() { return merge( innerBarrelGeomDets(), TIDGeomDets() ); }
  /// Return inner barrel GeomDets 
  Alignables& innerBarrelGeomDets() { return alignableLists_.find("TIBModule"); }
  /// Return outer barrel GeomDets
  Alignables& outerBarrelGeomDets() { return alignableLists_.find("TOBModule"); }
  /// Return pixel barrel GeomDets
  Alignables& pixelHalfBarrelGeomDets() { return alignableLists_.find("TPBModule"); }
  /// Return endcap  GeomDets
  Alignables& endcapGeomDets() { return alignableLists_.find("TECModule"); }
  /// Return TID  GeomDets  
  Alignables& TIDGeomDets() { return alignableLists_.find("TIDModule"); }
  /// Return pixel endcap GeomDets
  Alignables& pixelEndcapGeomDets() { return alignableLists_.find("TPEModule"); }
  
  /// Return inner and outer barrel rods
  Alignables barrelRods() { return merge( innerBarrelRods(), outerBarrelRods() ); }
  /// Return inner barrel rods
  Alignables& innerBarrelRods() { return alignableLists_.find("TIBString"); }
  /// Return outer barrel rods
  Alignables& outerBarrelRods() { return alignableLists_.find("TOBRod"); }
  /// Return pixel half barrel ladders (implemented as AlignableRods)
  Alignables& pixelHalfBarrelLadders() { return alignableLists_.find("TPBLadder"); }
  /// Return encap petals
  Alignables& endcapPetals() { return alignableLists_.find("TECPetal"); }
  /// Return TID rings
  Alignables& TIDRings() { return alignableLists_.find("TIDRing"); }
  /// Return pixel endcap petals
  Alignables& pixelEndcapPetals() { return alignableLists_.find("TPEPanel"); }
		     
  /// Return inner and outer barrel layers
  Alignables barrelLayers() { return merge( innerBarrelLayers(), outerBarrelLayers() ); }
  /// Return inner barrel layers
  Alignables& innerBarrelLayers() { return alignableLists_.find("TIBLayer"); }
  /// Return outer barrel layers
  Alignables& outerBarrelLayers() { return alignableLists_.find("TOBLayer"); }
  /// Return pixel half barrel layers
  Alignables& pixelHalfBarrelLayers() { return alignableLists_.find("TPBLayer"); }
  /// Return endcap layers
  Alignables& endcapLayers() { return alignableLists_.find("TECDisk"); }
  /// Return TID layers
  Alignables& TIDLayers() { return alignableLists_.find("TIDDisk"); }
  /// Return pixel endcap layers
  Alignables& pixelEndcapLayers() { return alignableLists_.find("TPEHalfDisk"); }

  /// Return alignments, sorted by DetId
  Alignments* alignments() const;

  /// Return alignment errors, sorted by DetId
  AlignmentErrors* alignmentErrors() const;

  private:

  /// Build a barrel for a given sub-detector (TPB, TIB, TOB).
  void buildBarrel( const std::string& subDet );  // prefix for sub-detector 
  
  /// Create list of lower-level modules
  void detsToAlignables(const TrackingGeometry::DetContainer& dets,
                        const std::string& moduleName );

  void buildTPB();
  void buildTPE();
  void buildTIB();
  void buildTID();
  void buildTOB();
  void buildTEC();
  void buildTRK();

  Alignables merge( const Alignables& list1, const Alignables& list2 ) const;

  AlignSetup<Alignables> alignableLists_; //< Lists of alignables
  AlignSetup<Alignable*> alignables_;     //< Hierarchy

  TrackerCounters tkCounters_;
  

};

#endif //AlignableTracker_H
