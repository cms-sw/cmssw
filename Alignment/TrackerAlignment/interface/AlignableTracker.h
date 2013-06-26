#ifndef Alignment_TrackerAlignment_AlignableTracker_H
#define Alignment_TrackerAlignment_AlignableTracker_H

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignSetup.h"
#include "Alignment/TrackerAlignment/interface/TrackerCounters.h"

class GeometricDet;
class TrackerGeometry;
class TrackerTopology;


class AlignableTracker: public AlignableComposite 
{

public:
  
  /// Constructor (builds the full hierarchy)
  explicit AlignableTracker(const TrackerGeometry *tracker, const TrackerTopology *tTopo);

  /// Return alignables of subdet and hierarchy level determined by name
  /// as defined in tracker part of Alignment/CommonAlignment/StructureType.h
  Alignables& subStructures(const std::string &subStructName) {
    return alignableLists_.find(subStructName);
  }

  /// Return TOB half barrels
  Alignables& outerHalfBarrels() { return this->subStructures("TOBHalfBarrel");}
  /// Return TIB half barrels
  Alignables& innerHalfBarrels() { return this->subStructures("TIBHalfBarrel");}
  /// Return Pixel half barrels
  Alignables& pixelHalfBarrels() { return this->subStructures("TPBHalfBarrel");}
  /// Return TECs
  Alignables& endCaps() { return this->subStructures("TECEndcap");}
  /// Return TPEs
  Alignables& pixelEndCaps() { return this->subStructures("TPEEndcap");}
  /// Return TIDs
  Alignables& TIDs() { return this->subStructures("TIDEndcap");}

  /// Return inner and outer barrel GeomDets together 
  Alignables barrelGeomDets() { return this->merge(this->innerBarrelGeomDets(),
						   this->outerBarrelGeomDets());}
  /// Return inner barrel and TID GeomDets together 
  Alignables TIBTIDGeomDets() { return this->merge(this->innerBarrelGeomDets(),
						   this->TIDGeomDets());
  }
  /// Return inner barrel GeomDets 
  Alignables& innerBarrelGeomDets() { return this->subStructures("TIBModule");}
  /// Return outer barrel GeomDets
  Alignables& outerBarrelGeomDets() { return this->subStructures("TOBModule");}
  /// Return pixel barrel GeomDets
  Alignables& pixelHalfBarrelGeomDets() { return this->subStructures("TPBModule");}
  /// Return endcap  GeomDets
  Alignables& endcapGeomDets() { return this->subStructures("TECModule");}
  /// Return TID  GeomDets  
  Alignables& TIDGeomDets() { return this->subStructures("TIDModule");}
  /// Return pixel endcap GeomDets
  Alignables& pixelEndcapGeomDets() { return this->subStructures("TPEModule");}
  
  /// Return inner and outer barrel rods
  Alignables barrelRods() { return this->merge(this->innerBarrelRods(), this->outerBarrelRods());}
  /// Return inner barrel rods
  Alignables& innerBarrelRods() { return this->subStructures("TIBString");}
  /// Return outer barrel rods
  Alignables& outerBarrelRods() { return this->subStructures("TOBRod");}
  /// Return pixel half barrel ladders (implemented as AlignableRods)
  Alignables& pixelHalfBarrelLadders() { return this->subStructures("TPBLadder");}
  /// Return encap petals
  Alignables& endcapPetals() { return this->subStructures("TECPetal");}
  /// Return TID rings
  Alignables& TIDRings() { return this->subStructures("TIDRing");}
  /// Return pixel endcap petals
  Alignables& pixelEndcapPetals() { return this->subStructures("TPEPanel");}
		     
  /// Return inner and outer barrel layers
  Alignables barrelLayers() { return this->merge(this->innerBarrelLayers(),
						 this->outerBarrelLayers() );
  }
  /// Return inner barrel layers
  Alignables& innerBarrelLayers() { return this->subStructures("TIBLayer");}
  /// Return outer barrel layers
  Alignables& outerBarrelLayers() { return this->subStructures("TOBLayer");}
  /// Return pixel half barrel layers
  Alignables& pixelHalfBarrelLayers() { return this->subStructures("TPBLayer");}
  /// Return endcap layers
  Alignables& endcapLayers() { return this->subStructures("TECDisk");}
  /// Return TID layers
  Alignables& TIDLayers() { return this->subStructures("TIDDisk");}
  /// Return pixel endcap layers
  Alignables& pixelEndcapLayers() { return this->subStructures("TPEHalfDisk");}

  /// Return alignments, sorted by DetId
  Alignments* alignments() const;

  /// Return alignment errors, sorted by DetId
  AlignmentErrors* alignmentErrors() const;

  /// Returns tracker topology
  const TrackerTopology* trackerTopology() const { return tTopo_;}
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

  AlignSetup<Alignables> alignableLists_; //< kind of map of lists of alignables

  TrackerCounters tkCounters_;
  
  const TrackerTopology* tTopo_;

};

#endif //AlignableTracker_H
