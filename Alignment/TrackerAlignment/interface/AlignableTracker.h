#ifndef Alignment_TrackerAlignment_AlignableTracker_H
#define Alignment_TrackerAlignment_AlignableTracker_H

// Original Author:  ?
//     Last Update:  Max Stark
//            Date:  Mon, 15 Feb 2016 09:32:12 CET

// alignment
#include "Alignment/CommonAlignment/interface/AlignableMap.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/TrackerAlignment/interface/TrackerNameSpace.h"

class TrackerGeometry;
class TrackerTopology;



class AlignableTracker : public AlignableComposite {

  /// grant access for the tracker-alignables builder
  friend class AlignableTrackerBuilder;

public:

  AlignableTracker(const TrackerGeometry*, const TrackerTopology*);
  virtual ~AlignableTracker() { /* TODO: delete all tracker-alignables? */ };

  /// Return alignables of subdet and hierarchy level determined by name
  /// as defined in tracker part of Alignment/CommonAlignment/StructureType.h
  Alignables& subStructures(const std::string &subStructName) {
    return alignableMap.find(subStructName);
  }

  /// Return TOB half barrels
  Alignables& outerHalfBarrels() {
    return this->subStructures(alignableObjectId_.typeToName(align::TOBHalfBarrel));
  }
  /// Return TIB half barrels
  Alignables& innerHalfBarrels() {
    return this->subStructures(alignableObjectId_.typeToName(align::TIBHalfBarrel));
  }
  /// Return Pixel half barrels
  Alignables& pixelHalfBarrels() {
    return this->subStructures(alignableObjectId_.typeToName(align::TPBHalfBarrel));
  }
  /// Return TECs
  Alignables& endCaps() {
    return this->subStructures(alignableObjectId_.typeToName(align::TECEndcap));
  }
  /// Return TPEs
  Alignables& pixelEndCaps() {
    return this->subStructures(alignableObjectId_.typeToName(align::TPEEndcap));
  }
  /// Return TIDs
  Alignables& TIDs() {
    return this->subStructures(alignableObjectId_.typeToName(align::TIDEndcap));
  }

  /// Return inner and outer barrel GeomDets together 
  Alignables barrelGeomDets() { return this->merge(this->innerBarrelGeomDets(),
						   this->outerBarrelGeomDets());}
  /// Return inner barrel and TID GeomDets together 
  Alignables TIBTIDGeomDets() { return this->merge(this->innerBarrelGeomDets(),
						   this->TIDGeomDets());
  }
  /// Return inner barrel GeomDets 
  Alignables& innerBarrelGeomDets() {
    return this->subStructures(alignableObjectId_.typeToName(align::TIBModule));
  }
  /// Return outer barrel GeomDets
  Alignables& outerBarrelGeomDets() {
    return this->subStructures(alignableObjectId_.typeToName(align::TOBModule));
  }
  /// Return pixel barrel GeomDets
  Alignables& pixelHalfBarrelGeomDets() {
    return this->subStructures(alignableObjectId_.typeToName(align::TPBModule));
  }
  /// Return endcap  GeomDets
  Alignables& endcapGeomDets() {
    return this->subStructures(alignableObjectId_.typeToName(align::TECModule));
  }
  /// Return TID  GeomDets  
  Alignables& TIDGeomDets() {
    return this->subStructures(alignableObjectId_.typeToName(align::TIDModule));
  }
  /// Return pixel endcap GeomDets
  Alignables& pixelEndcapGeomDets() {
    return this->subStructures(alignableObjectId_.typeToName(align::TPEModule));
  }
  
  /// Return inner and outer barrel rods
  Alignables barrelRods() { return this->merge(this->innerBarrelRods(), this->outerBarrelRods());}
  /// Return inner barrel rods
  Alignables& innerBarrelRods() {
    return this->subStructures(alignableObjectId_.typeToName(align::TIBString));
  }
  /// Return outer barrel rods
  Alignables& outerBarrelRods() {
    return this->subStructures(alignableObjectId_.typeToName(align::TOBRod));
  }
  /// Return pixel half barrel ladders (implemented as AlignableRods)
  Alignables& pixelHalfBarrelLadders() {
    return this->subStructures(alignableObjectId_.typeToName(align::TPBLadder));
  }
  /// Return encap petals
  Alignables& endcapPetals() {
    return this->subStructures(alignableObjectId_.typeToName(align::TECPetal));
  }
  /// Return TID rings
  Alignables& TIDRings() {
    return this->subStructures(alignableObjectId_.typeToName(align::TIDRing));
  }
  /// Return pixel endcap petals
  Alignables& pixelEndcapPetals() {
    return this->subStructures(alignableObjectId_.typeToName(align::TPEPanel));
  }
		     
  /// Return inner and outer barrel layers
  Alignables barrelLayers() { return this->merge(this->innerBarrelLayers(),
						 this->outerBarrelLayers() );
  }
  /// Return inner barrel layers
  Alignables& innerBarrelLayers() {
    return this->subStructures(alignableObjectId_.typeToName(align::TIBLayer));
  }
  /// Return outer barrel layers
  Alignables& outerBarrelLayers() {
    return this->subStructures(alignableObjectId_.typeToName(align::TOBLayer));
  }
  /// Return pixel half barrel layers
  Alignables& pixelHalfBarrelLayers() {
    return this->subStructures(alignableObjectId_.typeToName(align::TPBLayer));
  }
  /// Return endcap layers
  Alignables& endcapLayers() {
    return this->subStructures(alignableObjectId_.typeToName(align::TECDisk));
  }
  /// Return TID layers
  Alignables& TIDLayers() {
    return this->subStructures(alignableObjectId_.typeToName(align::TIDDisk));
  }
  /// Return pixel endcap layers
  Alignables& pixelEndcapLayers() {
    return this->subStructures(alignableObjectId_.typeToName(align::TPEHalfDisk));
  }



  /// Return alignments, sorted by DetId
  Alignments* alignments() const;

  /// Return alignment errors, sorted by DetId
  AlignmentErrorsExtended* alignmentErrors() const;

  /// Return tracker topology used to build AlignableTracker
  const TrackerTopology* trackerTopology() const { return tTopo_;}

  /// Return tracker name space derived from the tracker's topology
  const align::TrackerNameSpace& trackerNameSpace() const { return trackerNameSpace_; }

  /// Return tracker alignable object ID provider derived from the tracker's geometry
  const AlignableObjectId& objectIdProvider() const { return alignableObjectId_; }
private:
  Alignables merge( const Alignables& list1, const Alignables& list2 ) const;

  const TrackerTopology* tTopo_;
  align::TrackerNameSpace trackerNameSpace_;
  AlignableObjectId alignableObjectId_;
  AlignableMap alignableMap;

};

#endif //AlignableTracker_H
