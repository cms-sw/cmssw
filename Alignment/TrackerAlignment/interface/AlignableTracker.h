#ifndef Alignment_TrackerAlignment_AlignableTracker_H
#define Alignment_TrackerAlignment_AlignableTracker_H

// Original Author:  ?
//     Last Update:  Max Stark
//            Date:  Mon, 15 Feb 2016 09:32:12 CET

// alignment
#include "Alignment/CommonAlignment/interface/AlignableMap.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

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
    return this->subStructures(AlignableObjectId::typeToName(align::TOBHalfBarrel));
  }
  /// Return TIB half barrels
  Alignables& innerHalfBarrels() {
    return this->subStructures(AlignableObjectId::typeToName(align::TIBHalfBarrel));
  }
  /// Return Pixel half barrels
  Alignables& pixelHalfBarrels() {
    return this->subStructures(AlignableObjectId::typeToName(align::TPBHalfBarrel));
  }
  /// Return TECs
  Alignables& endCaps() {
    return this->subStructures(AlignableObjectId::typeToName(align::TECEndcap));
  }
  /// Return TPEs
  Alignables& pixelEndCaps() {
    return this->subStructures(AlignableObjectId::typeToName(align::TPEEndcap));
  }
  /// Return TIDs
  Alignables& TIDs() {
    return this->subStructures(AlignableObjectId::typeToName(align::TIDEndcap));
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
    return this->subStructures(AlignableObjectId::typeToName(align::TIBModule));
  }
  /// Return outer barrel GeomDets
  Alignables& outerBarrelGeomDets() {
    return this->subStructures(AlignableObjectId::typeToName(align::TOBModule));
  }
  /// Return pixel barrel GeomDets
  Alignables& pixelHalfBarrelGeomDets() {
    return this->subStructures(AlignableObjectId::typeToName(align::TPBModule));
  }
  /// Return endcap  GeomDets
  Alignables& endcapGeomDets() {
    return this->subStructures(AlignableObjectId::typeToName(align::TECModule));
  }
  /// Return TID  GeomDets  
  Alignables& TIDGeomDets() {
    return this->subStructures(AlignableObjectId::typeToName(align::TIDModule));
  }
  /// Return pixel endcap GeomDets
  Alignables& pixelEndcapGeomDets() {
    return this->subStructures(AlignableObjectId::typeToName(align::TPEModule));
  }
  
  /// Return inner and outer barrel rods
  Alignables barrelRods() { return this->merge(this->innerBarrelRods(), this->outerBarrelRods());}
  /// Return inner barrel rods
  Alignables& innerBarrelRods() {
    return this->subStructures(AlignableObjectId::typeToName(align::TIBString));
  }
  /// Return outer barrel rods
  Alignables& outerBarrelRods() {
    return this->subStructures(AlignableObjectId::typeToName(align::TOBRod));
  }
  /// Return pixel half barrel ladders (implemented as AlignableRods)
  Alignables& pixelHalfBarrelLadders() {
    return this->subStructures(AlignableObjectId::typeToName(align::TPBLadder));
  }
  /// Return encap petals
  Alignables& endcapPetals() {
    return this->subStructures(AlignableObjectId::typeToName(align::TECPetal));
  }
  /// Return TID rings
  Alignables& TIDRings() {
    return this->subStructures(AlignableObjectId::typeToName(align::TIDRing));
  }
  /// Return pixel endcap petals
  Alignables& pixelEndcapPetals() {
    return this->subStructures(AlignableObjectId::typeToName(align::TPEPanel));
  }
		     
  /// Return inner and outer barrel layers
  Alignables barrelLayers() { return this->merge(this->innerBarrelLayers(),
						 this->outerBarrelLayers() );
  }
  /// Return inner barrel layers
  Alignables& innerBarrelLayers() {
    return this->subStructures(AlignableObjectId::typeToName(align::TIBLayer));
  }
  /// Return outer barrel layers
  Alignables& outerBarrelLayers() {
    return this->subStructures(AlignableObjectId::typeToName(align::TOBLayer));
  }
  /// Return pixel half barrel layers
  Alignables& pixelHalfBarrelLayers() {
    return this->subStructures(AlignableObjectId::typeToName(align::TPBLayer));
  }
  /// Return endcap layers
  Alignables& endcapLayers() {
    return this->subStructures(AlignableObjectId::typeToName(align::TECDisk));
  }
  /// Return TID layers
  Alignables& TIDLayers() {
    return this->subStructures(AlignableObjectId::typeToName(align::TIDDisk));
  }
  /// Return pixel endcap layers
  Alignables& pixelEndcapLayers() {
    return this->subStructures(AlignableObjectId::typeToName(align::TPEHalfDisk));
  }



  /// Return alignments, sorted by DetId
  Alignments* alignments() const;

  /// Return alignment errors, sorted by DetId
  AlignmentErrorsExtended* alignmentErrors() const;

  /// Return tracker topology used to build AlignableTracker
  const TrackerTopology* trackerTopology() const { return tTopo_;}

private:
  Alignables merge( const Alignables& list1, const Alignables& list2 ) const;

  const TrackerTopology* tTopo_;
  AlignableMap alignableMap;

};

#endif //AlignableTracker_H
