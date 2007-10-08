#ifndef Alignment_TrackerAlignment_AlignableTracker_H
#define Alignment_TrackerAlignment_AlignableTracker_H

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignSetup.h"

class GeometricDet;
class TrackerGeometry;

class AlignableTracker: public AlignableComposite 
{

public:
  
  /// Constructor (builds the full hierarchy)
  AlignableTracker(
		   const GeometricDet* = 0,   // dummy for backward compatibility
		   const TrackerGeometry* = 0 // dummy for backward compatibility
		   ); 

  /// Return TOB half barrels
  const Alignables& outerHalfBarrels() const { return AlignSetup<Alignables>::find("TOBHalfBarrel"); }
  /// Return TIB half barrels
  const Alignables& innerHalfBarrels() const { return AlignSetup<Alignables>::find("TIBHalfBarrel"); }
  /// Return Pixel half barrels
  const Alignables& pixelHalfBarrels() const { return AlignSetup<Alignables>::find("TPBHalfBarrel"); }
  /// Return TECs
  const Alignables& endCaps() const { return AlignSetup<Alignables>::find("TECEndcap"); }
  /// Return TPEs
  const Alignables& pixelEndCaps() const { return AlignSetup<Alignables>::find("TPEEndcap"); }
  /// Return TIDs
  const Alignables& TIDs() const { return AlignSetup<Alignables>::find("TIDEndcap"); }

  /// Return inner and outer barrel GeomDets together 
  Alignables barrelGeomDets() const { return merge( innerBarrelGeomDets(), outerBarrelGeomDets() ); }
  /// Return inner barrel and TID GeomDets together 
  Alignables TIBTIDGeomDets() const { return merge( innerBarrelGeomDets(), TIDGeomDets() ); }
  /// Return inner barrel GeomDets 
  const Alignables& innerBarrelGeomDets() const { return AlignSetup<Alignables>::find("TIBModule"); }
  /// Return outer barrel GeomDets
  const Alignables& outerBarrelGeomDets() const { return AlignSetup<Alignables>::find("TOBModule"); }
  /// Return pixel barrel GeomDets
  const Alignables& pixelHalfBarrelGeomDets() const { return AlignSetup<Alignables>::find("TPBModule"); }
  /// Return endcap  GeomDets
  const Alignables& endcapGeomDets() const { return AlignSetup<Alignables>::find("TECModule"); }
  /// Return TID  GeomDets  
  const Alignables& TIDGeomDets() const { return AlignSetup<Alignables>::find("TIDModule"); }
  /// Return pixel endcap GeomDets
  const Alignables& pixelEndcapGeomDets() const { return AlignSetup<Alignables>::find("TPEModule"); }
  
  /// Return inner and outer barrel rods
  Alignables barrelRods() const { return merge( innerBarrelRods(), outerBarrelRods() ); }
  /// Return inner barrel rods
  const Alignables& innerBarrelRods() const { return AlignSetup<Alignables>::find("TIBString"); }
  /// Return outer barrel rods
  const Alignables& outerBarrelRods() const { return AlignSetup<Alignables>::find("TOBRod"); }
  /// Return pixel half barrel ladders (implemented as AlignableRods)
  const Alignables& pixelHalfBarrelLadders() const { return AlignSetup<Alignables>::find("TPBLadder"); }
  /// Return encap petals
  const Alignables& endcapPetals() const { return AlignSetup<Alignables>::find("TECPetal"); }
  /// Return TID rings
  const Alignables& TIDRings() const { return AlignSetup<Alignables>::find("TIDRing"); }
  /// Return pixel endcap petals
  const Alignables& pixelEndcapPetals() const { return AlignSetup<Alignables>::find("TPEPanel"); }
		     
//   /// Return inner and outer barrel layers
  Alignables barrelLayers() const { return merge( innerBarrelLayers(), outerBarrelLayers() ); }
  /// Return inner barrel layers
  const Alignables& innerBarrelLayers() const { return AlignSetup<Alignables>::find("TIBLayer"); }
  /// Return outer barrel layers
  const Alignables& outerBarrelLayers() const { return AlignSetup<Alignables>::find("TOBLayer"); }
  /// Return pixel half barrel layers
  const Alignables& pixelHalfBarrelLayers() const { return AlignSetup<Alignables>::find("TPBLayer"); }
  /// Return endcap layers
  const Alignables& endcapLayers() const { return AlignSetup<Alignables>::find("TECDisk"); }
  /// Return TID layers
  const Alignables& TIDLayers() const { return AlignSetup<Alignables>::find("TIDDisk"); }
  /// Return pixel endcap layers
  const Alignables& pixelEndcapLayers() const { return AlignSetup<Alignables>::find("TPEHalfDisk"); }

  /// Return alignments, sorted by DetId
  Alignments* alignments() const;

  /// Return alignment errors, sorted by DetId
  AlignmentErrors* alignmentErrors() const;

  private:

  /// Build a barrel for a given sub-detector (TPB, TIB, TOB).
  void buildBarrel(
		   const std::string& subDet // prefix for sub-detector
		   ) const;

  void buildTPB() const;
  void buildTPE() const;
  void buildTIB() const;
  void buildTID() const;
  void buildTOB() const;
  void buildTEC() const;
  void buildTRK();

  Alignables merge(
		   const Alignables& list1,
		   const Alignables& list2
		   ) const;
};

#endif //AlignableTracker_H
