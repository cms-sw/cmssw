#ifndef DataFormats_ForwardDetId_HGCScintillatorDetId_H
#define DataFormats_ForwardDetId_HGCScintillatorDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

/* \brief description of the bit assigment
   [0:8]   iphi index wrt x-axis on +z side
   [9:16]  |radius| index (starting from a minimum radius depending on type)
   [17:21] Layer #
   [22:24] Reserved for future extension
   [25:25] z-side (0 for +z; 1 for -z)
   [26:27] Type (0 fine divisions of scintillators;
                 1 coarse divisions of scintillators)
   [28:31] Detector type (HGCalHSc)
*/

class HGCScintillatorDetId : public DetId {

public:

  /** Create a null cellid*/
  HGCScintillatorDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HGCScintillatorDetId(uint32_t rawid);
  /** Constructor from subdetector, zplus, layer, module, cell numbers */
  HGCScintillatorDetId(int type, int layer, int iradius, int iphi);
  /** Constructor from a generic cell id */
  HGCScintillatorDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HGCScintillatorDetId& operator=(const DetId& id);
  
  /** Converter for a geometry cell id */
  HGCScintillatorDetId geometryCell () const {return HGCScintillatorDetId (0, layer(), iradius(), iphi());}

  /// get the subdetector
  DetId::Detector subdet() const { return det(); }

  /// get the type
  int type() const { return (id_>>kHGCalTypeOffset)&kHGCalTypeMask; }

  /// get the z-side of the cell (1/-1)
  int zside() const { return (((id_>>kHGCalZsideOffset) & kHGCalZsideMask) ? -1 : 1); }

  /// get the layer #
  int layer() const { return (id_>>kHGCalLayerOffset)&kHGCalLayerMask; }

  /// get the eta index
  int iradiusAbs() const { return (id_>>kHGCalRadiusOffset)&kHGCalRadiusMask; }
  int iradius()    const { return zside()*iradiusAbs(); }
  int ietaAbs()    const { return (id_>>kHGCalRadiusOffset)&kHGCalRadiusMask; }
  int ieta()       const { return zside()*ietaAbs(); }

  /// get the phi index
  int iphi() const { return (id_>>kHGCalPhiOffset)&kHGCalPhiMask; }
  std::pair<int,int> ietaphi() const { return std::pair<int,int>(ieta(),iphi()); }
  std::pair<int,int> iradiusphi() const { return std::pair<int,int>(iradius(),iphi()); }

  /// consistency check : no bits left => no overhead
  bool isEE()      const { return false; }
  bool isHE()      const { return true; }
  bool isForward() const { return true; }
  
  static const HGCScintillatorDetId Undefined;

private:

  static const int kHGCalPhiOffset      = 0;
  static const int kHGCalPhiMask        = 0x1FF;
  static const int kHGCalRadiusOffset   = 9;
  static const int kHGCalRadiusMask     = 0xFF;
  static const int kHGCalLayerOffset    = 17;
  static const int kHGCalLayerMask      = 0x1F;
  static const int kHGCalZsideOffset    = 25;
  static const int kHGCalZsideMask      = 0x1;
  static const int kHGCalTypeOffset     = 26;
  static const int kHGCalTypeMask       = 0x3;
};

std::ostream& operator<<(std::ostream&,const HGCScintillatorDetId& id);

#endif
