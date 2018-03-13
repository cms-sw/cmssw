#ifndef DataFormats_ForwardDetId_HGCScintillatorDetId_H
#define DataFormats_ForwardDetId_HGCScintillatorDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

/* \brief description of the bit assigment
   [0:8]   iphi index wrt x-axis on +z side
   [9:16]  |ieta| index (starting from |etamin|)
   [17:21] Layer #
   [22:22] z-side (0 for +z; 1 for -z)
   [23:23] Type (0 fine divisions; 1 for coarse division)
   [24:24] Reserved for future extension
   [28:31] Detector type (HGCalHSc)
*/

class HGCScintillatorDetId : public DetId {

public:

  /** Create a null cellid*/
  HGCScintillatorDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HGCScintillatorDetId(uint32_t rawid);
  /** Constructor from subdetector, zplus, layer, module, cell numbers */
  HGCScintillatorDetId(int type, int layer, int ieta, int iphi);
  /** Constructor from a generic cell id */
  HGCScintillatorDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HGCScintillatorDetId& operator=(const DetId& id);
  
  /** Converter for a geometry cell id */
  HGCScintillatorDetId geometryCell () const {return HGCScintillatorDetId (type(), layer(), ieta(), 0);}

  /// get the subdetector
  ForwardSubdetector subdet() const { return HGCHEB; }

  /// get the type
  int type() const { return (id_>>kHGCalTypeOffset)&kHGCalTypeMask; }

  /// get the z-side of the cell (1/-1)
  int zside() const { return (((id_>>kHGCalZsideOffset) & kHGCalZsideMask) ? -1 : 1); }

  /// get the layer #
  int layer() const { return (id_>>kHGCalLayerOffset)&kHGCalLayerMask; }

  /// get the eta index
  int ietaAbs() const { return (id_>>kHGCalEtaOffset)&kHGCalEtaMask; }
  int ieta()    const { return zside()*ietaAbs(); }

  /// get the phi index
  int iphi() const { return (id_>>kHGCalPhiOffset)&kHGCalPhiMask; }
  std::pair<int,int> ietaphi() const { return std::pair<int,int>(ieta(),iphi()); }

  /// consistency check : no bits left => no overhead
  bool isEE()      const { return false; }
  bool isHE()      const { return true; }
  bool isForward() const { return true; }
  
  static const HGCScintillatorDetId Undefined;

private:

  static const int kHGCalPhiOffset      = 0;
  static const int kHGCalPhiMask        = 0x1FF;
  static const int kHGCalEtaOffset      = 9;
  static const int kHGCalEtaMask        = 0xFF;
  static const int kHGCalLayerOffset    = 17;
  static const int kHGCalLayerMask      = 0x1F;
  static const int kHGCalZsideOffset    = 22;
  static const int kHGCalZsideMask      = 0x1;
  static const int kHGCalZsideMask2     = 0x400000;
  static const int kHGCalTypeOffset     = 23;
  static const int kHGCalTypeMask       = 0x1;
};

std::ostream& operator<<(std::ostream&,const HGCScintillatorDetId& id);

#endif
