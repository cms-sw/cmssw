#ifndef DataFormats_ForwardDetId_HFNoseDetId_H
#define DataFormats_ForwardDetId_HFNoseDetId_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/WaferDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

/* \brief description of the bit assigment
   [0:4]   u-coordinate of the cell (measured from the lower left
   [5:9]   v-coordinate of the cell  corner of the wafer)
   [10:12] abs(u) of the wafer (u-axis points along -x axis)
   [13:13] sign of u (0:+u; 1:-u) (u=0 is at the center of beam line)
   [14:16] abs(v) of the wafer (v-axis points 60-degree wrt x-axis)
   [17:17] sign of v (0:+v; 1:-v) (v=0 is at the center of beam line)
   [18:21] layer number 
   [22:22] z-side (0 for +z; 1 for -z)
   [23:24] Type (0 fine divisions of wafer with 120 mum thick silicon
                 1 coarse divisions of wafer with 200 mum thick silicon
                 2 coarse divisions of wafer with 300 mum thick silicon)
   [25:27] Subdetector type (HFNose)
   [28:31] Detector type (Forward)
*/
class HFNoseDetId : public WaferDetId {

public:

  enum hfNoseWaferType {HFNoseFine=0, HFNoseCoarseThin=1, HFNoseCoarseThick=2};
  static const int HFNoseFineN  =12;
  static const int HFNoseCoarseN=8;
  static const int HFNoseFineTrigger  =3;
  static const int HFNoseCoarseTrigger=2;

  /** Create a null cellid*/
  HFNoseDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HFNoseDetId(uint32_t rawid);
  /** Constructor from subdetector, zplus, layer, module, cell numbers */
  HFNoseDetId(int zp, int type, int layer, int waferU, int waferV, int cellU,
	      int cellV);
  /** Constructor from a generic cell id */
  HFNoseDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HFNoseDetId& operator=(const DetId& id);

  /// get the subdetector
  ForwardSubdetector subdet() const { return HFNose; }
  
  /** Converter for a geometry cell id */
  HFNoseDetId geometryCell () const {return HFNoseDetId (zside(), 0, layer(), waferU(), waferV(), 0, 0);}

  /// get the type
  int type() const override { return (id_>>kHFNoseTypeOffset)&kHFNoseTypeMask; }

  /// get the z-side of the cell (1/-1)
  int zside() const override { return (((id_>>kHFNoseZsideOffset) & kHFNoseZsideMask) ? -1 : 1); }

  /// get the layer #
  int layer() const override { return (id_>>kHFNoseLayerOffset)&kHFNoseLayerMask; }

  /// get the cell #'s in u,v or in x,y
  int cellU() const override { return (id_>>kHFNoseCellUOffset)&kHFNoseCellUMask; }
  int cellV() const override { return (id_>>kHFNoseCellVOffset)&kHFNoseCellVMask; }
  int getN()  const override { return ((type() == HFNoseFine) ? HFNoseFineN : HFNoseCoarseN); }

  /// get the wafer #'s in u,v or in x,y
  int waferUAbs() const override { return (id_>>kHFNoseWaferUOffset)&kHFNoseWaferUMask; }
  int waferVAbs() const override { return (id_>>kHFNoseWaferVOffset)&kHFNoseWaferVMask; }
  int waferU() const override { return (((id_>>kHFNoseWaferUSignOffset) & kHFNoseWaferUSignMask) ? -waferUAbs() : waferUAbs()); }
  int waferV() const override { return (((id_>>kHFNoseWaferVSignOffset) & kHFNoseWaferVSignMask) ? -waferVAbs() : waferVAbs()); }

  /// consistency check : no bits left => no overhead
  bool isEE()      const override { return (layer() <= kHFNoseLayerEEmax); }
  bool isHE()      const override { return (layer() >  kHFNoseLayerEEmax); }
  bool isForward() const { return true; }
  
  static const HFNoseDetId Undefined;

private:

  static const int kHFNoseLayerEEmax       = 6;
  static const int kHFNoseCellUOffset      = 0;
  static const int kHFNoseCellUMask        = 0x1F;
  static const int kHFNoseCellVOffset      = 5;
  static const int kHFNoseCellVMask        = 0x1F;
  static const int kHFNoseWaferUOffset     = 10;
  static const int kHFNoseWaferUMask       = 0x7;
  static const int kHFNoseWaferUSignOffset = 13;
  static const int kHFNoseWaferUSignMask   = 0x1;
  static const int kHFNoseWaferVOffset     = 14;
  static const int kHFNoseWaferVMask       = 0x7;
  static const int kHFNoseWaferVSignOffset = 17;
  static const int kHFNoseWaferVSignMask   = 0x1;
  static const int kHFNoseLayerOffset      = 18;
  static const int kHFNoseLayerMask        = 0xF;
  static const int kHFNoseZsideOffset      = 22;
  static const int kHFNoseZsideMask        = 0x1;
  static const int kHFNoseTypeOffset       = 23;
  static const int kHFNoseTypeMask         = 0x3;
};

std::ostream& operator<<(std::ostream&,const HFNoseDetId& id);

#endif
