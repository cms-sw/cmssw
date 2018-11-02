#ifndef DataFormats_ForwardDetId_HGCSiliconDetId_H
#define DataFormats_ForwardDetId_HGCSiliconDetId_H 1

#include <iosfwd>
#include "DataFormats/ForwardDetId/interface/WaferDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

/* \brief description of the bit assigment
   [0:4]   u-coordinate of the cell (measured from the lower left
   [5:9]   v-coordinate of the cell  corner of the wafer)
   [10:13] abs(u) of the wafer (u-axis points along -x axis)
   [14:14] sign of u (0:+u; 1:-u) (u=0 is at the center of beam line)
   [15:18] abs(v) of the wafer (v-axis points 60-degree wrt x-axis)
   [19:19] sign of v (0:+v; 1:-v) (v=0 is at the center of beam line)
   [20:24] layer number 
   [25:25] z-side (0 for +z; 1 for -z)
   [26:27] Type (0 fine divisions of wafer with 120 mum thick silicon
                 1 coarse divisions of wafer with 200 mum thick silicon
                 2 coarse divisions of wafer with 300 mum thick silicon)
   [28:31] Detector type (HGCalEE or HGCalHSi)
*/
class HGCSiliconDetId : public WaferDetId {

public:

  enum waferType {HGCalFine=0, HGCalCoarseThin=1, HGCalCoarseThick=2};
  static const int HGCalFineN  =12;
  static const int HGCalCoarseN=8;
  static const int HGCalFineTrigger  =3;
  static const int HGCalCoarseTrigger=2;

  /** Create a null cellid*/
  HGCSiliconDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  HGCSiliconDetId(uint32_t rawid);
  /** Constructor from subdetector, zplus, layer, module, cell numbers */
  HGCSiliconDetId(DetId::Detector det, int zp, int type, int layer, int waferU,
	     int waferV, int cellU, int cellV);
  /** Constructor from a generic cell id */
  HGCSiliconDetId(const DetId& id);
  /** Assignment from a generic cell id */
  HGCSiliconDetId& operator=(const DetId& id);
  
  /** Converter for a geometry cell id */
  HGCSiliconDetId geometryCell () const {return HGCSiliconDetId (det(), zside(), 0, layer(), waferU(), waferV(), 0, 0);}

  /// get the subdetector
  DetId::Detector subdet() const { return det(); }

  /// get the type
  int type() const override { return (id_>>kHGCalTypeOffset)&kHGCalTypeMask; }

  /// get the z-side of the cell (1/-1)
  int zside() const override { return (((id_>>kHGCalZsideOffset) & kHGCalZsideMask) ? -1 : 1); }

  /// get the layer #
  int layer() const override { return (id_>>kHGCalLayerOffset)&kHGCalLayerMask; }

  /// get the cell #'s in u,v or in x,y
  int cellU() const override { return (id_>>kHGCalCellUOffset)&kHGCalCellUMask; }
  int cellV() const override { return (id_>>kHGCalCellVOffset)&kHGCalCellVMask; }
  int getN()  const override { return ((type() == HGCalFine) ? HGCalFineN : HGCalCoarseN); }

  /// get the wafer #'s in u,v or in x,y
  int waferUAbs() const override { return (id_>>kHGCalWaferUOffset)&kHGCalWaferUMask; }
  int waferVAbs() const override { return (id_>>kHGCalWaferVOffset)&kHGCalWaferVMask; }
  int waferU()    const override { return (((id_>>kHGCalWaferUSignOffset) & kHGCalWaferUSignMask) ? -waferUAbs() : waferUAbs()); }
  int waferV()    const override { return (((id_>>kHGCalWaferVSignOffset) & kHGCalWaferVSignMask) ? -waferVAbs() : waferVAbs()); }

  // get trigger cell u,v
  int triggerCellU() const {
    int N  = (type() == HGCalFine) ? HGCalFineN : HGCalCoarseN;
    int NT = (type() == HGCalFine) ? HGCalFineTrigger : HGCalCoarseTrigger;
    return (cellU() >= N && cellV() >= N) ? cellU()/NT :
      ((cellU() < N && cellU() <= cellV()) ? cellU()/NT : 
       (1+(cellU()-(cellV()%NT+1))/NT));
  }
  int triggerCellV() const {
    int N  = (type() == HGCalFine) ? HGCalFineN : HGCalCoarseN;
    int NT = (type() == HGCalFine) ? HGCalFineTrigger : HGCalCoarseTrigger;
    return (cellU() >= N && cellV() >= N) ? cellV()/NT :
      ((cellU() < N && cellU() <= cellV()) ? ((cellV()-cellU())/NT+cellU()/NT) : 
       cellV()/NT);
  }
  std::pair<int,int> triggerCellUV() const {
    return std::pair<int,int>(triggerCellU(),triggerCellV());
  }

  /// consistency check : no bits left => no overhead
  bool isEE() const override { return (det() == HGCalEE); }
  bool isHE() const override { return (det() == HGCalHSi); }
  bool isForward() const { return true; }
  
  static const HGCSiliconDetId Undefined;

public:

  static const int kHGCalCellUOffset      = 0;
  static const int kHGCalCellUMask        = 0x1F;
  static const int kHGCalCellVOffset      = 5;
  static const int kHGCalCellVMask        = 0x1F;
  static const int kHGCalWaferUOffset     = 10;
  static const int kHGCalWaferUMask       = 0xF;
  static const int kHGCalWaferUSignOffset = 14;
  static const int kHGCalWaferUSignMask   = 0x1;
  static const int kHGCalWaferVOffset     = 15;
  static const int kHGCalWaferVMask       = 0xF;
  static const int kHGCalWaferVSignOffset = 19;
  static const int kHGCalWaferVSignMask   = 0x1;
  static const int kHGCalLayerOffset      = 20;
  static const int kHGCalLayerMask        = 0x1F;
  static const int kHGCalZsideOffset      = 25;
  static const int kHGCalZsideMask        = 0x1;
  static const int kHGCalTypeOffset       = 26;
  static const int kHGCalTypeMask         = 0x3;
};

std::ostream& operator<<(std::ostream&,const HGCSiliconDetId& id);

#endif
