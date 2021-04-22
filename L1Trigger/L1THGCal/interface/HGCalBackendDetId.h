#ifndef L1Trigger_L1THGCal_HGCalBackendDetId_H
#define L1Trigger_L1THGCal_HGCalBackendDetId_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "L1Trigger/L1THGCal/interface/HGCalBackendCommon.h"

/* \brief description of the bit assigment
   [0:10]  ID of the lpGBT or Stage 1 FPGA in sector 0 
   [11:12] sector (0,1,2 counter-clockwise from u-axis)
   [13:15] Type (0 lpGBT
                 1 Stage 1 FPGA
                 2 Stage 1 link
                 3 Stage 2 FPGA)
   [16:16] z-side (0 for +z; 1 for -z)
   [19:23] reserved for future use
   [24:24] Class identifier (0 for HGCalModuleDetID, 1 for HGCalBackendDetID)
   [25:27] Subdetector Type (HGCTrigger)
   [28:31] Detector type (Forward)
*/

class HGCalBackendDetId : public DetId {
public:
  /** Create a null backend id*/
  HGCalBackendDetId();
  /** Create backend id from raw id (0=invalid id) */
  HGCalBackendDetId(uint32_t rawid);
  /** Constructor from zplus, type, sector, label */
  HGCalBackendDetId(int zp, int type, int sector, int label);
  /** Constructor from a generic det id */
  HGCalBackendDetId(const DetId& id);
  /** Assignment from a generic det id */
  HGCalBackendDetId& operator=(const DetId& id);

  /// get the class
  int classId() const { return (id_ >> kHGCalClassIdentifierOffset) & kHGCalClassIdentifierMask; }

  /// get the type
  int type() const { return (id_ >> kHGCalTypeOffset) & kHGCalTypeMask; }

  /// get the z-side of the backend object (1/-1)
  int zside() const { return ((id_ >> kHGCalZsideOffset) & kHGCalZsideMask ? -1 : 1); }

  /// get the sector #
  int sector() const { return (id_ >> kHGCalSectorOffset) & kHGCalSectorMask; }

  /// get the value
  int label() const { return (id_ >> kHGCalLabelOffset) & kHGCalLabelMask; }

  bool isLpGBT() const { return (type() == BackendType::LpGBT); }
  bool isStage1FPGA() const { return (type() == BackendType::Stage1FPGA); }
  bool isStage1Link() const { return (type() == BackendType::Stage1Link); }
  bool isStage2FPGA() const { return (type() == BackendType::Stage2FPGA); }
  bool isForward() const { return true; }
  bool isHGCalModuleDetId() const { return (classId() == HGCalClassIdentifier::ModuleDetId); }
  bool isHGCalBackendDetId() const { return (classId() == HGCalClassIdentifier::BackendDetId); }

  static const HGCalBackendDetId Undefined;

  static const int kHGCalLabelOffset = 0;
  static const int kHGCalLabelMask = 0x7FF;
  static const int kHGCalSectorOffset = 11;
  static const int kHGCalSectorMask = 0x3;
  static const int kHGCalTypeOffset = 13;
  static const int kHGCalTypeMask = 0x7;
  static const int kHGCalZsideOffset = 16;
  static const int kHGCalZsideMask = 0x1;
  static const int kHGCalClassIdentifierOffset = 24;
  static const int kHGCalClassIdentifierMask = 0x1;

  enum BackendType { LpGBT, Stage1FPGA, Stage1Link, Stage2FPGA };
};

std::ostream& operator<<(std::ostream&, const HGCalBackendDetId& id);

#endif
