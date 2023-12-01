#ifndef DataFormats_ForwardDetId_HGCalTriggerModuleDetId_H
#define DataFormats_ForwardDetId_HGCalTriggerModuleDetId_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerBackendCommon.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

/* \brief description of the bit assigment
   [0:3]  u-coordinate of the silicon module (u-axis points along -x axis)
          or eta-coordinate of the scintillator module
   [4:7]  v-coordinate of the silicon module (v-axis points 60-degree wrt x-axis)
          or phi-coordinate of the scintillator module
   [8:9] sector (0,1,2 counter-clockwise from u-axis)

   [10:13] reserved for future use

   [14:18] layer number 
   [19:20] Type (0 fine divisions of wafer with 120 mum thick silicon
                 1 coarse divisions of wafer with 200 mum thick silicon
                 2 coarse divisions of wafer with 300 mum thick silicon
                 0 fine divisions of scintillators
                 1 coarse divisions of scintillators)

   [21:21] z-side (0 for +z; 1 for -z)
   [22:23] Trigger Subdetector Type(HGCEE/HGCHEF/HGCHEB/HFNose) 
   [24:24] Class identifier (0 for HGCalTriggerModuleDetID, 1 for HGCalTriggerBackendDetID)
   [25:27] Subdetector Type (HGCTrigger)
   [28:31] Detector type (Forward)
*/

class HGCalTriggerModuleDetId : public DetId {
public:
  /** Create a null module id*/
  HGCalTriggerModuleDetId();
  /** Create module id from raw id (0=invalid id) */
  HGCalTriggerModuleDetId(uint32_t rawid);
  /** Constructor from subdetector, zplus, type, layer, sector, module numbers */
  HGCalTriggerModuleDetId(
      HGCalTriggerSubdetector subdet, int zp, int type, int layer, int sector, int moduleU, int moduleV);
  /** Constructor from a generic det id */
  HGCalTriggerModuleDetId(const DetId& id);
  /** Assignment from a generic det id */
  HGCalTriggerModuleDetId& operator=(const DetId& id);

  /// get the trigger sub-detector
  int triggerSubdetId() const { return (id_ >> kHGCalTriggerSubdetOffset) & kHGCalTriggerSubdetMask; }

  /// get the class
  int classId() const { return (id_ >> kHGCalTriggerClassIdentifierOffset) & kHGCalTriggerClassIdentifierMask; }

  /// get the type
  int type() const { return (id_ >> kHGCalTypeOffset) & kHGCalTypeMask; }

  /// get the z-side of the module (1/-1)
  int zside() const { return ((id_ >> kHGCalZsideOffset) & kHGCalZsideMask ? -1 : 1); }

  /// get the layer #
  int layer() const { return (id_ >> kHGCalLayerOffset) & kHGCalLayerMask; }

  /// get the sector #
  int sector() const { return (id_ >> kHGCalSectorOffset) & kHGCalSectorMask; }

  /// get the module U
  int moduleU() const { return (id_ >> kHGCalModuleUOffset) & kHGCalModuleUMask; }

  /// get the module V
  int moduleV() const { return (id_ >> kHGCalModuleVOffset) & kHGCalModuleVMask; }

  /// get the scintillator panel eta
  int eta() const { return moduleU(); }

  /// get the scintillator panel phi
  int phi() const { return moduleV(); }

  /// consistency check : no bits left => no overhead
  bool isHFNose() const { return (triggerSubdetId() == HFNoseTrigger); }
  bool isEE() const { return (triggerSubdetId() == HGCalEETrigger); }
  bool isHSilicon() const { return (triggerSubdetId() == HGCalHSiTrigger); }
  bool isHScintillator() const { return (triggerSubdetId() == HGCalHScTrigger); }
  bool isForward() const { return true; }
  bool isHGCTrigger() const { return true; }
  bool isHGCalModuleDetId() const { return (classId() == HGCalTriggerClassIdentifier::ModuleDetId); }
  bool isHGCalBackendDetId() const { return (classId() == HGCalTriggerClassIdentifier::BackendDetId); }

  static const HGCalTriggerModuleDetId Undefined;

  static const int kHGCalModuleUOffset = 0;
  static const int kHGCalModuleUMask = 0xF;
  static const int kHGCalModuleVOffset = 4;
  static const int kHGCalModuleVMask = 0xF;
  static const int kHGCalSectorOffset = 8;
  static const int kHGCalSectorMask = 0x3;
  static const int kHGCalLayerOffset = 14;
  static const int kHGCalLayerMask = 0x1F;
  static const int kHGCalTypeOffset = 19;
  static const int kHGCalTypeMask = 0x3;
  static const int kHGCalZsideOffset = 21;
  static const int kHGCalZsideMask = 0x1;
  static const int kHGCalTriggerSubdetOffset = 22;
  static const int kHGCalTriggerSubdetMask = 0x3;
  static const int kHGCalTriggerClassIdentifierOffset = 24;
  static const int kHGCalTriggerClassIdentifierMask = 0x1;
};

std::ostream& operator<<(std::ostream&, const HGCalTriggerModuleDetId& id);

#endif
