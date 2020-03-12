#ifndef __L1Trigger_L1THGCal_HGCalCoarseTriggerCellMapping_h__
#define __L1Trigger_L1THGCal_HGCalCoarseTriggerCellMapping_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetIdToROC.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalCoarseTriggerCellMapping {
public:
  HGCalCoarseTriggerCellMapping(const std::vector<unsigned>& ctcSize);
  uint32_t getRepresentativeDetId(uint32_t tcid) const;
  std::vector<uint32_t> getConstituentTriggerCells(uint32_t ctcId) const;
  GlobalPoint getCoarseTriggerCellPosition(uint32_t ctcId) const;
  uint32_t getCoarseTriggerCellId(uint32_t detid) const;
  void checkSizeValidity(int ctcSize) const;
  void eventSetup(const edm::EventSetup& es) { triggerTools_.eventSetup(es); }

  static constexpr int kCTCsizeCoarse_ = 16;
  static constexpr int kCTCsizeMid_ = 8;
  static constexpr int kCTCsizeFine_ = 4;
  static constexpr int kCTCsizeVeryFine_ = 2;
  static constexpr int kCTCsizeIndividual_ = 1;

private:
  static const std::map<int, int> kSplit_;
  static const std::map<int, int> kSplit_v9_;
  static const std::map<int, int> kSplit_v9_Scin_;
  static constexpr int kSTCidMaskInv_ = ~0x3f;
  static constexpr int kSTCidMaskInv_v9_ = ~0xf;
  static constexpr int kSplit_v8_Coarse_ = 0x30;
  static constexpr int kSplit_v8_Mid_ = 0x38;
  static constexpr int kSplit_v8_Fine_ = 0x3a;
  static constexpr int kSplit_v8_VeryFine_ = 0x3e;
  static constexpr int kSplit_v8_Individual_ = 0x3f;
  static constexpr int kNThicknesses_ = 4;
  static constexpr int kNHGCalLayersMax_ = 52;
  static constexpr int kSplit_v9_Coarse_ = 0;
  static constexpr int kSplit_v9_Mid_ = 0x2;
  static constexpr int kSplit_v9_Fine_ = 0xa;
  static constexpr int kSplit_v9_VeryFine_ = 0xb;
  static constexpr int kSplit_v9_Individual_ = 0xf;

  static constexpr int kSplit_v9_Scin_Coarse_ = 0x1f9fc;
  static constexpr int kSplit_v9_Scin_Mid_ = 0x1fdfc;
  static constexpr int kSplit_v9_Scin_Fine_ = 0x1fdfe;
  static constexpr int kSplit_v9_Scin_VeryFine_ = 0x1fffe;
  static constexpr int kSplit_v9_Scin_Individual_ = 0x1ffff;

  //For coarse TCs
  static constexpr int kRocShift_ = 4;
  static constexpr int kRocMask_ = 0xf;
  static constexpr int kRotate4_ = 4;
  static constexpr int kRotate7_ = 7;
  static constexpr int kUShift_ = 2;
  static constexpr int kVShift_ = 0;
  static constexpr int kUMask_ = 0x3;
  static constexpr int kVMask_ = 0x3;
  static constexpr int kHGCalCellMaskV9Inv_ = ~0xff;
  static constexpr int kHGCalScinCellMaskInv_ = ~0x1ffff;

  static constexpr int kRoc0deg_ = 1;
  static constexpr int kRoc120deg_ = 2;
  static constexpr int kRoc240deg_ = 3;

  HGCalTriggerTools triggerTools_;
  HGCSiliconDetIdToROC detIdToROC_;
  std::vector<unsigned> ctcSize_;
};

#endif
