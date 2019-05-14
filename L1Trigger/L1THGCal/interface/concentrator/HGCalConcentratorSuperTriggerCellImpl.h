#ifndef __L1Trigger_L1THGCal_HGCalConcentratorSuperTriggerCellImpl_h__
#define __L1Trigger_L1THGCal_HGCalConcentratorSuperTriggerCellImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetIdToROC.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

#include <array>
#include <vector>

class HGCalConcentratorSuperTriggerCellImpl {
public:
  HGCalConcentratorSuperTriggerCellImpl(const edm::ParameterSet& conf);

  void superTriggerCellSelectImpl(const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput,
                                  std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput);
  void eventSetup(const edm::EventSetup& es) { triggerTools_.eventSetup(es); }

private:
  int getSuperTriggerCellId(int detid) const;
  static const std::map<int, int> kSplit_;
  static constexpr int kWafer_offset_ = 6;
  static constexpr int kSTCsizeCoarse_ = 16;
  static constexpr int kSTCsizeFine_ = 4;
  static constexpr int kSplit_v8_Coarse_ = 0x30;
  static constexpr int kSplit_v8_Fine_ = 0x3a;
  static constexpr int kNLayers_ = 3;
  static constexpr int kSplit_v9_ = 0x36;

  static constexpr int kRocShift_ = 6;
  static constexpr int kRotate4_ = 4;
  static constexpr int kUShift_ = 3;

  HGCalTriggerTools triggerTools_;
  HGCSiliconDetIdToROC detIdToROC_;
  std::vector<unsigned> stcSize_;

  class SuperTriggerCell {
  private:
    float sumPt_, sumMipPt_;
    int sumHwPt_, maxHwPt_;
    unsigned maxId_;

  public:
    SuperTriggerCell() { sumPt_ = 0, sumMipPt_ = 0, sumHwPt_ = 0, maxHwPt_ = 0, maxId_ = 0; }
    void add(const l1t::HGCalTriggerCell& c) {
      sumPt_ += c.pt();
      sumMipPt_ += c.mipPt();
      sumHwPt_ += c.hwPt();
      if (maxId_ == 0 || c.hwPt() > maxHwPt_) {
        maxHwPt_ = c.hwPt();
        maxId_ = c.detId();
      }
    }
    void assignEnergy(l1t::HGCalTriggerCell& c) const {
      c.setHwPt(sumHwPt_);
      c.setMipPt(sumMipPt_);
      c.setPt(sumPt_);
    }
    unsigned GetMaxId() const { return maxId_; }
  };
};

#endif
