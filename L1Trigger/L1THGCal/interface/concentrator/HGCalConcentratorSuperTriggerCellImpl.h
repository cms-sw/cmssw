#ifndef __L1Trigger_L1THGCal_HGCalConcentratorSuperTriggerCellImpl_h__
#define __L1Trigger_L1THGCal_HGCalConcentratorSuperTriggerCellImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include "L1Trigger/L1THGCal/interface/HGCalCoarseTriggerCellMapping.h"

#include <array>
#include <vector>

class HGCalConcentratorSuperTriggerCellImpl {
public:
  HGCalConcentratorSuperTriggerCellImpl(const edm::ParameterSet& conf);

  void select(const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput,
              std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput);
  void eventSetup(const edm::EventSetup& es) {
    triggerTools_.eventSetup(es);
    coarseTCmapping_.eventSetup(es);
    superTCmapping_.eventSetup(es);
  }

private:
  enum EnergyDivisionType {
    superTriggerCell,
    oneBitFraction,
    equalShare,
  };
  EnergyDivisionType energyDivisionType_;
  static constexpr int kHighDensityThickness_ = 0;
  static constexpr int kOddNumberMask_ = 1;

  HGCalTriggerTools triggerTools_;
  bool fixedDataSizePerHGCROC_;
  HGCalCoarseTriggerCellMapping coarseTCmapping_;
  HGCalCoarseTriggerCellMapping superTCmapping_;

  //Parameters for energyDivisionType_ = oneBitFraction
  double oneBitFractionThreshold_;
  double oneBitFractionLowValue_;
  double oneBitFractionHighValue_;

  //Parameters for energyDivisionType_ = equalShare
  static constexpr int kTriggerCellsForDivision_ = 4;

  class SuperTriggerCell {
  private:
    float sumPt_, sumMipPt_, maxMipPt_, fracsum_;
    int sumHwPt_;
    uint32_t maxId_, stcId_;
    std::map<uint32_t, float> tc_pts_;

  public:
    SuperTriggerCell() : sumPt_(0), sumMipPt_(0), maxMipPt_(0), fracsum_(0), sumHwPt_(0), maxId_(0), stcId_(0){};

    void add(const l1t::HGCalTriggerCell& c, uint32_t stcId) {
      sumPt_ += c.pt();
      sumMipPt_ += c.mipPt();
      sumHwPt_ += c.hwPt();
      if (maxId_ == 0 || c.mipPt() > maxMipPt_) {
        maxMipPt_ = c.mipPt();
        maxId_ = c.detId();
      }

      if (stcId_ == 0) {
        stcId_ = stcId;
      }
      tc_pts_[c.detId()] = c.mipPt();
    }
    void addToFractionSum(float frac) {
      fracsum_ += frac;
      if (fracsum_ > 1) {
        throw cms::Exception("HGCalConcentratorSuperTriggerCellError")
            << "Sum of Trigger Cell fractions should not be greater than 1";
      }
    }
    uint32_t getMaxId() const { return maxId_; }
    uint32_t getSTCId() const { return stcId_; }
    float getSumMipPt() const { return sumMipPt_; }
    int getSumHwPt() const { return sumHwPt_; }
    float getSumPt() const { return sumPt_; }
    float getFractionSum() const { return fracsum_; }
    float getTCpt(uint32_t tcid) const {
      const auto pt = tc_pts_.find(tcid);
      return (pt == tc_pts_.end() ? 0 : pt->second);
    }
    int size() const { return tc_pts_.size(); }
  };
  void createAllTriggerCells(std::unordered_map<unsigned, SuperTriggerCell>& STCs,
                             std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput) const;
  void assignSuperTriggerCellEnergyAndPosition(l1t::HGCalTriggerCell& c, const SuperTriggerCell& stc) const;
  float getTriggerCellOneBitFraction(float tcPt, float sumPt) const;
};

#endif
