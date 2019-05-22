#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorSuperTriggerCellImpl.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"

#include <unordered_map>

HGCalConcentratorSuperTriggerCellImpl::HGCalConcentratorSuperTriggerCellImpl(const edm::ParameterSet& conf)
    : stcSize_(conf.getParameter<std::vector<unsigned> >("stcSize")) {
  if (stcSize_.size() != kNLayers_) {
    throw cms::Exception("HGCTriggerParameterError")
        << "Inconsistent size of super trigger cell size vector" << stcSize_.size();
  }
  for (auto stc : stcSize_) {
    if (stc != kSTCsizeFine_ && stc != kSTCsizeCoarse_) {
      throw cms::Exception("HGCTriggerParameterError")
          << "Super Trigger Cell should be of size " << kSTCsizeFine_ << " or " << kSTCsizeCoarse_;
    }
  }
}

const std::map<int, int> HGCalConcentratorSuperTriggerCellImpl::kSplit_ = {{kSTCsizeFine_, kSplit_v8_Fine_},
                                                                           {kSTCsizeCoarse_, kSplit_v8_Coarse_}};

int HGCalConcentratorSuperTriggerCellImpl::getSuperTriggerCellId(int detid) const {
  DetId TC_id(detid);
  if (TC_id.det() == DetId::Forward) {  //V8

    HGCalDetId TC_idV8(detid);

    if (triggerTools_.isScintillator(detid)) {
      return TC_idV8.cell();  //scintillator
    } else {
      int TC_wafer = TC_idV8.wafer();
      int thickness = triggerTools_.thicknessIndex(detid, true);
      int TC_split = (TC_idV8.cell() & kSplit_.at(stcSize_.at(thickness)));

      return TC_wafer << kWafer_offset_ | TC_split;
    }

  }

  else if (TC_id.det() == DetId::HGCalTrigger) {  //V9

    if (triggerTools_.isScintillator(detid)) {
      HGCScintillatorDetId TC_idV9(detid);
      return TC_idV9.ietaAbs() << HGCScintillatorDetId::kHGCalPhiOffset | TC_idV9.iphi();  //scintillator
    } else {
      HGCalTriggerDetId TC_idV9(detid);

      int TC_wafer = TC_idV9.waferU() << kWafer_offset_ | TC_idV9.waferV();
      int thickness = triggerTools_.thicknessIndex(detid);

      int TC_12th = 0;
      int Uprime = 0;
      int Vprime = 0;
      int rocnum = detIdToROC_.getROCNumber(TC_idV9.triggerCellU(), TC_idV9.triggerCellV(), 1);

      if (rocnum == 1) {
        Uprime = TC_idV9.triggerCellU();
        Vprime = TC_idV9.triggerCellV() - TC_idV9.triggerCellU();

      } else if (rocnum == 2) {
        Uprime = TC_idV9.triggerCellU() - TC_idV9.triggerCellV() - 1;
        Vprime = TC_idV9.triggerCellV();

      } else if (rocnum == 3) {
        Uprime = TC_idV9.triggerCellU() - kRotate4_;
        Vprime = TC_idV9.triggerCellV() - kRotate4_;
      }

      TC_12th = (rocnum << kRocShift_) | ((Uprime << kUShift_ | Vprime) & kSplit_v9_);

      int TC_split = TC_12th;
      if (stcSize_.at(thickness) == kSTCsizeCoarse_) {
        TC_split = rocnum;
      }

      return TC_wafer << kWafer_offset_ | TC_split;
    }

  } else {
    return -1;
  }
}

void HGCalConcentratorSuperTriggerCellImpl::superTriggerCellSelectImpl(
    const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput, std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput) {
  std::unordered_map<unsigned, SuperTriggerCell> STCs;

  // first pass, fill the super trigger cells
  for (const l1t::HGCalTriggerCell& tc : trigCellVecInput) {
    if (tc.subdetId() == HGCHEB)
      continue;
    STCs[getSuperTriggerCellId(tc.detId())].add(tc);
  }

  // second pass, write them out
  for (const l1t::HGCalTriggerCell& tc : trigCellVecInput) {
    //If scintillator use a simple threshold cut
    if (tc.subdetId() == HGCHEB) {
      trigCellVecOutput.push_back(tc);
    } else {
      const auto& stc = STCs[getSuperTriggerCellId(tc.detId())];
      if (tc.detId() == stc.GetMaxId()) {
        trigCellVecOutput.push_back(tc);
        stc.assignEnergy(trigCellVecOutput.back());
      }
    }

  }  // end of second loop
}
