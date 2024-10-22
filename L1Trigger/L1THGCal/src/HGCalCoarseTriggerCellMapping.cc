#include "L1Trigger/L1THGCal/interface/HGCalCoarseTriggerCellMapping.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"

HGCalCoarseTriggerCellMapping::HGCalCoarseTriggerCellMapping(const std::vector<unsigned>& ctcSize)
    : ctcSize_(!ctcSize.empty() ? ctcSize
                                : std::vector<unsigned>{kNHGCalLayersMax_ * kNThicknesses_, kCTCsizeVeryFine_}) {
  if (ctcSize_.size() != (kNHGCalLayersMax_ + 1) * kNThicknesses_) {
    throw cms::Exception("HGCTriggerParameterError")
        << "Inconsistent size of coarse trigger cell size vector " << ctcSize_.size();
  }

  for (auto ctc : ctcSize_)
    checkSizeValidity(ctc);
}

const std::map<int, int> HGCalCoarseTriggerCellMapping::kSplit_ = {
    {kCTCsizeIndividual_, kSplit_Individual_},
    {kCTCsizeVeryFine_, kSplit_VeryFine_},
    {kCTCsizeFine_, kSplit_Fine_},
    {kCTCsizeMid_, kSplit_Mid_},
    {kCTCsizeCoarse_, kSplit_Coarse_},
};

const std::map<int, int> HGCalCoarseTriggerCellMapping::kSplit_Scin_ = {
    {kCTCsizeIndividual_, kSplit_Scin_Individual_},
    {kCTCsizeVeryFine_, kSplit_Scin_VeryFine_},
    {kCTCsizeFine_, kSplit_Scin_Fine_},
    {kCTCsizeMid_, kSplit_Scin_Mid_},
    {kCTCsizeCoarse_, kSplit_Scin_Coarse_},
};

void HGCalCoarseTriggerCellMapping::checkSizeValidity(int ctcSize) const {
  if (ctcSize != kCTCsizeFine_ && ctcSize != kCTCsizeCoarse_ && ctcSize != kCTCsizeMid_ &&
      ctcSize != kCTCsizeVeryFine_ && ctcSize != kCTCsizeIndividual_) {
    throw cms::Exception("HGCTriggerParameterError")
        << "Coarse Trigger Cell should be of size " << kCTCsizeIndividual_ << " or " << kCTCsizeVeryFine_ << " or "
        << kCTCsizeFine_ << " or " << kCTCsizeMid_ << " or " << kCTCsizeCoarse_;
  }
}

uint32_t HGCalCoarseTriggerCellMapping::getRepresentativeDetId(uint32_t tcid) const {
  uint32_t representativeid = 0;
  //Firstly get coarse trigger cell Id
  uint32_t ctcId = getCoarseTriggerCellId(tcid);
  //Get list of the constituent TCs, and choose (arbitrarily) the first one
  representativeid = getConstituentTriggerCells(ctcId).at(0);
  if (triggerTools_.getTriggerGeometry()->validTriggerCell(representativeid)) {
    return representativeid;
  } else {
    return tcid;
  }
}

uint32_t HGCalCoarseTriggerCellMapping::getCoarseTriggerCellId(uint32_t detid) const {
  unsigned int layer = triggerTools_.layerWithOffset(detid);
  int thickness = triggerTools_.thicknessIndex(detid);

  int ctcSize = ctcSize_.at(thickness * kNHGCalLayersMax_ + layer);

  DetId tc_Id(detid);
  if (tc_Id.det() == DetId::HGCalTrigger || tc_Id.det() == DetId::HGCalHSc) {
    if (triggerTools_.isScintillator(detid)) {
      HGCScintillatorDetId tc_Id(detid);

      int tcSplit = (((tc_Id.ietaAbs() - 1) << HGCScintillatorDetId::kHGCalRadiusOffset) | (tc_Id.iphi() - 1)) &
                    kSplit_Scin_.at(ctcSize);

      detid = (detid & kHGCalScinCellMaskInv_) | tcSplit;

      return detid;

    } else {
      HGCalTriggerDetId tc_Id(detid);

      int uPrime = 0;
      int vPrime = 0;
      int rocnum = detIdToROC_.getROCNumber(tc_Id.triggerCellU(), tc_Id.triggerCellV(), 1);

      if (rocnum == kRoc0deg_) {
        uPrime = tc_Id.triggerCellU();
        vPrime = tc_Id.triggerCellV() - tc_Id.triggerCellU();

      } else if (rocnum == kRoc120deg_) {
        uPrime = tc_Id.triggerCellU() - tc_Id.triggerCellV() - 1;
        vPrime = tc_Id.triggerCellV();

      } else if (rocnum == kRoc240deg_) {
        uPrime = tc_Id.triggerCellV() - kRotate4_;
        vPrime = kRotate7_ - tc_Id.triggerCellU();
      }

      int tcSplit = (rocnum << kRocShift_) | ((uPrime << kUShift_ | vPrime) & kSplit_.at(ctcSize));
      detid = (detid & kHGCalCellMaskInv_) | tcSplit;
      return detid;
    }

  } else {
    return 0;
  }
}

std::vector<uint32_t> HGCalCoarseTriggerCellMapping::getConstituentTriggerCells(uint32_t ctcId) const {
  int thickness = triggerTools_.thicknessIndex(ctcId);
  unsigned int layer = triggerTools_.layerWithOffset(ctcId);
  int ctcSize = ctcSize_.at(thickness * kNHGCalLayersMax_ + layer);

  std::vector<uint32_t> output_ids;
  DetId tc_Id(ctcId);

  if (tc_Id.det() == DetId::HGCalTrigger || tc_Id.det() == DetId::HGCalHSc) {
    if (triggerTools_.isScintillator(ctcId)) {
      int splitInv = ~(kHGCalScinCellMaskInv_ | kSplit_Scin_.at(ctcSize));
      for (int i = 0; i < splitInv + 1; i++) {
        if ((i & splitInv) != i)
          continue;

        HGCScintillatorDetId prime = (ctcId | i);
        unsigned outid = (ctcId & kHGCalScinCellMaskInv_) |
                         (((prime.iradiusAbs() + 1) << HGCScintillatorDetId::kHGCalRadiusOffset) | (prime.iphi() + 1));

        if (triggerTools_.getTriggerGeometry()->validTriggerCell(outid)) {
          output_ids.emplace_back(outid);
        }
      }

    } else {
      int splitInv = ~(kSTCidMaskInv_ | kSplit_.at(ctcSize));
      for (int i = 0; i < splitInv + 1; i++) {
        if ((i & splitInv) != i)
          continue;
        int uPrime = ((ctcId | i) >> kUShift_) & kUMask_;
        int vPrime = ((ctcId | i) >> kVShift_) & kVMask_;
        int rocnum = (ctcId >> kRocShift_) & kRocMask_;

        int u = 0;
        int v = 0;

        if (rocnum == kRoc0deg_) {
          u = uPrime;
          v = vPrime + u;
        } else if (rocnum == kRoc120deg_) {
          u = uPrime + vPrime + 1;
          v = vPrime;
        } else if (rocnum == kRoc240deg_) {
          u = kRotate7_ - vPrime;
          v = uPrime + kRotate4_;
        }

        uint32_t outid = ctcId & kHGCalCellMaskInv_;
        outid |= (((u & HGCalTriggerDetId::kHGCalCellUMask) << HGCalTriggerDetId::kHGCalCellUOffset) |
                  ((v & HGCalTriggerDetId::kHGCalCellVMask) << HGCalTriggerDetId::kHGCalCellVOffset));

        if (triggerTools_.getTriggerGeometry()->validTriggerCell(outid)) {
          output_ids.emplace_back(outid);
        }
      }
    }
  }
  return output_ids;
}

GlobalPoint HGCalCoarseTriggerCellMapping::getCoarseTriggerCellPosition(uint32_t ctcId) const {
  std::vector<uint32_t> constituentTCs = getConstituentTriggerCells(ctcId);
  Basic3DVector<float> average_vector(0., 0., 0.);

  for (const auto constituent : constituentTCs) {
    average_vector += triggerTools_.getTCPosition(constituent).basicVector();
  }

  GlobalPoint average_point(average_vector / constituentTCs.size());
  return average_point;
}
