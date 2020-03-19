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

const std::map<int, int> HGCalCoarseTriggerCellMapping::kSplit_ = {{kCTCsizeIndividual_, kSplit_v8_Individual_},
                                                                   {kCTCsizeVeryFine_, kSplit_v8_VeryFine_},
                                                                   {kCTCsizeFine_, kSplit_v8_Fine_},
                                                                   {kCTCsizeMid_, kSplit_v8_Mid_},
                                                                   {kCTCsizeCoarse_, kSplit_v8_Coarse_}};

const std::map<int, int> HGCalCoarseTriggerCellMapping::kSplit_v9_ = {
    {kCTCsizeIndividual_, kSplit_v9_Individual_},
    {kCTCsizeVeryFine_, kSplit_v9_VeryFine_},
    {kCTCsizeFine_, kSplit_v9_Fine_},
    {kCTCsizeMid_, kSplit_v9_Mid_},
    {kCTCsizeCoarse_, kSplit_v9_Coarse_},
};

const std::map<int, int> HGCalCoarseTriggerCellMapping::kSplit_v9_Scin_ = {
    {kCTCsizeIndividual_, kSplit_v9_Scin_Individual_},
    {kCTCsizeVeryFine_, kSplit_v9_Scin_VeryFine_},
    {kCTCsizeFine_, kSplit_v9_Scin_Fine_},
    {kCTCsizeMid_, kSplit_v9_Scin_Mid_},
    {kCTCsizeCoarse_, kSplit_v9_Scin_Coarse_},
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
  int thickness = triggerTools_.thicknessIndex(detid, true);

  int ctcSize = ctcSize_.at(thickness * kNHGCalLayersMax_ + layer);

  DetId tc_Id(detid);
  if (tc_Id.det() == DetId::Forward) {  //V8

    HGCalDetId tc_IdV8(detid);

    if (triggerTools_.isScintillator(detid)) {
      return detid;  //stc not available in scintillator for v8
    } else {
      int tcSplit = (tc_IdV8.cell() & kSplit_.at(ctcSize));
      detid = (detid & ~(HGCalDetId::kHGCalCellMask)) | tcSplit;
      return detid;
    }

  }

  else if (tc_Id.det() == DetId::HGCalTrigger || tc_Id.det() == DetId::HGCalHSc) {  //V9
    if (triggerTools_.isScintillator(detid)) {
      HGCScintillatorDetId tc_IdV9(detid);

      int tcSplit = (((tc_IdV9.ietaAbs() - 1) << HGCScintillatorDetId::kHGCalRadiusOffset) | (tc_IdV9.iphi() - 1)) &
                    kSplit_v9_Scin_.at(ctcSize);

      detid = (detid & kHGCalScinCellMaskInv_) | tcSplit;

      return detid;

    } else {
      HGCalTriggerDetId tc_IdV9(detid);

      int uPrime = 0;
      int vPrime = 0;
      int rocnum = detIdToROC_.getROCNumber(tc_IdV9.triggerCellU(), tc_IdV9.triggerCellV(), 1);

      if (rocnum == kRoc0deg_) {
        uPrime = tc_IdV9.triggerCellU();
        vPrime = tc_IdV9.triggerCellV() - tc_IdV9.triggerCellU();

      } else if (rocnum == kRoc120deg_) {
        uPrime = tc_IdV9.triggerCellU() - tc_IdV9.triggerCellV() - 1;
        vPrime = tc_IdV9.triggerCellV();

      } else if (rocnum == kRoc240deg_) {
        uPrime = tc_IdV9.triggerCellV() - kRotate4_;
        vPrime = kRotate7_ - tc_IdV9.triggerCellU();
      }

      int tcSplit = (rocnum << kRocShift_) | ((uPrime << kUShift_ | vPrime) & kSplit_v9_.at(ctcSize));
      detid = (detid & kHGCalCellMaskV9Inv_) | tcSplit;
      return detid;
    }

  } else {
    return 0;
  }
}

std::vector<uint32_t> HGCalCoarseTriggerCellMapping::getConstituentTriggerCells(uint32_t ctcId) const {
  int thickness = triggerTools_.thicknessIndex(ctcId, true);
  unsigned int layer = triggerTools_.layerWithOffset(ctcId);
  int ctcSize = ctcSize_.at(thickness * kNHGCalLayersMax_ + layer);

  std::vector<uint32_t> output_ids;
  DetId tc_Id(ctcId);

  if (tc_Id.det() == DetId::Forward) {  //V8

    if (triggerTools_.isScintillator(ctcId)) {
      output_ids.emplace_back(ctcId);  //stc not available in scintillator for v8
    } else {
      int splitInv = ~(kSTCidMaskInv_ | kSplit_.at(ctcSize));
      for (int i = 0; i < splitInv + 1; i++) {
        if ((i & splitInv) != i)
          continue;

        output_ids.emplace_back(ctcId | i);
      }
    }
  } else if (tc_Id.det() == DetId::HGCalTrigger || tc_Id.det() == DetId::HGCalHSc) {  //V9

    if (triggerTools_.isScintillator(ctcId)) {
      int splitInv = ~(kHGCalScinCellMaskInv_ | kSplit_v9_Scin_.at(ctcSize));
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
      int splitInv = ~(kSTCidMaskInv_v9_ | kSplit_v9_.at(ctcSize));
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

        uint32_t outid = ctcId & kHGCalCellMaskV9Inv_;
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
