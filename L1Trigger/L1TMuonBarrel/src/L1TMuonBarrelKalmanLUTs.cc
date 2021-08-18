#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanLUTs.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

L1TMuonBarrelKalmanLUTs::L1TMuonBarrelKalmanLUTs(const std::string& filename) {
  edm::FileInPath path(filename);
  lutFile_ = new TFile(path.fullPath().c_str());

  lut_[3 * 64 + 8] = (TH1*)lutFile_->Get("gain_8_3");
  lut_[2 * 64 + 8] = (TH1*)lutFile_->Get("gain_8_2");
  lut_[2 * 64 + 12] = (TH1*)lutFile_->Get("gain_12_2");
  lut_[2 * 64 + 4] = (TH1*)lutFile_->Get("gain_4_2");
  lut_[1 * 64 + 12] = (TH1*)lutFile_->Get("gain_12_1");
  lut_[1 * 64 + 10] = (TH1*)lutFile_->Get("gain_10_1");
  lut_[1 * 64 + 6] = (TH1*)lutFile_->Get("gain_6_1");
  lut_[1 * 64 + 14] = (TH1*)lutFile_->Get("gain_14_1");
  lut_[3] = (TH1*)lutFile_->Get("gain_3_0");
  lut_[5] = (TH1*)lutFile_->Get("gain_5_0");
  lut_[6] = (TH1*)lutFile_->Get("gain_6_0");
  lut_[7] = (TH1*)lutFile_->Get("gain_7_0");
  lut_[9] = (TH1*)lutFile_->Get("gain_9_0");
  lut_[10] = (TH1*)lutFile_->Get("gain_10_0");
  lut_[11] = (TH1*)lutFile_->Get("gain_11_0");
  lut_[12] = (TH1*)lutFile_->Get("gain_12_0");
  lut_[13] = (TH1*)lutFile_->Get("gain_13_0");
  lut_[14] = (TH1*)lutFile_->Get("gain_14_0");
  lut_[15] = (TH1*)lutFile_->Get("gain_15_0");

  lut2HH_[3 * 64 + 8] = (TH1*)lutFile_->Get("gain2_8_3_HH");
  lut2HH_[2 * 64 + 8] = (TH1*)lutFile_->Get("gain2_8_2_HH");
  lut2HH_[2 * 64 + 4] = (TH1*)lutFile_->Get("gain2_4_2_HH");
  lut2HH_[1 * 64 + 8] = (TH1*)lutFile_->Get("gain2_8_1_HH");
  lut2HH_[1 * 64 + 4] = (TH1*)lutFile_->Get("gain2_4_1_HH");
  lut2HH_[1 * 64 + 2] = (TH1*)lutFile_->Get("gain2_2_1_HH");

  lut2LH_[3 * 64 + 8] = (TH1*)lutFile_->Get("gain2_8_3_LH");
  lut2LH_[2 * 64 + 8] = (TH1*)lutFile_->Get("gain2_8_2_LH");
  lut2LH_[2 * 64 + 4] = (TH1*)lutFile_->Get("gain2_4_2_LH");
  lut2LH_[1 * 64 + 8] = (TH1*)lutFile_->Get("gain2_8_1_LH");
  lut2LH_[1 * 64 + 4] = (TH1*)lutFile_->Get("gain2_4_1_LH");
  lut2LH_[1 * 64 + 2] = (TH1*)lutFile_->Get("gain2_2_1_LH");

  lut2HL_[3 * 64 + 8] = (TH1*)lutFile_->Get("gain2_8_3_HL");
  lut2HL_[2 * 64 + 8] = (TH1*)lutFile_->Get("gain2_8_2_HL");
  lut2HL_[2 * 64 + 4] = (TH1*)lutFile_->Get("gain2_4_2_HL");
  lut2HL_[1 * 64 + 8] = (TH1*)lutFile_->Get("gain2_8_1_HL");
  lut2HL_[1 * 64 + 4] = (TH1*)lutFile_->Get("gain2_4_1_HL");
  lut2HL_[1 * 64 + 2] = (TH1*)lutFile_->Get("gain2_2_1_HL");

  lut2LL_[3 * 64 + 8] = (TH1*)lutFile_->Get("gain2_8_3_LL");
  lut2LL_[2 * 64 + 8] = (TH1*)lutFile_->Get("gain2_8_2_LL");
  lut2LL_[2 * 64 + 4] = (TH1*)lutFile_->Get("gain2_4_2_LL");
  lut2LL_[1 * 64 + 8] = (TH1*)lutFile_->Get("gain2_8_1_LL");
  lut2LL_[1 * 64 + 4] = (TH1*)lutFile_->Get("gain2_4_1_LL");
  lut2LL_[1 * 64 + 2] = (TH1*)lutFile_->Get("gain2_2_1_LL");

  coarseEta_[3] = (TH1*)lutFile_->Get("coarseEta_3");
  coarseEta_[5] = (TH1*)lutFile_->Get("coarseEta_5");
  coarseEta_[6] = (TH1*)lutFile_->Get("coarseEta_6");
  coarseEta_[7] = (TH1*)lutFile_->Get("coarseEta_7");
  coarseEta_[9] = (TH1*)lutFile_->Get("coarseEta_9");

  coarseEta_[10] = (TH1*)lutFile_->Get("coarseEta_10");
  coarseEta_[11] = (TH1*)lutFile_->Get("coarseEta_11");
  coarseEta_[12] = (TH1*)lutFile_->Get("coarseEta_12");
  coarseEta_[13] = (TH1*)lutFile_->Get("coarseEta_13");
  coarseEta_[14] = (TH1*)lutFile_->Get("coarseEta_14");
  coarseEta_[15] = (TH1*)lutFile_->Get("coarseEta_15");
}

L1TMuonBarrelKalmanLUTs::~L1TMuonBarrelKalmanLUTs() {
  lutFile_->Close();
  if (lutFile_ != nullptr)
    delete lutFile_;
}

std::vector<float> L1TMuonBarrelKalmanLUTs::trackGain(uint step, uint bitmask, uint K) {
  std::vector<float> gain(4, 0.0);
  const TH1* h = lut_[64 * step + bitmask];
  gain[0] = h->GetBinContent(K + 1);
  gain[2] = h->GetBinContent(1024 + K + 1);
  return gain;
}

std::vector<float> L1TMuonBarrelKalmanLUTs::trackGain2(uint step, uint bitmask, uint K, uint qual1, uint qual2) {
  std::vector<float> gain(4, 0.0);

  //  printf("Track gain %d %d %d\n",step,bitmask,K);
  const TH1* h;
  if (qual1 < 4) {
    if (qual2 < 4)
      h = lut2LL_[64 * step + bitmask];
    else
      h = lut2LH_[64 * step + bitmask];
  } else {
    if (qual2 < 4)
      h = lut2HL_[64 * step + bitmask];
    else
      h = lut2HH_[64 * step + bitmask];
  }
  gain[0] = h->GetBinContent(K + 1);
  gain[1] = h->GetBinContent(512 + K + 1);
  gain[2] = h->GetBinContent(2 * 512 + K + 1);
  gain[3] = h->GetBinContent(3 * 512 + K + 1);
  return gain;
}

std::pair<float, float> L1TMuonBarrelKalmanLUTs::vertexGain(uint bitmask, uint K) {
  const TH1* h = lut_[bitmask];
  std::pair<float, float> gain(-h->GetBinContent(K + 1), -h->GetBinContent(1024 + K + 1));
  return gain;
}

uint L1TMuonBarrelKalmanLUTs::coarseEta(uint pattern, uint mask) {
  const TH1* h = coarseEta_[pattern];
  return uint(h->GetBinContent(h->GetXaxis()->FindBin(mask)));
}
