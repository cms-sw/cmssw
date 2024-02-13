#ifndef L1Trigger_Phase2L1GMT_KMTFLUTS_h
#define L1Trigger_Phase2L1GMT_KMTFLUTS_h
#include <cstdlib>
#include "TH1.h"
#include "TFile.h"
#include <map>
#include "FWCore/ParameterSet/interface/FileInPath.h"

namespace Phase2L1GMT {

  class KMTFLUTs {
  public:
    KMTFLUTs(const std::string &filename) {
      edm::FileInPath path(filename);
      lutFile_ = new TFile(path.fullPath().c_str());
      lut_[3 * 64 + 8] = (TH1 *)lutFile_->Get("gain_8_3");
      lut_[2 * 64 + 8] = (TH1 *)lutFile_->Get("gain_8_2");
      lut_[2 * 64 + 12] = (TH1 *)lutFile_->Get("gain_12_2");
      lut_[2 * 64 + 4] = (TH1 *)lutFile_->Get("gain_4_2");
      lut_[1 * 64 + 12] = (TH1 *)lutFile_->Get("gain_12_1");
      lut_[1 * 64 + 10] = (TH1 *)lutFile_->Get("gain_10_1");
      lut_[1 * 64 + 6] = (TH1 *)lutFile_->Get("gain_6_1");
      lut_[1 * 64 + 14] = (TH1 *)lutFile_->Get("gain_14_1");
      lut_[3] = (TH1 *)lutFile_->Get("gain_3_0");
      lut_[5] = (TH1 *)lutFile_->Get("gain_5_0");
      lut_[6] = (TH1 *)lutFile_->Get("gain_6_0");
      lut_[7] = (TH1 *)lutFile_->Get("gain_7_0");
      lut_[9] = (TH1 *)lutFile_->Get("gain_9_0");
      lut_[10] = (TH1 *)lutFile_->Get("gain_10_0");
      lut_[11] = (TH1 *)lutFile_->Get("gain_11_0");
      lut_[12] = (TH1 *)lutFile_->Get("gain_12_0");
      lut_[13] = (TH1 *)lutFile_->Get("gain_13_0");
      lut_[14] = (TH1 *)lutFile_->Get("gain_14_0");
      lut_[15] = (TH1 *)lutFile_->Get("gain_15_0");

      lut2HH_[3 * 64 + 8] = (TH1 *)lutFile_->Get("gain2_8_3_HH");
      lut2HH_[2 * 64 + 8] = (TH1 *)lutFile_->Get("gain2_8_2_HH");
      lut2HH_[2 * 64 + 4] = (TH1 *)lutFile_->Get("gain2_4_2_HH");
      lut2HH_[1 * 64 + 8] = (TH1 *)lutFile_->Get("gain2_8_1_HH");
      lut2HH_[1 * 64 + 4] = (TH1 *)lutFile_->Get("gain2_4_1_HH");
      lut2HH_[1 * 64 + 2] = (TH1 *)lutFile_->Get("gain2_2_1_HH");

      lut2LH_[3 * 64 + 8] = (TH1 *)lutFile_->Get("gain2_8_3_LH");
      lut2LH_[2 * 64 + 8] = (TH1 *)lutFile_->Get("gain2_8_2_LH");
      lut2LH_[2 * 64 + 4] = (TH1 *)lutFile_->Get("gain2_4_2_LH");
      lut2LH_[1 * 64 + 8] = (TH1 *)lutFile_->Get("gain2_8_1_LH");
      lut2LH_[1 * 64 + 4] = (TH1 *)lutFile_->Get("gain2_4_1_LH");
      lut2LH_[1 * 64 + 2] = (TH1 *)lutFile_->Get("gain2_2_1_LH");

      lut2HL_[3 * 64 + 8] = (TH1 *)lutFile_->Get("gain2_8_3_HL");
      lut2HL_[2 * 64 + 8] = (TH1 *)lutFile_->Get("gain2_8_2_HL");
      lut2HL_[2 * 64 + 4] = (TH1 *)lutFile_->Get("gain2_4_2_HL");
      lut2HL_[1 * 64 + 8] = (TH1 *)lutFile_->Get("gain2_8_1_HL");
      lut2HL_[1 * 64 + 4] = (TH1 *)lutFile_->Get("gain2_4_1_HL");
      lut2HL_[1 * 64 + 2] = (TH1 *)lutFile_->Get("gain2_2_1_HL");

      lut2LL_[3 * 64 + 8] = (TH1 *)lutFile_->Get("gain2_8_3_LL");
      lut2LL_[2 * 64 + 8] = (TH1 *)lutFile_->Get("gain2_8_2_LL");
      lut2LL_[2 * 64 + 4] = (TH1 *)lutFile_->Get("gain2_4_2_LL");
      lut2LL_[1 * 64 + 8] = (TH1 *)lutFile_->Get("gain2_8_1_LL");
      lut2LL_[1 * 64 + 4] = (TH1 *)lutFile_->Get("gain2_4_1_LL");
      lut2LL_[1 * 64 + 2] = (TH1 *)lutFile_->Get("gain2_2_1_LL");

      coarseEta_ = (TH1 *)lutFile_->Get("coarseETALUT");
    }

    ~KMTFLUTs() {
      lutFile_->Close();
      if (lutFile_ != nullptr)
        delete lutFile_;
    }

    std::vector<float> trackGain(uint step, uint bitmask, uint K) {
      std::vector<float> gain(4, 0.0);
      const TH1 *h = lut_[64 * step + bitmask];
      gain[0] = h->GetBinContent(K + 1);
      gain[2] = h->GetBinContent(1024 + K + 1);
      return gain;
    }

    std::vector<float> trackGain2(uint step, uint bitmask, uint K, uint qual1, uint qual2) {
      std::vector<float> gain(4, 0.0);
      const TH1 *h;
      if (qual1 < 6) {
        if (qual2 < 6)
          h = lut2LL_[64 * step + bitmask];
        else
          h = lut2LH_[64 * step + bitmask];
      } else {
        if (qual2 < 6)
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

    std::pair<float, float> vertexGain(uint bitmask, uint K) {
      const TH1 *h = lut_[bitmask];
      std::pair<float, float> gain(-h->GetBinContent(K + 1), -h->GetBinContent(1024 + K + 1));
      return gain;
    }

    uint coarseEta(uint mask) {
      //    printf("histo=%f  out=%d\n\n",coarseEta_->GetBinContent(coarseEta_->GetXaxis()->FindBin(mask)),uint((1<<12)*coarseEta_->GetBinContent(coarseEta_->GetXaxis()->FindBin(mask))/M_PI));
      return uint((1 << 12) * coarseEta_->GetBinContent(coarseEta_->GetXaxis()->FindBin(mask)) / M_PI);
    }

    TFile *lutFile_;
    std::map<uint, const TH1 *> lut_;
    std::map<uint, const TH1 *> lut2HH_;
    std::map<uint, const TH1 *> lut2LH_;
    std::map<uint, const TH1 *> lut2HL_;
    std::map<uint, const TH1 *> lut2LL_;
    const TH1 *coarseEta_;
  };

}  // namespace Phase2L1GMT
#endif
