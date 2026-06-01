#ifndef L1Trigger_Phase2L1GMT_KMTFLUTS_h
#define L1Trigger_Phase2L1GMT_KMTFLUTS_h
#include <cstdlib>
#include <cmath>
#include "TH1.h"
#include "TFile.h"
#include <map>
#include "FWCore/ParameterSet/interface/FileInPath.h"

namespace Phase2L1GMT {

  class KMTFLUTs {
  public:
    KMTFLUTs(const std::string &filename, const std::string &ThetaFilename) {
      edm::FileInPath path(filename);
      edm::FileInPath pathTheta(ThetaFilename);
      lutFile_ = new TFile(path.fullPath().c_str());
      lutThetaFile_ = new TFile(pathTheta.fullPath().c_str());
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
		
	  lutTheta1D_[1 * 64 + 0] = (TH1 *)lutThetaFile_->Get("gain1D_0_1");
	  lutTheta1D_[1 * 64 + 2] = (TH1 *)lutThetaFile_->Get("gain1D_2_1");
	  lutTheta1D_[1 * 64 + 4] = (TH1 *)lutThetaFile_->Get("gain1D_4_1");
	  lutTheta1D_[1 * 64 + 6] = (TH1 *)lutThetaFile_->Get("gain1D_6_1");
	  lutTheta1D_[2 * 64 + 4] = (TH1 *)lutThetaFile_->Get("gain1D_4_2");
	  lutTheta1D_[3 * 64 + 0] = (TH1 *)lutThetaFile_->Get("gain1D_0_3");
	  lutTheta2D_[1 * 64 + 0] = (TH1 *)lutThetaFile_->Get("gain2D_0_1");
	  lutTheta2D_[1 * 64 + 2] = (TH1 *)lutThetaFile_->Get("gain2D_2_1");
	  lutTheta2D_[1 * 64 + 4] = (TH1 *)lutThetaFile_->Get("gain2D_4_1");
	  lutTheta2D_[1 * 64 + 6] = (TH1 *)lutThetaFile_->Get("gain2D_6_1");
	  lutTheta2D_[2 * 64 + 0] = (TH1 *)lutThetaFile_->Get("gain2D_0_2");
	  lutTheta2D_[2 * 64 + 4] = (TH1 *)lutThetaFile_->Get("gain2D_4_2");
	  lutTheta2D_[3 * 64 + 0] = (TH1 *)lutThetaFile_->Get("gain2D_0_3");
	  lutTheta1D11_[2 * 64 + 0] = (TH1 *)lutThetaFile_->Get("gain1D_0_2_phi1100");
	  lutTheta1D01_[2 * 64 + 0]= (TH1 *)lutThetaFile_->Get("gain1D_0_2_phi0100");
	  lutTheta1D10_[2 * 64 + 0] = (TH1 *)lutThetaFile_->Get("gain1D_0_2_phi1000");
    }

    ~KMTFLUTs() {
      lutFile_->Close();
      lutThetaFile_->Close();
      if (lutFile_ != nullptr){
        delete lutFile_;
	  }
	  if (lutThetaFile_ != nullptr){
        delete lutThetaFile_;
      }
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
      return uint((1 << 12) * coarseEta_->GetBinContent(coarseEta_->GetXaxis()->FindBin(mask)) / M_PI);
    }

	std::vector<float> trackGainTheta(uint step, uint bitmask, uint K, bool is2D) {
      std::vector<float> gain(4, 0.0);
      const TH1 *h;
	  if (is2D){
      	h = lutTheta2D_[64 * step + bitmask];
		}
	  else {
      	h = lutTheta1D_[64 * step + bitmask];
		}
      gain[0] = h->GetBinContent(K + 1);
      gain[1] = h->GetBinContent(512 + K + 1);
      gain[2] = h->GetBinContent(2 * 512 + K + 1);
      gain[3] = h->GetBinContent(3 * 512 + K + 1);
      return gain;
    }

    std::vector<float> trackGainTheta2(uint step, uint bitmask, uint phiBitmask, uint K) {
      std::vector<float> gain(4, 0.0);
      const TH1 *h;
	  if (phiBitmask == 0b1100){
      	h = lutTheta1D11_[64 * step + bitmask];
		}
	  else if (phiBitmask == 0b1000){
      	h = lutTheta1D10_[64 * step + bitmask];
		}
	  else {
      	h = lutTheta1D01_[64 * step + bitmask];
		}
      gain[0] = h->GetBinContent(K + 1);
      gain[1] = h->GetBinContent(512 + K + 1);
      gain[2] = h->GetBinContent(2 * 512 + K + 1);
      gain[3] = h->GetBinContent(3 * 512 + K + 1);
      return gain;
    }

    TFile *lutFile_;
    TFile *lutThetaFile_;
    std::map<uint, const TH1 *> lut_;
    std::map<uint, const TH1 *> lut2HH_;
    std::map<uint, const TH1 *> lut2LH_;
    std::map<uint, const TH1 *> lut2HL_;
    std::map<uint, const TH1 *> lut2LL_;

    std::map<uint, const TH1 *> lutTheta1D_;
    std::map<uint, const TH1 *> lutTheta2D_;
    std::map<uint, const TH1 *> lutTheta1D11_;
    std::map<uint, const TH1 *> lutTheta1D10_;
    std::map<uint, const TH1 *> lutTheta1D01_;
    const TH1 *coarseEta_;
  };

}  // namespace Phase2L1GMT
#endif
