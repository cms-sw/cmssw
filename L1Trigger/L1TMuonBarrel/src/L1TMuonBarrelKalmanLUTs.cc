#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanLUTs.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

L1TMuonBarrelKalmanLUTs::L1TMuonBarrelKalmanLUTs(const std::string& filename) {
  edm::FileInPath path(filename);
  lutFile_ = new TFile(path.fullPath().c_str());


  lut_[15*100+3] = (TH1*)lutFile_->Get("gain_15_3");
  lut_[15*100+2] = (TH1*)lutFile_->Get("gain_15_2");
  lut_[15*100+1] = (TH1*)lutFile_->Get("gain_15_1");
  lut_[15*100] = (TH1*)lutFile_->Get("gain_15_0");
  lut_[14*100+3] = (TH1*)lutFile_->Get("gain_14_3");
  lut_[14*100+2] = (TH1*)lutFile_->Get("gain_14_2");
  lut_[14*100] = (TH1*)lutFile_->Get("gain_14_0");

  lut_[13*100+3] = (TH1*)lutFile_->Get("gain_13_3");
  lut_[13*100+1] = (TH1*)lutFile_->Get("gain_13_1");
  lut_[13*100] = (TH1*)lutFile_->Get("gain_13_0");

  lut_[12*100+3] = (TH1*)lutFile_->Get("gain_12_3");
  lut_[12*100] = (TH1*)lutFile_->Get("gain_12_0");

  lut_[11*100+2] = (TH1*)lutFile_->Get("gain_11_2");
  lut_[11*100+1] = (TH1*)lutFile_->Get("gain_11_1");
  lut_[11*100] = (TH1*)lutFile_->Get("gain_11_0");

  lut_[10*100+2] = (TH1*)lutFile_->Get("gain_10_2");
  lut_[10*100] = (TH1*)lutFile_->Get("gain_10_0");

  lut_[9*100+1] = (TH1*)lutFile_->Get("gain_9_1");
  lut_[9*100] = (TH1*)lutFile_->Get("gain_9_0");

  lut_[7*100+2] = (TH1*)lutFile_->Get("gain_7_2");
  lut_[7*100+1] = (TH1*)lutFile_->Get("gain_7_1");
  lut_[7*100] = (TH1*)lutFile_->Get("gain_7_0");

  lut_[6*100+2] = (TH1*)lutFile_->Get("gain_6_2");
  lut_[6*100] = (TH1*)lutFile_->Get("gain_6_0");

  lut_[5*100+1] = (TH1*)lutFile_->Get("gain_5_1");
  lut_[5*100] = (TH1*)lutFile_->Get("gain_5_0");
  lut_[3*100+1] = (TH1*)lutFile_->Get("gain_3_1");
  lut_[3*100] = (TH1*)lutFile_->Get("gain_3_0");

}

L1TMuonBarrelKalmanLUTs::~L1TMuonBarrelKalmanLUTs() {
  lutFile_->Close();
  if (lutFile_ !=0)
    delete lutFile_;
}


std::vector<float> L1TMuonBarrelKalmanLUTs::trackGain(uint step,uint bitmask,uint K) {
  std::vector<float> gain(4,0.0);
  //  printf("Track gain %d %d %d\n",step,bitmask,K);
  const TH1* h = lut_[100*bitmask+step];
  gain[0] = h->GetBinContent(K+1);
  gain[1] = h->GetBinContent(1024+K+1);
  gain[2] = -h->GetBinContent(2*1024+K+1);
  gain[3] = h->GetBinContent(3*1024+K+1);
  return gain;
}


std::pair<float,float> L1TMuonBarrelKalmanLUTs::vertexGain(uint bitmask,uint K) {
  const TH1* h = lut_[100*bitmask];
  std::pair<float,float> gain(-h->GetBinContent(K+1),-h->GetBinContent(1024+K+1) );
  return gain;
}
