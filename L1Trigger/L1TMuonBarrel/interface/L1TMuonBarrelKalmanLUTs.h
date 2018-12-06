#ifndef L1Trigger_L1TMuonBarrel_L1TMuonBarrelKalmanLUTs_h
#define L1Trigger_L1TMuonBarrel_L1TMuonBarrelKalmanLUTs_h

#include <cstdlib>
#include "TH1.h"
#include "TFile.h"
#include <map>

class L1TMuonBarrelKalmanLUTs {
 public:
  L1TMuonBarrelKalmanLUTs(const std::string&);
  ~L1TMuonBarrelKalmanLUTs();

  std::vector<float> trackGain(uint,uint, uint);
  std::vector<float> trackGain2(uint,uint, uint);
  std::pair<float,float> vertexGain(uint, uint);
  uint coarseEta(uint, uint);

 private:
  TFile *lutFile_;
  std::map<uint,const TH1*> lut_;
  std::map<uint,const TH1*> lut2_;
  std::map<uint,const TH1*> coarseEta_;

};

#endif
