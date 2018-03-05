#ifndef __L1TMuonBarelKalmanLUTs_h
#define __L1TMuonBarrelKalmanLUTs_h

#include <stdlib.h>
#include "TH1.h"
#include "TFile.h"
#include <map>

class L1TMuonBarrelKalmanLUTs {
 public:
  L1TMuonBarrelKalmanLUTs(const std::string&);
  ~L1TMuonBarrelKalmanLUTs();

  std::vector<float> trackGain(uint,uint, uint);
  std::pair<float,float> vertexGain(uint, uint);

 private:
  TFile *lutFile_;
  std::map<uint,const TH1*> lut_;

};

#endif
