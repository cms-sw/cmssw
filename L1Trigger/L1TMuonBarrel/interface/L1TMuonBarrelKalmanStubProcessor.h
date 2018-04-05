#ifndef L1TMUONBARRELKALMANSTUBPROCESSOR
#define L1TMUONBARRELKALMANSTUBPROCESSOR

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1TMuon/interface/L1MuKBMTCombinedStub.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsRcd.h"

class L1MuDTTFMasks;


class L1TMuonBarrelKalmanStubProcessor {
 public:
  L1TMuonBarrelKalmanStubProcessor();
  L1TMuonBarrelKalmanStubProcessor(const edm::ParameterSet&);
  
  ~L1TMuonBarrelKalmanStubProcessor();


  L1MuKBMTCombinedStubCollection makeStubs(const L1MuDTChambPhContainer*,const L1MuDTChambThContainer*);

  
 private:
  bool isGoodPhiStub(const L1MuDTChambPhDigi*); 
  std::pair<bool,bool> isGoodThetaStub(const L1MuDTChambThDigi*,uint pos1,uint pos2=0); 
  L1MuKBMTCombinedStub buildStub(const L1MuDTChambPhDigi*,const L1MuDTChambThDigi*);
  int minPhiQuality_;
  int minThetaQuality_;
  int minBX_;
  int maxBX_;

  std::vector<int>  etaLUT_minus_2_1;
  std::vector<int>  etaLUT_minus_2_2;
  std::vector<int>  etaLUT_minus_2_3;
  std::vector<int>  etaLUT_minus_1_1;
  std::vector<int>  etaLUT_minus_1_2;
  std::vector<int>  etaLUT_minus_1_3;
  std::vector<int>  etaLUT_0_1;
  std::vector<int>  etaLUT_0_2;
  std::vector<int>  etaLUT_0_3;
  std::vector<int>  etaLUT_plus_1_1;
  std::vector<int>  etaLUT_plus_1_2;
  std::vector<int>  etaLUT_plus_1_3;
  std::vector<int>  etaLUT_plus_2_1;
  std::vector<int>  etaLUT_plus_2_2;
  std::vector<int>  etaLUT_plus_2_3;
  std::vector<int>  etaCoarseLUT_minus_2;
  std::vector<int>  etaCoarseLUT_minus_1;
  std::vector<int>  etaCoarseLUT_0;
  std::vector<int>  etaCoarseLUT_plus_1;
  std::vector<int>  etaCoarseLUT_plus_2;

  int verbose_;


  //    edm::ESHandle< L1TMuonBarrelParams > bmtfParamsHandle;
  //    L1MuDTTFMasks       masks_;


};


#endif
