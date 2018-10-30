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


  L1MuKBMTCombinedStubCollection makeStubs(const L1MuDTChambPhContainer*,const L1MuDTChambThContainer*,const L1TMuonBarrelParams&);
  void makeInputPattern(const L1MuDTChambPhContainer* phiContainer,const L1MuDTChambThContainer* etaContainer,int sector);

  
 private:
  bool isGoodPhiStub(const L1MuDTChambPhDigi*); 
  L1MuKBMTCombinedStub buildStub(const L1MuDTChambPhDigi&,const L1MuDTChambThDigi*);
  L1MuKBMTCombinedStub buildStubNoEta(const L1MuDTChambPhDigi&);


  int calculateEta(uint, int,uint,uint);  
  int minPhiQuality_;
  int minBX_;
  int maxBX_;
  std::vector<int> eta1_;
  std::vector<int> eta2_;
  std::vector<int> eta3_;


  bool disableMasks_;
  int verbose_;


  //    edm::ESHandle< L1TMuonBarrelParams > bmtfParamsHandle;
  //    L1MuDTTFMasks       masks_;





};


#endif
