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
  void printWord(const L1MuDTChambPhContainer* phiContainer,const L1MuDTChambThContainer* etaContainer,int,int);
  
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



  //For Making Patterns
  typedef struct {
    int ts1_st1_phi;
    int ts1_st1_phib;
    int ts1_st1_q;
    int ts1_st1_rpc;
    int ts1_st1_cal;

    int ts1_st2_phi;
    int ts1_st2_phib;
    int  ts1_st2_q;
    int ts1_st2_rpc;
    int ts1_st2_cal;

    int ts1_st3_phi;
    int ts1_st3_phib;
    int ts1_st3_q;
    int ts1_st3_rpc;
    int ts1_st3_cal;

    int ts1_st4_phi;
    int ts1_st4_phib;
    int ts1_st4_q;
    int ts1_st4_rpc;
    int ts1_st4_cal;

    int eta_hit_st1;
    int eta_hit_st2;
    int eta_hit_st3;

    int bcnt_1a;
    int bcnt_1b;
    int bcnt_1c;
    int bcnt_1d;
    int bcnt_1e;
    int bcnt_1f;

    int bc0_1;


    int ts2_st1_phi;
    int ts2_st1_phib;
    int ts2_st1_q;
    int ts2_st1_rpc;
    int ts2_st1_cal;

    int ts2_st2_phi;
    int ts2_st2_phib;
    int ts2_st2_q;
    int ts2_st2_rpc;
    int ts2_st2_cal;

    int ts2_st3_phi;
    int ts2_st3_phib;
    int ts2_st3_q;
    int ts2_st3_rpc;
    int ts2_st3_cal;

    int ts2_st4_phi;
    int ts2_st4_phib;
    int ts2_st4_q;
    int ts2_st4_rpc;
    int ts2_st4_cal;

    int eta_qbit_st1;
    int eta_qbit_st2;
    int eta_qbit_st3;

    int bcnt_2a;
    int bcnt_2b;
    int bcnt_2c;
    int bcnt_2d;
    int bcnt_2e;
    int bcnt_2f;
    int bc0_2;
  }bmtf_in;

  bmtf_in makePattern(const L1MuDTChambPhContainer* phiContainer,const L1MuDTChambThContainer* etaContainer,int sector, int wheel);

};


#endif
