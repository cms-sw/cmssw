#ifndef L1TMUONBARRELKALMANSECTORPROCESSOR_H
#define L1TMUONBARRELKALMANSECTORPROCESSOR_H

#include "DataFormats/L1TMuon/interface/L1MuKBMTCombinedStub.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanAlgo.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanRegionModule.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class L1TMuonBarrelKalmanSectorProcessor {
 public:
  L1TMuonBarrelKalmanSectorProcessor(const edm::ParameterSet&,int sector);
  ~L1TMuonBarrelKalmanSectorProcessor();

  L1MuKBMTrackCollection process(L1TMuonBarrelKalmanAlgo*,const L1MuKBMTCombinedStubRefVector& stubs,int bx);
  void verbose(L1TMuonBarrelKalmanAlgo*,const L1MuKBMTrackCollection&);
 private:
  int verbose_;
  int sector_;

  std::vector<L1TMuonBarrelKalmanRegionModule> regions_;


  //For patterns
  typedef struct {
    int pt_1;
    int qual_1;
    int eta_1;
    int HF_1;
    int phi_1;
    int bx0_1;
    int charge_1;
    int chargeValid_1;
    int dxy_1;
    int addr1_1;
    int addr2_1;
    int addr3_1;
    int addr4_1;
    int reserved_1;
    int wheel_1;
    int ptSTA_1;
    int SE_1;

    int pt_2;
    int qual_2;
    int eta_2;
    int HF_2;
    int phi_2;
    int bx0_2;
    int charge_2;
    int chargeValid_2;
    int dxy_2;
    int addr1_2;
    int addr2_2;
    int addr3_2;
    int addr4_2;
    int reserved_2;
    int wheel_2;
    int ptSTA_2;
    int SE_2;

    int pt_3;
    int qual_3;
    int eta_3;
    int HF_3;
    int phi_3;
    int bx0_3;
    int charge_3;
    int chargeValid_3;
    int dxy_3;
    int addr1_3;
    int addr2_3;
    int addr3_3;
    int addr4_3;
    int reserved_3;
    int wheel_3;
    int ptSTA_3;
    int SE_3;

  } bmtf_out;


  bmtf_out makeWord(L1TMuonBarrelKalmanAlgo*,const L1MuKBMTrackCollection&);



};



#endif
