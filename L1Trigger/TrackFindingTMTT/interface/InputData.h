#ifndef __INPUTDATA_H__
#define __INPUTDATA_H__

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <vector>

using namespace std;

namespace TMTT {

class Settings;

//=== Unpacks stub & tracking particle (truth) data into user-friendlier format in Stub & TP classes.
//=== Also makes B-field available to Settings class.

class InputData {

public:
  
  InputData(const edm::Event& iEvent, const edm::EventSetup& iSetup, Settings* settings,
  const edm::EDGetTokenT<TrackingParticleCollection> tpInputTag,
  const edm::EDGetTokenT<DetSetVec> stubInputTag,
  const edm::EDGetTokenT<TTStubAssMap> stubTruthInputTag,
  const edm::EDGetTokenT<TTClusterAssMap> clusterTruthInputTag,
  const edm::EDGetTokenT< reco::GenJetCollection > genJetInputTag
   );

  // Get tracking particles
  const vector<TP>&          getTPs()      const {return vTPs_;}
  // Get stubs that would be output by the front-end readout electronics 
  const vector<const Stub*>& getStubs()    const {return vStubs_;}

  //--- of minor importance ...

  // Get number of stubs prior to applying tighted front-end readout electronics cuts specified in section StubCuts of Analyze_Defaults_cfi.py. (Only used to measure the efficiency of these cuts).
  const vector<Stub>&        getAllStubs() const {return vAllStubs_;}

private:
  // const edm::EDGetTokenT<TrackingParticleCollection> inputTag;

  // Can optionally be used to sort stubs by bend.
  struct SortStubsInBend {
     inline bool operator() (const Stub* stub1, const Stub* stub2) {
        return(fabs(stub1->bend()) < fabs(stub2->bend()));
     }
  };

private:

  bool enableMCtruth_; // Notes if job will use MC truth info.

  vector<TP> vTPs_; // tracking particles
  vector<const Stub*> vStubs_; // stubs that would be output by the front-end readout electronics.

  //--- of minor importance ...

  vector<Stub> vAllStubs_; // all stubs, even those that would fail any tightened front-end readout electronic cuts specified in section StubCuts of Analyze_Defaults_cfi.py. (Only used to measure the efficiency of these cuts).
};

}
#endif

