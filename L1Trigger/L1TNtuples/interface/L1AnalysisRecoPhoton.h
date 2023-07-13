#ifndef __L1Analysis_L1AnalysisRecoPhoton_H__
#define __L1Analysis_L1AnalysisRecoPhoton_H__

//-------------------------------------------------------------------------------
// Original code : L1Trigger/L1TNtuples/L1RecoJetNtupleProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include "L1AnalysisRecoPhotonDataFormat.h"

//photons
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace L1Analysis {
  class L1AnalysisRecoPhoton {
  public:
    L1AnalysisRecoPhoton();
    ~L1AnalysisRecoPhoton();

    //void Print(std::ostream &os = std::cout) const;
    void SetPhoton(const edm::Event& event,
                     const edm::EventSetup& setup,
                     const edm::Handle<reco::PhotonCollection> photons,
                     const std::vector<edm::Handle<edm::ValueMap<bool> > > phoVIDDecisionHandles,
                     const unsigned& maxPhoton);

    L1AnalysisRecoPhotonDataFormat* getData() { return &recoPhoton_; }
    void Reset() { recoPhoton_.Reset(); }

  private:
    L1AnalysisRecoPhotonDataFormat recoPhoton_;
  };
}  // namespace L1Analysis
#endif
