#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoPhoton.h"

using namespace std;

L1Analysis::L1AnalysisRecoPhoton::L1AnalysisRecoPhoton() {}

L1Analysis::L1AnalysisRecoPhoton::~L1AnalysisRecoPhoton() {}

void L1Analysis::L1AnalysisRecoPhoton::SetPhoton(
    const edm::Event& event,
    const edm::EventSetup& setup,
    edm::Handle<reco::PhotonCollection> photons,
    std::vector<edm::Handle<edm::ValueMap<bool> > > phoVIDDecisionHandles,
    const unsigned& maxPhoton)

{
  recoPhoton_.nPhotons = 0;

  for (reco::PhotonCollection::const_iterator ph = photons->begin();
       ph != photons->end() && recoPhoton_.nPhotons < maxPhoton;
       ++ph) {
    recoPhoton_.e.push_back(ph->energy());
    recoPhoton_.pt.push_back(ph->pt());
    recoPhoton_.et.push_back(ph->et());
    recoPhoton_.eta.push_back(ph->eta());
    recoPhoton_.phi.push_back(ph->phi());
    recoPhoton_.r9.push_back(ph->r9());
    recoPhoton_.hasPixelSeed.push_back(ph->hasPixelSeed());

    edm::Ref<reco::PhotonCollection> photonEdmRef(photons, recoPhoton_.nPhotons);

    recoPhoton_.isTightPhoton.push_back((*(phoVIDDecisionHandles[0]))[photonEdmRef]);
    recoPhoton_.isLoosePhoton.push_back((*(phoVIDDecisionHandles[1]))[photonEdmRef]);

    recoPhoton_.nPhotons++;
  }
}
