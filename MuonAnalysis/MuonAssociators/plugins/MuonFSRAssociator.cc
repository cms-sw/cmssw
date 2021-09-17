// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/Common/interface/Association.h"

//
// class declaration
//

class MuonFSRAssociator : public edm::global::EDProducer<> {
public:
  explicit MuonFSRAssociator(const edm::ParameterSet& iConfig)
      :

        photons_{consumes<pat::GenericParticleCollection>(iConfig.getParameter<edm::InputTag>("photons"))},
        muons_{consumes<edm::View<reco::Muon>>(iConfig.getParameter<edm::InputTag>("muons"))}

  {
    produces<edm::Association<std::vector<pat::GenericParticle>>>();
    produces<edm::ValueMap<int>>("fsrIndex");
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("photons")->setComment("FSR photon collection to associate with muons");
    desc.add<edm::InputTag>("muons")->setComment("collection of muons to associate with FSR photons");

    descriptions.addWithDefaultLabel(desc);
  }
  ~MuonFSRAssociator() override {}

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<pat::GenericParticleCollection> photons_;
  const edm::EDGetTokenT<edm::View<reco::Muon>> muons_;
};

void MuonFSRAssociator::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace std;

  edm::Handle<pat::GenericParticleCollection> photons;
  iEvent.getByToken(photons_, photons);
  edm::Handle<edm::View<reco::Muon>> muons;
  iEvent.getByToken(muons_, muons);

  std::vector<int> muonMapping(muons->size(), -1);
  // loop over all muons
  for (auto muon = muons->begin(); muon != muons->end(); ++muon) {
    for (auto iter_pho = photons->begin(); iter_pho != photons->end(); iter_pho++) {
      if (iter_pho->hasUserCand("associatedMuon") and
          iter_pho->userCand("associatedMuon") == reco::CandidatePtr(muons, muon - muons->begin()))
        muonMapping[muon - muons->begin()] = (iter_pho - photons->begin());
    }
  }

  auto muon2photon = std::make_unique<edm::Association<std::vector<pat::GenericParticle>>>(photons);
  edm::Association<std::vector<pat::GenericParticle>>::Filler muon2photonFiller(*muon2photon);
  muon2photonFiller.insert(muons, muonMapping.begin(), muonMapping.end());
  muon2photonFiller.fill();
  iEvent.put(std::move(muon2photon));

  std::unique_ptr<edm::ValueMap<int>> bareIdx(new edm::ValueMap<int>());
  edm::ValueMap<int>::Filler fillerBareIdx(*bareIdx);
  fillerBareIdx.insert(muons, muonMapping.begin(), muonMapping.end());
  fillerBareIdx.fill();
  iEvent.put(std::move(bareIdx), "fsrIndex");
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonFSRAssociator);
