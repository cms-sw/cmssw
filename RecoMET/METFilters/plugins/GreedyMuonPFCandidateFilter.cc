

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"

//
// class declaration
//

class GreedyMuonPFCandidateFilter : public edm::global::EDFilter<> {
public:
  explicit GreedyMuonPFCandidateFilter(const edm::ParameterSet&);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  const edm::EDGetTokenT<reco::PFCandidateCollection> tokenPFCandidates_;
  // ----------member data ---------------------------

  const double eOverPMax_;

  const bool debug_;

  const bool taggingMode_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
GreedyMuonPFCandidateFilter::GreedyMuonPFCandidateFilter(const edm::ParameterSet& iConfig)
    //now do what ever initialization is needed
    : tokenPFCandidates_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("PFCandidates"))),
      eOverPMax_(iConfig.getParameter<double>("eOverPMax")),
      debug_(iConfig.getParameter<bool>("debug")),
      taggingMode_(iConfig.getParameter<bool>("taggingMode")) {
  produces<bool>();
  produces<reco::PFCandidateCollection>("muons");
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool GreedyMuonPFCandidateFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace std;
  using namespace edm;

  Handle<reco::PFCandidateCollection> pfCandidates;
  iEvent.getByToken(tokenPFCandidates_, pfCandidates);

  bool foundMuon = false;

  auto pOutputCandidateCollection = std::make_unique<reco::PFCandidateCollection>();

  for (unsigned i = 0; i < pfCandidates->size(); i++) {
    const reco::PFCandidate& cand = (*pfCandidates)[i];

    //    if( cand.particleId() != 3 ) // not a muon
    if (cand.particleId() != reco::PFCandidate::mu)  // not a muon
      continue;

    if (!PFMuonAlgo::isIsolatedMuon(cand.muonRef()))  // muon is not isolated
      continue;

    double totalCaloEnergy = cand.rawEcalEnergy() + cand.rawHcalEnergy();
    double eOverP = totalCaloEnergy / cand.p();

    if (eOverP < eOverPMax_)
      continue;

    foundMuon = true;

    pOutputCandidateCollection->push_back(cand);

    if (debug_) {
      cout << cand << " HCAL E=" << endl;
      cout << "\t"
           << "ECAL energy " << cand.rawEcalEnergy() << endl;
      cout << "\t"
           << "HCAL energy " << cand.rawHcalEnergy() << endl;
      cout << "\t"
           << "E/p " << eOverP << endl;
    }
  }

  iEvent.put(std::move(pOutputCandidateCollection), "muons");

  bool pass = !foundMuon;

  iEvent.put(std::make_unique<bool>(pass));

  return taggingMode_ || pass;
}

//define this as a plug-in
DEFINE_FWK_MODULE(GreedyMuonPFCandidateFilter);
