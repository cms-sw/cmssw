/** \class HLTMhtProducer
 *
 * See header file for documentation
 *
 *  \author Steven Lowette
 *
 */

#include "HLTrigger/JetMET/interface/HLTMhtProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"

// Constructor
HLTMhtProducer::HLTMhtProducer(const edm::ParameterSet& iConfig)
    : usePt_(iConfig.getParameter<bool>("usePt")),
      excludePFMuons_(iConfig.getParameter<bool>("excludePFMuons")),
      minNJet_(iConfig.getParameter<int>("minNJet")),
      minPtJet_(iConfig.getParameter<double>("minPtJet")),
      maxEtaJet_(iConfig.getParameter<double>("maxEtaJet")),
      jetsLabel_(iConfig.getParameter<edm::InputTag>("jetsLabel")),
      pfCandidatesLabel_(iConfig.getParameter<edm::InputTag>("pfCandidatesLabel")) {
  m_theJetToken = consumes<reco::CandidateView>(jetsLabel_);
  if (pfCandidatesLabel_.label().empty())
    excludePFMuons_ = false;
  if (excludePFMuons_)
    m_thePFCandidateToken = consumes<reco::PFCandidateCollection>(pfCandidatesLabel_);

  // Register the products
  produces<reco::METCollection>();
}

// Destructor
HLTMhtProducer::~HLTMhtProducer() = default;

// Fill descriptions
void HLTMhtProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // Current default is for hltPFMET
  edm::ParameterSetDescription desc;
  desc.add<bool>("usePt", true);
  desc.add<bool>("excludePFMuons", false);
  desc.add<int>("minNJet", 0);
  desc.add<double>("minPtJet", 0.);
  desc.add<double>("maxEtaJet", 999.);
  desc.add<edm::InputTag>("jetsLabel", edm::InputTag("hltAntiKT4PFJets"));
  desc.add<edm::InputTag>("pfCandidatesLabel", edm::InputTag("hltParticleFlow"));
  descriptions.add("hltMhtProducer", desc);
}

// Produce the products
void HLTMhtProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Create a pointer to the products
  std::unique_ptr<reco::METCollection> result(new reco::METCollection());

  edm::Handle<reco::CandidateView> jets;
  iEvent.getByToken(m_theJetToken, jets);

  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  if (excludePFMuons_)
    iEvent.getByToken(m_thePFCandidateToken, pfCandidates);

  int nj = 0;
  double sumet = 0., mhx = 0., mhy = 0.;

  for (auto const& aJet : *jets) {
    double const pt = usePt_ ? aJet.pt() : aJet.et();
    double const eta = aJet.eta();
    double const phi = aJet.phi();
    double const px = usePt_ ? aJet.px() : aJet.et() * cos(phi);
    double const py = usePt_ ? aJet.py() : aJet.et() * sin(phi);

    if (pt > minPtJet_ && std::abs(eta) < maxEtaJet_) {
      mhx -= px;
      mhy -= py;
      sumet += pt;
      ++nj;
    }
  }

  if (excludePFMuons_) {
    for (auto const& aCand : *pfCandidates) {
      if (std::abs(aCand.pdgId()) == 13) {
        mhx += aCand.px();
        mhy += aCand.py();
      }
    }
  }

  if (nj < minNJet_) {
    sumet = 0;
    mhx = 0;
    mhy = 0;
  }

  reco::MET::LorentzVector p4(mhx, mhy, 0, sqrt(mhx * mhx + mhy * mhy));
  reco::MET::Point vtx(0, 0, 0);
  reco::MET mht(sumet, p4, vtx);
  result->push_back(mht);

  // Put the products into the Event
  iEvent.put(std::move(result));
}
