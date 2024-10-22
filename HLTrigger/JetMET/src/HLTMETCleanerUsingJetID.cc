/** \class HLTMETCleanerUsingJetID
 *
 * See header file for more information.
 *
 *  \author a Jet/MET person
 *
 */

#include "HLTrigger/JetMET/interface/HLTMETCleanerUsingJetID.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

// Constructor
HLTMETCleanerUsingJetID::HLTMETCleanerUsingJetID(const edm::ParameterSet& iConfig)
    : minPt_(iConfig.getParameter<double>("minPt")),
      maxEta_(iConfig.getParameter<double>("maxEta")),
      metLabel_(iConfig.getParameter<edm::InputTag>("metLabel")),
      jetsLabel_(iConfig.getParameter<edm::InputTag>("jetsLabel")),
      goodJetsLabel_(iConfig.getParameter<edm::InputTag>("goodJetsLabel")) {
  m_theMETToken = consumes<reco::CaloMETCollection>(metLabel_);
  m_theJetToken = consumes<reco::CaloJetCollection>(jetsLabel_);
  m_theGoodJetToken = consumes<reco::CaloJetCollection>(goodJetsLabel_);

  // Register the products
  produces<reco::CaloMETCollection>();
}

// Destructor
HLTMETCleanerUsingJetID::~HLTMETCleanerUsingJetID() = default;

// Fill descriptions
void HLTMETCleanerUsingJetID::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("minPt", 20.);
  desc.add<double>("maxEta", 5.);
  desc.add<edm::InputTag>("metLabel", edm::InputTag("hltMet"));
  desc.add<edm::InputTag>("jetsLabel", edm::InputTag("hltAntiKT4CaloJets"));
  desc.add<edm::InputTag>("goodJetsLabel", edm::InputTag("hltCaloJetIDPassed"));
  descriptions.add("hltMETCleanerUsingJetID", desc);
}

// Produce the products
void HLTMETCleanerUsingJetID::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Create a pointer to the products
  std::unique_ptr<reco::CaloMETCollection> result(new reco::CaloMETCollection);

  edm::Handle<reco::CaloMETCollection> met;
  edm::Handle<reco::CaloJetCollection> jets;
  edm::Handle<reco::CaloJetCollection> goodJets;

  iEvent.getByToken(m_theMETToken, met);
  iEvent.getByToken(m_theJetToken, jets);
  iEvent.getByToken(m_theGoodJetToken, goodJets);

  double mex_jets = 0.;
  double mey_jets = 0.;
  //double sumet_jets = 0.;
  if (!jets->empty()) {
    for (auto const& j : *jets) {
      double pt = j.pt();
      double eta = j.eta();
      double px = j.px();
      double py = j.py();

      if (pt > minPt_ && std::abs(eta) < maxEta_) {
        mex_jets -= px;
        mey_jets -= py;
        //sumet_jets += pt;
      }
    }
  }

  double mex_goodJets = 0.;
  double mey_goodJets = 0.;
  //double sumet_goodJets = 0.;
  if (!goodJets->empty()) {
    for (auto const& j : *goodJets) {
      double pt = j.pt();
      double eta = j.eta();
      double px = j.px();
      double py = j.py();

      if (pt > minPt_ && std::abs(eta) < maxEta_) {
        mex_goodJets -= px;
        mey_goodJets -= py;
        //sumet_goodJets += pt;
      }
    }
  }

  if (!met->empty()) {
    double mex_diff = mex_goodJets - mex_jets;
    double mey_diff = mey_goodJets - mey_jets;
    //double sumet_diff = sumet_goodJets - sumet_jets;  // cannot set sumet...
    reco::Candidate::LorentzVector p4_clean(met->front().px() + mex_diff,
                                            mey_diff + met->front().py(),
                                            0,
                                            sqrt((met->front().px() + mex_diff) * (met->front().px() + mex_diff) +
                                                 (met->front().py() + mey_diff) * (met->front().py() + mey_diff)));

    reco::CaloMET cleanmet = met->front();
    cleanmet.setP4(p4_clean);
    result->push_back(cleanmet);
  }

  iEvent.put(std::move(result));
}
