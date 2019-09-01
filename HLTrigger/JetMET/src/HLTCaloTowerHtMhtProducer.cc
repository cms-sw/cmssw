/** \class HLTCaloTowerHtMhtProducer
 *
 * See header file for documentation
 *
 *  \author Steven Lowette
 *  \author Thiago Tomei
 *
 */

#include "HLTrigger/JetMET/interface/HLTCaloTowerHtMhtProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"

// Constructor
HLTCaloTowerHtMhtProducer::HLTCaloTowerHtMhtProducer(const edm::ParameterSet& iConfig)
    : usePt_(iConfig.getParameter<bool>("usePt")),
      minPtTowerHt_(iConfig.getParameter<double>("minPtTowerHt")),
      minPtTowerMht_(iConfig.getParameter<double>("minPtTowerMht")),
      maxEtaTowerHt_(iConfig.getParameter<double>("maxEtaTowerHt")),
      maxEtaTowerMht_(iConfig.getParameter<double>("maxEtaTowerMht")),
      towersLabel_(iConfig.getParameter<edm::InputTag>("towersLabel")) {
  m_theTowersToken = consumes<CaloTowerCollection>(towersLabel_);

  // Register the products
  produces<reco::METCollection>();
}

// Destructor
HLTCaloTowerHtMhtProducer::~HLTCaloTowerHtMhtProducer() = default;

// Fill descriptions
void HLTCaloTowerHtMhtProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // Current default is for hltHtMht
  edm::ParameterSetDescription desc;
  desc.add<bool>("usePt", false);
  desc.add<double>("minPtTowerHt", 1.);
  desc.add<double>("minPtTowerMht", 1.);
  desc.add<double>("maxEtaTowerHt", 5.);
  desc.add<double>("maxEtaTowerMht", 5.);
  desc.add<edm::InputTag>("towersLabel", edm::InputTag("hltTowerMakerForAll"));
  descriptions.add("hltCaloTowerHtMhtProducer", desc);
}

// Produce the products
void HLTCaloTowerHtMhtProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Create a pointer to the products
  std::unique_ptr<reco::METCollection> result(new reco::METCollection());

  edm::Handle<CaloTowerCollection> towers;
  iEvent.getByToken(m_theTowersToken, towers);

  double ht = 0., mhx = 0., mhy = 0.;

  if (!towers->empty()) {
    for (auto const& j : *towers) {
      double pt = usePt_ ? j.pt() : j.et();
      double eta = j.eta();
      double phi = j.phi();
      double px = usePt_ ? j.px() : j.et() * cos(phi);
      double py = usePt_ ? j.py() : j.et() * sin(phi);

      if (pt > minPtTowerHt_ && std::abs(eta) < maxEtaTowerHt_) {
        ht += pt;
      }

      if (pt > minPtTowerMht_ && std::abs(eta) < maxEtaTowerMht_) {
        mhx -= px;
        mhy -= py;
      }
    }
  }

  reco::MET::LorentzVector p4(mhx, mhy, 0, sqrt(mhx * mhx + mhy * mhy));
  reco::MET::Point vtx(0, 0, 0);
  reco::MET htmht(ht, p4, vtx);
  result->push_back(htmht);

  // Put the products into the Event
  iEvent.put(std::move(result));
}
