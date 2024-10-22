/** \class HLTPFJetIDProducer
 *
 * See header file for documentation
 *
 *  \author a Jet/MET person
 *
 */

#include "HLTrigger/JetMET/interface/HLTPFJetIDProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"

// Constructor
HLTPFJetIDProducer::HLTPFJetIDProducer(const edm::ParameterSet& iConfig)
    : minPt_(iConfig.getParameter<double>("minPt")),
      maxEta_(iConfig.getParameter<double>("maxEta")),
      CHF_(iConfig.getParameter<double>("CHF")),
      NHF_(iConfig.getParameter<double>("NHF")),
      CEF_(iConfig.getParameter<double>("CEF")),
      NEF_(iConfig.getParameter<double>("NEF")),
      maxCF_(iConfig.getParameter<double>("maxCF")),
      NCH_(iConfig.getParameter<int>("NCH")),
      NTOT_(iConfig.getParameter<int>("NTOT")),
      inputTag_(iConfig.getParameter<edm::InputTag>("jetsInput")) {
  m_thePFJetToken = consumes<reco::PFJetCollection>(inputTag_);

  // Register the products
  produces<reco::PFJetCollection>();
}

// Destructor
HLTPFJetIDProducer::~HLTPFJetIDProducer() = default;

// Fill descriptions
void HLTPFJetIDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("minPt", 20.);
  desc.add<double>("maxEta", 1e99);
  desc.add<double>("CHF", -99.);
  desc.add<double>("NHF", 99.);
  desc.add<double>("CEF", 99.);
  desc.add<double>("NEF", 99.);
  desc.add<double>("maxCF", 99.);
  desc.add<int>("NCH", 0);
  desc.add<int>("NTOT", 0);
  desc.add<edm::InputTag>("jetsInput", edm::InputTag("hltAntiKT4PFJets"));
  descriptions.add("hltPFJetIDProducer", desc);
}

// Produce the products
void HLTPFJetIDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Create a pointer to the products
  std::unique_ptr<reco::PFJetCollection> result(new reco::PFJetCollection());

  edm::Handle<reco::PFJetCollection> pfjets;
  iEvent.getByToken(m_thePFJetToken, pfjets);

  for (auto const& j : *pfjets) {
    bool pass = false;
    double pt = j.pt();
    double eta = j.eta();
    double abseta = std::abs(eta);

    if (!(pt > 0.))
      continue;  // skip jets with zero or negative pt

    if (pt < minPt_) {
      pass = true;

    } else if (abseta >= maxEta_) {
      pass = true;

    } else {
      double chf = j.chargedHadronEnergyFraction();
      //double nhf  = j->neutralHadronEnergyFraction() + j->HFHadronEnergyFraction();
      double nhf = j.neutralHadronEnergyFraction();
      double cef = j.chargedEmEnergyFraction();
      double nef = j.neutralEmEnergyFraction();
      double cftot = chf + cef + j.chargedMuEnergyFraction();
      int nch = j.chargedMultiplicity();
      int ntot = j.numberOfDaughters();

      pass = true;
      pass = pass && (ntot > NTOT_);
      pass = pass && (nef < NEF_);
      pass = pass && (nhf < NHF_ || abseta >= 2.4);  //NHF-cut does not work in HF anymore with recent PF
      pass = pass && (cef < CEF_ || abseta >= 2.4);
      pass = pass && (chf > CHF_ || abseta >= 2.4);
      pass = pass && (nch > NCH_ || abseta >= 2.4);
      pass = pass && (cftot < maxCF_ || abseta >= 2.4);
    }

    if (pass)
      result->push_back(j);
  }

  // Put the products into the Event
  iEvent.put(std::move(result));
}
