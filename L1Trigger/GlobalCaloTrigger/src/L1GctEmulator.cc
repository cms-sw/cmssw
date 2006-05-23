// system include files
#include <memory>

// EDM include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// GCT include files
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmulator.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

// data format include files
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctDigis.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"

#include <memory>

L1GctEmulator::L1GctEmulator(const edm::ParameterSet& ps) {

  // list of products
  produces<L1GctIsoEmCollection>();
  produces<L1GctNonIsoEmCollection>();
  produces<L1GctCenJetCollection>();
  produces<L1GctForJetCollection>();
  produces<L1GctTauJetCollection>();
  produces<L1GctEtTotal>();
  produces<L1GctEtHad>();
  produces<L1GctEtMiss>();

  // instantiate the GCT
  m_gct = new L1GlobalCaloTrigger();

}

L1GctEmulator::~L1GctEmulator() {
  delete m_gct;
}


void L1GctEmulator::produce(edm::Event& e, const edm::EventSetup& c) {

  // create the ED Products
  std::auto_ptr<L1GctIsoEmCollection> isoEmResult(new L1GctIsoEmCollection);
  std::auto_ptr<L1GctNonIsoEmCollection> nonIsoEmResult(new L1GctNonIsoEmCollection);
  std::auto_ptr<L1GctCenJetCollection> cenJetResult(new L1GctCenJetCollection);
  std::auto_ptr<L1GctForJetCollection> forJetResult(new L1GctForJetCollection);
  std::auto_ptr<L1GctTauJetCollection> tauJetResult(new L1GctTauJetCollection);
  std::auto_ptr<L1GctEtTotal> etTotResult(new L1GctEtTotal);
  std::auto_ptr<L1GctEtHad> etHadResult(new L1GctEtHad);
  std::auto_ptr<L1GctEtMiss> etMissResult(new L1GctEtMiss);

  // reset the GCT internal buffers
  // m_gct->reset;


  // process the event
  //  m_gct->process()

  // fill the ED Products with data
  for (int i=0; i<4; i++) {

    isoEmResult->push_back(m_gct->getIsoElectrons()[i].makeIsoEm());
    nonIsoEmResult->push_back(m_gct->getNonIsoElectrons()[i].makeNonIsoEm());
    cenJetResult->push_back(m_gct->getCentralJets()[i].makeCenJet());
    forJetResult->push_back(m_gct->getForwardJets()[i].makeForJet());
    tauJetResult->push_back(m_gct->getTauJets()[i].makeTauJet());

  }


  // ... fill the collection ...
  e.put(isoEmResult);
  e.put(nonIsoEmResult);
  e.put(cenJetResult);
  e.put(forJetResult);
  e.put(tauJetResult);
  e.put(etTotResult);
  e.put(etHadResult);
  e.put(etMissResult);

}

