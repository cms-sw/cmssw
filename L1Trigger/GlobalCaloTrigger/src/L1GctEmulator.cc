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

L1GctEmulator::L1GctEmulator(const edm::ParameterSet& ps) :
  m_verbose(ps.getUntrackedParameter<bool>("verbose", false)),
  m_jetEtLut_A( ps.getParameter<double>("jetEtLutA") ),
  m_jetEtLut_B( ps.getParameter<double>("jetEtLutB") ),
  m_jetEtLut_C( ps.getParameter<double>("jetEtLutC") )
{

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
  m_gct = new L1GlobalCaloTrigger(true);

  // set parameters
  // m_gct->setVerbose(m_verbose);
  //  m_gct->getJetEtCalibLut();

  // for now, call the debug print output
  m_gct->print();

}

L1GctEmulator::~L1GctEmulator() {
  delete m_gct;
}


void L1GctEmulator::produce(edm::Event& e, const edm::EventSetup& c) {

  // reset the GCT internal buffers
  m_gct->reset();

  // process the event
  m_gct->process();

  // create the em and jet collections
  std::auto_ptr<L1GctIsoEmCollection> isoEmResult(new L1GctIsoEmCollection);
  std::auto_ptr<L1GctNonIsoEmCollection> nonIsoEmResult(new L1GctNonIsoEmCollection);
  std::auto_ptr<L1GctCenJetCollection> cenJetResult(new L1GctCenJetCollection);
  std::auto_ptr<L1GctForJetCollection> forJetResult(new L1GctForJetCollection);
  std::auto_ptr<L1GctTauJetCollection> tauJetResult(new L1GctTauJetCollection);

  // fill the em and jet collections with digis
  for (int i=0; i<4; i++) {
    isoEmResult->push_back(m_gct->getIsoElectrons()[i].makeIsoEm());
    nonIsoEmResult->push_back(m_gct->getNonIsoElectrons()[i].makeNonIsoEm());
    cenJetResult->push_back(m_gct->getCentralJets()[i].makeCenJet());
    forJetResult->push_back(m_gct->getForwardJets()[i].makeForJet());
    tauJetResult->push_back(m_gct->getTauJets()[i].makeTauJet());
  }

  // create the energy sum digis
  std::auto_ptr<L1GctEtTotal> etTotResult(new L1GctEtTotal(m_gct->getEtSum().value(), false) );
  std::auto_ptr<L1GctEtHad> etHadResult(new L1GctEtHad(m_gct->getEtHad().value(), false) );
  std::auto_ptr<L1GctEtMiss> etMissResult(new L1GctEtMiss(m_gct->getEtMiss().value(), m_gct->getEtMissPhi().value(), false) );

  // put the collections into the event
  e.put(isoEmResult);
  e.put(nonIsoEmResult);
  e.put(cenJetResult);
  e.put(forJetResult);
  e.put(tauJetResult);
  e.put(etTotResult);
  e.put(etHadResult);
  e.put(etMissResult);

}

