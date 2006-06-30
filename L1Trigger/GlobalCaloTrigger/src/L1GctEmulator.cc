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

// input RCT data
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

// output GCT data
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include <memory>

L1GctEmulator::L1GctEmulator(const edm::ParameterSet& ps) :
  m_verbose(ps.getUntrackedParameter<bool>("verbose", false)),
  m_jetEtLut_A( ps.getParameter<double>("jetEtLutA") ),
  m_jetEtLut_B( ps.getParameter<double>("jetEtLutB") ),
  m_jetEtLut_C( ps.getParameter<double>("jetEtLutC") )
{

  // list of products
  produces<L1GctEmCandCollection>();
  produces<L1GctJetCandCollection>();
  produces<L1GctEtTotal>();
  produces<L1GctEtHad>();
  produces<L1GctEtMiss>();
  produces<L1GctJetCounts>();

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

  // get the RCT data
  edm::Handle<L1CaloEmCollection> em;
  edm::Handle<L1CaloRegionCollection> rgn;
  e.getByType(em);
  e.getByType(rgn);

  // reset the GCT internal buffers
  m_gct->reset();
  
  // fill the GCT source cards
  m_gct->fillEmCands(*em);
  m_gct->fillRegions(*rgn);
  
  // process the event
  m_gct->process();

  // create the em and jet collections
  std::auto_ptr<L1GctEmCandCollection> isoEmResult(new L1GctEmCandCollection);
  std::auto_ptr<L1GctEmCandCollection> nonIsoEmResult(new L1GctEmCandCollection);
  std::auto_ptr<L1GctJetCandCollection> cenJetResult(new L1GctJetCandCollection);
  std::auto_ptr<L1GctJetCandCollection> forJetResult(new L1GctJetCandCollection);
  std::auto_ptr<L1GctJetCandCollection> tauJetResult(new L1GctJetCandCollection);

  // fill the em and jet collections with digis
  for (int i=0; i<4; i++) {
    isoEmResult->push_back(m_gct->getIsoElectrons().at(i));
    nonIsoEmResult->push_back(m_gct->getNonIsoElectrons().at(i));
    cenJetResult->push_back(m_gct->getCentralJets().at(i).makeJetCand());
    forJetResult->push_back(m_gct->getForwardJets().at(i).makeJetCand());
    tauJetResult->push_back(m_gct->getTauJets().at(i).makeJetCand());
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

