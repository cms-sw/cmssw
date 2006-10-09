#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmulator.h"

// system includes
#include <memory>

// EDM includes
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Trigger includes
#include "L1Trigger/L1Scales/interface/L1CaloEtScale.h"
#include "L1Trigger/L1Scales/interface/L1JetEtScaleRcd.h"

// GCT include files
#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

// RCT data includes
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

// GCT data includes
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

using std::cout;
using std::endl;


L1GctEmulator::L1GctEmulator(const edm::ParameterSet& ps) :
  m_verbose(ps.getUntrackedParameter<bool>("verbose", false))
 {

  // list of products
  produces<L1GctEmCandCollection>("isoEm");
  produces<L1GctEmCandCollection>("nonIsoEm");
  produces<L1GctJetCandCollection>("cenJets");
  produces<L1GctJetCandCollection>("forJets");
  produces<L1GctJetCandCollection>("tauJets");
  produces<L1GctEtTotal>();
  produces<L1GctEtHad>();
  produces<L1GctEtMiss>();
  produces<L1GctJetCounts>();

  // Get the filename for the Jet Et LUT
  edm::FileInPath fp = ps.getParameter<edm::FileInPath>("jetEtLutFile");

  // instantiate the GCT
  m_gct = new L1GlobalCaloTrigger(false,L1GctJetLeafCard::tdrJetFinder,fp.fullPath());

  // set parameters
  //  m_gct->setVerbose(m_verbose);

  // for now, call the debug print output
  if (m_verbose) {
    m_gct->print();
  }

}

L1GctEmulator::~L1GctEmulator() {
  delete m_gct;
}


void L1GctEmulator::produce(edm::Event& e, const edm::EventSetup& c) {

  // get data from EventSetup
  edm::ESHandle< L1CaloEtScale > jetScale ;
  c.get< L1JetEtScaleRcd >().get( jetScale ) ; // which record?

  // tell the jet Et LUT about the scale
  m_gct->getJetEtCalibLut()->setOutputEtScale(jetScale.product());

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
  std::auto_ptr<L1GctEtTotal> etTotResult(new L1GctEtTotal(m_gct->getEtSum().value(), m_gct->getEtSum().overFlow() ) );
  std::auto_ptr<L1GctEtHad> etHadResult(new L1GctEtHad(m_gct->getEtHad().value(), m_gct->getEtHad().overFlow() ) );
  std::auto_ptr<L1GctEtMiss> etMissResult(new L1GctEtMiss(m_gct->getEtMiss().value(), m_gct->getEtMissPhi().value(), m_gct->getEtMiss().overFlow() ) );

  // put the collections into the event
  e.put(isoEmResult,"isoEm");
  e.put(nonIsoEmResult,"nonIsoEm");
  e.put(cenJetResult,"cenJets");
  e.put(forJetResult,"forJets");
  e.put(tauJetResult,"tauJets");
  e.put(etTotResult);
  e.put(etHadResult);
  e.put(etMissResult);

}

