#include "L1Trigger/GlobalCaloTrigger/plugins/L1GctEmulator.h"

// system includes
#include <memory>
#include <vector>
#include <iostream>

// EDM includes
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/Exception.h"

// Trigger configuration includes
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"
#include "CondFormats/L1TObjects/interface/L1GctJetCounterSetup.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetCalibFunRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetCounterPositiveEtaRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetCounterNegativeEtaRcd.h"

// GCT include files
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

// RCT data includes
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

// GCT data includes
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtTotal.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtHad.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtMiss.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

using std::vector;

L1GctEmulator::L1GctEmulator(const edm::ParameterSet& ps) :
  m_jetEtCalibLut(new L1GctJetEtCalibrationLut()),
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

  // get the input label
  edm::InputTag inputTag  = ps.getParameter<edm::InputTag>("inputLabel");
  m_inputLabel = inputTag.label();

  // instantiate the GCT. Argument selects the type of jetFinder to be used.
  L1GctJetLeafCard::jetFinderType jfType=L1GctJetLeafCard::hardwareJetFinder;
  std::string jfTypeStr = ps.getParameter<std::string>("jetFinderType");
  if (jfTypeStr == "tdrJetFinder") { jfType = L1GctJetLeafCard::tdrJetFinder; }
  else if (jfTypeStr != "hardwareJetFinder") {
    edm::LogWarning ("L1GctEmulatorSetup") << "Unrecognised jetFinder option " << jfTypeStr
                                           << "\nHardware jetFinder will be used";
  }
  m_gct = new L1GlobalCaloTrigger(jfType);

  // set verbosity (not implemented yet!)
  //  m_gct->setVerbose(m_verbose);

  // print debug info?
  if (m_verbose) {
    m_gct->print();

  }
}

L1GctEmulator::~L1GctEmulator() {
  if (m_jetEtCalibLut != 0) delete m_jetEtCalibLut;
  if (m_gct != 0) delete m_gct;
}


void L1GctEmulator::beginJob(const edm::EventSetup& c)
{
}

void L1GctEmulator::endJob()
{
}

void L1GctEmulator::configureGct(const edm::EventSetup& c)
{
  assert(&c!=0);

  // get data from EventSetup
  edm::ESHandle< L1GctJetFinderParams > jfPars ;
  c.get< L1GctJetFinderParamsRcd >().get( jfPars ) ; // which record?
  edm::ESHandle< L1GctJetCounterSetup > jcPosPars ;
  c.get< L1GctJetCounterPositiveEtaRcd >().get( jcPosPars ) ; // which record?
  edm::ESHandle< L1GctJetCounterSetup > jcNegPars ;
  c.get< L1GctJetCounterNegativeEtaRcd >().get( jcNegPars ) ; // which record?
  edm::ESHandle< L1GctJetEtCalibrationFunction > calibFun ;
  c.get< L1GctJetCalibFunRcd >().get( calibFun ) ; // which record?
  edm::ESHandle< L1CaloEtScale > etScale ;
  c.get< L1JetEtScaleRcd >().get( etScale ) ; // which record?

  if (jfPars.product() == 0) {
    throw cms::Exception("L1GctConfigError")
      << "Failed to find a L1GctJetFinderParamsRcd:L1GctJetFinderParams in EventSetup!" << std::endl
      << "Cannot continue without these parameters" << std::endl;
  }

  if (calibFun.product() == 0) {
    throw cms::Exception("L1GctConfigError")
      << "Failed to find a L1GctJetCalibFunRcd:L1GctJetEtCalibrationFunction in EventSetup!" << std::endl
      << "Cannot continue without this function" << std::endl;
  }

  m_gct->setJetFinderParams(jfPars.product());

  // tell the jet Et Lut about the scales
  m_jetEtCalibLut->setFunction(calibFun.product());
  m_jetEtCalibLut->setOutputEtScale(etScale.product());
  m_gct->setJetEtCalibrationLut(m_jetEtCalibLut);
  m_gct->setupJetCounterLuts(jcPosPars.product(), jcNegPars.product());
  
}

void L1GctEmulator::produce(edm::Event& e, const edm::EventSetup& c) {

  // get config data from EventSetup
  configureGct(c);

  // get the RCT data
  edm::Handle<L1CaloEmCollection> em;
  edm::Handle<L1CaloRegionCollection> rgn;
  e.getByLabel(m_inputLabel, em);
  e.getByLabel(m_inputLabel, rgn);

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
    cenJetResult->push_back(m_gct->getCentralJets().at(i));
    forJetResult->push_back(m_gct->getForwardJets().at(i));
    tauJetResult->push_back(m_gct->getTauJets().at(i));
  }

  // create the energy sum digis
  std::auto_ptr<L1GctEtTotal> etTotResult(new L1GctEtTotal(m_gct->getEtSum().value(), m_gct->getEtSum().overFlow() ) );
  std::auto_ptr<L1GctEtHad> etHadResult(new L1GctEtHad(m_gct->getEtHad().value(), m_gct->getEtHad().overFlow() ) );
  std::auto_ptr<L1GctEtMiss> etMissResult(new L1GctEtMiss(m_gct->getEtMiss().value(),
                           m_gct->getEtMissPhi().value(), m_gct->getEtMiss().overFlow() ) );

  // create the jet counts digis
  std::auto_ptr<L1GctJetCounts> jetCountResult(new L1GctJetCounts(m_gct->getJetCountValues() ) );

  // put the collections into the event
  e.put(isoEmResult,"isoEm");
  e.put(nonIsoEmResult,"nonIsoEm");
  e.put(cenJetResult,"cenJets");
  e.put(forJetResult,"forJets");
  e.put(tauJetResult,"tauJets");
  e.put(etTotResult);
  e.put(etHadResult);
  e.put(etMissResult);
  e.put(jetCountResult);

}

DEFINE_FWK_MODULE(L1GctEmulator);

