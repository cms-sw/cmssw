#include "L1Trigger/GlobalCaloTrigger/plugins/L1GctEmulator.h"

// system includes
#include <memory>
#include <vector>

// EDM includes
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Trigger configuration includes
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/L1TObjects/interface/L1GctJetCounterSetup.h"
#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"
#include "CondFormats/L1TObjects/interface/L1GctHfLutSetup.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetCounterPositiveEtaRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetCounterNegativeEtaRcd.h"
#include "CondFormats/DataRecord/interface/L1GctChannelMaskRcd.h"
#include "CondFormats/DataRecord/interface/L1GctHfLutSetupRcd.h"

// GCT include files
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"

// RCT data includes
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

// GCT data includes
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

using std::vector;

L1GctEmulator::L1GctEmulator(const edm::ParameterSet& ps) :
  m_jetEtCalibLuts(),
  m_verbose(ps.getUntrackedParameter<bool>("verbose", false))
 {

  // list of products
  produces<L1GctEmCandCollection>("isoEm");
  produces<L1GctEmCandCollection>("nonIsoEm");
  produces<L1GctJetCandCollection>("cenJets");
  produces<L1GctJetCandCollection>("forJets");
  produces<L1GctJetCandCollection>("tauJets");
  produces<L1GctInternJetDataCollection>();
  produces<L1GctEtTotalCollection>();
  produces<L1GctEtHadCollection>();
  produces<L1GctEtMissCollection>();
  produces<L1GctHtMissCollection>();
  produces<L1GctJetCountsCollection>();
  produces<L1GctHFBitCountsCollection>();
  produces<L1GctHFRingEtSumsCollection>();

  // get the input label
  edm::InputTag inputTag  = ps.getParameter<edm::InputTag>("inputLabel");
  m_inputLabel = inputTag.label();

  // Get the number of bunch crossings to be processed
  int firstBx = -ps.getParameter<unsigned>("preSamples");
  int  lastBx =  ps.getParameter<unsigned>("postSamples");

  // instantiate the GCT. Argument selects the type of jetFinder to be used.
  L1GctJetLeafCard::jetFinderType jfType=L1GctJetLeafCard::hardwareJetFinder;
  std::string jfTypeStr = ps.getParameter<std::string>("jetFinderType");
  if (jfTypeStr == "tdrJetFinder") { jfType = L1GctJetLeafCard::tdrJetFinder; }
  else if (jfTypeStr != "hardwareJetFinder") {
    edm::LogWarning ("L1GctEmulatorSetup") << "Unrecognised jetFinder option " << jfTypeStr
                                           << "\nHardware jetFinder will be used";
  }
  m_gct = new L1GlobalCaloTrigger(jfType);
  m_gct->setBxRange(firstBx, lastBx);

  // Fill the jetEtCalibLuts vector
  lutPtr nextLut( new L1GctJetEtCalibrationLut() );

  for (unsigned ieta=0; ieta<L1GctJetFinderBase::COL_OFFSET; ieta++) {
    nextLut->setEtaBin(ieta);
    m_jetEtCalibLuts.push_back(nextLut);
    nextLut.reset ( new L1GctJetEtCalibrationLut() );
  }

  // Setup the tau algorithm parameters
  bool useImprovedTauAlgo            = ps.getParameter<bool>("useImprovedTauAlgorithm");
  bool ignoreTauVetoBitsForIsolation = ps.getParameter<bool>("ignoreRCTTauVetoBitsForIsolation");
  m_gct->setupTauAlgo(useImprovedTauAlgo, ignoreTauVetoBitsForIsolation);

  // set verbosity (not implemented yet!)
  //  m_gct->setVerbose(m_verbose);

  // print debug info?
  if (m_verbose) {
    m_gct->print();

  }
}

L1GctEmulator::~L1GctEmulator() {
  if (m_gct != 0) delete m_gct;
}


void L1GctEmulator::beginJob(const edm::EventSetup& c)
{
}

void L1GctEmulator::endJob()
{
}

int L1GctEmulator::configureGct(const edm::EventSetup& c)
{
  int success = 0;
  if (&c==0) {
    success = -1;
    if (m_verbose) {
      edm::LogWarning("L1GctConfigFailure") << "Cannot find EventSetup information." << std::endl;
    }
  }

  if (success == 0) {
    // get data from EventSetup
    edm::ESHandle< L1GctJetFinderParams > jfPars ;
    c.get< L1GctJetFinderParamsRcd >().get( jfPars ) ; // which record?
    edm::ESHandle< L1GctJetCounterSetup > jcPosPars ;
    c.get< L1GctJetCounterPositiveEtaRcd >().get( jcPosPars ) ; // which record?
    edm::ESHandle< L1GctJetCounterSetup > jcNegPars ;
    c.get< L1GctJetCounterNegativeEtaRcd >().get( jcNegPars ) ; // which record?
    edm::ESHandle< L1GctHfLutSetup > hfLSetup ;
    c.get< L1GctHfLutSetupRcd >().get( hfLSetup ) ; // which record?
    edm::ESHandle< L1GctChannelMask > chanMask ;
    c.get< L1GctChannelMaskRcd >().get( chanMask ) ; // which record?
    edm::ESHandle< L1CaloEtScale > etScale ;
    c.get< L1JetEtScaleRcd >().get( etScale ) ; // which record?

    if (jfPars.product() == 0) {
      success = -1;
      if (m_verbose) {
	edm::LogWarning("L1GctConfigFailure")
	  << "Failed to find a L1GctJetFinderParamsRcd:L1GctJetFinderParams in EventSetup!" << std::endl;
      }
    }

    if (hfLSetup.product() == 0) {
      success = -1;
      if (m_verbose) {
	edm::LogWarning("L1GctConfigFailure")
	  << "Failed to find a L1GctHfLutSetupRcd:L1GctHfLutSetup in EventSetup!" << std::endl;
      }
    }

    if (chanMask.product() == 0) {
      success = -1;
      if (m_verbose) {
	edm::LogWarning("L1GctConfigFailure")
	  << "Failed to find a L1GctChannelMaskRcd:L1GctChannelMask in EventSetup!" << std::endl;
      }
    }

    if (success==0) {
      // tell the jet Et Luts about the scales
      for (unsigned ieta=0; ieta<m_jetEtCalibLuts.size(); ieta++) {
	m_jetEtCalibLuts.at(ieta)->setFunction(jfPars.product());
	m_jetEtCalibLuts.at(ieta)->setOutputEtScale(etScale.product());
      }

      // pass all the setup info to the gct
      m_gct->setJetEtCalibrationLuts(m_jetEtCalibLuts);
      m_gct->setJetFinderParams(jfPars.product());
      m_gct->setupJetCounterLuts(jcPosPars.product(), jcNegPars.product());
      m_gct->setupHfSumLuts(hfLSetup.product());
      m_gct->setChannelMask(chanMask.product());
    }
  }

  if (success != 0 && m_verbose) {
    edm::LogError("L1GctConfigError")
      << "Configuration failed - GCT emulator will not be run" << std::endl;
  }
  return success;
}

void L1GctEmulator::produce(edm::Event& e, const edm::EventSetup& c) {

  // get config data from EventSetup.
  // check this has been done successfully before proceeding
  if (configureGct(c) == 0) { 

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
    std::auto_ptr<L1GctEmCandCollection> isoEmResult   (new L1GctEmCandCollection(m_gct->getIsoElectrons() ) );
    std::auto_ptr<L1GctEmCandCollection> nonIsoEmResult(new L1GctEmCandCollection(m_gct->getNonIsoElectrons() ) );
    std::auto_ptr<L1GctJetCandCollection> cenJetResult(new L1GctJetCandCollection(m_gct->getCentralJets() ) );
    std::auto_ptr<L1GctJetCandCollection> forJetResult(new L1GctJetCandCollection(m_gct->getForwardJets() ) );
    std::auto_ptr<L1GctJetCandCollection> tauJetResult(new L1GctJetCandCollection(m_gct->getTauJets() ) );

    std::auto_ptr<L1GctInternJetDataCollection> internalJetResult(new L1GctInternJetDataCollection(m_gct->getInternalJets() ) );

    // create the energy sum digis
    std::auto_ptr<L1GctEtTotalCollection> etTotResult (new L1GctEtTotalCollection(m_gct->getEtSumCollection() ) );
    std::auto_ptr<L1GctEtHadCollection>   etHadResult (new L1GctEtHadCollection  (m_gct->getEtHadCollection() ) );
    std::auto_ptr<L1GctEtMissCollection>  etMissResult(new L1GctEtMissCollection (m_gct->getEtMissCollection() ) );
    std::auto_ptr<L1GctHtMissCollection>  htMissResult(new L1GctHtMissCollection (m_gct->getHtMissCollection() ) );

    // create the jet counts digis
    std::auto_ptr<L1GctJetCountsCollection> jetCountResult(new L1GctJetCountsCollection(m_gct->getJetCountsCollection() ) );

    // create the Hf sums digis
    std::auto_ptr<L1GctHFBitCountsCollection>  hfBitCountResult (new L1GctHFBitCountsCollection (m_gct->getHFBitCountsCollection () ) );
    std::auto_ptr<L1GctHFRingEtSumsCollection> hfRingEtSumResult(new L1GctHFRingEtSumsCollection(m_gct->getHFRingEtSumsCollection() ) );

    // put the collections into the event
    e.put(isoEmResult,"isoEm");
    e.put(nonIsoEmResult,"nonIsoEm");
    e.put(cenJetResult,"cenJets");
    e.put(forJetResult,"forJets");
    e.put(tauJetResult,"tauJets");
    e.put(internalJetResult);
    e.put(etTotResult);
    e.put(etHadResult);
    e.put(etMissResult);
    e.put(htMissResult);
    e.put(jetCountResult);
    e.put(hfBitCountResult);
    e.put(hfRingEtSumResult);
  }
}

DEFINE_FWK_MODULE(L1GctEmulator);

