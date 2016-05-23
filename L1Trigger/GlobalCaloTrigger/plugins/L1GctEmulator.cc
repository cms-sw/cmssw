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
#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1GctChannelMaskRcd.h"

// GCT include files
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"

// RCT data includes
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

// GCT data includes
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

using std::vector;

L1GctEmulator::L1GctEmulator(const edm::ParameterSet& ps) :
  m_jetEtCalibLuts(),
  m_writeInternalData(ps.getParameter<bool>("writeInternalData")),
  m_verbose(ps.getUntrackedParameter<bool>("verbose", false)),
  m_conditionsLabel(ps.getParameter<std::string>("conditionsLabel"))
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
  produces<L1GctInternEtSumCollection>();
  produces<L1GctInternHtMissCollection>();
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
  bool hwTest = ps.getParameter<bool>("hardwareTest");
  if (hwTest) {
    unsigned mask = ps.getUntrackedParameter<unsigned>("jetLeafMask", 0);
    m_gct = new L1GlobalCaloTrigger(jfType, mask);
    edm::LogWarning ("L1GctEmulatorSetup") << "Emulator has been configured in hardware test mode with mask " << mask
					   << "\nThis mode should NOT be used for Physics studies!";
  } else {
    m_gct = new L1GlobalCaloTrigger(jfType);
  }
  m_gct->setBxRange(firstBx, lastBx);

  // Fill the jetEtCalibLuts vector
  lutPtr nextLut( new L1GctJetEtCalibrationLut() );

  for (unsigned ieta=0; ieta<L1GctJetFinderParams::NUMBER_ETA_VALUES; ieta++) {
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
  consumes<L1CaloEmCollection>(m_inputLabel);
  consumes<L1CaloRegionCollection>(m_inputLabel);
}

L1GctEmulator::~L1GctEmulator() {
  if (m_gct != 0) delete m_gct;
}


void L1GctEmulator::beginJob()
{
}

void L1GctEmulator::endJob()
{
}

int L1GctEmulator::configureGct(const edm::EventSetup& c)
{
  int success = 0;

  if (success == 0) {
    // get data from EventSetup
    edm::ESHandle< L1GctJetFinderParams > jfPars ;
    c.get< L1GctJetFinderParamsRcd >().get( m_conditionsLabel, jfPars ) ; // which record?
    edm::ESHandle< L1GctChannelMask > chanMask ;
    c.get< L1GctChannelMaskRcd >().get( m_conditionsLabel, chanMask ) ; // which record?
    edm::ESHandle< L1CaloEtScale > etScale ;
    c.get< L1JetEtScaleRcd >().get( m_conditionsLabel, etScale ) ; // which record?
    edm::ESHandle< L1CaloEtScale > htMissScale ;
    c.get< L1HtMissScaleRcd >().get( m_conditionsLabel, htMissScale ) ; // which record?
    edm::ESHandle< L1CaloEtScale > hfRingEtScale ;
    c.get< L1HfRingEtScaleRcd >().get( m_conditionsLabel, hfRingEtScale ) ; // which record?


    if (jfPars.product() == 0) {
      success = -1;
      if (m_verbose) {
	edm::LogWarning("L1GctConfigFailure")
	  << "Failed to find a L1GctJetFinderParamsRcd:L1GctJetFinderParams in EventSetup!" << std::endl;
      }
    }

    if (chanMask.product() == 0) {
      success = -1;
      if (m_verbose) {
	edm::LogWarning("L1GctConfigFailure")
	  << "Failed to find a L1GctChannelMaskRcd:L1GctChannelMask in EventSetup!" << std::endl;
      }
    }

    if (hfRingEtScale.product() == 0) {
      success = -1;
      if (m_verbose) {
	edm::LogWarning("L1GctConfigFailure")
	  << "Failed to find a L1HfRingEtScaleRcd:L1HfRingEtScaleRcd in EventSetup!" << std::endl;
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
      m_gct->setHtMissScale(htMissScale.product());
      m_gct->setupHfSumLuts(hfRingEtScale.product());
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

  // The emulator will always produce output collections, which get filled as long as
  // the setup and input data are present. Start by making empty output collections.

  // create the em and jet collections
  std::auto_ptr<L1GctEmCandCollection> isoEmResult   (new L1GctEmCandCollection( ) );
  std::auto_ptr<L1GctEmCandCollection> nonIsoEmResult(new L1GctEmCandCollection( ) );
  std::auto_ptr<L1GctJetCandCollection> cenJetResult(new L1GctJetCandCollection( ) );
  std::auto_ptr<L1GctJetCandCollection> forJetResult(new L1GctJetCandCollection( ) );
  std::auto_ptr<L1GctJetCandCollection> tauJetResult(new L1GctJetCandCollection( ) );

  // create the energy sum digis
  std::auto_ptr<L1GctEtTotalCollection> etTotResult (new L1GctEtTotalCollection( ) );
  std::auto_ptr<L1GctEtHadCollection>   etHadResult (new L1GctEtHadCollection  ( ) );
  std::auto_ptr<L1GctEtMissCollection>  etMissResult(new L1GctEtMissCollection ( ) );
  std::auto_ptr<L1GctHtMissCollection>  htMissResult(new L1GctHtMissCollection ( ) );

  // create the Hf sums digis
  std::auto_ptr<L1GctHFBitCountsCollection>  hfBitCountResult (new L1GctHFBitCountsCollection ( ) );
  std::auto_ptr<L1GctHFRingEtSumsCollection> hfRingEtSumResult(new L1GctHFRingEtSumsCollection( ) );

  // create internal data collections
  std::auto_ptr<L1GctInternJetDataCollection> internalJetResult   (new L1GctInternJetDataCollection( ));
  std::auto_ptr<L1GctInternEtSumCollection>   internalEtSumResult (new L1GctInternEtSumCollection  ( ));
  std::auto_ptr<L1GctInternHtMissCollection>  internalHtMissResult(new L1GctInternHtMissCollection ( ));

  // get config data from EventSetup.
  // check this has been done successfully before proceeding
  if (configureGct(c) == 0) { 

    // get the RCT data
    edm::Handle<L1CaloEmCollection> em;
    edm::Handle<L1CaloRegionCollection> rgn;
    bool gotEm  = e.getByLabel(m_inputLabel, em);
    bool gotRgn = e.getByLabel(m_inputLabel, rgn);

    // check the data
    if (!gotEm && m_verbose) {
      edm::LogError("L1GctInputFailedError")
	<< "Failed to get em candidates with label " << m_inputLabel << " - GCT emulator will not be run" << std::endl;
    }

    if (!gotRgn && m_verbose) {
      edm::LogError("L1GctInputFailedError")
	<< "Failed to get calo regions with label " << m_inputLabel << " - GCT emulator will not be run" << std::endl;
    }

    if (gotEm && !em.isValid()) {
      gotEm = false;
      if (m_verbose) {
	edm::LogError("L1GctInputFailedError")
	  << "isValid() flag set to false for em candidates with label " << m_inputLabel << " - GCT emulator will not be run" << std::endl;
      }
    }

    if (gotRgn && !rgn.isValid()) {
      gotRgn = false;
      if (m_verbose) {
	edm::LogError("L1GctInputFailedError")
	  << "isValid() flag set to false for calo regions with label " << m_inputLabel << " - GCT emulator will not be run" << std::endl;
      }
    }

    // if all is ok, proceed with GCT processing
    if (gotEm && gotRgn) {
      // reset the GCT internal buffers
      m_gct->reset();

      // fill the GCT source cards
      m_gct->fillEmCands(*em);
      m_gct->fillRegions(*rgn);
  
      // process the event
      m_gct->process();

      // fill the em and jet collections
      *isoEmResult    = m_gct->getIsoElectrons();
      *nonIsoEmResult = m_gct->getNonIsoElectrons();
      *cenJetResult   = m_gct->getCentralJets();
      *forJetResult   = m_gct->getForwardJets();
      *tauJetResult   = m_gct->getTauJets();

      // fill the energy sum digis
      *etTotResult  = m_gct->getEtSumCollection();
      *etHadResult  = m_gct->getEtHadCollection();
      *etMissResult = m_gct->getEtMissCollection();
      *htMissResult = m_gct->getHtMissCollection();

      // fill the Hf sums digis
      *hfBitCountResult  = m_gct->getHFBitCountsCollection ();
      *hfRingEtSumResult = m_gct->getHFRingEtSumsCollection();

      // fill internal data collections if required
      if (m_writeInternalData) {
	*internalJetResult    = m_gct->getInternalJets();
	*internalEtSumResult  = m_gct->getInternalEtSums();
	*internalHtMissResult = m_gct->getInternalHtMiss();
      }
    }
  }

  // put the collections into the event
  e.put(isoEmResult,"isoEm");
  e.put(nonIsoEmResult,"nonIsoEm");
  e.put(cenJetResult,"cenJets");
  e.put(forJetResult,"forJets");
  e.put(tauJetResult,"tauJets");
  e.put(etTotResult);
  e.put(etHadResult);
  e.put(etMissResult);
  e.put(htMissResult);
  e.put(hfBitCountResult);
  e.put(hfRingEtSumResult);

  e.put(internalJetResult);
  e.put(internalEtSumResult);
  e.put(internalHtMissResult);

}

DEFINE_FWK_MODULE(L1GctEmulator);

