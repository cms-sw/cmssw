/**
 * \class L1GlobalTrigger
 *
 *
 * Description: see header file.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// this class header
#include "L1GlobalTrigger.h"

// system include files
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>

// user include files
//#include
//"DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHFBitCounts.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHFRingEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFDL.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtfeExtWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1TcsWord.h"

#include "DataFormats/Common/interface/RefProd.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cstdint>

// constructors

L1GlobalTrigger::L1GlobalTrigger(const edm::ParameterSet &parSet)
    : m_muGmtInputTag(parSet.getParameter<edm::InputTag>("GmtInputTag")),
      m_caloGctInputTag(parSet.getParameter<edm::InputTag>("GctInputTag")),
      m_castorInputTag(parSet.getParameter<edm::InputTag>("CastorInputTag")),
      m_technicalTriggersInputTags(parSet.getParameter<std::vector<edm::InputTag>>("TechnicalTriggersInputTags")),
      m_produceL1GtDaqRecord(parSet.getParameter<bool>("ProduceL1GtDaqRecord")),
      m_produceL1GtEvmRecord(parSet.getParameter<bool>("ProduceL1GtEvmRecord")),
      m_produceL1GtObjectMapRecord(parSet.getParameter<bool>("ProduceL1GtObjectMapRecord")),
      m_writePsbL1GtDaqRecord(parSet.getParameter<bool>("WritePsbL1GtDaqRecord")),
      m_readTechnicalTriggerRecords(parSet.getParameter<bool>("ReadTechnicalTriggerRecords")),
      m_emulateBxInEvent(parSet.getParameter<int>("EmulateBxInEvent")),
      m_recordLength(parSet.getParameter<std::vector<int>>("RecordLength")),
      m_alternativeNrBxBoardDaq(parSet.getParameter<unsigned int>("AlternativeNrBxBoardDaq")),
      m_alternativeNrBxBoardEvm(parSet.getParameter<unsigned int>("AlternativeNrBxBoardEvm")),
      m_psBstLengthBytes(parSet.getParameter<int>("BstLengthBytes")),
      m_algorithmTriggersUnprescaled(parSet.getParameter<bool>("AlgorithmTriggersUnprescaled")),
      m_algorithmTriggersUnmasked(parSet.getParameter<bool>("AlgorithmTriggersUnmasked")),
      m_technicalTriggersUnprescaled(parSet.getParameter<bool>("TechnicalTriggersUnprescaled")),
      m_technicalTriggersUnmasked(parSet.getParameter<bool>("TechnicalTriggersUnmasked")),
      m_technicalTriggersVetoUnmasked(parSet.getParameter<bool>("TechnicalTriggersVetoUnmasked")),
      m_verbosity(parSet.getUntrackedParameter<int>("Verbosity", 0)),
      m_isDebugEnabled(edm::isDebugEnabled()),
      m_l1GtStableParToken(esConsumes<L1GtStableParameters, L1GtStableParametersRcd>()),
      m_l1GtParToken(esConsumes<L1GtParameters, L1GtParametersRcd>()),
      m_l1GtBMToken(esConsumes<L1GtBoardMaps, L1GtBoardMapsRcd>()),
      m_l1GtPfAlgoToken(esConsumes<L1GtPrescaleFactors, L1GtPrescaleFactorsAlgoTrigRcd>()),
      m_l1GtPfTechToken(esConsumes<L1GtPrescaleFactors, L1GtPrescaleFactorsTechTrigRcd>()),
      m_l1GtTmAlgoToken(esConsumes<L1GtTriggerMask, L1GtTriggerMaskAlgoTrigRcd>()),
      m_l1GtTmTechToken(esConsumes<L1GtTriggerMask, L1GtTriggerMaskTechTrigRcd>()),
      m_l1GtTmVetoAlgoToken(esConsumes<L1GtTriggerMask, L1GtTriggerMaskVetoAlgoTrigRcd>()),
      m_l1GtTmVetoTechToken(esConsumes<L1GtTriggerMask, L1GtTriggerMaskVetoTechTrigRcd>()) {
  if (m_verbosity) {
    LogDebug("L1GlobalTrigger") << std::endl;

    LogTrace("L1GlobalTrigger") << "\nInput tag for muon collection from GMT:         " << m_muGmtInputTag
                                << "\nInput tag for calorimeter collections from GCT: " << m_caloGctInputTag
                                << "\nInput tag for CASTOR record:                    " << m_castorInputTag
                                << "\nInput tag for technical triggers:               " << std::endl;

    // loop over all producers of technical trigger records
    for (std::vector<edm::InputTag>::const_iterator it = m_technicalTriggersInputTags.begin();
         it != m_technicalTriggersInputTags.end();
         it++) {
      LogTrace("L1GlobalTrigger") << "\n  " << (*it) << std::endl;
    }

    LogTrace("L1GlobalTrigger")
        << "\nProduce the L1 GT DAQ readout record:           " << m_produceL1GtDaqRecord
        << "\nProduce the L1 GT EVM readout record:           " << m_produceL1GtEvmRecord
        << "\nProduce the L1 GT Object Map record:            " << m_produceL1GtObjectMapRecord << " \n"
        << "\nWrite Psb content to L1 GT DAQ Record:          " << m_writePsbL1GtDaqRecord << " \n"
        << "\nRead technical trigger records:                 " << m_readTechnicalTriggerRecords << " \n"
        << "\nNumber of BxInEvent to be emulated:             " << m_emulateBxInEvent
        << "\nNumber of BXs corresponding to alternative 0:   " << m_recordLength.at(0)
        << "\nNumber of BXs corresponding to alternative 1:   " << m_recordLength.at(1) << " \n"
        << "\nAlternative for number of BX in GT DAQ record:   0x" << std::hex << m_alternativeNrBxBoardDaq
        << "\nAlternative for number of BX in GT EVM record:   0x" << std::hex << m_alternativeNrBxBoardEvm << std::dec
        << " \n"
        << "\nLength of BST message [bytes]:                  " << m_psBstLengthBytes << "\n"
        << "\nRun algorithm triggers unprescaled:             " << m_algorithmTriggersUnprescaled
        << "\nRun algorithm triggers unmasked (all enabled):  " << m_algorithmTriggersUnmasked << "\n"
        << "\nRun technical triggers unprescaled:             " << m_technicalTriggersUnprescaled
        << "\nRun technical triggers unmasked (all enabled):  " << m_technicalTriggersUnmasked
        << "\nRun technical triggers veto unmasked (no veto): " << m_technicalTriggersUnmasked << "\n"
        << std::endl;
  }

  if ((m_emulateBxInEvent > 0) && ((m_emulateBxInEvent % 2) == 0)) {
    m_emulateBxInEvent = m_emulateBxInEvent - 1;

    if (m_verbosity) {
      edm::LogWarning("L1GlobalTrigger") << "\nWARNING: Number of bunch crossing to be emulated rounded to: "
                                         << m_emulateBxInEvent << "\n         The number must be an odd number!\n"
                                         << std::endl;
    }
  }

  int requiredRecordLength = std::max(m_recordLength.at(0), m_recordLength.at(1));
  if ((m_emulateBxInEvent >= 0) && (m_emulateBxInEvent < requiredRecordLength)) {
    m_emulateBxInEvent = requiredRecordLength;

    if (m_verbosity) {
      edm::LogWarning("L1GlobalTrigger") << "\nWARNING: Number of bunch crossing required to be emulated ( "
                                         << m_emulateBxInEvent << " BX) smaller as required in RecordLength:"
                                         << "\n  Number of BXs corresponding to alternative 0:   "
                                         << m_recordLength.at(0)
                                         << "\n  Number of BXs corresponding to alternative 1:   "
                                         << m_recordLength.at(1) << "\nEmulating " << requiredRecordLength << " BX!"
                                         << "\n"
                                         << std::endl;
    }
  }

  // register products
  if (m_produceL1GtDaqRecord) {
    produces<L1GlobalTriggerReadoutRecord>();
  }

  if (m_produceL1GtEvmRecord) {
    produces<L1GlobalTriggerEvmReadoutRecord>();
  }

  if (m_produceL1GtObjectMapRecord) {
    produces<L1GlobalTriggerObjectMapRecord>();
  }

  // create new PSBs
  m_gtPSB = new L1GlobalTriggerPSB(m_caloGctInputTag, m_technicalTriggersInputTags, consumesCollector());
  m_gtPSB->setVerbosity(m_verbosity);

  // create new GTL
  m_gtGTL = new L1GlobalTriggerGTL(m_muGmtInputTag, consumesCollector());
  m_gtGTL->setVerbosity(m_verbosity);

  // create new FDL
  m_gtFDL = new L1GlobalTriggerFDL();
  m_gtFDL->setVerbosity(m_verbosity);

  // initialize cached IDs

  //
  m_l1GtStableParCacheID = 0ULL;

  m_numberPhysTriggers = 0;
  m_numberTechnicalTriggers = 0;
  m_numberDaqPartitions = 0;

  m_nrL1Mu = 0;

  m_nrL1NoIsoEG = 0;
  m_nrL1IsoEG = 0;

  m_nrL1CenJet = 0;
  m_nrL1ForJet = 0;
  m_nrL1TauJet = 0;

  m_nrL1JetCounts = 0;

  m_ifMuEtaNumberBits = 0;
  m_ifCaloEtaNumberBits = 0;

  //
  m_l1GtParCacheID = 0ULL;

  m_totalBxInEvent = 0;

  m_activeBoardsGtDaq = 0;
  m_activeBoardsGtEvm = 0;
  m_bstLengthBytes = 0;

  //
  m_l1GtBMCacheID = 0ULL;

  //
  m_l1GtPfAlgoCacheID = 0ULL;
  m_l1GtPfTechCacheID = 0ULL;

  m_l1GtTmAlgoCacheID = 0ULL;
  m_l1GtTmTechCacheID = 0ULL;

  m_l1GtTmVetoAlgoCacheID = 0ULL;
  m_l1GtTmVetoTechCacheID = 0ULL;

  consumes<L1MuGMTReadoutCollection>(m_muGmtInputTag);
}

// destructor
L1GlobalTrigger::~L1GlobalTrigger() {
  delete m_gtPSB;
  delete m_gtGTL;
  delete m_gtFDL;
}

// member functions

// method called to produce the data
void L1GlobalTrigger::produce(edm::Event &iEvent, const edm::EventSetup &evSetup) {
  // process event iEvent

  // get / update the stable parameters from the EventSetup
  // local cache & check on cacheIdentifier

  unsigned long long l1GtStableParCacheID = evSetup.get<L1GtStableParametersRcd>().cacheIdentifier();

  if (m_l1GtStableParCacheID != l1GtStableParCacheID) {
    edm::ESHandle<L1GtStableParameters> l1GtStablePar = evSetup.getHandle(m_l1GtStableParToken);
    m_l1GtStablePar = l1GtStablePar.product();

    // number of physics triggers
    m_numberPhysTriggers = m_l1GtStablePar->gtNumberPhysTriggers();

    // number of technical triggers
    m_numberTechnicalTriggers = m_l1GtStablePar->gtNumberTechnicalTriggers();

    // number of DAQ partitions
    m_numberDaqPartitions = 8;  // FIXME add it to stable parameters

    // number of objects of each type
    m_nrL1Mu = static_cast<int>(m_l1GtStablePar->gtNumberL1Mu());

    m_nrL1NoIsoEG = static_cast<int>(m_l1GtStablePar->gtNumberL1NoIsoEG());
    m_nrL1IsoEG = static_cast<int>(m_l1GtStablePar->gtNumberL1IsoEG());

    m_nrL1CenJet = static_cast<int>(m_l1GtStablePar->gtNumberL1CenJet());
    m_nrL1ForJet = static_cast<int>(m_l1GtStablePar->gtNumberL1ForJet());
    m_nrL1TauJet = static_cast<int>(m_l1GtStablePar->gtNumberL1TauJet());

    m_nrL1JetCounts = static_cast<int>(m_l1GtStablePar->gtNumberL1JetCounts());

    // ... the rest of the objects are global

    m_ifMuEtaNumberBits = static_cast<int>(m_l1GtStablePar->gtIfMuEtaNumberBits());
    m_ifCaloEtaNumberBits = static_cast<int>(m_l1GtStablePar->gtIfCaloEtaNumberBits());

    // (re)initialize L1GlobalTriggerGTL
    m_gtGTL->init(m_nrL1Mu, m_numberPhysTriggers);

    // (re)initialize L1GlobalTriggerPSB
    m_gtPSB->init(m_nrL1NoIsoEG, m_nrL1IsoEG, m_nrL1CenJet, m_nrL1ForJet, m_nrL1TauJet, m_numberTechnicalTriggers);

    //
    m_l1GtStableParCacheID = l1GtStableParCacheID;
  }

  // get / update the parameters from the EventSetup
  // local cache & check on cacheIdentifier

  unsigned long long l1GtParCacheID = evSetup.get<L1GtParametersRcd>().cacheIdentifier();

  if (m_l1GtParCacheID != l1GtParCacheID) {
    edm::ESHandle<L1GtParameters> l1GtPar = evSetup.getHandle(m_l1GtParToken);
    m_l1GtPar = l1GtPar.product();

    //    total number of Bx's in the event coming from EventSetup
    m_totalBxInEvent = m_l1GtPar->gtTotalBxInEvent();

    //    active boards in L1 GT DAQ record and in L1 GT EVM record
    m_activeBoardsGtDaq = m_l1GtPar->gtDaqActiveBoards();
    m_activeBoardsGtEvm = m_l1GtPar->gtEvmActiveBoards();

    ///   length of BST message (in bytes) for L1 GT EVM record
    m_bstLengthBytes = m_l1GtPar->gtBstLengthBytes();

    m_l1GtParCacheID = l1GtParCacheID;
  }

  // negative value: emulate TotalBxInEvent as given in EventSetup
  if (m_emulateBxInEvent < 0) {
    m_emulateBxInEvent = m_totalBxInEvent;
  }

  int minBxInEvent = (m_emulateBxInEvent + 1) / 2 - m_emulateBxInEvent;
  int maxBxInEvent = (m_emulateBxInEvent + 1) / 2 - 1;

  int recordLength0 = m_recordLength.at(0);
  int recordLength1 = m_recordLength.at(1);

  if ((recordLength0 < 0) || (recordLength1 < 0)) {
    // take them from event setup
    // FIXME implement later - temporary solution

    recordLength0 = m_emulateBxInEvent;
    recordLength1 = m_emulateBxInEvent;
  }

  if (m_verbosity) {
    LogDebug("L1GlobalTrigger") << "\nTotal number of BX to emulate in the GT readout record: " << m_emulateBxInEvent
                                << " = "
                                << "[" << minBxInEvent << ", " << maxBxInEvent << "] BX\n"
                                << "\nNumber of BX for alternative 0:  " << recordLength0
                                << "\nNumber of BX for alternative 1:  " << recordLength1
                                << "\nActive boards in L1 GT DAQ record (hex format) = " << std::hex
                                << std::setw(sizeof(m_activeBoardsGtDaq) * 2) << std::setfill('0')
                                << m_activeBoardsGtDaq << std::dec << std::setfill(' ')
                                << "\nActive boards in L1 GT EVM record (hex format) = " << std::hex
                                << std::setw(sizeof(m_activeBoardsGtEvm) * 2) << std::setfill('0')
                                << m_activeBoardsGtEvm << std::dec << std::setfill(' ') << "\n"
                                << std::endl;
  }

  // get / update the board maps from the EventSetup
  // local cache & check on cacheIdentifier

  typedef std::vector<L1GtBoard>::const_iterator CItBoardMaps;

  unsigned long long l1GtBMCacheID = evSetup.get<L1GtBoardMapsRcd>().cacheIdentifier();

  if (m_l1GtBMCacheID != l1GtBMCacheID) {
    edm::ESHandle<L1GtBoardMaps> l1GtBM = evSetup.getHandle(m_l1GtBMToken);
    m_l1GtBM = l1GtBM.product();

    m_l1GtBMCacheID = l1GtBMCacheID;
  }

  // TODO need changes in CondFormats to cache the maps
  const std::vector<L1GtBoard> &boardMaps = m_l1GtBM->gtBoardMaps();

  // get / update the prescale factors from the EventSetup
  // local cache & check on cacheIdentifier

  unsigned long long l1GtPfAlgoCacheID = evSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().cacheIdentifier();

  if (m_l1GtPfAlgoCacheID != l1GtPfAlgoCacheID) {
    edm::ESHandle<L1GtPrescaleFactors> l1GtPfAlgo = evSetup.getHandle(m_l1GtPfAlgoToken);
    m_l1GtPfAlgo = l1GtPfAlgo.product();

    m_prescaleFactorsAlgoTrig = &(m_l1GtPfAlgo->gtPrescaleFactors());

    m_l1GtPfAlgoCacheID = l1GtPfAlgoCacheID;
  }

  unsigned long long l1GtPfTechCacheID = evSetup.get<L1GtPrescaleFactorsTechTrigRcd>().cacheIdentifier();

  if (m_l1GtPfTechCacheID != l1GtPfTechCacheID) {
    edm::ESHandle<L1GtPrescaleFactors> l1GtPfTech = evSetup.getHandle(m_l1GtPfTechToken);
    m_l1GtPfTech = l1GtPfTech.product();

    m_prescaleFactorsTechTrig = &(m_l1GtPfTech->gtPrescaleFactors());

    m_l1GtPfTechCacheID = l1GtPfTechCacheID;
  }

  // get / update the trigger mask from the EventSetup
  // local cache & check on cacheIdentifier

  unsigned long long l1GtTmAlgoCacheID = evSetup.get<L1GtTriggerMaskAlgoTrigRcd>().cacheIdentifier();

  if (m_l1GtTmAlgoCacheID != l1GtTmAlgoCacheID) {
    edm::ESHandle<L1GtTriggerMask> l1GtTmAlgo = evSetup.getHandle(m_l1GtTmAlgoToken);
    m_l1GtTmAlgo = l1GtTmAlgo.product();

    m_triggerMaskAlgoTrig = m_l1GtTmAlgo->gtTriggerMask();

    m_l1GtTmAlgoCacheID = l1GtTmAlgoCacheID;
  }

  unsigned long long l1GtTmTechCacheID = evSetup.get<L1GtTriggerMaskTechTrigRcd>().cacheIdentifier();

  if (m_l1GtTmTechCacheID != l1GtTmTechCacheID) {
    edm::ESHandle<L1GtTriggerMask> l1GtTmTech = evSetup.getHandle(m_l1GtTmTechToken);
    m_l1GtTmTech = l1GtTmTech.product();

    m_triggerMaskTechTrig = m_l1GtTmTech->gtTriggerMask();

    m_l1GtTmTechCacheID = l1GtTmTechCacheID;
  }

  unsigned long long l1GtTmVetoAlgoCacheID = evSetup.get<L1GtTriggerMaskVetoAlgoTrigRcd>().cacheIdentifier();

  if (m_l1GtTmVetoAlgoCacheID != l1GtTmVetoAlgoCacheID) {
    edm::ESHandle<L1GtTriggerMask> l1GtTmVetoAlgo = evSetup.getHandle(m_l1GtTmVetoAlgoToken);
    m_l1GtTmVetoAlgo = l1GtTmVetoAlgo.product();

    m_triggerMaskVetoAlgoTrig = m_l1GtTmVetoAlgo->gtTriggerMask();

    m_l1GtTmVetoAlgoCacheID = l1GtTmVetoAlgoCacheID;
  }

  unsigned long long l1GtTmVetoTechCacheID = evSetup.get<L1GtTriggerMaskVetoTechTrigRcd>().cacheIdentifier();

  if (m_l1GtTmVetoTechCacheID != l1GtTmVetoTechCacheID) {
    edm::ESHandle<L1GtTriggerMask> l1GtTmVetoTech = evSetup.getHandle(m_l1GtTmVetoTechToken);
    m_l1GtTmVetoTech = l1GtTmVetoTech.product();

    m_triggerMaskVetoTechTrig = m_l1GtTmVetoTech->gtTriggerMask();

    m_l1GtTmVetoTechCacheID = l1GtTmVetoTechCacheID;
  }

  // loop over blocks in the GT DAQ record receiving data, count them if they
  // are active all board type are defined in CondFormats/L1TObjects/L1GtFwd
  // enum L1GtBoardType { GTFE, FDL, PSB, GMT, TCS, TIM };
  // &
  // set the active flag for each object type received from GMT and GCT
  // all objects in the GT system are defined in enum L1GtObject from
  // DataFormats/L1Trigger/L1GlobalTriggerReadoutSetupFwd

  int daqNrFdlBoards = 0;
  int daqNrPsbBoards = 0;

  //
  bool receiveMu = false;
  bool receiveNoIsoEG = false;
  bool receiveIsoEG = false;
  bool receiveCenJet = false;
  bool receiveForJet = false;
  bool receiveTauJet = false;
  bool receiveETM = false;
  bool receiveETT = false;
  bool receiveHTT = false;
  bool receiveHTM = false;
  bool receiveJetCounts = false;
  bool receiveHfBitCounts = false;
  bool receiveHfRingEtSums = false;

  bool receiveExternal = false;

  bool receiveTechTr = false;

  for (CItBoardMaps itBoard = boardMaps.begin(); itBoard != boardMaps.end(); ++itBoard) {
    int iPosition = itBoard->gtPositionDaqRecord();
    if (iPosition > 0) {
      int iActiveBit = itBoard->gtBitDaqActiveBoards();
      bool activeBoard = false;

      if (iActiveBit >= 0) {
        activeBoard = m_activeBoardsGtDaq & (1 << iActiveBit);
      }

      // use board if: in the record, but not in ActiveBoardsMap (iActiveBit <
      // 0)
      //               in the record and ActiveBoardsMap, and active
      if ((iActiveBit < 0) || activeBoard) {
        switch (itBoard->gtBoardType()) {
          case FDL: {
            daqNrFdlBoards++;
          }

          break;
          case PSB: {
            daqNrPsbBoards++;

            // get the objects coming to this PSB
            std::vector<L1GtPsbQuad> quadInPsb = itBoard->gtQuadInPsb();
            for (std::vector<L1GtPsbQuad>::const_iterator itQuad = quadInPsb.begin(); itQuad != quadInPsb.end();
                 ++itQuad) {
              switch (*itQuad) {
                case TechTr: {
                  receiveTechTr = true;
                }

                break;
                case NoIsoEGQ: {
                  receiveNoIsoEG = true;
                }

                break;
                case IsoEGQ: {
                  receiveIsoEG = true;
                }

                break;
                case CenJetQ: {
                  receiveCenJet = true;
                }

                break;
                case ForJetQ: {
                  receiveForJet = true;
                }

                break;
                case TauJetQ: {
                  receiveTauJet = true;
                }

                break;
                case ESumsQ: {
                  receiveETM = true;
                  receiveETT = true;
                  receiveHTT = true;
                  receiveHTM = true;
                }

                break;
                case JetCountsQ: {
                  receiveJetCounts = true;
                }

                break;
                case CastorQ: {
                  // obsolete
                }

                break;
                case BptxQ: {
                  // obsolete
                }

                break;
                case GtExternalQ: {
                  receiveExternal = true;
                }

                break;
                case HfQ: {
                  receiveHfBitCounts = true;
                  receiveHfRingEtSums = true;
                }

                break;
                  // FIXME add MIP/Iso bits
                default: {
                  // do nothing
                }

                break;
              }
            }

          }

          break;
          default: {
            // do nothing, all blocks are given in GtBoardType enum
          }

          break;
        }
      }
    }
  }

  // produce the L1GlobalTriggerReadoutRecord now, after we found how many
  // BxInEvent the record has and how many boards are active
  std::unique_ptr<L1GlobalTriggerReadoutRecord> gtDaqReadoutRecord(
      new L1GlobalTriggerReadoutRecord(m_emulateBxInEvent, daqNrFdlBoards, daqNrPsbBoards));

  // * produce the L1GlobalTriggerEvmReadoutRecord
  std::unique_ptr<L1GlobalTriggerEvmReadoutRecord> gtEvmReadoutRecord(
      new L1GlobalTriggerEvmReadoutRecord(m_emulateBxInEvent, daqNrFdlBoards));
  // daqNrFdlBoards OK, just reserve memory at this point

  // * produce the L1GlobalTriggerObjectMapRecord
  std::unique_ptr<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord(new L1GlobalTriggerObjectMapRecord());

  // fill the boards not depending on the BxInEvent in the L1 GT DAQ record
  // GMT, PSB and FDL depend on BxInEvent

  // fill in emulator the same bunch crossing (12 bits - hardwired number of
  // bits...) and the same local bunch crossing for all boards
  int bxCross = iEvent.bunchCrossing();
  uint16_t bxCrossHw = 0;
  if ((bxCross & 0xFFF) == bxCross) {
    bxCrossHw = static_cast<uint16_t>(bxCross);
  } else {
    bxCrossHw = 0;  // Bx number too large, set to 0!
    if (m_verbosity) {
      LogDebug("L1GlobalTrigger") << "\nBunch cross number [hex] = " << std::hex << bxCross
                                  << "\n  larger than 12 bits. Set to 0! \n"
                                  << std::dec << std::endl;
    }
  }

  if (m_produceL1GtDaqRecord) {
    for (CItBoardMaps itBoard = boardMaps.begin(); itBoard != boardMaps.end(); ++itBoard) {
      int iPosition = itBoard->gtPositionDaqRecord();
      if (iPosition > 0) {
        int iActiveBit = itBoard->gtBitDaqActiveBoards();
        bool activeBoard = false;

        if (iActiveBit >= 0) {
          activeBoard = m_activeBoardsGtDaq & (1 << iActiveBit);
        }

        // use board if: in the record, but not in ActiveBoardsMap (iActiveBit <
        // 0)
        //               in the record and ActiveBoardsMap, and active
        if ((iActiveBit < 0) || activeBoard) {
          switch (itBoard->gtBoardType()) {
            case GTFE: {
              L1GtfeWord gtfeWordValue;

              gtfeWordValue.setBoardId(itBoard->gtBoardId());

              // cast int to uint16_t
              // there are normally 3 or 5 BxInEvent
              gtfeWordValue.setRecordLength(static_cast<uint16_t>(recordLength0));

              gtfeWordValue.setRecordLength1(static_cast<uint16_t>(recordLength1));

              // bunch crossing
              gtfeWordValue.setBxNr(bxCrossHw);

              // set the list of active boards
              gtfeWordValue.setActiveBoards(m_activeBoardsGtDaq);

              // set alternative for number of BX per board
              gtfeWordValue.setAltNrBxBoard(static_cast<uint16_t>(m_alternativeNrBxBoardDaq));

              // set the TOTAL_TRIGNR as read from iEvent
              // TODO check again - PTC stuff

              gtfeWordValue.setTotalTriggerNr(static_cast<uint32_t>(iEvent.id().event()));

              // ** fill L1GtfeWord in GT DAQ record

              gtDaqReadoutRecord->setGtfeWord(gtfeWordValue);
            }

            break;
            case TCS: {
              // nothing
            }

            break;
            case TIM: {
              // nothing
            }

            break;
            default: {
              // do nothing, all blocks are given in GtBoardType enum
            }

            break;
          }
        }
      }
    }
  }

  // fill the boards not depending on the BxInEvent in the L1 GT EVM record

  int evmNrFdlBoards = 0;

  if (m_produceL1GtEvmRecord) {
    // get the length of the BST message from parameter set or from event setup

    int bstLengthBytes = 0;

    if (m_psBstLengthBytes < 0) {
      // length from event setup
      bstLengthBytes = static_cast<int>(m_bstLengthBytes);

    } else {
      // length from parameter set
      bstLengthBytes = m_psBstLengthBytes;
    }

    if (m_verbosity) {
      LogTrace("L1GlobalTrigger") << "\n Length of BST message (in bytes): " << bstLengthBytes << "\n" << std::endl;
    }

    for (CItBoardMaps itBoard = boardMaps.begin(); itBoard != boardMaps.end(); ++itBoard) {
      int iPosition = itBoard->gtPositionEvmRecord();
      if (iPosition > 0) {
        int iActiveBit = itBoard->gtBitEvmActiveBoards();
        bool activeBoard = false;

        if (iActiveBit >= 0) {
          activeBoard = m_activeBoardsGtEvm & (1 << iActiveBit);
        }

        // use board if: in the record, but not in ActiveBoardsMap (iActiveBit <
        // 0)
        //               in the record and ActiveBoardsMap, and active
        if ((iActiveBit < 0) || activeBoard) {
          switch (itBoard->gtBoardType()) {
            case GTFE: {
              L1GtfeExtWord gtfeWordValue(bstLengthBytes);

              gtfeWordValue.setBoardId(itBoard->gtBoardId());

              // cast int to uint16_t
              // there are normally 3 or 5 BxInEvent
              gtfeWordValue.setRecordLength(static_cast<uint16_t>(recordLength0));

              gtfeWordValue.setRecordLength1(static_cast<uint16_t>(recordLength1));

              // bunch crossing
              gtfeWordValue.setBxNr(bxCrossHw);

              // set the list of active boards
              gtfeWordValue.setActiveBoards(m_activeBoardsGtEvm);

              // set alternative for number of BX per board
              gtfeWordValue.setAltNrBxBoard(static_cast<uint16_t>(m_alternativeNrBxBoardEvm));

              // set the TOTAL_TRIGNR as read from iEvent
              // TODO check again - PTC stuff

              gtfeWordValue.setTotalTriggerNr(static_cast<uint32_t>(iEvent.id().event()));

              // set the GPS time to the value read from Timestamp
              edm::TimeValue_t evTime = iEvent.time().value();

              gtfeWordValue.setGpsTime(evTime);

              // LogDebug("L1GlobalTrigger")
              //<< "\nEvent timestamp value [hex] = " << std::hex << evTime
              //<< "\nBST retrieved value [hex]   = " << gtfeWordValue.gpsTime()
              //<< std::dec << std::endl;

              // source of BST message: DDDD simulated data
              uint16_t bstSourceVal = 0xDDDD;
              gtfeWordValue.setBstSource(bstSourceVal);

              // ** fill L1GtfeWord in GT EVM record

              gtEvmReadoutRecord->setGtfeWord(gtfeWordValue);
            }

            break;
            case FDL: {
              evmNrFdlBoards++;
            }

            break;
            case TCS: {
              L1TcsWord tcsWordValue;

              tcsWordValue.setBoardId(itBoard->gtBoardId());

              // bunch crossing
              tcsWordValue.setBxNr(bxCrossHw);

              uint16_t trigType = 0x5;  // 0101 simulated event
              tcsWordValue.setTriggerType(trigType);

              // luminosity segment number
              tcsWordValue.setLuminositySegmentNr(static_cast<uint16_t>(iEvent.luminosityBlock()));

              // set the Event_Nr as read from iEvent
              tcsWordValue.setEventNr(static_cast<uint32_t>(iEvent.id().event()));

              // orbit number
              tcsWordValue.setOrbitNr(static_cast<uint64_t>(iEvent.orbitNumber()));

              // ** fill L1TcsWord in the EVM record

              gtEvmReadoutRecord->setTcsWord(tcsWordValue);

            }

            break;
            case TIM: {
              // nothing
            }

            break;
            default: {
              // do nothing, all blocks are given in GtBoardType enum
            }

            break;
          }
        }
      }
    }
  }

  // get the prescale factor set used in the actual luminosity segment
  int pfAlgoSetIndex = 0;  // FIXME
  const std::vector<int> &prescaleFactorsAlgoTrig = (*m_prescaleFactorsAlgoTrig).at(pfAlgoSetIndex);

  int pfTechSetIndex = 0;  // FIXME
  const std::vector<int> &prescaleFactorsTechTrig = (*m_prescaleFactorsTechTrig).at(pfTechSetIndex);

  //

  // loop over BxInEvent
  for (int iBxInEvent = minBxInEvent; iBxInEvent <= maxBxInEvent; ++iBxInEvent) {
    // * receive GCT object data via PSBs
    // LogDebug("L1GlobalTrigger")
    //<< "\nL1GlobalTrigger : receiving PSB data for bx = " << iBxInEvent <<
    //"\n"
    //<< std::endl;

    m_gtPSB->receiveGctObjectData(iEvent,
                                  m_caloGctInputTag,
                                  iBxInEvent,
                                  receiveNoIsoEG,
                                  m_nrL1NoIsoEG,
                                  receiveIsoEG,
                                  m_nrL1IsoEG,
                                  receiveCenJet,
                                  m_nrL1CenJet,
                                  receiveForJet,
                                  m_nrL1ForJet,
                                  receiveTauJet,
                                  m_nrL1TauJet,
                                  receiveETM,
                                  receiveETT,
                                  receiveHTT,
                                  receiveHTM,
                                  receiveJetCounts,
                                  receiveHfBitCounts,
                                  receiveHfRingEtSums);

    /// receive technical trigger
    if (m_readTechnicalTriggerRecords) {
      m_gtPSB->receiveTechnicalTriggers(
          iEvent, m_technicalTriggersInputTags, iBxInEvent, receiveTechTr, m_numberTechnicalTriggers);
    }

    if (receiveExternal) {
      // FIXME read the external conditions
    }

    if (m_produceL1GtDaqRecord && m_writePsbL1GtDaqRecord) {
      m_gtPSB->fillPsbBlock(iEvent,
                            m_activeBoardsGtDaq,
                            recordLength0,
                            recordLength1,
                            m_alternativeNrBxBoardDaq,
                            boardMaps,
                            iBxInEvent,
                            gtDaqReadoutRecord.get());
    }

    // * receive GMT object data via GTL
    // LogDebug("L1GlobalTrigger")
    //<< "\nL1GlobalTrigger : receiving GMT data for bx = " << iBxInEvent <<
    //"\n"
    //<< std::endl;

    m_gtGTL->receiveGmtObjectData(iEvent, m_muGmtInputTag, iBxInEvent, receiveMu, m_nrL1Mu);

    // * run GTL
    // LogDebug("L1GlobalTrigger")
    //<< "\nL1GlobalTrigger : running GTL for bx = " << iBxInEvent << "\n"
    //<< std::endl;

    m_gtGTL->run(iEvent,
                 evSetup,
                 m_gtPSB,
                 m_produceL1GtObjectMapRecord,
                 iBxInEvent,
                 gtObjectMapRecord.get(),
                 m_numberPhysTriggers,
                 m_nrL1Mu,
                 m_nrL1NoIsoEG,
                 m_nrL1IsoEG,
                 m_nrL1CenJet,
                 m_nrL1ForJet,
                 m_nrL1TauJet,
                 m_nrL1JetCounts,
                 m_ifMuEtaNumberBits,
                 m_ifCaloEtaNumberBits);

    // LogDebug("L1GlobalTrigger")
    //<< "\n AlgorithmOR\n" << m_gtGTL->getAlgorithmOR() << "\n"
    //<< std::endl;

    // * run FDL
    // LogDebug("L1GlobalTrigger")
    //<< "\nL1GlobalTrigger : running FDL for bx = " << iBxInEvent << "\n"
    //<< std::endl;

    m_gtFDL->run(iEvent,
                 prescaleFactorsAlgoTrig,
                 prescaleFactorsTechTrig,
                 m_triggerMaskAlgoTrig,
                 m_triggerMaskTechTrig,
                 m_triggerMaskVetoAlgoTrig,
                 m_triggerMaskVetoTechTrig,
                 boardMaps,
                 m_emulateBxInEvent,
                 iBxInEvent,
                 m_numberPhysTriggers,
                 m_numberTechnicalTriggers,
                 m_numberDaqPartitions,
                 m_gtGTL,
                 m_gtPSB,
                 pfAlgoSetIndex,
                 pfTechSetIndex,
                 m_algorithmTriggersUnprescaled,
                 m_algorithmTriggersUnmasked,
                 m_technicalTriggersUnprescaled,
                 m_technicalTriggersUnmasked,
                 m_technicalTriggersVetoUnmasked);

    if (m_produceL1GtDaqRecord && (daqNrFdlBoards > 0)) {
      m_gtFDL->fillDaqFdlBlock(iBxInEvent,
                               m_activeBoardsGtDaq,
                               recordLength0,
                               recordLength1,
                               m_alternativeNrBxBoardDaq,
                               boardMaps,
                               gtDaqReadoutRecord.get());
    }

    if (m_produceL1GtEvmRecord && (evmNrFdlBoards > 0)) {
      m_gtFDL->fillEvmFdlBlock(iBxInEvent,
                               m_activeBoardsGtEvm,
                               recordLength0,
                               recordLength1,
                               m_alternativeNrBxBoardEvm,
                               boardMaps,
                               gtEvmReadoutRecord.get());
    }

    // reset
    m_gtPSB->reset();
    m_gtGTL->reset();
    m_gtFDL->reset();

    // LogDebug("L1GlobalTrigger") << "\n Reset PSB, GTL, FDL\n" << std::endl;
  }

  if (receiveMu) {
    // LogDebug("L1GlobalTrigger")
    //<< "\n**** "
    //<< "\n  Persistent reference for L1MuGMTReadoutCollection with input tag:
    //"
    //<< m_muGmtInputTag
    //<< "\n**** \n"
    //<< std::endl;

    // get L1MuGMTReadoutCollection reference and set it in GT record

    edm::Handle<L1MuGMTReadoutCollection> gmtRcHandle;
    iEvent.getByLabel(m_muGmtInputTag, gmtRcHandle);

    if (!gmtRcHandle.isValid()) {
      if (m_verbosity) {
        edm::LogWarning("L1GlobalTrigger") << "\nWarning: L1MuGMTReadoutCollection with input tag " << m_muGmtInputTag
                                           << "\nrequested in configuration, but not found in the event.\n"
                                           << std::endl;
      }
    } else {
      gtDaqReadoutRecord->setMuCollectionRefProd(gmtRcHandle);
    }
  }

  if (m_verbosity && m_isDebugEnabled) {
    std::ostringstream myCoutStream;
    gtDaqReadoutRecord->print(myCoutStream);
    LogTrace("L1GlobalTrigger") << "\n The following L1 GT DAQ readout record was produced:\n"
                                << myCoutStream.str() << "\n"
                                << std::endl;

    myCoutStream.str("");
    myCoutStream.clear();

    gtEvmReadoutRecord->print(myCoutStream);
    LogTrace("L1GlobalTrigger") << "\n The following L1 GT EVM readout record was produced:\n"
                                << myCoutStream.str() << "\n"
                                << std::endl;

    myCoutStream.str("");
    myCoutStream.clear();

    const std::vector<L1GlobalTriggerObjectMap> objMapVec = gtObjectMapRecord->gtObjectMap();

    for (std::vector<L1GlobalTriggerObjectMap>::const_iterator it = objMapVec.begin(); it != objMapVec.end(); ++it) {
      (*it).print(myCoutStream);
    }

    LogDebug("L1GlobalTrigger") << "Test gtObjectMapRecord in L1GlobalTrigger \n\n"
                                << myCoutStream.str() << "\n\n"
                                << std::endl;

    myCoutStream.str("");
    myCoutStream.clear();
  }

  // **
  // register products
  if (m_produceL1GtDaqRecord) {
    iEvent.put(std::move(gtDaqReadoutRecord));
  }

  if (m_produceL1GtEvmRecord) {
    iEvent.put(std::move(gtEvmReadoutRecord));
  }

  if (m_produceL1GtObjectMapRecord) {
    iEvent.put(std::move(gtObjectMapRecord));
  }
}

// static data members

DEFINE_FWK_MODULE(L1GlobalTrigger);
