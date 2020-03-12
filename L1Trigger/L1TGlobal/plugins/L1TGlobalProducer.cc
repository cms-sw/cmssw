// L1TGlobalProducer.cc
//author:   Brian Winer - Ohio State
//          Vladimir Rekovic - extend for overlap removal

#include "L1Trigger/L1TGlobal/plugins/L1TGlobalProducer.h"

// system include files
#include <memory>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "FWCore/Utilities/interface/typedefs.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1TGlobalParameters.h"

#include "CondFormats/DataRecord/interface/L1TGlobalParametersRcd.h"
#include "L1Trigger/L1TGlobal/interface/GlobalParamsHelper.h"
#include "CondFormats/L1TObjects/interface/L1TGlobalPrescalesVetos.h"
#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosRcd.h"

#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalObjectMapRecord.h"

#include "L1Trigger/L1TGlobal/interface/TriggerMenu.h"

#include "TriggerMenuParser.h"

using namespace l1t;

void L1TGlobalProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // These parameters are part of the L1T/HLT interface, avoid changing if possible::
  desc.add<edm::InputTag>("MuonInputTag", edm::InputTag(""))
      ->setComment("InputTag for Global Muon Trigger (required parameter:  default value is invalid)");
  desc.add<edm::InputTag>("EGammaInputTag", edm::InputTag(""))
      ->setComment("InputTag for Calo Trigger EGamma (required parameter:  default value is invalid)");
  desc.add<edm::InputTag>("TauInputTag", edm::InputTag(""))
      ->setComment("InputTag for Calo Trigger Tau (required parameter:  default value is invalid)");
  desc.add<edm::InputTag>("JetInputTag", edm::InputTag(""))
      ->setComment("InputTag for Calo Trigger Jet (required parameter:  default value is invalid)");
  desc.add<edm::InputTag>("EtSumInputTag", edm::InputTag(""))
      ->setComment("InputTag for Calo Trigger EtSum (required parameter:  default value is invalid)");
  desc.add<edm::InputTag>("ExtInputTag", edm::InputTag(""))
      ->setComment("InputTag for external conditions (not required, but recommend to specify explicitly in config)");
  desc.add<edm::InputTag>("AlgoBlkInputTag", edm::InputTag("gtDigis"))
      ->setComment("InputTag for unpacked Algblk (required only if GetPrescaleColumnFromData set to true)");
  desc.add<bool>("GetPrescaleColumnFromData", false)
      ->setComment("Get prescale column from unpacked GlobalAlgBck. Otherwise use value specified in PrescaleSet");
  desc.add<bool>("AlgorithmTriggersUnprescaled", false)
      ->setComment("not required, but recommend to specify explicitly in config");
  desc.add<bool>("AlgorithmTriggersUnmasked", false)
      ->setComment("not required, but recommend to specify explicitly in config");
  // These parameters have well defined  default values and are not currently
  // part of the L1T/HLT interface.  They can be cleaned up or updated at will:
  desc.add<bool>("ProduceL1GtDaqRecord", true);
  desc.add<bool>("ProduceL1GtObjectMapRecord", true);
  desc.add<int>("EmulateBxInEvent", 1);
  desc.add<int>("L1DataBxInEvent", 5);
  desc.add<unsigned int>("AlternativeNrBxBoardDaq", 0);
  desc.add<int>("BstLengthBytes", -1);
  desc.add<unsigned int>("PrescaleSet", 1);
  desc.addUntracked<int>("Verbosity", 0);
  desc.addUntracked<bool>("PrintL1Menu", false);
  desc.add<std::string>("TriggerMenuLuminosity", "startup");
  desc.add<std::string>("PrescaleCSVFile", "prescale_L1TGlobal.csv");
  descriptions.add("L1TGlobalProducer", desc);
}

// constructors

L1TGlobalProducer::L1TGlobalProducer(const edm::ParameterSet& parSet)
    : m_muInputTag(parSet.getParameter<edm::InputTag>("MuonInputTag")),
      m_egInputTag(parSet.getParameter<edm::InputTag>("EGammaInputTag")),
      m_tauInputTag(parSet.getParameter<edm::InputTag>("TauInputTag")),
      m_jetInputTag(parSet.getParameter<edm::InputTag>("JetInputTag")),
      m_sumInputTag(parSet.getParameter<edm::InputTag>("EtSumInputTag")),
      m_extInputTag(parSet.getParameter<edm::InputTag>("ExtInputTag")),

      m_produceL1GtDaqRecord(parSet.getParameter<bool>("ProduceL1GtDaqRecord")),
      m_produceL1GtObjectMapRecord(parSet.getParameter<bool>("ProduceL1GtObjectMapRecord")),

      m_emulateBxInEvent(parSet.getParameter<int>("EmulateBxInEvent")),
      m_L1DataBxInEvent(parSet.getParameter<int>("L1DataBxInEvent")),

      m_alternativeNrBxBoardDaq(parSet.getParameter<unsigned int>("AlternativeNrBxBoardDaq")),
      m_psBstLengthBytes(parSet.getParameter<int>("BstLengthBytes")),

      m_prescaleSet(parSet.getParameter<unsigned int>("PrescaleSet")),

      m_algorithmTriggersUnprescaled(parSet.getParameter<bool>("AlgorithmTriggersUnprescaled")),
      m_algorithmTriggersUnmasked(parSet.getParameter<bool>("AlgorithmTriggersUnmasked")),

      m_verbosity(parSet.getUntrackedParameter<int>("Verbosity")),
      m_printL1Menu(parSet.getUntrackedParameter<bool>("PrintL1Menu")),
      m_isDebugEnabled(edm::isDebugEnabled()),
      m_getPrescaleColumnFromData(parSet.getParameter<bool>("GetPrescaleColumnFromData")),
      m_algoblkInputTag(parSet.getParameter<edm::InputTag>("AlgoBlkInputTag")) {
  m_egInputToken = consumes<BXVector<EGamma>>(m_egInputTag);
  m_tauInputToken = consumes<BXVector<Tau>>(m_tauInputTag);
  m_jetInputToken = consumes<BXVector<Jet>>(m_jetInputTag);
  m_sumInputToken = consumes<BXVector<EtSum>>(m_sumInputTag);
  m_muInputToken = consumes<BXVector<Muon>>(m_muInputTag);
  m_extInputToken = consumes<BXVector<GlobalExtBlk>>(m_extInputTag);
  if (m_getPrescaleColumnFromData)
    m_algoblkInputToken = consumes<BXVector<GlobalAlgBlk>>(m_algoblkInputTag);

  if (m_verbosity) {
    LogTrace("L1TGlobalProducer") << "\nInput tag for muon collection from uGMT:         " << m_muInputTag
                                  << "\nInput tag for calorimeter jet collections from Cal: " << m_jetInputTag
                                  << "\nInput tag for external conditions     :         " << m_extInputTag << std::endl;

    LogTrace("L1TGlobalProducer") << "\nProduce the L1 uGT DAQ readout record:          " << m_produceL1GtDaqRecord
                                  << "\nProduce the L1 uGT Object Map record:           "
                                  << m_produceL1GtObjectMapRecord << " \n"
                                  << "\nNumber of BxInEvent to be emulated:             " << m_emulateBxInEvent << " \n"
                                  << "\nAlternative for number of BX in GT DAQ record:   0x" << std::hex
                                  << m_alternativeNrBxBoardDaq << " \n"
                                  << "\nRun algorithm triggers unprescaled:             "
                                  << m_algorithmTriggersUnprescaled
                                  << "\nRun algorithm triggers unmasked (all enabled):  " << m_algorithmTriggersUnmasked
                                  << "\n"
                                  << std::endl;
  }

  if ((m_emulateBxInEvent > 0) && ((m_emulateBxInEvent % 2) == 0)) {
    m_emulateBxInEvent = m_emulateBxInEvent - 1;

    if (m_verbosity) {
      edm::LogWarning("L1TGlobalProducer")
          << "\nWARNING: Number of bunch crossing to be emulated rounded to: " << m_emulateBxInEvent
          << "\n         The number must be an odd number!\n"
          << std::endl;
    }
  }

  if ((m_L1DataBxInEvent > 0) && ((m_L1DataBxInEvent % 2) == 0)) {
    m_L1DataBxInEvent = m_L1DataBxInEvent - 1;

    if (m_verbosity) {
      edm::LogWarning("L1TGlobalProducer")
          << "\nWARNING: Number of bunch crossing for incoming L1 Data rounded to: " << m_L1DataBxInEvent
          << "\n         The number must be an odd number!\n"
          << std::endl;
    }
  } else if (m_L1DataBxInEvent < 0) {
    m_L1DataBxInEvent = 1;

    if (m_verbosity) {
      edm::LogWarning("L1TGlobalProducer")
          << "\nWARNING: Number of bunch crossing for incoming L1 Data was changed to: " << m_L1DataBxInEvent
          << "\n         The number must be an odd positive number!\n"
          << std::endl;
    }
  }

  // register products
  if (m_produceL1GtDaqRecord) {
    produces<GlobalAlgBlkBxCollection>();
    //blwEXT produces<GlobalExtBlkBxCollection>();
  }

  if (m_produceL1GtObjectMapRecord) {
    produces<GlobalObjectMapRecord>();
  }

  // create new uGt Board
  m_uGtBrd = std::make_unique<GlobalBoard>();
  m_uGtBrd->setVerbosity(m_verbosity);

  // initialize cached IDs

  //
  m_l1GtParCacheID = 0ULL;
  m_l1GtMenuCacheID = 0ULL;

  m_numberPhysTriggers = 0;
  m_numberDaqPartitions = 0;

  m_nrL1Mu = 0;
  m_nrL1EG = 0;
  m_nrL1Tau = 0;

  m_nrL1Jet = 0;

  m_ifMuEtaNumberBits = 0;
  m_ifCaloEtaNumberBits = 0;

  //
  m_l1GtParCacheID = 0ULL;

  m_totalBxInEvent = 0;

  m_activeBoardsGtDaq = 0;
  m_bstLengthBytes = 0;

  //
  m_l1GtBMCacheID = 0ULL;

  //
  m_l1GtPfAlgoCacheID = 0ULL;

  m_l1GtTmAlgoCacheID = 0ULL;

  m_l1GtTmVetoAlgoCacheID = 0ULL;

  m_currentLumi = 0;

  // Set default, initial, dummy prescale factor table
  std::vector<std::vector<int>> temp_prescaleTable;

  temp_prescaleTable.push_back(std::vector<int>());
  m_initialPrescaleFactorsAlgoTrig = temp_prescaleTable;
}

// destructor
L1TGlobalProducer::~L1TGlobalProducer() {}

// member functions

// method called to produce the data
void L1TGlobalProducer::produce(edm::Event& iEvent, const edm::EventSetup& evSetup) {
  // process event iEvent
  // get / update the stable parameters from the EventSetup
  // local cache & check on cacheIdentifier

  unsigned long long l1GtParCacheID = evSetup.get<L1TGlobalParametersRcd>().cacheIdentifier();

  if (m_l1GtParCacheID != l1GtParCacheID) {
    edm::ESHandle<L1TGlobalParameters> l1GtStablePar;
    evSetup.get<L1TGlobalParametersRcd>().get(l1GtStablePar);
    m_l1GtStablePar = l1GtStablePar.product();
    const GlobalParamsHelper* data = GlobalParamsHelper::readFromEventSetup(m_l1GtStablePar);

    // number of bx
    m_totalBxInEvent = data->totalBxInEvent();

    // number of physics triggers
    m_numberPhysTriggers = data->numberPhysTriggers();

    // number of objects of each type
    m_nrL1Mu = data->numberL1Mu();

    // EG
    m_nrL1EG = data->numberL1EG();

    // jets
    m_nrL1Jet = data->numberL1Jet();

    // taus
    m_nrL1Tau = data->numberL1Tau();

    if (m_L1DataBxInEvent < 1)
      m_L1DataBxInEvent = 1;
    int minL1DataBxInEvent = (m_L1DataBxInEvent + 1) / 2 - m_L1DataBxInEvent;
    int maxL1DataBxInEvent = (m_L1DataBxInEvent + 1) / 2 - 1;

    // Initialize Board
    m_uGtBrd->init(
        m_numberPhysTriggers, m_nrL1Mu, m_nrL1EG, m_nrL1Tau, m_nrL1Jet, minL1DataBxInEvent, maxL1DataBxInEvent);

    //
    m_l1GtParCacheID = l1GtParCacheID;
  }

  if (m_emulateBxInEvent < 0) {
    m_emulateBxInEvent = m_totalBxInEvent;
  }

  if (m_emulateBxInEvent < 1)
    m_emulateBxInEvent = 1;
  int minEmulBxInEvent = (m_emulateBxInEvent + 1) / 2 - m_emulateBxInEvent;
  int maxEmulBxInEvent = (m_emulateBxInEvent + 1) / 2 - 1;

  // get / update the trigger menu from the EventSetup
  // local cache & check on cacheIdentifier
  unsigned long long l1GtMenuCacheID = evSetup.get<L1TUtmTriggerMenuRcd>().cacheIdentifier();

  if (m_l1GtMenuCacheID != l1GtMenuCacheID) {
    const GlobalParamsHelper* data = GlobalParamsHelper::readFromEventSetup(m_l1GtStablePar);

    edm::ESHandle<L1TUtmTriggerMenu> l1GtMenu;
    evSetup.get<L1TUtmTriggerMenuRcd>().get(l1GtMenu);
    const L1TUtmTriggerMenu* utml1GtMenu = l1GtMenu.product();

    // Instantiate Parser
    TriggerMenuParser gtParser = TriggerMenuParser();

    gtParser.setGtNumberConditionChips(data->numberChips());
    gtParser.setGtPinsOnConditionChip(data->pinsOnChip());
    gtParser.setGtOrderConditionChip(data->orderOfChip());
    gtParser.setGtNumberPhysTriggers(data->numberPhysTriggers());

    //Parse menu into emulator classes
    gtParser.parseCondFormats(utml1GtMenu);

    // transfer the condition map and algorithm map from parser to L1uGtTriggerMenu
    m_l1GtMenu = std::make_unique<TriggerMenu>(gtParser.gtTriggerMenuName(),
                                               data->numberChips(),
                                               gtParser.vecMuonTemplate(),
                                               gtParser.vecCaloTemplate(),
                                               gtParser.vecEnergySumTemplate(),
                                               gtParser.vecExternalTemplate(),
                                               gtParser.vecCorrelationTemplate(),
                                               gtParser.vecCorrelationWithOverlapRemovalTemplate(),
                                               gtParser.corMuonTemplate(),
                                               gtParser.corCaloTemplate(),
                                               gtParser.corEnergySumTemplate());

    m_l1GtMenu->setGtTriggerMenuInterface(gtParser.gtTriggerMenuInterface());
    m_l1GtMenu->setGtTriggerMenuImplementation(gtParser.gtTriggerMenuImplementation());
    m_l1GtMenu->setGtScaleDbKey(gtParser.gtScaleDbKey());
    m_l1GtMenu->setGtScales(gtParser.gtScales());
    m_l1GtMenu->setGtTriggerMenuUUID(gtParser.gtTriggerMenuUUID());

    m_l1GtMenu->setGtAlgorithmMap(gtParser.gtAlgorithmMap());
    m_l1GtMenu->setGtAlgorithmAliasMap(gtParser.gtAlgorithmAliasMap());

    m_l1GtMenu->buildGtConditionMap();

    int printV = 2;
    if (m_printL1Menu)
      m_l1GtMenu->print(std::cout, printV);

    m_l1GtMenuCacheID = l1GtMenuCacheID;
  }

  // get / update the board maps from the EventSetup
  // local cache & check on cacheIdentifier

  /*   *** Drop L1GtBoard Maps for now
    typedef std::vector<L1GtBoard>::const_iterator CItBoardMaps;

    unsigned long long l1GtBMCacheID = evSetup.get<L1GtBoardMapsRcd>().cacheIdentifier();
*/

  /*  ** Drop board mapping for now
    if (m_l1GtBMCacheID != l1GtBMCacheID) {

        edm::ESHandle< L1GtBoardMaps > l1GtBM;
        evSetup.get< L1GtBoardMapsRcd >().get( l1GtBM );
        m_l1GtBM = l1GtBM.product();

        m_l1GtBMCacheID = l1GtBMCacheID;

    }
   

    // TODO need changes in CondFormats to cache the maps
    const std::vector<L1GtBoard>& boardMaps = m_l1GtBM->gtBoardMaps();
*/
  // get / update the prescale factors from the EventSetup
  // local cache & check on cacheIdentifier

  // Only get event record if not unprescaled and not unmasked
  if (!(m_algorithmTriggersUnprescaled && m_algorithmTriggersUnmasked)) {
    unsigned long long l1GtPfAlgoCacheID = evSetup.get<L1TGlobalPrescalesVetosRcd>().cacheIdentifier();

    if (m_l1GtPfAlgoCacheID != l1GtPfAlgoCacheID) {
      edm::ESHandle<L1TGlobalPrescalesVetos> l1GtPrescalesVetoes;
      evSetup.get<L1TGlobalPrescalesVetosRcd>().get(l1GtPrescalesVetoes);
      const L1TGlobalPrescalesVetos* es = l1GtPrescalesVetoes.product();
      m_l1GtPrescalesVetoes = PrescalesVetosHelper::readFromEventSetup(es);

      m_prescaleFactorsAlgoTrig = &(m_l1GtPrescalesVetoes->prescaleTable());
      m_triggerMaskVetoAlgoTrig = &(m_l1GtPrescalesVetoes->triggerMaskVeto());

      m_l1GtPfAlgoCacheID = l1GtPfAlgoCacheID;
    }
    if (m_getPrescaleColumnFromData &&
        (m_currentLumi != iEvent.luminosityBlock())) {  // get prescale column from unpacked data

      m_currentLumi = iEvent.luminosityBlock();

      edm::Handle<BXVector<GlobalAlgBlk>> m_uGtAlgBlk;
      iEvent.getByToken(m_algoblkInputToken, m_uGtAlgBlk);

      if (m_uGtAlgBlk.isValid() && !m_uGtAlgBlk->isEmpty(0)) {
        std::vector<GlobalAlgBlk>::const_iterator algBlk = m_uGtAlgBlk->begin(0);
        m_prescaleSet = static_cast<unsigned int>(algBlk->getPreScColumn());
      } else {
        m_prescaleSet = 1;
        edm::LogError("L1TGlobalProduce")
            << "Could not find valid algo block. Setting prescale column to 1" << std::endl;
      }
    }
  } else {
    // Set Prescale factors to initial dummy values
    m_prescaleSet = 0;
    m_prescaleFactorsAlgoTrig = &m_initialPrescaleFactorsAlgoTrig;
    m_triggerMaskAlgoTrig = &m_initialTriggerMaskAlgoTrig;
    m_triggerMaskVetoAlgoTrig = &m_initialTriggerMaskVetoAlgoTrig;
  }

  // get / update the trigger mask from the EventSetup
  // local cache & check on cacheIdentifier

  /*  **** For now Leave out Masks  *****
    unsigned long long l1GtTmAlgoCacheID =
        evSetup.get<L1GtTriggerMaskAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtTmAlgoCacheID != l1GtTmAlgoCacheID) {

        edm::ESHandle< L1GtTriggerMask > l1GtTmAlgo;
        evSetup.get< L1GtTriggerMaskAlgoTrigRcd >().get( l1GtTmAlgo );
        m_l1GtTmAlgo = l1GtTmAlgo.product();

        m_triggerMaskAlgoTrig = m_l1GtTmAlgo->gtTriggerMask();

        m_l1GtTmAlgoCacheID = l1GtTmAlgoCacheID;

    }
*/

  /*  **** For now Leave out Veto Masks  *****
    unsigned long long l1GtTmVetoAlgoCacheID =
        evSetup.get<L1GtTriggerMaskVetoAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtTmVetoAlgoCacheID != l1GtTmVetoAlgoCacheID) {

        edm::ESHandle< L1GtTriggerMask > l1GtTmVetoAlgo;
        evSetup.get< L1GtTriggerMaskVetoAlgoTrigRcd >().get( l1GtTmVetoAlgo );
        m_l1GtTmVetoAlgo = l1GtTmVetoAlgo.product();

        m_triggerMaskVetoAlgoTrig = m_l1GtTmVetoAlgo->gtTriggerMask();

        m_l1GtTmVetoAlgoCacheID = l1GtTmVetoAlgoCacheID;

    }
*/

  // ******  Board Maps Need to be redone....hard code for now ******
  // loop over blocks in the GT DAQ record receiving data, count them if they are active
  // all board type are defined in CondFormats/L1TObjects/L1GtFwd
  // &
  // set the active flag for each object type received from GMT and GCT
  // all objects in the GT system

  //
  bool receiveMu = true;
  bool receiveEG = true;
  bool receiveTau = true;
  bool receiveJet = true;
  bool receiveEtSums = true;
  bool receiveExt = true;

  /*  *** Boards need redefining *****
    for (CItBoardMaps
            itBoard = boardMaps.begin();
            itBoard != boardMaps.end(); ++itBoard) {

        int iPosition = itBoard->gtPositionDaqRecord();
        if (iPosition > 0) {

            int iActiveBit = itBoard->gtBitDaqActiveBoards();
            bool activeBoard = false;

            if (iActiveBit >= 0) {
                activeBoard = m_activeBoardsGtDaq & (1 << iActiveBit);
            }

            // use board if: in the record, but not in ActiveBoardsMap (iActiveBit < 0)
            //               in the record and ActiveBoardsMap, and active
            if ((iActiveBit < 0) || activeBoard) {

// ******  Decide what board manipulation (if any we want here)

            }
        }

    }
*/

  // Produce the Output Records for the GT
  std::unique_ptr<GlobalAlgBlkBxCollection> uGtAlgRecord(
      new GlobalAlgBlkBxCollection(0, minEmulBxInEvent, maxEmulBxInEvent));

  // * produce the GlobalObjectMapRecord
  std::unique_ptr<GlobalObjectMapRecord> gtObjectMapRecord(new GlobalObjectMapRecord());

  // fill the boards not depending on the BxInEvent in the L1 GT DAQ record
  // GMT, PSB and FDL depend on BxInEvent

  // fill in emulator the same bunch crossing (12 bits - hardwired number of bits...)
  // and the same local bunch crossing for all boards
  int bxCross = iEvent.bunchCrossing();
  uint16_t bxCrossHw = 0;
  if ((bxCross & 0xFFF) == bxCross) {
    bxCrossHw = static_cast<uint16_t>(bxCross);
  } else {
    bxCrossHw = 0;  // Bx number too large, set to 0!
    if (m_verbosity) {
      LogDebug("L1TGlobalProducer") << "\nBunch cross number [hex] = " << std::hex << bxCross
                                    << "\n  larger than 12 bits. Set to 0! \n"
                                    << std::dec << std::endl;
    }
  }
  LogDebug("L1TGlobalProducer") << "HW BxCross " << bxCrossHw << std::endl;

  // get the prescale factor from the configuration for now
  // prescale set index counts from zero
  unsigned int pfAlgoSetIndex = m_prescaleSet;

  auto max = (*m_prescaleFactorsAlgoTrig).size() - 1;
  if (pfAlgoSetIndex > max) {
    edm::LogWarning("L1TGlobalProducer") << "\nAttempting to access prescale algo set: " << m_prescaleSet
                                         << "\nNumber of prescale algo sets available: 0.." << max
                                         << "Setting former to latter." << std::endl;
    pfAlgoSetIndex = max;
  }

  const std::vector<int>& prescaleFactorsAlgoTrig = (*m_prescaleFactorsAlgoTrig).at(pfAlgoSetIndex);

  // For now, set masks according to prescale value of 0
  m_initialTriggerMaskAlgoTrig.clear();
  for (unsigned int iAlgo = 0; iAlgo < prescaleFactorsAlgoTrig.size(); iAlgo++) {
    unsigned int value = prescaleFactorsAlgoTrig[iAlgo];
    value = (value == 0) ? 0 : 1;
    m_initialTriggerMaskAlgoTrig.push_back(value);
  }
  m_triggerMaskAlgoTrig = &m_initialTriggerMaskAlgoTrig;

  const std::vector<unsigned int>& triggerMaskAlgoTrig = *m_triggerMaskAlgoTrig;
  const std::vector<int>& triggerMaskVetoAlgoTrig = *m_triggerMaskVetoAlgoTrig;

  LogDebug("L1TGlobalProducer") << "Size of prescale vector" << prescaleFactorsAlgoTrig.size() << std::endl;

  // Load the calorimeter input onto the uGt Board
  m_uGtBrd->receiveCaloObjectData(iEvent,
                                  m_egInputToken,
                                  m_tauInputToken,
                                  m_jetInputToken,
                                  m_sumInputToken,
                                  receiveEG,
                                  m_nrL1EG,
                                  receiveTau,
                                  m_nrL1Tau,
                                  receiveJet,
                                  m_nrL1Jet,
                                  receiveEtSums);

  m_uGtBrd->receiveMuonObjectData(iEvent, m_muInputToken, receiveMu, m_nrL1Mu);

  m_uGtBrd->receiveExternalData(iEvent, m_extInputToken, receiveExt);

  // loop over BxInEvent
  for (int iBxInEvent = minEmulBxInEvent; iBxInEvent <= maxEmulBxInEvent; ++iBxInEvent) {
    //  run GTL
    LogDebug("L1TGlobalProducer") << "\nL1TGlobalProducer : running GTL  for bx = " << iBxInEvent << "\n" << std::endl;

    //  Run the GTL for this BX
    m_uGtBrd->runGTL(iEvent,
                     evSetup,
                     m_l1GtMenu.get(),
                     m_produceL1GtObjectMapRecord,
                     iBxInEvent,
                     gtObjectMapRecord,
                     m_numberPhysTriggers,
                     m_nrL1Mu,
                     m_nrL1EG,
                     m_nrL1Tau,
                     m_nrL1Jet);

    //  run FDL
    LogDebug("L1TGlobalProducer") << "\nL1TGlobalProducer : running FDL for bx = " << iBxInEvent << "\n" << std::endl;

    //  Run the Final Decision Logic for this BX
    m_uGtBrd->runFDL(iEvent,
                     iBxInEvent,
                     m_totalBxInEvent,
                     m_numberPhysTriggers,
                     prescaleFactorsAlgoTrig,
                     triggerMaskAlgoTrig,
                     triggerMaskVetoAlgoTrig,
                     m_algorithmTriggersUnprescaled,
                     m_algorithmTriggersUnmasked);

    // Fill in the DAQ Records
    if (m_produceL1GtDaqRecord) {
      m_uGtBrd->fillAlgRecord(iBxInEvent,
                              uGtAlgRecord,
                              m_prescaleSet,
                              m_l1GtMenu->gtTriggerMenuUUID(),
                              m_l1GtMenu->gtTriggerMenuImplementation());
    }

  }  //End Loop over Bx

  // Add explicit reset of Board
  m_uGtBrd->reset();

  if (m_verbosity && m_isDebugEnabled) {
    std::ostringstream myCoutStream;

    for (int bx = minEmulBxInEvent; bx < maxEmulBxInEvent; bx++) {
      /// Needs error checking that something exists at this bx.
      (uGtAlgRecord->at(bx, 0)).print(myCoutStream);
    }

    LogTrace("L1TGlobalProducer") << "\n The following L1 GT DAQ readout record was produced:\n"
                                  << myCoutStream.str() << "\n"
                                  << std::endl;

    myCoutStream.str("");
    myCoutStream.clear();

    const std::vector<GlobalObjectMap> objMapVec = gtObjectMapRecord->gtObjectMap();

    for (std::vector<GlobalObjectMap>::const_iterator it = objMapVec.begin(); it != objMapVec.end(); ++it) {
      (*it).print(myCoutStream);
    }

    LogDebug("L1TGlobalProducer") << "Test gtObjectMapRecord in L1TGlobalProducer \n\n"
                                  << myCoutStream.str() << "\n\n"
                                  << std::endl;

    myCoutStream.str("");
    myCoutStream.clear();
  }

  // register products
  if (m_produceL1GtDaqRecord) {
    iEvent.put(std::move(uGtAlgRecord));
  }

  if (m_produceL1GtObjectMapRecord) {
    iEvent.put(std::move(gtObjectMapRecord));
  }
}

//define this as a plug-in
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <cstdint>
DEFINE_FWK_MODULE(L1TGlobalProducer);
