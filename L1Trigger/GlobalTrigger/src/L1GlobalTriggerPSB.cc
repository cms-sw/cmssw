/**
 * \class L1GlobalTriggerPSB
 *
 *
 * Description: Pipelined Synchronising Buffer, see header file for details.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: M. Fierro            - HEPHY Vienna - ORCA version
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version
 *
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

// system include files
#include <bitset>
#include <iomanip>
#include <iostream>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHFBitCounts.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHFRingEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

// forward declarations

// constructor
L1GlobalTriggerPSB::L1GlobalTriggerPSB(const edm::InputTag &m_caloGctInputTag,
                                       const std::vector<edm::InputTag> &m_technicalTriggersInputTags,
                                       edm::ConsumesCollector &&iC)
    : m_candL1NoIsoEG(new std::vector<const L1GctCand *>),
      m_candL1IsoEG(new std::vector<const L1GctCand *>),
      m_candL1CenJet(new std::vector<const L1GctCand *>),
      m_candL1ForJet(new std::vector<const L1GctCand *>),
      m_candL1TauJet(new std::vector<const L1GctCand *>),
      m_candETM(nullptr),
      m_candETT(nullptr),
      m_candHTT(nullptr),
      m_candHTM(nullptr),
      m_candJetCounts(nullptr),
      m_candHfBitCounts(nullptr),
      m_candHfRingEtSums(nullptr),
      m_isDebugEnabled(edm::isDebugEnabled())

{
  iC.consumes<L1GctEmCandCollection>(edm::InputTag(m_caloGctInputTag.label(), "nonIsoEm", ""));
  iC.consumes<L1GctEmCandCollection>(edm::InputTag(m_caloGctInputTag.label(), "isoEm", ""));
  iC.consumes<L1GctJetCandCollection>(edm::InputTag(m_caloGctInputTag.label(), "cenJets", ""));
  iC.consumes<L1GctJetCandCollection>(edm::InputTag(m_caloGctInputTag.label(), "forJets", ""));
  iC.consumes<L1GctJetCandCollection>(edm::InputTag(m_caloGctInputTag.label(), "tauJets", ""));
  iC.consumes<L1GctEtMissCollection>(m_caloGctInputTag);
  iC.consumes<L1GctEtTotalCollection>(m_caloGctInputTag);
  iC.consumes<L1GctEtHadCollection>(m_caloGctInputTag);
  iC.consumes<L1GctHtMissCollection>(m_caloGctInputTag);
  iC.consumes<L1GctJetCountsCollection>(m_caloGctInputTag);
  iC.consumes<L1GctHFBitCountsCollection>(m_caloGctInputTag);
  iC.consumes<L1GctHFRingEtSumsCollection>(m_caloGctInputTag);

  for (std::vector<edm::InputTag>::const_iterator it = m_technicalTriggersInputTags.begin();
       it != m_technicalTriggersInputTags.end();
       it++) {
    iC.consumes<L1GtTechnicalTriggerRecord>((*it));
  }
  // empty
}

// destructor
L1GlobalTriggerPSB::~L1GlobalTriggerPSB() {
  reset();

  delete m_candL1NoIsoEG;
  delete m_candL1IsoEG;
  delete m_candL1CenJet;
  delete m_candL1ForJet;
  delete m_candL1TauJet;
}

// operations
void L1GlobalTriggerPSB::init(const int nrL1NoIsoEG,
                              const int nrL1IsoEG,
                              const int nrL1CenJet,
                              const int nrL1ForJet,
                              const int nrL1TauJet,
                              const int numberTechnicalTriggers) {
  m_candL1NoIsoEG->reserve(nrL1NoIsoEG);
  m_candL1IsoEG->reserve(nrL1IsoEG);
  m_candL1CenJet->reserve(nrL1CenJet);
  m_candL1ForJet->reserve(nrL1ForJet);
  m_candL1TauJet->reserve(nrL1TauJet);

  m_gtTechnicalTriggers.reserve(numberTechnicalTriggers);
  m_gtTechnicalTriggers.assign(numberTechnicalTriggers, false);
}

// receive input data

void L1GlobalTriggerPSB::receiveGctObjectData(edm::Event &iEvent,
                                              const edm::InputTag &caloGctInputTag,
                                              const int iBxInEvent,
                                              const bool receiveNoIsoEG,
                                              const int nrL1NoIsoEG,
                                              const bool receiveIsoEG,
                                              const int nrL1IsoEG,
                                              const bool receiveCenJet,
                                              const int nrL1CenJet,
                                              const bool receiveForJet,
                                              const int nrL1ForJet,
                                              const bool receiveTauJet,
                                              const int nrL1TauJet,
                                              const bool receiveETM,
                                              const bool receiveETT,
                                              const bool receiveHTT,
                                              const bool receiveHTM,
                                              const bool receiveJetCounts,
                                              const bool receiveHfBitCounts,
                                              const bool receiveHfRingEtSums) {
  // LogDebug("L1GlobalTrigger")
  //        << "\n**** L1GlobalTriggerPSB receiving calorimeter data for
  //        BxInEvent = "
  //        << iBxInEvent << "\n     from " << caloGctInputTag << "\n"
  //        << std::endl;

  reset();

  std::ostringstream warningsStream;
  bool warningEnabled = edm::isWarningEnabled();

  if (receiveNoIsoEG) {
    // get GCT NoIsoEG
    edm::Handle<L1GctEmCandCollection> emCands;
    iEvent.getByLabel(caloGctInputTag.label(), "nonIsoEm", emCands);

    if (!emCands.isValid()) {
      if (warningEnabled) {
        warningsStream << "\nWarning: L1GctEmCandCollection with input label " << caloGctInputTag.label()
                       << " and instance \"nonIsoEm\" \n"
                       << "requested in configuration, but not found in the event.\n"
                       << std::endl;
      }
    } else {
      for (L1GctEmCandCollection::const_iterator it = emCands->begin(); it != emCands->end(); it++) {
        if ((*it).bx() == iBxInEvent) {
          (*m_candL1NoIsoEG).push_back(&(*it));
          // LogTrace("L1GlobalTrigger") << "NoIsoEG:  " << (*it) << std::endl;
        }
      }
    }
  }

  if (receiveIsoEG) {
    // get GCT IsoEG
    edm::Handle<L1GctEmCandCollection> isoEmCands;
    iEvent.getByLabel(caloGctInputTag.label(), "isoEm", isoEmCands);

    if (!isoEmCands.isValid()) {
      if (warningEnabled) {
        warningsStream << "\nWarning: L1GctEmCandCollection with input label " << caloGctInputTag.label()
                       << " and instance \"isoEm\" \n"
                       << "requested in configuration, but not found in the event.\n"
                       << std::endl;
      }
    } else {
      for (L1GctEmCandCollection::const_iterator it = isoEmCands->begin(); it != isoEmCands->end(); it++) {
        if ((*it).bx() == iBxInEvent) {
          (*m_candL1IsoEG).push_back(&(*it));
          // LogTrace("L1GlobalTrigger") << "IsoEG:    " <<  (*it) << std::endl;
        }
      }
    }
  }

  if (receiveCenJet) {
    // get GCT CenJet
    edm::Handle<L1GctJetCandCollection> cenJets;
    iEvent.getByLabel(caloGctInputTag.label(), "cenJets", cenJets);

    if (!cenJets.isValid()) {
      if (warningEnabled) {
        warningsStream << "\nWarning: L1GctJetCandCollection with input label " << caloGctInputTag.label()
                       << " and instance \"cenJets\" \n"
                       << "requested in configuration, but not found in the event.\n"
                       << std::endl;
      }
    } else {
      for (L1GctJetCandCollection::const_iterator it = cenJets->begin(); it != cenJets->end(); it++) {
        if ((*it).bx() == iBxInEvent) {
          (*m_candL1CenJet).push_back(&(*it));
          // LogTrace("L1GlobalTrigger") << "CenJet    " <<  (*it) << std::endl;
        }
      }
    }
  }

  if (receiveForJet) {
    // get GCT ForJet
    edm::Handle<L1GctJetCandCollection> forJets;
    iEvent.getByLabel(caloGctInputTag.label(), "forJets", forJets);

    if (!forJets.isValid()) {
      if (warningEnabled) {
        warningsStream << "\nWarning: L1GctJetCandCollection with input label " << caloGctInputTag.label()
                       << " and instance \"forJets\" \n"
                       << "requested in configuration, but not found in the event.\n"
                       << std::endl;
      }
    } else {
      for (L1GctJetCandCollection::const_iterator it = forJets->begin(); it != forJets->end(); it++) {
        if ((*it).bx() == iBxInEvent) {
          (*m_candL1ForJet).push_back(&(*it));
          // LogTrace("L1GlobalTrigger") << "ForJet    " <<  (*it) << std::endl;
        }
      }
    }
  }

  if (receiveTauJet) {
    // get GCT TauJet
    edm::Handle<L1GctJetCandCollection> tauJets;
    iEvent.getByLabel(caloGctInputTag.label(), "tauJets", tauJets);

    if (!tauJets.isValid()) {
      if (warningEnabled) {
        warningsStream << "\nWarning: L1GctJetCandCollection with input label " << caloGctInputTag.label()
                       << " and instance \"tauJets\" \n"
                       << "requested in configuration, but not found in the event.\n"
                       << std::endl;
      }
    } else {
      for (L1GctJetCandCollection::const_iterator it = tauJets->begin(); it != tauJets->end(); it++) {
        if ((*it).bx() == iBxInEvent) {
          (*m_candL1TauJet).push_back(&(*it));
          // LogTrace("L1GlobalTrigger") << "TauJet    " <<  (*it) << std::endl;
        }
      }
    }
  }

  // get GCT ETM
  if (receiveETM) {
    edm::Handle<L1GctEtMissCollection> missEtColl;
    iEvent.getByLabel(caloGctInputTag, missEtColl);

    if (!missEtColl.isValid()) {
      if (warningEnabled) {
        warningsStream << "\nWarning: L1GctEtMissCollection with input tag " << caloGctInputTag
                       << "\nrequested in configuration, but not found in the event.\n"
                       << std::endl;
      }
    } else {
      for (L1GctEtMissCollection::const_iterator it = missEtColl->begin(); it != missEtColl->end(); it++) {
        if ((*it).bx() == iBxInEvent) {
          m_candETM = &(*it);
          // LogTrace("L1GlobalTrigger") << "ETM      " << (*it) << std::endl;
        }
      }
    }
  }

  // get GCT ETT
  if (receiveETT) {
    edm::Handle<L1GctEtTotalCollection> sumEtColl;
    iEvent.getByLabel(caloGctInputTag, sumEtColl);

    if (!sumEtColl.isValid()) {
      if (warningEnabled) {
        warningsStream << "\nWarning: L1GctEtTotalCollection with input tag " << caloGctInputTag
                       << "\nrequested in configuration, but not found in the event.\n"
                       << std::endl;
      }
    } else {
      for (L1GctEtTotalCollection::const_iterator it = sumEtColl->begin(); it != sumEtColl->end(); it++) {
        if ((*it).bx() == iBxInEvent) {
          m_candETT = &(*it);
          // LogTrace("L1GlobalTrigger") << "ETT      " << (*it) << std::endl;
        }
      }
    }
  }

  // get GCT HTT
  if (receiveHTT) {
    edm::Handle<L1GctEtHadCollection> sumHtColl;
    iEvent.getByLabel(caloGctInputTag, sumHtColl);

    if (!sumHtColl.isValid()) {
      if (warningEnabled) {
        warningsStream << "\nWarning: L1GctEtHadCollection with input tag " << caloGctInputTag
                       << "\nrequested in configuration, but not found in the event.\n"
                       << std::endl;
      }
    } else {
      for (L1GctEtHadCollection::const_iterator it = sumHtColl->begin(); it != sumHtColl->end(); it++) {
        if ((*it).bx() == iBxInEvent) {
          m_candHTT = &(*it);
          // LogTrace("L1GlobalTrigger") << "HTT      "  << (*it) << std::endl;
        }
      }
    }
  }

  // get GCT HTM
  if (receiveHTM) {
    edm::Handle<L1GctHtMissCollection> missHtColl;
    iEvent.getByLabel(caloGctInputTag, missHtColl);

    if (!missHtColl.isValid()) {
      if (warningEnabled) {
        warningsStream << "\nWarning: L1GctHtMissCollection with input tag " << caloGctInputTag
                       << "\nrequested in configuration, but not found in the event.\n"
                       << std::endl;
      }
    } else {
      for (L1GctHtMissCollection::const_iterator it = missHtColl->begin(); it != missHtColl->end(); it++) {
        if ((*it).bx() == iBxInEvent) {
          m_candHTM = &(*it);
          // LogTrace("L1GlobalTrigger") << "HTM      " << (*it) << std::endl;
        }
      }
    }
  }

  // get GCT JetCounts
  if (receiveJetCounts) {
    edm::Handle<L1GctJetCountsCollection> jetCountColl;
    iEvent.getByLabel(caloGctInputTag, jetCountColl);

    if (!jetCountColl.isValid()) {
      if (warningEnabled) {
        warningsStream << "\nWarning: L1GctJetCountsCollection with input tag " << caloGctInputTag
                       << "\nrequested in configuration, but not found in the event.\n"
                       << std::endl;
      }
    } else {
      for (L1GctJetCountsCollection::const_iterator it = jetCountColl->begin(); it != jetCountColl->end(); it++) {
        if ((*it).bx() == iBxInEvent) {
          m_candJetCounts = &(*it);
          // LogTrace("L1GlobalTrigger") << (*it) << std::endl;
        }
      }
    }
  }

  // get GCT HfBitCounts
  if (receiveHfBitCounts) {
    edm::Handle<L1GctHFBitCountsCollection> hfBitCountsColl;
    iEvent.getByLabel(caloGctInputTag, hfBitCountsColl);

    if (!hfBitCountsColl.isValid()) {
      if (warningEnabled) {
        warningsStream << "\nWarning: L1GctHFBitCountsCollection with input tag " << caloGctInputTag
                       << "\nrequested in configuration, but not found in the event.\n"
                       << std::endl;
      }
    } else {
      for (L1GctHFBitCountsCollection::const_iterator it = hfBitCountsColl->begin(); it != hfBitCountsColl->end();
           it++) {
        if ((*it).bx() == iBxInEvent) {
          m_candHfBitCounts = &(*it);
          // LogTrace("L1GlobalTrigger") << "L1GctHFBitCountsCollection: "
          //<< (*it) << std::endl;
        }
      }
    }
  }

  // get GCT HfRingEtSums
  if (receiveHfRingEtSums) {
    edm::Handle<L1GctHFRingEtSumsCollection> hfRingEtSums;
    iEvent.getByLabel(caloGctInputTag, hfRingEtSums);

    if (!hfRingEtSums.isValid()) {
      if (warningEnabled) {
        warningsStream << "\nWarning: L1GctHFRingEtSumsCollection with input tag " << caloGctInputTag
                       << "\nrequested in configuration, but not found in the event.\n"
                       << std::endl;
      }
    } else {
      for (L1GctHFRingEtSumsCollection::const_iterator it = hfRingEtSums->begin(); it != hfRingEtSums->end(); it++) {
        if ((*it).bx() == iBxInEvent) {
          m_candHfRingEtSums = &(*it);
          // LogTrace("L1GlobalTrigger") << "L1GctHFRingEtSumsCollection: "
          //<< (*it) << std::endl;
        }
      }
    }
  }

  if (m_verbosity && warningEnabled) {
    if (warningsStream.tellp() > 0) {
      edm::LogWarning("L1GlobalTrigger") << warningsStream.str();
    }
  }

  if (m_verbosity && m_isDebugEnabled) {
    LogDebug("L1GlobalTrigger") << "\n**** L1GlobalTriggerPSB receiving calorimeter data for BxInEvent "
                                   "= "
                                << iBxInEvent << "\n     from " << caloGctInputTag << "\n"
                                << std::endl;

    printGctObjectData(iBxInEvent);
  }
}

// receive CASTOR objects
void L1GlobalTriggerPSB::receiveCastorData(edm::Event &iEvent,
                                           const edm::InputTag &castorInputTag,
                                           const int iBxInEvent,
                                           const bool receiveCastor,
                                           const bool readFromPsb) {
  // get the CASTOR record

  // bool castorConditionFlag = true;
  // FIXME remove the following line and uncomment the line above
  //       when the L1CastorRecord is available
  //       until then, all CASTOR conditions are set to false
  // bool castorConditionFlag = false;

  // edm::Handle<L1CastorRecord > castorData;
  // iEvent.getByLabel(castorInputTag, castorData);

  // if (receiveCastor) {
  //
  //    if (!castorData.isValid()) {
  //        edm::LogWarning("L1GlobalTrigger")
  //        << "\nWarning: CASTOR record with input tag " << castorInputTag
  //        << "\nrequested in configuration, but not found in the event.\n"
  //        << std::endl;
  //
  //        castorConditionFlag = false;
  //    } else {
  //            LogTrace("L1GlobalTrigger") << *(castorData.product()) <<
  //            std::endl;
  //
  //    }
  //
  //} else {
  //
  //    // channel for CASTOR blocked - set all CASTOR to false
  //    // MUST NEVER BLOCK CASTOR CHANNEL AND USE OPERATOR "NOT" WITH CASTOR
  //    CONDITION
  //    //     ==> FALSE RESULTS!
  //    castorConditionResult = false;
  //
  //}
}

// receive BPTX objects
//   from a GT record with bptxInputTag - if readFromPsb is true
//   otherwise, generate them from randomly
void L1GlobalTriggerPSB::receiveBptxData(edm::Event &iEvent,
                                         const edm::InputTag &bptxInputTag,
                                         const int iBxInEvent,
                                         const bool receiveBptx,
                                         const bool readFromPsb) {}

// receive External objects
//   from a GT record with ExternalInputTag - if readFromPsb is true
//   otherwise, generate them from randomly
void L1GlobalTriggerPSB::receiveExternalData(edm::Event &iEvent,
                                             const std::vector<edm::InputTag> &externalInputTags,
                                             const int iBxInEvent,
                                             const bool receiveExternal,
                                             const bool readFromPsb) {}

// receive technical triggers
// each L1GtTechnicalTriggerRecord can have more than one technical trigger bit,
// such that a single producer per system can be used (if desired)
void L1GlobalTriggerPSB::receiveTechnicalTriggers(edm::Event &iEvent,
                                                  const std::vector<edm::InputTag> &technicalTriggersInputTags,
                                                  const int iBxInEvent,
                                                  const bool receiveTechTr,
                                                  const int nrL1TechTr) {
  std::ostringstream warningsStream;
  bool warningEnabled = edm::isWarningEnabled();

  // reset the technical trigger bits
  m_gtTechnicalTriggers = std::vector<bool>(nrL1TechTr, false);

  if (receiveTechTr) {
    // get the technical trigger bits from the records and write them in
    // the decision word for technical triggers

    // loop over all producers of technical trigger records
    for (std::vector<edm::InputTag>::const_iterator it = technicalTriggersInputTags.begin();
         it != technicalTriggersInputTags.end();
         it++) {
      edm::Handle<L1GtTechnicalTriggerRecord> techTrigRecord;
      iEvent.getByLabel((*it), techTrigRecord);

      if (!techTrigRecord.isValid()) {
        if (warningEnabled) {
          warningsStream << "\nWarning: L1GtTechnicalTriggerRecord with input tag " << (*it)
                         << "\nrequested in configuration, but not found in the event.\n"
                         << std::endl;
        }
      } else {
        const std::vector<L1GtTechnicalTrigger> &ttVec = techTrigRecord->gtTechnicalTrigger();
        size_t ttVecSize = ttVec.size();

        for (size_t iTT = 0; iTT < ttVecSize; ++iTT) {
          const L1GtTechnicalTrigger &ttBxRecord = ttVec[iTT];
          int ttBxInEvent = ttBxRecord.bxInEvent();

          if (ttBxInEvent == iBxInEvent) {
            int ttBitNumber = ttBxRecord.gtTechnicalTriggerBitNumber();
            bool ttResult = ttBxRecord.gtTechnicalTriggerResult();

            m_gtTechnicalTriggers.at(ttBitNumber) = ttResult;

            if (m_verbosity) {
              LogTrace("L1GlobalTrigger") << "Add for BxInEvent " << iBxInEvent << " the technical trigger produced by "
                                          << (*it) << " : name " << (ttBxRecord.gtTechnicalTriggerName())
                                          << " , bit number " << ttBitNumber << " and result " << ttResult << std::endl;
            }
          }
        }
      }
    }
  }

  if (m_verbosity && warningEnabled) {
    if (warningsStream.tellp() > 0) {
      edm::LogWarning("L1GlobalTrigger") << warningsStream.str();
    }
  }

  if (m_verbosity && m_isDebugEnabled) {
    LogDebug("L1GlobalTrigger") << "\n**** L1GlobalTriggerPSB receiving technical triggers: " << std::endl;

    int sizeW64 = 64;  // 64 bits words
    int iBit = 0;

    std::ostringstream myCout;

    for (std::vector<bool>::reverse_iterator ritBit = m_gtTechnicalTriggers.rbegin();
         ritBit != m_gtTechnicalTriggers.rend();
         ++ritBit) {
      myCout << (*ritBit ? '1' : '0');

      if ((((iBit + 1) % 16) == (sizeW64 % 16)) && (iBit != 63)) {
        myCout << " ";
      }

      iBit++;
    }

    LogTrace("L1GlobalTrigger") << myCout.str() << "\n" << std::endl;
  }
}

// fill the content of active PSB boards
void L1GlobalTriggerPSB::fillPsbBlock(edm::Event &iEvent,
                                      const uint16_t &activeBoardsGtDaq,
                                      const int recordLength0,
                                      const int recordLength1,
                                      const unsigned int altNrBxBoardDaq,
                                      const std::vector<L1GtBoard> &boardMaps,
                                      const int iBxInEvent,
                                      L1GlobalTriggerReadoutRecord *gtDaqReadoutRecord) {
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

  typedef std::vector<L1GtBoard>::const_iterator CItBoardMaps;

  // loop over PSB blocks in the GT DAQ record and fill them
  // with the content of the object list

  for (CItBoardMaps itBoard = boardMaps.begin(); itBoard != boardMaps.end(); ++itBoard) {
    int iPosition = itBoard->gtPositionDaqRecord();
    if (iPosition > 0) {
      int iActiveBit = itBoard->gtBitDaqActiveBoards();
      bool activeBoard = false;
      bool writeBoard = false;

      int recLength = -1;

      if (iActiveBit >= 0) {
        activeBoard = activeBoardsGtDaq & (1 << iActiveBit);

        int altNrBxBoard = (altNrBxBoardDaq & (1 << iActiveBit)) >> iActiveBit;

        if (altNrBxBoard == 1) {
          recLength = recordLength1;
        } else {
          recLength = recordLength0;
        }

        int lowBxInEvent = (recLength + 1) / 2 - recLength;
        int uppBxInEvent = (recLength + 1) / 2 - 1;

        if ((iBxInEvent >= lowBxInEvent) && (iBxInEvent <= uppBxInEvent)) {
          writeBoard = true;
        }

        // LogTrace("L1GlobalTrigger")
        //    << "\nBoard " << std::hex << (itBoard->gtBoardId()) << std::dec
        //    << "\naltNrBxBoard = " << altNrBxBoard << " recLength " <<
        //    recLength
        //    << " lowBxInEvent " << lowBxInEvent
        //    << " uppBxInEvent " << uppBxInEvent
        //    << std::endl;
      }

      // LogTrace("L1GlobalTrigger")
      //    << "\nBoard " << std::hex << (itBoard->gtBoardId()) << std::dec
      //    << "\niBxInEvent = " << iBxInEvent << " iActiveBit " << iActiveBit
      //    << " activeBoard " << activeBoard
      //    << " writeBoard " << writeBoard
      //    << std::endl;

      if (activeBoard && writeBoard && (itBoard->gtBoardType() == PSB)) {
        L1GtPsbWord psbWordValue;

        // set board ID
        psbWordValue.setBoardId(itBoard->gtBoardId());

        // set bunch cross in the GT event record
        psbWordValue.setBxInEvent(iBxInEvent);

        // set bunch cross number of the actual bx
        uint16_t bxNrValue = bxCrossHw;
        psbWordValue.setBxNr(bxNrValue);

        // set event number since last L1 reset generated in PSB
        psbWordValue.setEventNr(static_cast<uint32_t>(iEvent.id().event()));

        // set local bunch cross number of the actual bx
        // set identical to bxCrossHw - other solution?
        uint16_t localBxNrValue = bxCrossHw;
        psbWordValue.setLocalBxNr(localBxNrValue);

        // get the objects coming to this PSB and the quadruplet index

        // two objects writen one after another from the same quadruplet
        int nrObjRow = 2;

        std::vector<L1GtPsbQuad> quadInPsb = itBoard->gtQuadInPsb();
        int nrCables = quadInPsb.size();

        uint16_t aDataVal = 0;
        uint16_t bDataVal = 0;

        int iCable = -1;
        for (std::vector<L1GtPsbQuad>::const_iterator itQuad = quadInPsb.begin(); itQuad != quadInPsb.end(); ++itQuad) {
          iCable++;

          int iAB = (nrCables - iCable - 1) * nrObjRow;

          switch (*itQuad) {
            case TechTr: {
              // LogTrace("L1GlobalTrigger")
              //<< "\nL1GlobalTriggerPSB: write TechTr for BxInEvent = "
              //<< iBxInEvent
              //<< "\n PSB " << std::hex << itBoard->gtBoardId() << std::dec
              //<< " Cable " << iCable << " Quad " << (*itQuad)
              //<< std::endl;

              // order: 16-bit words
              int bitsPerWord = 16;

              //
              int iPair = 0;
              aDataVal = 0;

              int iBit = 0;
              uint16_t bitVal = 0;

              for (int i = 0; i < bitsPerWord; ++i) {
                if (m_gtTechnicalTriggers[iBit]) {
                  bitVal = 1;
                } else {
                  bitVal = 0;
                }

                aDataVal = aDataVal | (bitVal << i);
                iBit++;
              }
              psbWordValue.setAData(aDataVal, iAB + iPair);

              //
              bDataVal = 0;

              for (int i = 0; i < bitsPerWord; ++i) {
                if (m_gtTechnicalTriggers[iBit]) {
                  bitVal = 1;
                } else {
                  bitVal = 0;
                }

                bDataVal = bDataVal | (bitVal << i);
                iBit++;
              }
              psbWordValue.setBData(bDataVal, iAB + iPair);

              //
              iPair = 1;
              aDataVal = 0;

              for (int i = 0; i < bitsPerWord; ++i) {
                if (m_gtTechnicalTriggers[iBit]) {
                  bitVal = 1;
                } else {
                  bitVal = 0;
                }

                aDataVal = aDataVal | (bitVal << i);
                iBit++;
              }
              psbWordValue.setAData(aDataVal, iAB + iPair);

              bDataVal = 0;

              for (int i = 0; i < bitsPerWord; ++i) {
                if (m_gtTechnicalTriggers[iBit]) {
                  bitVal = 1;
                } else {
                  bitVal = 0;
                }

                bDataVal = bDataVal | (bitVal << i);
                iBit++;
              }
              psbWordValue.setBData(bDataVal, iAB + iPair);
            }

            break;
            case NoIsoEGQ: {
              // LogTrace("L1GlobalTrigger")
              //<< "\nL1GlobalTriggerPSB: write NoIsoEGQ for BxInEvent = "
              //<< iBxInEvent
              //<< "\n PSB " << std::hex << itBoard->gtBoardId() << std::dec
              //<< " Cable " << iCable << " Quad " << (*itQuad)
              //<< std::endl;

              int recL1NoIsoEG = m_candL1NoIsoEG->size();
              for (int iPair = 0; iPair < nrObjRow; ++iPair) {
                if (iPair < recL1NoIsoEG) {
                  aDataVal = (static_cast<const L1GctEmCand *>((*m_candL1NoIsoEG)[iPair]))->raw();
                } else {
                  aDataVal = 0;
                }
                psbWordValue.setAData(aDataVal, iAB + iPair);

                if ((iPair + nrObjRow) < recL1NoIsoEG) {
                  bDataVal = (static_cast<const L1GctEmCand *>((*m_candL1NoIsoEG)[iPair + nrObjRow]))->raw();
                } else {
                  bDataVal = 0;
                }
                psbWordValue.setBData(bDataVal, iAB + iPair);
              }
            }

            break;
            case IsoEGQ: {
              // LogTrace("L1GlobalTrigger")
              //<< "\nL1GlobalTriggerPSB: write IsoEGQ for BxInEvent = "
              //<< iBxInEvent
              //<< "\n PSB " << std::hex << itBoard->gtBoardId() << std::dec
              //<< " Cable " << iCable << " Quad " << (*itQuad)
              //<< std::endl;

              int recL1IsoEG = m_candL1IsoEG->size();
              for (int iPair = 0; iPair < nrObjRow; ++iPair) {
                if (iPair < recL1IsoEG) {
                  aDataVal = (static_cast<const L1GctEmCand *>((*m_candL1IsoEG)[iPair]))->raw();
                } else {
                  aDataVal = 0;
                }
                psbWordValue.setAData(aDataVal, iAB + iPair);

                if ((iPair + nrObjRow) < recL1IsoEG) {
                  bDataVal = (static_cast<const L1GctEmCand *>((*m_candL1IsoEG)[iPair + nrObjRow]))->raw();
                } else {
                  bDataVal = 0;
                }
                psbWordValue.setBData(bDataVal, iAB + iPair);
              }

            }

            break;
            case CenJetQ: {
              // LogTrace("L1GlobalTrigger")
              //<< "\nL1GlobalTriggerPSB: write CenJetQ for BxInEvent = "
              //<< iBxInEvent
              //<< "\n PSB " << std::hex << itBoard->gtBoardId() << std::dec
              //<< " Cable " << iCable << " Quad " << (*itQuad)
              //<< std::endl;

              int recL1CenJet = m_candL1CenJet->size();
              for (int iPair = 0; iPair < nrObjRow; ++iPair) {
                if (iPair < recL1CenJet) {
                  aDataVal = (static_cast<const L1GctJetCand *>((*m_candL1CenJet)[iPair]))->raw();
                } else {
                  aDataVal = 0;
                }
                psbWordValue.setAData(aDataVal, iAB + iPair);

                if ((iPair + nrObjRow) < recL1CenJet) {
                  bDataVal = (static_cast<const L1GctJetCand *>((*m_candL1CenJet)[iPair + nrObjRow]))->raw();
                } else {
                  bDataVal = 0;
                }
                psbWordValue.setBData(bDataVal, iAB + iPair);
              }
            }

            break;
            case ForJetQ: {
              // LogTrace("L1GlobalTrigger")
              //<< "\nL1GlobalTriggerPSB: write ForJetQ for BxInEvent = "
              //<< iBxInEvent
              //<< "\n PSB " << std::hex << itBoard->gtBoardId() << std::dec
              //<< " Cable " << iCable << " Quad " << (*itQuad)
              //<< std::endl;

              int recL1ForJet = m_candL1ForJet->size();
              for (int iPair = 0; iPair < nrObjRow; ++iPair) {
                if (iPair < recL1ForJet) {
                  aDataVal = (static_cast<const L1GctJetCand *>((*m_candL1ForJet)[iPair]))->raw();
                } else {
                  aDataVal = 0;
                }
                psbWordValue.setAData(aDataVal, iAB + iPair);

                if ((iPair + nrObjRow) < recL1ForJet) {
                  bDataVal = (static_cast<const L1GctJetCand *>((*m_candL1ForJet)[iPair + nrObjRow]))->raw();
                } else {
                  bDataVal = 0;
                }
                psbWordValue.setBData(bDataVal, iAB + iPair);
              }

            }

            break;
            case TauJetQ: {
              // LogTrace("L1GlobalTrigger")
              //<< "\nL1GlobalTriggerPSB: write TauJetQ for BxInEvent = "
              //<< iBxInEvent
              //<< "\n PSB " << std::hex << itBoard->gtBoardId() << std::dec
              //<< " Cable " << iCable << " Quad " << (*itQuad)
              //<< std::endl;

              int recL1TauJet = m_candL1TauJet->size();
              for (int iPair = 0; iPair < nrObjRow; ++iPair) {
                if (iPair < recL1TauJet) {
                  aDataVal = (static_cast<const L1GctJetCand *>((*m_candL1TauJet)[iPair]))->raw();
                } else {
                  aDataVal = 0;
                }
                psbWordValue.setAData(aDataVal, iAB + iPair);

                if ((iPair + nrObjRow) < recL1TauJet) {
                  bDataVal = (static_cast<const L1GctJetCand *>((*m_candL1TauJet)[iPair + nrObjRow]))->raw();
                } else {
                  bDataVal = 0;
                }
                psbWordValue.setBData(bDataVal, iAB + iPair);

                // LogTrace("L1GlobalTrigger")
                //        << "\n aDataVal[" << (iAB + iPair)
                //        << "] = 0x" << std::hex << aDataVal << std::dec
                //        << " (object " << iPair << ")"
                //        << "\n bDataVal[" << (iAB + iPair)
                //        << "] = 0x" << std::hex << bDataVal << std::dec
                //        << " (object " << (iPair + nrObjRow) << ")"
                //        << std::endl;
              }

            }

            break;
            case ESumsQ: {
              // LogTrace("L1GlobalTrigger")
              //<< "\nL1GlobalTriggerPSB: write ESumsQ for BxInEvent = "
              //<< iBxInEvent
              //<< "\n PSB " << std::hex << itBoard->gtBoardId() << std::dec
              //<< " Cable " << iCable << " Quad " << (*itQuad)
              //<< std::endl;

              // order: ETT, ETM et, HTT, ETM phi... hardcoded here
              int iPair = 0;

              if (m_candETT) {
                aDataVal = m_candETT->raw();
              } else {
                aDataVal = 0;
              }
              psbWordValue.setAData(aDataVal, iAB + iPair);

              if (m_candHTT) {
                bDataVal = m_candHTT->raw();
              } else {
                bDataVal = 0;
              }
              psbWordValue.setBData(bDataVal, iAB + iPair);

              // LogTrace("L1GlobalTrigger")
              //        << "\n aDataVal[" << (iAB + iPair)
              //        << "] = 0x" << std::hex << aDataVal << std::dec
              //        << "\n bDataVal[" << (iAB + iPair)
              //        << "] = 0x" << std::hex << bDataVal << std::dec
              //        << std::endl;
              //
              iPair = 1;
              if (m_candETM) {
                // bits 0:15
                aDataVal = m_candETM->raw() & 0x0000FFFF;

                // LogTrace("L1GlobalTrigger") << std::hex
                //        << "\n ETM et        = "
                //        << m_candETM->et()
                //        << "\n ETM overFlow  = "
                //       << m_candETM->overFlow() << std::dec
                //       << std::endl;
              } else {
                aDataVal = 0;
              }
              psbWordValue.setAData(aDataVal, iAB + iPair);

              if (m_candETM) {
                // bits 16:31
                bDataVal = (m_candETM->raw() & 0xFFFF0000) >> 16;

                // LogTrace("L1GlobalTrigger") << std::hex
                //        << "\n ETM phi  = " << m_candETM->phi()
                //        << std::dec << std::endl;

              } else {
                bDataVal = 0;
              }
              psbWordValue.setBData(bDataVal, iAB + iPair);

              // FIXME add HTM

              // LogTrace("L1GlobalTrigger")
              //        << "\n aDataVal[" << (iAB + iPair)
              //        << "] = 0x" << std::hex << aDataVal << std::dec
              //       << "\n bDataVal[" << (iAB + iPair)
              //        << "] = 0x" << std::hex << bDataVal << std::dec
              //       << std::endl;

            }

            break;
            case JetCountsQ: {
              // LogTrace("L1GlobalTrigger")
              //<< "\nL1GlobalTriggerPSB: write JetCountsQ for BxInEvent = "
              //<< iBxInEvent
              //<< "\n PSB " << std::hex << itBoard->gtBoardId() << std::dec
              //<< " Cable " << iCable << " Quad " << (*itQuad)
              //<< std::endl;

              // order: 3 JetCounts per 16-bits word ... hardcoded here
              int jetCountsBits = 5;  // FIXME get it from event setup
              int countsPerWord = 3;

              //
              int iPair = 0;
              aDataVal = 0;
              bDataVal = 0;

              int iCount = 0;

              if (m_candJetCounts) {
                for (int i = 0; i < countsPerWord; ++i) {
                  aDataVal = aDataVal | ((m_candJetCounts->count(iCount)) << (jetCountsBits * i));
                  iCount++;
                }

                //

                for (int i = 0; i < countsPerWord; ++i) {
                  bDataVal = bDataVal | ((m_candJetCounts->count(iCount)) << (jetCountsBits * i));
                  iCount++;
                }
              }

              psbWordValue.setAData(aDataVal, iAB + iPair);
              psbWordValue.setBData(bDataVal, iAB + iPair);

              //
              iPair = 1;
              aDataVal = 0;
              bDataVal = 0;

              if (m_candJetCounts) {
                for (int i = 0; i < countsPerWord; ++i) {
                  aDataVal = aDataVal | ((m_candJetCounts->count(iCount)) << (jetCountsBits * i));
                  iCount++;
                }

                //

                for (int i = 0; i < countsPerWord; ++i) {
                  bDataVal = bDataVal | ((m_candJetCounts->count(iCount)) << (jetCountsBits * i));
                  iCount++;
                }
              }

              psbWordValue.setAData(aDataVal, iAB + iPair);
              psbWordValue.setBData(bDataVal, iAB + iPair);
            }

            break;
              // FIXME add MIP/Iso bits
            case HfQ: {
              // LogTrace("L1GlobalTrigger")
              //<< "\nL1GlobalTriggerPSB: write HfQ for BxInEvent = "
              //<< iBxInEvent
              //<< "\n PSB " << std::hex << itBoard->gtBoardId() << std::dec
              //<< " Cable " << iCable << " Quad " << (*itQuad)
              //<< std::endl;

              // FIXME get it from event setup?
              // 3 bits per Hf index
              // order hardcoded here
              // HfBitCounts first, followed by HfRingEtSum
              int hfBits = 3;

              L1GctHFBitCounts hfBitCounts;
              int nHfBitCounts = hfBitCounts.nCounts();

              L1GctHFRingEtSums hfRingEtSums;
              int nHfRingEtSums = hfRingEtSums.nSums();

              //
              int iPair = 0;
              aDataVal = 0;
              bDataVal = 0;

              // sizeof return in multiple of 8 bits
              int hfPerWord = sizeof(aDataVal) * 8 / hfBits;
              // LogTrace("L1GlobalTrigger")
              //<< "\n nHfBitCounts  = " << nHfBitCounts
              //<< "\n nHfRingEtSums = " << nHfRingEtSums
              //<< "\n hfPerWord     = " << hfPerWord
              //<< std::endl;

              int iHf = 0;
              bool aDataFlag = true;
              bool bDataFlag = false;

              if (m_candHfBitCounts) {
                for (int i = 0; i < nHfBitCounts; ++i) {
                  if (aDataFlag) {
                    if (iHf < hfPerWord) {
                      // aData (cycle 0) for iPair 0 (object 0)
                      aDataVal = aDataVal | ((m_candHfBitCounts->bitCount(i)) << (hfBits * iHf));
                      iHf++;
                      // LogTrace("L1GlobalTrigger")
                      //        << "\n Added HfBitCounts index " << i << " to "
                      //        << " aDataVal[" << (iAB + iPair) << "]"
                      //        << std::endl;
                    } else {
                      aDataFlag = false;
                      bDataFlag = true;
                      iHf = 0;
                    }
                  }

                  if (bDataFlag) {
                    if (iHf < hfPerWord) {
                      // bData (cycle 1) for iPair 0 (object 2)
                      bDataVal = bDataVal | ((m_candHfBitCounts->bitCount(i)) << (hfBits * iHf));
                      iHf++;
                      // LogTrace("L1GlobalTrigger")
                      //        << "\n Added HfBitCounts index " << i << " to "
                      //        << " bDataVal[" << (iAB + iPair) << "]"
                      //       << std::endl;
                    } else {
                      aDataFlag = false;
                      bDataFlag = false;
                      iHf = 0;
                    }
                  }
                }
              } else {
                iHf = nHfBitCounts % hfPerWord;
                // LogTrace("L1GlobalTrigger")
                //        << "\n No HfBitCounts collection - skip "
                //        << iHf*hfBits << " bits "
                //        << std::endl;
              }

              if (aDataFlag && bDataFlag) {
                LogTrace("L1GlobalTrigger")
                    << "\n HfBitCounts collection filled aData and bData [" << (iAB + iPair) << "]"
                    << "\n HfRingEtSums collection has no space to be written" << std::endl;
              }

              if (m_candHfRingEtSums) {
                for (int i = 0; i < nHfRingEtSums; ++i) {
                  if (aDataFlag) {
                    if (iHf < hfPerWord) {
                      // aData (cycle 0) for iPair 0 (object 0)
                      aDataVal = aDataVal | ((m_candHfRingEtSums->etSum(i)) << (hfBits * iHf));
                      iHf++;
                      // LogTrace("L1GlobalTrigger")
                      //        << "\n Added HfRingEtSums index " << i << " to "
                      //        << " aDataVal[" << (iAB + iPair) << "]"
                      //       << std::endl;
                    } else {
                      aDataFlag = false;
                      bDataFlag = true;
                      iHf = 0;
                    }
                  }

                  if (bDataFlag) {
                    if (iHf < hfPerWord) {
                      // bData (cycle 1) for iPair 0 (object 2)
                      bDataVal = bDataVal | ((m_candHfRingEtSums->etSum(i)) << (hfBits * iHf));
                      iHf++;
                      // LogTrace("L1GlobalTrigger")
                      //        << "\n Added HfRingEtSums index " << i << " to "
                      //        << " bDataVal[" << (iAB + iPair) << "]"
                      //        << std::endl;
                    } else {
                      aDataFlag = false;
                      bDataFlag = false;
                      iHf = 0;
                    }
                  }
                }
              } else {
                iHf = nHfRingEtSums % hfPerWord;
                // LogTrace("L1GlobalTrigger")
                //        << "\n No HfRingEtSums collection - skip "
                //        << iHf*hfBits << " bits "
                //        << std::endl;
              }

              psbWordValue.setAData(aDataVal, iAB + iPair);
              psbWordValue.setBData(bDataVal, iAB + iPair);

              // LogTrace("L1GlobalTrigger")
              //        << "\n aDataVal[" << iAB + iPair
              //        << "] = 0x" << std::hex << aDataVal << std::dec
              //        << "\n bDataVal[" << (iAB + iPair)
              //        << "] = 0x" << std::hex << bDataVal << std::dec
              //       << std::endl;

              if (aDataFlag && bDataFlag) {
                LogTrace("L1GlobalTrigger") << "\n aData and bData [" << (iAB + iPair) << "] full"
                                            << "\n HfRingEtSums collection has not enough space to be "
                                               "written"
                                            << std::endl;
              }

            }

            break;
            default: {
              // do nothing
            }

            break;
          }  // end switch (*itQuad)

        }  // end for: (itQuad)

        // ** fill L1PsbWord in GT DAQ record

        // LogTrace("L1GlobalTrigger")
        //<< "\nL1GlobalTriggerPSB: write psbWordValue"
        //<< std::endl;

        gtDaqReadoutRecord->setGtPsbWord(psbWordValue);

      }  // end if (active && PSB)

    }  // end if (iPosition)

  }  // end for (itBoard
}

// clear PSB

void L1GlobalTriggerPSB::reset() {
  m_candL1NoIsoEG->clear();
  m_candL1IsoEG->clear();
  m_candL1CenJet->clear();
  m_candL1ForJet->clear();
  m_candL1TauJet->clear();

  // no reset() available...
  m_candETM = nullptr;
  m_candETT = nullptr;
  m_candHTT = nullptr;
  m_candHTM = nullptr;

  m_candJetCounts = nullptr;

  m_candHfBitCounts = nullptr;
  m_candHfRingEtSums = nullptr;
}

// print Global Calorimeter Trigger data
// use int to bitset conversion to print
void L1GlobalTriggerPSB::printGctObjectData(const int iBxInEvent) const {
  LogTrace("L1GlobalTrigger") << "\nL1GlobalTrigger: GCT data [hex] received by PSBs for BxInEvent = " << iBxInEvent
                              << "\n"
                              << std::endl;

  std::vector<const L1GctCand *>::const_iterator iterConst;

  LogTrace("L1GlobalTrigger") << "   GCT NoIsoEG " << std::endl;
  for (iterConst = m_candL1NoIsoEG->begin(); iterConst != m_candL1NoIsoEG->end(); iterConst++) {
    LogTrace("L1GlobalTrigger") << std::hex << "Rank = " << (*iterConst)->rank()
                                << " Eta index = " << (*iterConst)->etaIndex()
                                << " Phi index = " << (*iterConst)->phiIndex() << std::dec << std::endl;
  }

  LogTrace("L1GlobalTrigger") << "   GCT IsoEG " << std::endl;
  for (iterConst = m_candL1IsoEG->begin(); iterConst != m_candL1IsoEG->end(); iterConst++) {
    LogTrace("L1GlobalTrigger") << std::hex << "Rank = " << (*iterConst)->rank()
                                << " Eta index = " << (*iterConst)->etaIndex()
                                << " Phi index = " << (*iterConst)->phiIndex() << std::dec << std::endl;
  }

  LogTrace("L1GlobalTrigger") << "   GCT CenJet " << std::endl;
  for (iterConst = m_candL1CenJet->begin(); iterConst != m_candL1CenJet->end(); iterConst++) {
    LogTrace("L1GlobalTrigger") << std::hex << "Rank = " << (*iterConst)->rank()
                                << " Eta index = " << (*iterConst)->etaIndex()
                                << " Phi index = " << (*iterConst)->phiIndex() << std::dec << std::endl;
  }

  LogTrace("L1GlobalTrigger") << "   GCT ForJet " << std::endl;
  for (iterConst = m_candL1ForJet->begin(); iterConst != m_candL1ForJet->end(); iterConst++) {
    LogTrace("L1GlobalTrigger") << std::hex << "Rank = " << (*iterConst)->rank()
                                << " Eta index = " << (*iterConst)->etaIndex()
                                << " Phi index = " << (*iterConst)->phiIndex() << std::dec << std::endl;
  }

  LogTrace("L1GlobalTrigger") << "   GCT TauJet " << std::endl;
  for (iterConst = m_candL1TauJet->begin(); iterConst != m_candL1TauJet->end(); iterConst++) {
    LogTrace("L1GlobalTrigger") << std::hex << "Rank = " << (*iterConst)->rank()
                                << " Eta index = " << (*iterConst)->etaIndex()
                                << " Phi index = " << (*iterConst)->phiIndex() << std::dec << std::endl;
  }

  LogTrace("L1GlobalTrigger") << "   GCT ETM " << std::endl;
  if (m_candETM) {
    LogTrace("L1GlobalTrigger") << std::hex << "ET  = " << m_candETM->et() << std::dec << std::endl;

    LogTrace("L1GlobalTrigger") << std::hex << "phi = " << m_candETM->phi() << std::dec << std::endl;
  }

  LogTrace("L1GlobalTrigger") << "   GCT ETT " << std::endl;
  if (m_candETT) {
    LogTrace("L1GlobalTrigger") << std::hex << "ET  = " << m_candETT->et() << std::dec << std::endl;
  }

  LogTrace("L1GlobalTrigger") << "   GCT HTT " << std::endl;
  if (m_candHTT) {
    LogTrace("L1GlobalTrigger") << std::hex << "ET  = " << m_candHTT->et() << std::dec << std::endl;
  }

  LogTrace("L1GlobalTrigger") << "   GCT HTM " << std::endl;
  if (m_candHTM) {
    LogTrace("L1GlobalTrigger") << std::hex << "ET  = " << m_candHTM->et() << std::dec << std::endl;

    LogTrace("L1GlobalTrigger") << std::hex << "phi = " << m_candHTM->phi() << std::dec << std::endl;
  }

  LogTrace("L1GlobalTrigger") << "   GCT JetCounts " << std::endl;
  if (m_candJetCounts) {
    LogTrace("L1GlobalTrigger") << (*m_candJetCounts) << std::endl;
  }

  LogTrace("L1GlobalTrigger") << "   GCT HfBitCounts " << std::endl;
  if (m_candHfBitCounts) {
    LogTrace("L1GlobalTrigger") << (*m_candHfBitCounts) << std::endl;
  }

  LogTrace("L1GlobalTrigger") << "   GCT HfRingEtSums " << std::endl;
  if (m_candHfRingEtSums) {
    LogTrace("L1GlobalTrigger") << (*m_candHfRingEtSums) << std::endl;
  }
}

// static data members
