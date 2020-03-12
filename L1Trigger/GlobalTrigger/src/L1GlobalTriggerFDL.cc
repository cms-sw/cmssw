/**
 * \class L1GlobalTriggerFDL
 *
 *
 * Description: Final Decision Logic board.
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
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFDL.h"

// system include files
#include <iostream>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"

#include "CondFormats/L1TObjects/interface/L1GtBoard.h"
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"

// forward declarations

// constructor
L1GlobalTriggerFDL::L1GlobalTriggerFDL()
    :  // logical switches
      m_firstEv(true),
      m_firstEvLumiSegment(true),
      m_firstEvRun(true),
      m_isDebugEnabled(edm::isDebugEnabled()) {
  // create empty FDL word
  m_gtFdlWord = new L1GtFdlWord();

  // can not reserve memory here for prescale counters - no access to EventSetup
}

// destructor
L1GlobalTriggerFDL::~L1GlobalTriggerFDL() {
  reset();
  delete m_gtFdlWord;
}

// Operations

// run FDL
void L1GlobalTriggerFDL::run(edm::Event &iEvent,
                             const std::vector<int> &prescaleFactorsAlgoTrig,
                             const std::vector<int> &prescaleFactorsTechTrig,
                             const std::vector<unsigned int> &triggerMaskAlgoTrig,
                             const std::vector<unsigned int> &triggerMaskTechTrig,
                             const std::vector<unsigned int> &triggerMaskVetoAlgoTrig,
                             const std::vector<unsigned int> &triggerMaskVetoTechTrig,
                             const std::vector<L1GtBoard> &boardMaps,
                             const int totalBxInEvent,
                             const int iBxInEvent,
                             const unsigned int numberPhysTriggers,
                             const unsigned int numberTechnicalTriggers,
                             const unsigned int numberDaqPartitions,
                             const L1GlobalTriggerGTL *ptrGTL,
                             const L1GlobalTriggerPSB *ptrPSB,
                             const int pfAlgoSetIndex,
                             const int pfTechSetIndex,
                             const bool algorithmTriggersUnprescaled,
                             const bool algorithmTriggersUnmasked,
                             const bool technicalTriggersUnprescaled,
                             const bool technicalTriggersUnmasked,
                             const bool technicalTriggersVetoUnmasked) {
  // FIXME get rid of bitset in GTL in order to use only EventSetup
  const unsigned int numberPhysTriggersSet = L1GlobalTriggerReadoutSetup::NumberPhysTriggers;

  // get gtlDecisionWord from GTL
  std::bitset<numberPhysTriggersSet> gtlDecisionWord = ptrGTL->getAlgorithmOR();

  // convert decision word from std::bitset to std::vector<bool>
  DecisionWord algoDecisionWord(numberPhysTriggers);

  for (unsigned int iBit = 0; iBit < numberPhysTriggers; ++iBit) {
    bool bitValue = gtlDecisionWord.test(iBit);
    algoDecisionWord[iBit] = bitValue;
  }

  // prescale counters are reset at the beginning of the luminosity segment

  if (m_firstEv) {
    // prescale counters: numberPhysTriggers counters per bunch cross
    m_prescaleCounterAlgoTrig.reserve(numberPhysTriggers * totalBxInEvent);

    for (int iBxInEvent = 0; iBxInEvent <= totalBxInEvent; ++iBxInEvent) {
      m_prescaleCounterAlgoTrig.push_back(prescaleFactorsAlgoTrig);
    }

    // prescale counters: numberTechnicalTriggers counters per bunch cross
    m_prescaleCounterTechTrig.reserve(numberTechnicalTriggers * totalBxInEvent);

    for (int iBxInEvent = 0; iBxInEvent <= totalBxInEvent; ++iBxInEvent) {
      m_prescaleCounterTechTrig.push_back(prescaleFactorsTechTrig);
    }

    m_firstEv = false;
  }

  // TODO FIXME find the beginning of the luminosity segment
  if (m_firstEvLumiSegment) {
    m_prescaleCounterAlgoTrig.clear();
    for (int iBxInEvent = 0; iBxInEvent <= totalBxInEvent; ++iBxInEvent) {
      m_prescaleCounterAlgoTrig.push_back(prescaleFactorsAlgoTrig);
    }

    m_prescaleCounterTechTrig.clear();
    for (int iBxInEvent = 0; iBxInEvent <= totalBxInEvent; ++iBxInEvent) {
      m_prescaleCounterTechTrig.push_back(prescaleFactorsTechTrig);
    }

    m_firstEvLumiSegment = false;
  }

  // prescale the algorithm, if necessary

  // iBxInEvent is ... -2 -1 0 1 2 ... while counters are 0 1 2 3 4 ...
  int inBxInEvent = totalBxInEvent / 2 + iBxInEvent;

  for (unsigned int iBit = 0; iBit < numberPhysTriggers; ++iBit) {
    if ((!algorithmTriggersUnprescaled) && (prescaleFactorsAlgoTrig.at(iBit) != 1)) {
      bool bitValue = algoDecisionWord.at(iBit);
      if (bitValue) {
        (m_prescaleCounterAlgoTrig.at(inBxInEvent).at(iBit))--;
        if (m_prescaleCounterAlgoTrig.at(inBxInEvent).at(iBit) == 0) {
          // bit already true in algoDecisionWord, just reset counter
          m_prescaleCounterAlgoTrig.at(inBxInEvent).at(iBit) = prescaleFactorsAlgoTrig.at(iBit);

          // LogTrace("L1GlobalTrigger")
          //<< "\nPrescaled algorithm: " << iBit << ". Reset counter to "
          //<< prescaleFactorsAlgoTrig.at(iBit) << "\n"
          //<< std::endl;

        } else {
          // change bit to false
          algoDecisionWord[iBit] = false;
          ;

          // LogTrace("L1GlobalTrigger")
          //<< "\nPrescaled algorithm: " << iBit << ". Result set to false"
          //<< std::endl;
        }
      }
    }
  }

  // algo decision word written in the FDL readout before the trigger mask
  // in order to allow multiple DAQ partitions

  //
  // technical triggers
  //

  std::vector<bool> techDecisionWord = *(ptrPSB->getGtTechnicalTriggers());

  // prescale the technical trigger, if necessary

  for (unsigned int iBit = 0; iBit < numberTechnicalTriggers; ++iBit) {
    if ((!technicalTriggersUnprescaled) && (prescaleFactorsTechTrig.at(iBit) != 1)) {
      bool bitValue = techDecisionWord.at(iBit);
      if (bitValue) {
        (m_prescaleCounterTechTrig.at(inBxInEvent).at(iBit))--;
        if (m_prescaleCounterTechTrig.at(inBxInEvent).at(iBit) == 0) {
          // bit already true in techDecisionWord, just reset counter
          m_prescaleCounterTechTrig.at(inBxInEvent).at(iBit) = prescaleFactorsTechTrig.at(iBit);

          // LogTrace("L1GlobalTrigger")
          //<< "\nPrescaled algorithm: " << iBit << ". Reset counter to "
          //<< prescaleFactorsTechTrig.at(iBit) << "\n"
          //<< std::endl;

        } else {
          // change bit to false
          techDecisionWord[iBit] = false;

          // LogTrace("L1GlobalTrigger")
          //<< "\nPrescaled technical trigger: " << iBit << ". Result set to
          // false"
          //<< std::endl;
        }
      }
    }
  }

  // technical trigger decision word written in the FDL readout before the
  // trigger mask in order to allow multiple DAQ partitions

  //
  // compute the final decision word per DAQ partition
  //

  uint16_t finalOrValue = 0;

  for (unsigned int iDaq = 0; iDaq < numberDaqPartitions; ++iDaq) {
    bool daqPartitionFinalOR = false;

    // starts with technical trigger veto mask to minimize computation
    // no algorithm trigger veto mask is implemented up to now in hardware,
    // therefore do not implement it here
    bool vetoTechTrig = false;

    // vetoTechTrig can change only when using trigger veto masks
    if (!technicalTriggersVetoUnmasked) {
      for (unsigned int iBit = 0; iBit < numberTechnicalTriggers; ++iBit) {
        int triggerMaskVetoTechTrigBit = triggerMaskVetoTechTrig[iBit] & (1 << iDaq);
        // LogTrace("L1GlobalTrigger")
        //<< "\nTechnical trigger bit: " << iBit
        //<< " mask = " << triggerMaskVetoTechTrigBit
        //<< " DAQ partition " << iDaq
        //<< std::endl;

        if (triggerMaskVetoTechTrigBit && techDecisionWord[iBit]) {
          daqPartitionFinalOR = false;
          vetoTechTrig = true;

          // LogTrace("L1GlobalTrigger")
          //<< "\nVeto mask technical trigger: " << iBit
          // << ". FinalOR for DAQ partition " << iDaq << " set to false"
          //<< std::endl;

          break;
        }
      }
    }

    // apply algorithm and technical trigger masks only if no veto from
    // technical trigger
    if (!vetoTechTrig) {
      // algorithm trigger mask
      bool algoFinalOr = false;

      for (unsigned int iBit = 0; iBit < numberPhysTriggers; ++iBit) {
        bool iBitDecision = false;

        int triggerMaskAlgoTrigBit = -1;

        if (algorithmTriggersUnmasked) {
          triggerMaskAlgoTrigBit = 0;
        } else {
          triggerMaskAlgoTrigBit = triggerMaskAlgoTrig[iBit] & (1 << iDaq);
        }
        // LogTrace("L1GlobalTrigger")
        //<< "\nAlgorithm trigger bit: " << iBit
        //<< " mask = " << triggerMaskAlgoTrigBit
        //<< " DAQ partition " << iDaq
        //<< std::endl;

        if (triggerMaskAlgoTrigBit) {
          iBitDecision = false;

          // LogTrace("L1GlobalTrigger")
          //<< "\nMasked algorithm trigger: " << iBit << ". Result set to false"
          //<< std::endl;
        } else {
          iBitDecision = algoDecisionWord[iBit];
        }

        algoFinalOr = algoFinalOr || iBitDecision;
      }

      // set the technical trigger mask: block the corresponding algorithm if
      // bit value is 1

      bool techFinalOr = false;

      for (unsigned int iBit = 0; iBit < numberTechnicalTriggers; ++iBit) {
        bool iBitDecision = false;

        int triggerMaskTechTrigBit = -1;

        if (technicalTriggersUnmasked) {
          triggerMaskTechTrigBit = 0;
        } else {
          triggerMaskTechTrigBit = triggerMaskTechTrig[iBit] & (1 << iDaq);
        }
        // LogTrace("L1GlobalTrigger")
        //<< "\nTechnical trigger bit: " << iBit
        //<< " mask = " << triggerMaskTechTrigBit
        //<< std::endl;

        if (triggerMaskTechTrigBit) {
          iBitDecision = false;

          // LogTrace("L1GlobalTrigger")
          //<< "\nMasked technical trigger: " << iBit << ". Result set to false"
          //<< std::endl;
        } else {
          iBitDecision = techDecisionWord[iBit];
        }

        techFinalOr = techFinalOr || iBitDecision;
      }

      daqPartitionFinalOR = algoFinalOr || techFinalOr;

    } else {
      daqPartitionFinalOR = false;  // vetoTechTrig
    }

    // push it in finalOrValue
    uint16_t daqPartitionFinalORValue = static_cast<uint16_t>(daqPartitionFinalOR);

    finalOrValue = finalOrValue | (daqPartitionFinalORValue << iDaq);
  }

  // fill everything we know in the L1GtFdlWord

  typedef std::vector<L1GtBoard>::const_iterator CItBoardMaps;
  for (CItBoardMaps itBoard = boardMaps.begin(); itBoard != boardMaps.end(); ++itBoard) {
    if ((itBoard->gtBoardType() == FDL)) {
      m_gtFdlWord->setBoardId(itBoard->gtBoardId());

      // BxInEvent
      m_gtFdlWord->setBxInEvent(iBxInEvent);

      // bunch crossing

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

      m_gtFdlWord->setBxNr(bxCrossHw);

      // set event number since last L1 reset generated in FDL
      m_gtFdlWord->setEventNr(static_cast<uint32_t>(iEvent.id().event()));

      // technical trigger decision word
      m_gtFdlWord->setGtTechnicalTriggerWord(techDecisionWord);

      // algorithm trigger decision word
      m_gtFdlWord->setGtDecisionWord(algoDecisionWord);

      // index of prescale factor set - technical triggers and algo
      m_gtFdlWord->setGtPrescaleFactorIndexTech(static_cast<uint16_t>(pfTechSetIndex));
      m_gtFdlWord->setGtPrescaleFactorIndexAlgo(static_cast<uint16_t>(pfAlgoSetIndex));

      // NoAlgo bit FIXME

      // finalOR
      m_gtFdlWord->setFinalOR(finalOrValue);

      // orbit number
      m_gtFdlWord->setOrbitNr(static_cast<uint32_t>(iEvent.orbitNumber()));

      // luminosity segment number
      m_gtFdlWord->setLumiSegmentNr(static_cast<uint16_t>(iEvent.luminosityBlock()));

      // local bunch crossing - set identical with absolute BxNr
      m_gtFdlWord->setLocalBxNr(bxCrossHw);
    }
  }
}

// fill the FDL block in the L1 GT DAQ record for iBxInEvent
void L1GlobalTriggerFDL::fillDaqFdlBlock(const int iBxInEvent,
                                         const uint16_t &activeBoardsGtDaq,
                                         const int recordLength0,
                                         const int recordLength1,
                                         const unsigned int altNrBxBoardDaq,
                                         const std::vector<L1GtBoard> &boardMaps,
                                         L1GlobalTriggerReadoutRecord *gtDaqReadoutRecord) {
  typedef std::vector<L1GtBoard>::const_iterator CItBoardMaps;
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
      }

      if (activeBoard && writeBoard && (itBoard->gtBoardType() == FDL)) {
        gtDaqReadoutRecord->setGtFdlWord(*m_gtFdlWord);
      }
    }
  }
}

// fill the FDL block in the L1 GT EVM record for iBxInEvent
void L1GlobalTriggerFDL::fillEvmFdlBlock(const int iBxInEvent,
                                         const uint16_t &activeBoardsGtEvm,
                                         const int recordLength0,
                                         const int recordLength1,
                                         const unsigned int altNrBxBoardEvm,
                                         const std::vector<L1GtBoard> &boardMaps,
                                         L1GlobalTriggerEvmReadoutRecord *gtEvmReadoutRecord) {
  typedef std::vector<L1GtBoard>::const_iterator CItBoardMaps;
  for (CItBoardMaps itBoard = boardMaps.begin(); itBoard != boardMaps.end(); ++itBoard) {
    int iPosition = itBoard->gtPositionEvmRecord();
    if (iPosition > 0) {
      int iActiveBit = itBoard->gtBitEvmActiveBoards();
      bool activeBoard = false;

      if (iActiveBit >= 0) {
        activeBoard = activeBoardsGtEvm & (1 << iActiveBit);
      }

      if (activeBoard && (itBoard->gtBoardType() == FDL)) {
        gtEvmReadoutRecord->setGtFdlWord(*m_gtFdlWord);
      }
    }
  }
}

// clear FDL
void L1GlobalTriggerFDL::reset() {
  m_gtFdlWord->reset();

  // do NOT reset the prescale counters
}
