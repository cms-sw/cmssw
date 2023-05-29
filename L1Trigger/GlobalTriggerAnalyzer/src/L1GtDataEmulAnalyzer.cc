/**
 * \class L1GtDataEmulAnalyzer
 * 
 * 
 * Description: compare hardware records with emulator records for L1 GT record.  
 *
 * Implementation:
 *    Get the L1 GT records from data and from emulator.   
 *    Compare every board between data and emulator.
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// this class header
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtDataEmulAnalyzer.h"

// system include files
#include <memory>
#include <iostream>
#include <iomanip>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"

#include "TH1.h"
#include "TH2.h"
#include "TTree.h"

// constructor(s)
L1GtDataEmulAnalyzer::L1GtDataEmulAnalyzer(const edm::ParameterSet& parSet) {
  usesResource(TFileService::kSharedResource);

  // input tag for the L1 GT hardware DAQ/EVM record
  m_l1GtDataInputTag = parSet.getParameter<edm::InputTag>("L1GtDataInputTag");

  // input tag for the L1 GT emulator DAQ/EVM record
  m_l1GtEmulInputTag = parSet.getParameter<edm::InputTag>("L1GtEmulInputTag");

  // input tag for the L1 GCT hardware record
  m_l1GctDataInputTag = parSet.getParameter<edm::InputTag>("L1GctDataInputTag");

  LogDebug("L1GtDataEmulAnalyzer") << "\nInput tag for the L1 GT hardware records:          " << m_l1GtDataInputTag
                                   << "\nInput tag for the L1 GT emulator records:          " << m_l1GtEmulInputTag
                                   << "\nInput tag for the L1 GCT hardware record:          " << m_l1GctDataInputTag
                                   << std::endl;

  // initialize counters
  m_nrDataEventError = 0;
  m_nrEmulEventError = 0;

  // cache
  m_l1GtMenuCacheID = 0ULL;

  m_l1GtTmAlgoCacheID = 0ULL;
  m_l1GtTmTechCacheID = 0ULL;

  // book histograms
  bookHistograms();

  m_l1GtMenuToken = esConsumes();
  m_l1GtTmAlgoToken = esConsumes();
  m_l1GtTmTechToken = esConsumes();
}

// destructor
L1GtDataEmulAnalyzer::~L1GtDataEmulAnalyzer() {
  // empty
}

// member functions

// method called once each job just before starting event loop
void L1GtDataEmulAnalyzer::beginJob() {
  // empty
}

//compare the GTFE board
void L1GtDataEmulAnalyzer::compareGTFE(const edm::Event& iEvent,
                                       const edm::EventSetup& evSetup,
                                       const L1GtfeWord& gtfeBlockData,
                                       const L1GtfeWord& gtfeBlockEmul) {
  if (gtfeBlockData == gtfeBlockEmul) {
    m_myCoutStream << "\nData and emulated GTFE blocks: identical.\n";
    gtfeBlockData.print(m_myCoutStream);
  } else {
    m_myCoutStream << "\nData and emulated GTFE blocks: different.\n";

    m_myCoutStream << "\nData: GTFE block\n";
    gtfeBlockData.print(m_myCoutStream);

    m_myCoutStream << "\nEmul: GTFE block\n";
    gtfeBlockEmul.print(m_myCoutStream);
  }

  LogDebug("L1GtDataEmulAnalyzer") << m_myCoutStream.str() << std::endl;

  m_myCoutStream.str("");
  m_myCoutStream.clear();

  // get BoardId value
  const uint16_t boardIdData = gtfeBlockData.boardId();
  const uint16_t boardIdEmul = gtfeBlockEmul.boardId();

  if (boardIdData == boardIdEmul) {
    m_myCoutStream << "\nData and emulated GTFE boardId identical.";
    m_myCoutStream << "\n boardId() = " << std::hex << "0x" << std::setw(4) << std::setfill('0') << boardIdData
                   << std::setfill(' ') << std::dec;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\nData and emulated GTFE boardId different.";
    m_myCoutStream << "\n Data: boardId() = " << std::hex << "0x" << std::setw(4) << std::setfill('0') << boardIdData
                   << std::setfill(' ') << std::dec;
    m_myCoutStream << "\n Emul: boardId() = " << std::hex << "0x" << std::setw(4) << std::setfill('0') << boardIdEmul
                   << std::setfill(' ') << std::dec;
    m_myCoutStream << "\n";
    m_gtfeDataEmul->Fill(0);
  }

  /// get record length: 3 bx for standard, 5 bx for debug
  const uint16_t recordLengthData = gtfeBlockData.recordLength();
  const uint16_t recordLengthEmul = gtfeBlockEmul.recordLength();

  if (recordLengthData == recordLengthEmul) {
    m_myCoutStream << "\nData and emulated GTFE recordLength identical.";
    m_myCoutStream << "\n recordLength() = " << recordLengthData;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\nData and emulated GTFE recordLength different.";
    m_myCoutStream << "\n Data: recordLength() = " << recordLengthData;
    m_myCoutStream << "\n Emul: recordLength() = " << recordLengthEmul;
    m_myCoutStream << "\n";
    m_gtfeDataEmul->Fill(1);
  }

  /// get bunch cross number as counted in the GTFE board
  const uint16_t bxNrData = gtfeBlockData.bxNr();
  const uint16_t bxNrEmul = gtfeBlockEmul.bxNr();

  if (bxNrData == bxNrEmul) {
    m_myCoutStream << "\nData and emulated GTFE bxNr identical.";
    m_myCoutStream << "\n bxNr() = " << bxNrData;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\nData and emulated GTFE bxNr different.";
    m_myCoutStream << "\n Data: bxNr() = " << bxNrData;
    m_myCoutStream << "\n Emul: bxNr() = " << bxNrEmul;
    m_myCoutStream << "\n";
    m_gtfeDataEmul->Fill(2);
  }

  /// get setup version
  const uint32_t setupVersionData = gtfeBlockData.setupVersion();
  const uint32_t setupVersionEmul = gtfeBlockEmul.setupVersion();

  if (setupVersionData == setupVersionEmul) {
    m_myCoutStream << "\nData and emulated GTFE setupVersion identical.";
    m_myCoutStream << "\n setupVersion() = " << setupVersionData;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\nData and emulated GTFE setupVersion different.";
    m_myCoutStream << "\n Data: setupVersion() = " << setupVersionData;
    m_myCoutStream << "\n Emul: setupVersion() = " << setupVersionEmul;
    m_myCoutStream << "\n";
    m_gtfeDataEmul->Fill(3);
  }

  /// get boards contributing to EVM respectively DAQ record
  const uint16_t activeBoardsData = gtfeBlockData.activeBoards();
  const uint16_t activeBoardsEmul = gtfeBlockEmul.activeBoards();

  if (activeBoardsData == activeBoardsEmul) {
    m_myCoutStream << "\nData and emulated GTFE activeBoards identical.";
    m_myCoutStream << "\n activeBoards() = " << std::hex << "0x" << std::setw(4) << std::setfill('0')
                   << activeBoardsData << std::setfill(' ') << std::dec;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\nData and emulated GTFE activeBoards different.";
    m_myCoutStream << "\n Data: activeBoards() = " << std::hex << "0x" << std::setw(4) << std::setfill('0')
                   << activeBoardsData << std::setfill(' ') << std::dec;
    m_myCoutStream << "\n Emul: activeBoards() = " << std::hex << "0x" << std::setw(4) << std::setfill('0')
                   << activeBoardsEmul << std::setfill(' ') << std::dec;
    m_myCoutStream << "\n";
    m_gtfeDataEmul->Fill(4);
  }

  /// get total number of L1A sent since start of run
  const uint32_t totalTriggerNrData = gtfeBlockData.totalTriggerNr();
  const uint32_t totalTriggerNrEmul = gtfeBlockEmul.totalTriggerNr();

  if (totalTriggerNrData == totalTriggerNrEmul) {
    m_myCoutStream << "\nData and emulated GTFE totalTriggerNr identical.";
    m_myCoutStream << "\n totalTriggerNr() = " << totalTriggerNrData;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\nData and emulated GTFE totalTriggerNr different.";
    m_myCoutStream << "\n Data: totalTriggerNr() = " << totalTriggerNrData;
    m_myCoutStream << "\n Emul: totalTriggerNr() = " << totalTriggerNrEmul;
    m_myCoutStream << "\n";
    m_gtfeDataEmul->Fill(5);
  }

  edm::LogInfo("L1GtDataEmulAnalyzer") << m_myCoutStream.str() << std::endl;
  m_myCoutStream.str("");
  m_myCoutStream.clear();
}

//compare the FDL board
void L1GtDataEmulAnalyzer::compareFDL(const edm::Event& iEvent,
                                      const edm::EventSetup& evSetup,
                                      const L1GtFdlWord& fdlBlockData,
                                      const L1GtFdlWord& fdlBlockEmul,
                                      const int iRec) {
  // index of physics partition
  int PhysicsPartition = 0;

  //
  std::string recString;
  if (iRec == 0) {
    recString = "Daq";
  } else {
    recString = "Evm";
  }

  if (fdlBlockData == fdlBlockEmul) {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL blocks: identical.\n";
    fdlBlockData.print(m_myCoutStream);

  } else {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL blocks: different.\n";

    m_myCoutStream << "\nData: FDL block\n";
    fdlBlockData.print(m_myCoutStream);

    m_myCoutStream << "\nEmul: FDL block\n";
    fdlBlockEmul.print(m_myCoutStream);
  }

  LogDebug("L1GtDataEmulAnalyzer") << m_myCoutStream.str() << std::endl;

  m_myCoutStream.str("");
  m_myCoutStream.clear();

  // get bunch cross in the GT event record -
  // move it first as histograms are BxInEvent dependent
  const int bxInEventData = fdlBlockData.bxInEvent();
  const int bxInEventEmul = fdlBlockEmul.bxInEvent();

  bool matchBxInEvent = false;

  if (bxInEventData == bxInEventEmul) {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL bxInEvent identical.";
    m_myCoutStream << "\n bxInEvent() = " << bxInEventData;
    m_myCoutStream << "\n";
    matchBxInEvent = true;

  } else {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL bxInEvent different.";
    m_myCoutStream << "\n Data: bxInEvent() = " << bxInEventData;
    m_myCoutStream << "\n Emul: bxInEvent() = " << bxInEventEmul;
    m_myCoutStream << "\n";

    m_fdlDataEmul_Err[iRec]->Fill(1);
  }

  LogDebug("L1GtDataEmulAnalyzer") << m_myCoutStream.str() << std::endl;
  m_myCoutStream.str("");
  m_myCoutStream.clear();

  // symmetrize
  bool validBxInEvent = false;
  int histIndex = bxInEventData + (TotalBxInEvent + 1) / 2 - 1;
  if ((histIndex <= TotalBxInEvent) && (histIndex >= 0)) {
    validBxInEvent = true;
  }

  // get / update the trigger menu from the EventSetup
  // local cache & check on cacheIdentifier

  unsigned long long l1GtMenuCacheID = evSetup.get<L1GtTriggerMenuRcd>().cacheIdentifier();

  if (m_l1GtMenuCacheID != l1GtMenuCacheID) {
    m_l1GtMenu = &evSetup.getData(m_l1GtMenuToken);

    m_l1GtMenuCacheID = l1GtMenuCacheID;
  }
  // get / update the trigger mask from the EventSetup
  // local cache & check on cacheIdentifier

  unsigned long long l1GtTmAlgoCacheID = evSetup.get<L1GtTriggerMaskAlgoTrigRcd>().cacheIdentifier();

  if (m_l1GtTmAlgoCacheID != l1GtTmAlgoCacheID) {
    m_l1GtTmAlgo = &evSetup.getData(m_l1GtTmAlgoToken);

    m_triggerMaskAlgoTrig = m_l1GtTmAlgo->gtTriggerMask();

    m_l1GtTmAlgoCacheID = l1GtTmAlgoCacheID;
  }

  unsigned long long l1GtTmTechCacheID = evSetup.get<L1GtTriggerMaskTechTrigRcd>().cacheIdentifier();

  if (m_l1GtTmTechCacheID != l1GtTmTechCacheID) {
    m_l1GtTmTech = &evSetup.getData(m_l1GtTmTechToken);

    m_triggerMaskTechTrig = m_l1GtTmTech->gtTriggerMask();

    m_l1GtTmTechCacheID = l1GtTmTechCacheID;
  }

  // loop over algorithms and increase the corresponding counters
  // FIXME put it back in cache
  // FIXME when the menu changes, make a copy of histograms, and clear the old one
  //       otherwise the labels are wrong
  const AlgorithmMap& algorithmMap = m_l1GtMenu->gtAlgorithmMap();

  for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {
    std::string aName = itAlgo->first;
    const char* algName = aName.c_str();
    int algBitNumber = (itAlgo->second).algoBitNumber();

    (m_fdlDataAlgoDecision[histIndex][iRec])->GetXaxis()->SetBinLabel(algBitNumber + 1, algName);
    m_fdlDataAlgoDecision_Err[iRec]->GetXaxis()->SetBinLabel(algBitNumber + 1, algName);

    m_fdlEmulAlgoDecision[histIndex][iRec]->GetXaxis()->SetBinLabel(algBitNumber + 1, algName);
    m_fdlEmulAlgoDecision_Err[iRec]->GetXaxis()->SetBinLabel(algBitNumber + 1, algName);

    m_fdlDataEmulAlgoDecision[histIndex][iRec]->GetXaxis()->SetBinLabel(algBitNumber + 1, algName);
    m_fdlDataEmulAlgoDecision_Err[iRec]->GetXaxis()->SetBinLabel(algBitNumber + 1, algName);
  }

  // get BoardId value
  const uint16_t boardIdData = fdlBlockData.boardId();
  const uint16_t boardIdEmul = fdlBlockEmul.boardId();

  if (boardIdData == boardIdEmul) {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL boardId identical.";
    m_myCoutStream << "\n boardId() = " << std::hex << "0x" << std::setw(4) << std::setfill('0') << boardIdData
                   << std::setfill(' ') << std::dec;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL boardId different.";
    m_myCoutStream << "\n Data: boardId() = " << std::hex << "0x" << std::setw(4) << std::setfill('0') << boardIdData
                   << std::setfill(' ') << std::dec;
    m_myCoutStream << "\n Emul: boardId() = " << std::hex << "0x" << std::setw(4) << std::setfill('0') << boardIdEmul
                   << std::setfill(' ') << std::dec;
    m_myCoutStream << "\n";

    if (matchBxInEvent && validBxInEvent) {
      m_fdlDataEmul[histIndex][iRec]->Fill(0);
    } else {
      m_fdlDataEmul_Err[iRec]->Fill(0);
    }
  }

  LogDebug("L1GtDataEmulAnalyzer") << m_myCoutStream.str() << std::endl;
  m_myCoutStream.str("");
  m_myCoutStream.clear();

  // get BxNr - bunch cross number of the actual bx
  const uint16_t bxNrData = fdlBlockData.bxNr();
  const uint16_t bxNrEmul = fdlBlockEmul.bxNr();

  if (bxNrData == bxNrEmul) {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL bxNr identical.";
    m_myCoutStream << "\n bxNr() = " << bxNrData;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL bxNr different.";
    m_myCoutStream << "\n Data: bxNr() = " << bxNrData;
    m_myCoutStream << "\n Emul: bxNr() = " << bxNrEmul;
    m_myCoutStream << "\n";

    if (matchBxInEvent && validBxInEvent) {
      m_fdlDataEmul[histIndex][iRec]->Fill(2);
    } else {
      m_fdlDataEmul_Err[iRec]->Fill(2);
    }
  }

  LogDebug("L1GtDataEmulAnalyzer") << m_myCoutStream.str() << std::endl;
  m_myCoutStream.str("");
  m_myCoutStream.clear();

  // get event number since last L1 reset generated in FDL
  const uint32_t eventNrData = fdlBlockData.eventNr();
  const uint32_t eventNrEmul = fdlBlockEmul.eventNr();

  if (eventNrData == eventNrEmul) {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL eventNr identical.";
    m_myCoutStream << "\n eventNr() = " << eventNrData;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL eventNr different.";
    m_myCoutStream << "\n Data: eventNr() = " << eventNrData;
    m_myCoutStream << "\n Emul: eventNr() = " << eventNrEmul;
    m_myCoutStream << "\n";

    if (matchBxInEvent && validBxInEvent) {
      m_fdlDataEmul[histIndex][iRec]->Fill(3);
    } else {
      m_fdlDataEmul_Err[iRec]->Fill(3);
    }
  }

  LogDebug("L1GtDataEmulAnalyzer") << m_myCoutStream.str() << std::endl;
  m_myCoutStream.str("");
  m_myCoutStream.clear();

  // get  technical trigger bits
  const TechnicalTriggerWord& gtTechnicalTriggerWordData = fdlBlockData.gtTechnicalTriggerWord();
  const TechnicalTriggerWord& gtTechnicalTriggerWordEmul = fdlBlockEmul.gtTechnicalTriggerWord();

  int nTechBits = gtTechnicalTriggerWordData.size();

  TechnicalTriggerWord gtTechnicalTriggerWordDataMask(nTechBits);
  TechnicalTriggerWord gtTechnicalTriggerWordEmulMask(nTechBits);

  unsigned int triggerMask = 0;
  unsigned int bitValue = 0;

  if (matchBxInEvent && validBxInEvent) {
    for (int iBit = 0; iBit < nTechBits; ++iBit) {
      triggerMask = (m_triggerMaskTechTrig.at(iBit)) & (1 << PhysicsPartition);

      if (gtTechnicalTriggerWordData[iBit]) {
        m_fdlDataTechDecision[histIndex][iRec]->Fill(iBit);

        bitValue = (triggerMask) ? 0 : 1;
        gtTechnicalTriggerWordDataMask[iBit] = bitValue;
        if (bitValue) {
          m_fdlDataTechDecisionMask[histIndex][iRec]->Fill(iBit);
        }
      }

      if (gtTechnicalTriggerWordEmul.at(iBit)) {
        m_fdlEmulTechDecision[histIndex][iRec]->Fill(iBit);

        bitValue = (triggerMask) ? 0 : 1;
        gtTechnicalTriggerWordEmulMask[iBit] = bitValue;
        if (bitValue) {
          m_fdlEmulTechDecisionMask[histIndex][iRec]->Fill(iBit);
        }
      }
    }
  } else {
    for (int iBit = 0; iBit < nTechBits; ++iBit) {
      if (gtTechnicalTriggerWordData[iBit]) {
        m_fdlDataTechDecision_Err[iRec]->Fill(iBit);
      }

      if (gtTechnicalTriggerWordEmul.at(iBit)) {
        m_fdlEmulTechDecision_Err[iRec]->Fill(iBit);
      }
    }
  }

  if (gtTechnicalTriggerWordData == gtTechnicalTriggerWordEmul) {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL gtTechnicalTriggerWord identical.\n";
    fdlBlockData.printGtTechnicalTriggerWord(m_myCoutStream);
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL gtTechnicalTriggerWord different.";
    m_myCoutStream << "\n Data: ";
    fdlBlockData.printGtTechnicalTriggerWord(m_myCoutStream);
    m_myCoutStream << "\n Emul: ";
    fdlBlockEmul.printGtTechnicalTriggerWord(m_myCoutStream);
    m_myCoutStream << "\n";

    if (matchBxInEvent && validBxInEvent) {
      m_fdlDataEmul[histIndex][iRec]->Fill(4);
    } else {
      m_fdlDataEmul_Err[iRec]->Fill(4);
    }

    if (matchBxInEvent && validBxInEvent) {
      for (int iBit = 0; iBit < nTechBits; ++iBit) {
        if (gtTechnicalTriggerWordData[iBit] != gtTechnicalTriggerWordEmul.at(iBit)) {
          m_fdlDataEmulTechDecision[histIndex][iRec]->Fill(iBit);
        }
      }
    } else {
      for (int iBit = 0; iBit < nTechBits; ++iBit) {
        if (gtTechnicalTriggerWordData[iBit] != gtTechnicalTriggerWordEmul.at(iBit)) {
          m_fdlDataEmulTechDecision_Err[iRec]->Fill(iBit);
        }
      }
    }
  }

  LogDebug("L1GtDataEmulAnalyzer") << m_myCoutStream.str() << std::endl;
  m_myCoutStream.str("");
  m_myCoutStream.clear();

  if (gtTechnicalTriggerWordDataMask == gtTechnicalTriggerWordEmulMask) {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL gtTechnicalTriggerWord after mask identical.\n";
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL gtTechnicalTriggerWord after mask different.";
    m_myCoutStream << "\n Data: ";
    m_myCoutStream << "\n Emul: ";
    m_myCoutStream << "\n";

    if (matchBxInEvent && validBxInEvent) {
      m_fdlDataEmul[histIndex][iRec]->Fill(5);
    } else {
      m_fdlDataEmul_Err[iRec]->Fill(5);
    }

    if (matchBxInEvent && validBxInEvent) {
      for (int iBit = 0; iBit < nTechBits; ++iBit) {
        if (gtTechnicalTriggerWordData[iBit] != gtTechnicalTriggerWordEmul.at(iBit)) {
          m_fdlDataEmulTechDecisionMask[histIndex][iRec]->Fill(iBit);
        }
      }
    }
  }

  LogDebug("L1GtDataEmulAnalyzer") << m_myCoutStream.str() << std::endl;
  m_myCoutStream.str("");
  m_myCoutStream.clear();

  // get algorithms bits (decision word)
  const DecisionWord& gtDecisionWordData = fdlBlockData.gtDecisionWord();
  const DecisionWord& gtDecisionWordEmul = fdlBlockEmul.gtDecisionWord();

  int nAlgoBits = gtDecisionWordData.size();

  DecisionWord gtDecisionWordDataMask(nAlgoBits);
  DecisionWord gtDecisionWordEmulMask(nAlgoBits);

  if (matchBxInEvent && validBxInEvent) {
    for (int iBit = 0; iBit < nAlgoBits; ++iBit) {
      triggerMask = (m_triggerMaskAlgoTrig.at(iBit)) & (1 << PhysicsPartition);

      if (gtDecisionWordData[iBit]) {
        m_fdlDataAlgoDecision[histIndex][iRec]->Fill(iBit);

        bitValue = (triggerMask) ? 0 : 1;
        gtDecisionWordDataMask[iBit] = bitValue;
        if (bitValue) {
          m_fdlDataAlgoDecisionMask[histIndex][iRec]->Fill(iBit);
        }
      }

      if (gtDecisionWordEmul.at(iBit)) {
        m_fdlEmulAlgoDecision[histIndex][iRec]->Fill(iBit);

        bitValue = (triggerMask) ? 0 : 1;
        gtDecisionWordEmulMask[iBit] = bitValue;
        if (bitValue) {
          m_fdlEmulAlgoDecisionMask[histIndex][iRec]->Fill(iBit);
        }
      }
    }
  } else {
    for (int iBit = 0; iBit < nAlgoBits; ++iBit) {
      if (gtDecisionWordData[iBit]) {
        m_fdlDataAlgoDecision_Err[iRec]->Fill(iBit);
      }
    }

    for (int iBit = 0; iBit < nAlgoBits; ++iBit) {
      if (gtDecisionWordEmul.at(iBit)) {
        m_fdlEmulAlgoDecision_Err[iRec]->Fill(iBit);
      }
    }
  }

  if (gtDecisionWordData == gtDecisionWordEmul) {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL gtDecisionWord identical.";
    fdlBlockData.printGtDecisionWord(m_myCoutStream);
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL gtDecisionWord different.";
    m_myCoutStream << "\n Data: ";
    fdlBlockData.printGtDecisionWord(m_myCoutStream);
    m_myCoutStream << "\n Emul: ";
    fdlBlockEmul.printGtDecisionWord(m_myCoutStream);
    m_myCoutStream << "\n";

    if (matchBxInEvent && validBxInEvent) {
      m_fdlDataEmul[histIndex][iRec]->Fill(6);
    } else {
      m_fdlDataEmul_Err[iRec]->Fill(6);
    }

    if (matchBxInEvent && validBxInEvent) {
      for (int iBit = 0; iBit < nAlgoBits; ++iBit) {
        if (gtDecisionWordData[iBit] != gtDecisionWordEmul.at(iBit)) {
          m_fdlDataEmulAlgoDecision[histIndex][iRec]->Fill(iBit);
        }
      }
    } else {
      for (int iBit = 0; iBit < nAlgoBits; ++iBit) {
        if (gtDecisionWordData[iBit] != gtDecisionWordEmul.at(iBit)) {
          m_fdlDataEmulAlgoDecision_Err[iRec]->Fill(iBit);
        }
      }
    }
  }

  if (gtDecisionWordDataMask == gtDecisionWordEmulMask) {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL gtDecisionWord after mask identical.";
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL gtDecisionWord after mask different.";
    m_myCoutStream << "\n Data: ";
    m_myCoutStream << "\n Emul: ";
    m_myCoutStream << "\n";

    if (matchBxInEvent && validBxInEvent) {
      m_fdlDataEmul[histIndex][iRec]->Fill(7);
    } else {
      m_fdlDataEmul_Err[iRec]->Fill(7);
    }

    if (matchBxInEvent && validBxInEvent) {
      for (int iBit = 0; iBit < nAlgoBits; ++iBit) {
        if (gtDecisionWordDataMask[iBit] != gtDecisionWordEmulMask.at(iBit)) {
          m_fdlDataEmulAlgoDecisionMask[histIndex][iRec]->Fill(iBit);
        }
      }
    }
  }

  // get  extended algorithms bits (extended decision word)
  const DecisionWordExtended& gtDecisionWordExtendedData = fdlBlockData.gtDecisionWordExtended();
  const DecisionWordExtended& gtDecisionWordExtendedEmul = fdlBlockEmul.gtDecisionWordExtended();

  if (gtDecisionWordExtendedData == gtDecisionWordExtendedEmul) {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL gtDecisionWordExtended identical.\n";
    fdlBlockData.printGtDecisionWordExtended(m_myCoutStream);
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL gtDecisionWordExtended different.\n";
    m_myCoutStream << "\n Data: ";
    fdlBlockData.printGtDecisionWordExtended(m_myCoutStream);
    m_myCoutStream << "\n Emul: ";
    fdlBlockEmul.printGtDecisionWordExtended(m_myCoutStream);
    m_myCoutStream << "\n";

    if (matchBxInEvent && validBxInEvent) {
      m_fdlDataEmul[histIndex][iRec]->Fill(8);
    } else {
      m_fdlDataEmul_Err[iRec]->Fill(8);
    }
  }

  // get  NoAlgo
  const uint16_t noAlgoData = fdlBlockData.noAlgo();
  const uint16_t noAlgoEmul = fdlBlockEmul.noAlgo();

  if (noAlgoData == noAlgoEmul) {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL noAlgo identical.";
    m_myCoutStream << "\n noAlgo() = " << noAlgoData;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL noAlgo different.";
    m_myCoutStream << "\n Data: noAlgo() = " << noAlgoData;
    m_myCoutStream << "\n Emul: noAlgo() = " << noAlgoEmul;
    m_myCoutStream << "\n";

    if (matchBxInEvent && validBxInEvent) {
      m_fdlDataEmul[histIndex][iRec]->Fill(9);
    } else {
      m_fdlDataEmul_Err[iRec]->Fill(9);
    }
  }

  // get  "Final OR" bits
  const uint16_t finalORData = fdlBlockData.finalOR();
  const uint16_t finalOREmul = fdlBlockEmul.finalOR();

  if (finalORData == finalOREmul) {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL finalOR identical.";
    m_myCoutStream << "\n finalOR() = " << std::hex << "0x" << std::setw(2) << std::setfill('0') << finalORData
                   << std::setfill(' ') << std::dec;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL finalOR different.";
    m_myCoutStream << "\n Data: finalOR() = " << std::hex << "0x" << std::setw(2) << std::setfill('0') << finalORData
                   << std::setfill(' ') << std::dec;
    m_myCoutStream << "\n Emul: finalOR() = " << std::hex << "0x" << std::setw(2) << std::setfill('0') << finalOREmul
                   << std::setfill(' ') << std::dec;
    m_myCoutStream << "\n";

    if (matchBxInEvent && validBxInEvent) {
      m_fdlDataEmul[histIndex][iRec]->Fill(10);
    } else {
      m_fdlDataEmul_Err[iRec]->Fill(10);
    }
  }

  // get  "Final OR" for physics partition
  const int finalORPhysData = finalORData & (1 << PhysicsPartition);
  const int finalORPhysEmul = finalOREmul & (1 << PhysicsPartition);

  if (finalORPhysData == finalORPhysEmul) {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL finalOR for the physics partition identical.";
    m_myCoutStream << "\n finalOR() = " << finalORPhysData;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL finalOR for the physics partition  different.";
    m_myCoutStream << "\n Data: finalOR() = " << finalORPhysData;
    m_myCoutStream << "\n Emul: finalOR() = " << finalORPhysEmul;
    m_myCoutStream << "\n";

    if (matchBxInEvent && validBxInEvent) {
      m_fdlDataEmul[histIndex][iRec]->Fill(11);
    } else {
      m_fdlDataEmul_Err[iRec]->Fill(11);
    }
  }

  // get  local bunch cross number of the actual bx
  const uint16_t localBxNrData = fdlBlockData.localBxNr();
  const uint16_t localBxNrEmul = fdlBlockEmul.localBxNr();

  if (localBxNrData == localBxNrEmul) {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL localBxNr identical.";
    m_myCoutStream << "\n localBxNr() = " << localBxNrData;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\n" << recString << " Data and emulated FDL localBxNr different.";
    m_myCoutStream << "\n Data: localBxNr() = " << localBxNrData;
    m_myCoutStream << "\n Emul: localBxNr() = " << localBxNrEmul;
    m_myCoutStream << "\n";

    if (matchBxInEvent && validBxInEvent) {
      m_fdlDataEmul[histIndex][iRec]->Fill(12);
    } else {
      m_fdlDataEmul_Err[iRec]->Fill(12);
    }
  }

  edm::LogInfo("L1GtDataEmulAnalyzer") << m_myCoutStream.str() << std::endl;
  m_myCoutStream.str("");
  m_myCoutStream.clear();
}

//compare the PSB board
void L1GtDataEmulAnalyzer::comparePSB(const edm::Event& iEvent,
                                      const edm::EventSetup& evSetup,
                                      const L1GtPsbWord& psbBlockData,
                                      const L1GtPsbWord& psbBlockEmul) {
  if (psbBlockData == psbBlockEmul) {
    m_myCoutStream << "\nData and emulated PSB blocks: identical.\n";
    psbBlockData.print(m_myCoutStream);

  } else {
    m_myCoutStream << "\nData and emulated PSB blocks: different.\n";

    m_myCoutStream << "\nData: PSB block\n";
    psbBlockData.print(m_myCoutStream);

    m_myCoutStream << "\nEmul: PSB block\n";
    psbBlockEmul.print(m_myCoutStream);
  }

  LogDebug("L1GtDataEmulAnalyzer") << m_myCoutStream.str() << std::endl;

  m_myCoutStream.str("");
  m_myCoutStream.clear();

  // get BoardId value
  const uint16_t boardIdData = psbBlockData.boardId();
  const uint16_t boardIdEmul = psbBlockEmul.boardId();

  if (boardIdData == boardIdEmul) {
    m_myCoutStream << "\nData and emulated PSB boardId identical.";
    m_myCoutStream << "\n boardId() = " << std::hex << "0x" << std::setw(4) << std::setfill('0') << boardIdData
                   << std::setfill(' ') << std::dec;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\nData and emulated PSB boardId different.";
    m_myCoutStream << "\n Data: boardId() = " << std::hex << "0x" << std::setw(4) << std::setfill('0') << boardIdData
                   << std::setfill(' ') << std::dec;
    m_myCoutStream << "\n Emul: boardId() = " << std::hex << "0x" << std::setw(4) << std::setfill('0') << boardIdEmul
                   << std::setfill(' ') << std::dec;
    m_myCoutStream << "\n";
  }

  // get bunch cross in the GT event record
  const int bxInEventData = psbBlockData.bxInEvent();
  const int bxInEventEmul = psbBlockEmul.bxInEvent();

  if (bxInEventData == bxInEventEmul) {
    m_myCoutStream << "\nData and emulated PSB bxInEvent identical.";
    m_myCoutStream << "\n bxInEvent() = " << bxInEventData;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\nData and emulated PSB bxInEvent different.";
    m_myCoutStream << "\n Data: bxInEvent() = " << bxInEventData;
    m_myCoutStream << "\n Emul: bxInEvent() = " << bxInEventEmul;
    m_myCoutStream << "\n";
  }

  // get BxNr - bunch cross number of the actual bx
  const uint16_t bxNrData = psbBlockData.bxNr();
  const uint16_t bxNrEmul = psbBlockEmul.bxNr();

  if (bxNrData == bxNrEmul) {
    m_myCoutStream << "\nData and emulated PSB bxNr identical.";
    m_myCoutStream << "\n bxNr() = " << bxNrData;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\nData and emulated PSB bxNr different.";
    m_myCoutStream << "\n Data: bxNr() = " << bxNrData;
    m_myCoutStream << "\n Emul: bxNr() = " << bxNrEmul;
    m_myCoutStream << "\n";
  }

  // get event number since last L1 reset generated in FDL
  const uint32_t eventNrData = psbBlockData.eventNr();
  const uint32_t eventNrEmul = psbBlockEmul.eventNr();

  if (eventNrData == eventNrEmul) {
    m_myCoutStream << "\nData and emulated PSB eventNr identical.";
    m_myCoutStream << "\n eventNr() = " << eventNrData;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\nData and emulated PSB eventNr different.";
    m_myCoutStream << "\n Data: eventNr() = " << eventNrData;
    m_myCoutStream << "\n Emul: eventNr() = " << eventNrEmul;
    m_myCoutStream << "\n";
  }

  /// get/set A_DATA_CH_IA
  uint16_t valData;
  uint16_t valEmul;

  for (int iA = 0; iA < psbBlockData.NumberAData; ++iA) {
    valData = psbBlockData.aData(iA);
    valEmul = psbBlockEmul.aData(iA);

    if (valData == valEmul) {
      m_myCoutStream << "\nData and emulated PSB aData(" << iA << ") identical.";
      m_myCoutStream << "\n aData(iA) = " << std::hex << "0x" << std::setw(4) << std::setfill('0') << valData
                     << std::setfill(' ') << std::dec;
      m_myCoutStream << "\n";

    } else {
      m_myCoutStream << "\nData and emulated PSB aData(" << iA << ") different.";
      m_myCoutStream << "\n Data: aData(iA) = " << std::hex << "0x" << std::setw(4) << std::setfill('0') << valData
                     << std::setfill(' ') << std::dec;
      m_myCoutStream << "\n Emul: aData(iA) = " << std::hex << "0x" << std::setw(4) << std::setfill('0') << valEmul
                     << std::setfill(' ') << std::dec;
      m_myCoutStream << "\n";
    }
  }

  /// get/set B_DATA_CH_IB
  for (int iB = 0; iB < psbBlockData.NumberBData; ++iB) {
    valData = psbBlockData.bData(iB);
    valEmul = psbBlockEmul.bData(iB);

    if (valData == valEmul) {
      m_myCoutStream << "\nData and emulated PSB bData(" << iB << ") identical.";
      m_myCoutStream << "\n bData(iA) = " << std::hex << "0x" << std::setw(4) << std::setfill('0') << valData
                     << std::setfill(' ') << std::dec;
      m_myCoutStream << "\n";

    } else {
      m_myCoutStream << "\nData and emulated PSB bData(" << iB << ") different.";
      m_myCoutStream << "\n Data: bData(iA) = " << std::hex << "0x" << std::setw(4) << std::setfill('0') << valData
                     << std::setfill(' ') << std::dec;
      m_myCoutStream << "\n Emul: bData(iA) = " << std::hex << "0x" << std::setw(4) << std::setfill('0') << valEmul
                     << std::setfill(' ') << std::dec;
      m_myCoutStream << "\n";
    }
  }

  // get  local bunch cross number of the actual bx
  const uint16_t localBxNrData = psbBlockData.localBxNr();
  const uint16_t localBxNrEmul = psbBlockEmul.localBxNr();

  if (localBxNrData == localBxNrEmul) {
    m_myCoutStream << "\nData and emulated PSB localBxNr identical.";
    m_myCoutStream << "\n localBxNr() = " << localBxNrData;
    m_myCoutStream << "\n";

  } else {
    m_myCoutStream << "\nData and emulated PSB localBxNr different.";
    m_myCoutStream << "\n Data: localBxNr() = " << localBxNrData;
    m_myCoutStream << "\n Emul: localBxNr() = " << localBxNrEmul;
    m_myCoutStream << "\n";
  }

  edm::LogInfo("L1GtDataEmulAnalyzer") << m_myCoutStream.str() << std::endl;
  m_myCoutStream.str("");
  m_myCoutStream.clear();
}

//compare the TCS board
void L1GtDataEmulAnalyzer::compareTCS(const edm::Event& iEvent,
                                      const edm::EventSetup& evSetup,
                                      const L1TcsWord&,
                                      const L1TcsWord&) {
  // empty
}

//L1 GT DAQ record comparison
void L1GtDataEmulAnalyzer::compareDaqRecord(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  // formal index for DAQ record
  int iRec = 0;

  // get the L1 GT hardware DAQ record L1GlobalTriggerReadoutRecord
  edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecordData;
  iEvent.getByLabel(m_l1GtDataInputTag, gtReadoutRecordData);

  bool validData = false;

  if (!gtReadoutRecordData.isValid()) {
    m_nrDataEventError++;
  } else {
    validData = true;
  }

  // get the L1 GT emulator DAQ record L1GlobalTriggerReadoutRecord
  edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecordEmul;
  iEvent.getByLabel(m_l1GtEmulInputTag, gtReadoutRecordEmul);

  bool validEmul = false;

  if (!gtReadoutRecordEmul.isValid()) {
    m_nrEmulEventError++;
  } else {
    validEmul = true;
  }

  if ((!validData) || (!validEmul)) {
    edm::LogWarning("L1GtDataEmulAnalyzer")
        << "\n Valid data:" << validData << "\n Valid emulator:" << validEmul << std::endl;
  }

  // compare GTFE
  const L1GtfeWord& gtfeBlockData = gtReadoutRecordData->gtfeWord();
  const L1GtfeWord& gtfeBlockEmul = gtReadoutRecordEmul->gtfeWord();

  compareGTFE(iEvent, evSetup, gtfeBlockData, gtfeBlockEmul);

  // FDL comparison
  const std::vector<L1GtFdlWord>& gtFdlVectorData = gtReadoutRecordData->gtFdlVector();
  const std::vector<L1GtFdlWord>& gtFdlVectorEmul = gtReadoutRecordEmul->gtFdlVector();

  int gtFdlVectorDataSize = gtFdlVectorData.size();
  int gtFdlVectorEmulSize = gtFdlVectorEmul.size();

  if (gtFdlVectorDataSize == gtFdlVectorEmulSize) {
    m_myCoutStream << "\nData and emulated FDL vector size: identical.\n";
    m_myCoutStream << "  Size: " << gtFdlVectorDataSize << std::endl;

    for (int iFdl = 0; iFdl < gtFdlVectorDataSize; ++iFdl) {
      const L1GtFdlWord& fdlBlockData = gtFdlVectorData[iFdl];
      const L1GtFdlWord& fdlBlockEmul = gtFdlVectorEmul[iFdl];

      compareFDL(iEvent, evSetup, fdlBlockData, fdlBlockEmul, iRec);
    }
  } else {
    m_myCoutStream << "\nData and emulated FDL vector size: different.\n";
    m_myCoutStream << "  Data: size = " << gtFdlVectorDataSize << std::endl;
    m_myCoutStream << "  Emul: size = " << gtFdlVectorEmulSize << std::endl;
  }

  LogDebug("L1GtDataEmulAnalyzer") << m_myCoutStream.str() << std::endl;

  m_myCoutStream.str("");
  m_myCoutStream.clear();

  // PSB comparison
  const std::vector<L1GtPsbWord>& gtPsbVectorData = gtReadoutRecordData->gtPsbVector();
  const std::vector<L1GtPsbWord>& gtPsbVectorEmul = gtReadoutRecordEmul->gtPsbVector();

  int gtPsbVectorDataSize = gtPsbVectorData.size();
  int gtPsbVectorEmulSize = gtPsbVectorEmul.size();

  if (gtPsbVectorDataSize == gtPsbVectorEmulSize) {
    m_myCoutStream << "\nData and emulated PSB vector size: identical.\n";
    m_myCoutStream << "  Size: " << gtPsbVectorDataSize << std::endl;
  } else {
    m_myCoutStream << "\nData and emulated PSB vector size: different.\n";
    m_myCoutStream << "  Data: size = " << gtPsbVectorDataSize << std::endl;
    m_myCoutStream << "  Emul: size = " << gtPsbVectorEmulSize << std::endl;
  }

  // the order of PSB block in the gtPsbVector is different in emulator and in data
  // in emulator: all active PSB in one BxInEvent, ordered L1A-1, L1A, L1A+1
  // in unpacker: every PSB in all BxInEvent
  for (int iPsb = 0; iPsb < gtPsbVectorDataSize; ++iPsb) {
    const L1GtPsbWord& psbBlockData = gtPsbVectorData[iPsb];
    const uint16_t boardIdData = psbBlockData.boardId();
    const int bxInEventData = psbBlockData.bxInEvent();

    // search the corresponding PSB in the emulated record using the
    // BoardId and the BxInEvent

    bool foundPSB = false;

    for (int iPsbF = 0; iPsbF < gtPsbVectorEmulSize; ++iPsbF) {
      const L1GtPsbWord& psbBlockEmul = gtPsbVectorEmul[iPsbF];
      const uint16_t boardIdEmul = psbBlockEmul.boardId();
      const int bxInEventEmul = psbBlockEmul.bxInEvent();

      if ((boardIdEmul == boardIdData) && (bxInEventData == bxInEventEmul)) {
        foundPSB = true;

        // compare the boards
        comparePSB(iEvent, evSetup, psbBlockData, psbBlockEmul);
      }
    }

    if (!foundPSB) {
      m_myCoutStream << "\nNo emulated PSB with boardId() = " << std::hex << "0x" << std::setw(4) << std::setfill('0')
                     << boardIdData << std::setfill(' ') << std::dec << " and BxInEvent = " << bxInEventData
                     << " was found";
    }
  }

  edm::LogInfo("L1GtDataEmulAnalyzer") << m_myCoutStream.str() << std::endl;

  m_myCoutStream.str("");
  m_myCoutStream.clear();
}

// L1 GT EVM record comparison
void L1GtDataEmulAnalyzer::compareEvmRecord(const edm::Event& iEvent, const edm::EventSetup&) {
  // FIXME
}

// compare the GCT collections obtained from L1 GT PSB with the input
// GCT collections
void L1GtDataEmulAnalyzer::compareGt_Gct(const edm::Event& iEvent, const edm::EventSetup&) {
  // FIXME
}

// analyze each event: event loop
void L1GtDataEmulAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  // L1 GT DAQ record comparison
  compareDaqRecord(iEvent, evSetup);

  // L1 GT EVM record comparison
  compareEvmRecord(iEvent, evSetup);

  // GCT collections from L1 GT PSB versus unpacked GCT
  compareGt_Gct(iEvent, evSetup);
}

// book all histograms for the module
void L1GtDataEmulAnalyzer::bookHistograms() {
  // histogram service
  edm::Service<TFileService> histServ;

  // histograms

  // GTFE histograms
  TFileDirectory gtfeHist = histServ->mkdir("GTFE");
  m_gtfeDataEmul = gtfeHist.make<TH1F>("gtfeDataEmul", "GTFE data vs emul", 6, 0., 6.);
  m_gtfeDataEmul->GetXaxis()->SetBinLabel(1, "BoardId");
  m_gtfeDataEmul->GetXaxis()->SetBinLabel(2, "RecordLength");
  m_gtfeDataEmul->GetXaxis()->SetBinLabel(3, "BxNr");
  m_gtfeDataEmul->GetXaxis()->SetBinLabel(4, "SetupVersion");
  m_gtfeDataEmul->GetXaxis()->SetBinLabel(5, "DaqActiveBoards");
  m_gtfeDataEmul->GetXaxis()->SetBinLabel(6, "TotalTriggerNr");

  // FDL histograms

  TFileDirectory fdlHist = histServ->mkdir("FDL");

  const unsigned int numberTechTriggers = L1GlobalTriggerReadoutSetup::NumberTechnicalTriggers;

  const unsigned int numberAlgoTriggers = L1GlobalTriggerReadoutSetup::NumberPhysTriggers;

  for (int iRec = 0; iRec < 2; ++iRec) {
    std::string recString;
    if (iRec == 0) {
      recString = "Daq";
    } else {
      recString = "Evm";
    }

    std::string hName;
    const char* histName;

    for (int iHist = 0; iHist < TotalBxInEvent; ++iHist) {
      // convert [0, TotalBxInEvent] to [-X, +X] and add to histogram name
      int iIndex = iHist - ((TotalBxInEvent + 1) / 2 - 1);
      int hIndex = (iIndex + 16) % 16;

      std::stringstream ss;
      std::string str;
      ss << std::uppercase << std::hex << hIndex;
      ss >> str;

      hName = recString + "FdlDataEmul_" + str;
      histName = hName.c_str();

      std::string hTitle = "FDL data vs emul mismatch for BxInEvent = " + str;
      const char* histTitle = hTitle.c_str();

      //

      m_fdlDataEmul[iHist][iRec] = fdlHist.make<TH1F>(histName, histTitle, 13, 0., 13.);
      m_fdlDataEmul[iHist][iRec]->GetXaxis()->SetBinLabel(1, "BoardId");
      m_fdlDataEmul[iHist][iRec]->GetXaxis()->SetBinLabel(2, "BxInEvent");
      m_fdlDataEmul[iHist][iRec]->GetXaxis()->SetBinLabel(3, "BxNr");
      m_fdlDataEmul[iHist][iRec]->GetXaxis()->SetBinLabel(4, "EventNr");
      m_fdlDataEmul[iHist][iRec]->GetXaxis()->SetBinLabel(5, "TechTrigger");
      m_fdlDataEmul[iHist][iRec]->GetXaxis()->SetBinLabel(6, "TechTriggerMask");
      m_fdlDataEmul[iHist][iRec]->GetXaxis()->SetBinLabel(7, "AlgoTrigger");
      m_fdlDataEmul[iHist][iRec]->GetXaxis()->SetBinLabel(8, "AlgoTriggerMask");
      m_fdlDataEmul[iHist][iRec]->GetXaxis()->SetBinLabel(9, "AlgoExtend");
      m_fdlDataEmul[iHist][iRec]->GetXaxis()->SetBinLabel(10, "NoAlgo");
      m_fdlDataEmul[iHist][iRec]->GetXaxis()->SetBinLabel(11, "FinalORAllParts");
      m_fdlDataEmul[iHist][iRec]->GetXaxis()->SetBinLabel(12, "FinalORPhysPart");
      m_fdlDataEmul[iHist][iRec]->GetXaxis()->SetBinLabel(13, "LocalBxNr");

      // algorithm decision
      //   data
      hName = recString + "FdlDataAlgoDecision_" + str;
      histName = hName.c_str();

      hTitle = "Data: algorithm decision word for BxInEvent = " + str;
      histTitle = hTitle.c_str();

      m_fdlDataAlgoDecision[iHist][iRec] =
          fdlHist.make<TH1F>(histName, histTitle, numberAlgoTriggers, 0., numberAlgoTriggers);

      //   emul
      hName = recString + "FdlEmulAlgoDecision_" + str;
      histName = hName.c_str();

      hTitle = "Emul: algorithm decision word for BxInEvent = " + str;
      histTitle = hTitle.c_str();

      m_fdlEmulAlgoDecision[iHist][iRec] =
          fdlHist.make<TH1F>(histName, histTitle, numberAlgoTriggers, 0., numberAlgoTriggers);

      // algorithm decision after masking (partition physics)
      //   data
      hName = recString + "FdlDataAlgoDecisionMask_" + str;
      histName = hName.c_str();

      hTitle = "Data, physics partition: algorithm decision word after mask for BxInEvent = " + str;
      histTitle = hTitle.c_str();

      m_fdlDataAlgoDecisionMask[iHist][iRec] =
          fdlHist.make<TH1F>(histName, histTitle, numberAlgoTriggers, 0., numberAlgoTriggers);

      //   emul
      hName = recString + "FdlEmulAlgoDecisionMask_" + str;
      histName = hName.c_str();

      hTitle = "Emul, physics partition: algorithm decision word after mask for BxInEvent =  " + str;
      histTitle = hTitle.c_str();

      m_fdlEmulAlgoDecisionMask[iHist][iRec] =
          fdlHist.make<TH1F>(histName, histTitle, numberAlgoTriggers, 0., numberAlgoTriggers);

      //
      hName = recString + "FdlDataEmulAlgoDecision_" + str;
      histName = hName.c_str();

      hTitle = "Data vs emul: non-matching algorithm decision word for BxInEvent = " + str;
      histTitle = hTitle.c_str();

      m_fdlDataEmulAlgoDecision[iHist][iRec] =
          fdlHist.make<TH1F>(histName, histTitle, numberAlgoTriggers, 0., numberAlgoTriggers);

      //
      hName = recString + "FdlDataEmulAlgoDecisionMask_" + str;
      histName = hName.c_str();

      hTitle =
          "Data vs emul, physics partition: non-matching algorithm decision word after mask for BxInEvent = " + str;
      histTitle = hTitle.c_str();

      m_fdlDataEmulAlgoDecisionMask[iHist][iRec] =
          fdlHist.make<TH1F>(histName, histTitle, numberAlgoTriggers, 0., numberAlgoTriggers);

      // technical trigger decision
      //   data
      hName = recString + "FdlDataTechDecision_" + str;
      histName = hName.c_str();

      hTitle = "Data technical trigger decision word for BxInEvent = " + str;
      histTitle = hTitle.c_str();

      m_fdlDataTechDecision[iHist][iRec] =
          fdlHist.make<TH1F>(histName, histTitle, numberTechTriggers, 0., numberTechTriggers);

      //   emul
      hName = recString + "FdlEmulTechDecision_" + str;
      histName = hName.c_str();

      hTitle = "Emul: technical trigger decision word for BxInEvent = " + str;
      histTitle = hTitle.c_str();

      m_fdlEmulTechDecision[iHist][iRec] =
          fdlHist.make<TH1F>(histName, histTitle, numberTechTriggers, 0., numberTechTriggers);

      // technical trigger decision after masking (partition physics)
      hName = recString + "FdlDataTechDecisionMask_" + str;
      histName = hName.c_str();

      hTitle = "Data technical trigger decision word after mask for BxInEvent = " + str;
      histTitle = hTitle.c_str();

      m_fdlDataTechDecisionMask[iHist][iRec] =
          fdlHist.make<TH1F>(histName, histTitle, numberTechTriggers, 0., numberTechTriggers);

      //
      hName = recString + "FdlEmulTechDecisionMask_" + str;
      histName = hName.c_str();

      hTitle = "Emul: technical trigger decision word after mask for BxInEvent = " + str;
      histTitle = hTitle.c_str();

      m_fdlEmulTechDecisionMask[iHist][iRec] =
          fdlHist.make<TH1F>(histName, histTitle, numberTechTriggers, 0., numberTechTriggers);

      //
      hName = recString + "FdlDataEmulTechDecision_" + str;
      histName = hName.c_str();

      hTitle = "Data vs emul: non-matching technical trigger decision word for BxInEvent = " + str;
      histTitle = hTitle.c_str();

      m_fdlDataEmulTechDecision[iHist][iRec] =
          fdlHist.make<TH1F>(histName, histTitle, numberTechTriggers, 0., numberTechTriggers);

      hName = recString + "FdlDataEmulTechDecisionMask_" + str;
      histName = hName.c_str();

      hTitle = "Data vs emul: non-matching technical trigger decision word after mask for BxInEvent = " + str;
      histTitle = hTitle.c_str();

      m_fdlDataEmulTechDecisionMask[iHist][iRec] =
          fdlHist.make<TH1F>(histName, histTitle, numberTechTriggers, 0., numberTechTriggers);
    }

    hName = recString + "FdlDataEmul_Err";
    histName = hName.c_str();

    m_fdlDataEmul_Err[iRec] = fdlHist.make<TH1F>(histName, "FDL data vs emul: non-matching BxInEvent", 13, 0., 13.);
    m_fdlDataEmul_Err[iRec]->GetXaxis()->SetBinLabel(1, "BoardId");
    m_fdlDataEmul_Err[iRec]->GetXaxis()->SetBinLabel(2, "BxInEvent");
    m_fdlDataEmul_Err[iRec]->GetXaxis()->SetBinLabel(3, "BxNr");
    m_fdlDataEmul_Err[iRec]->GetXaxis()->SetBinLabel(4, "EventNr");
    m_fdlDataEmul_Err[iRec]->GetXaxis()->SetBinLabel(5, "TechTrigger");
    m_fdlDataEmul_Err[iRec]->GetXaxis()->SetBinLabel(6, "TechTriggerMask");
    m_fdlDataEmul_Err[iRec]->GetXaxis()->SetBinLabel(7, "AlgoTrigger");
    m_fdlDataEmul_Err[iRec]->GetXaxis()->SetBinLabel(8, "AlgoTriggerMask");
    m_fdlDataEmul_Err[iRec]->GetXaxis()->SetBinLabel(9, "AlgoExtend");
    m_fdlDataEmul_Err[iRec]->GetXaxis()->SetBinLabel(10, "NoAlgo");
    m_fdlDataEmul_Err[iRec]->GetXaxis()->SetBinLabel(11, "FinalORAllParts");
    m_fdlDataEmul_Err[iRec]->GetXaxis()->SetBinLabel(12, "FinalORPhysPart");
    m_fdlDataEmul_Err[iRec]->GetXaxis()->SetBinLabel(13, "LocalBxNr");

    hName = recString + "FdlDataAlgoDecision_Err";
    histName = hName.c_str();

    m_fdlDataAlgoDecision_Err[iRec] =
        fdlHist.make<TH1F>(histName,
                           "Data: algorithm trigger decision word, non-matching BxInEvent",
                           numberAlgoTriggers,
                           0.,
                           numberAlgoTriggers);

    //
    hName = recString + "FdlEmulAlgoDecision_Err";
    histName = hName.c_str();

    m_fdlEmulAlgoDecision_Err[iRec] =
        fdlHist.make<TH1F>(histName,
                           "Emul: algorithm trigger decision word, non-matching BxInEvent",
                           numberAlgoTriggers,
                           0.,
                           numberAlgoTriggers);

    hName = recString + "FdlDataEmulAlgoDecision_Err";
    histName = hName.c_str();

    m_fdlDataEmulAlgoDecision_Err[iRec] =
        fdlHist.make<TH1F>(histName,
                           "Data vs emul: algorithm trigger decision word, non-matching BxInEvent",
                           numberAlgoTriggers,
                           0.,
                           numberAlgoTriggers);

    //
    hName = recString + "FdlDataTechDecision_Err";
    histName = hName.c_str();

    m_fdlDataTechDecision_Err[iRec] =
        fdlHist.make<TH1F>(histName,
                           "Data: technical trigger decision word, non-matching BxInEvent",
                           numberTechTriggers,
                           0.,
                           numberTechTriggers);

    hName = recString + "FdlEmulTechDecision_Err";
    histName = hName.c_str();

    m_fdlEmulTechDecision_Err[iRec] =
        fdlHist.make<TH1F>(histName,
                           "Emul: technical trigger decision word, non-matching BxInEvent",
                           numberTechTriggers,
                           0.,
                           numberTechTriggers);

    hName = recString + "FdlDataEmulTechDecision_Err";
    histName = hName.c_str();

    m_fdlDataEmulTechDecision_Err[iRec] =
        fdlHist.make<TH1F>(histName,
                           "Data vs emul: technical trigger decision word, non-matching BxInEvent",
                           numberTechTriggers,
                           0.,
                           numberTechTriggers);
  }
}

// method called once each job just after ending the event loop
void L1GtDataEmulAnalyzer::endJob() {
  // empty
}
