/**
 * \class L1GtPackUnpackAnalyzer
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
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtPackUnpackAnalyzer.h"

// system include files
#include <memory>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor(s)
L1GtPackUnpackAnalyzer::L1GtPackUnpackAnalyzer(const edm::ParameterSet& parSet) {
  // input tag for the initial GT DAQ record:
  m_initialDaqGtInputTag = parSet.getParameter<edm::InputTag>("InitialDaqGtInputTag");

  // input tag for the initial GMT readout collection:
  m_initialMuGmtInputTag = parSet.getParameter<edm::InputTag>("InitialMuGmtInputTag");

  // input tag for the final GT DAQ and GMT records:
  m_finalGtGmtInputTag = parSet.getParameter<edm::InputTag>("FinalGtGmtInputTag");

  edm::LogInfo("L1GtPackUnpackAnalyzer") << "\nInput tag for the initial GT DAQ record:          "
                                         << m_initialDaqGtInputTag << " \n"
                                         << "\nInput tag for the initial GMT readout collection: "
                                         << m_initialMuGmtInputTag << " \n"
                                         << "\nInput tag for the final GT DAQ and GMT records:   "
                                         << m_finalGtGmtInputTag << " \n"
                                         << std::endl;
}

// destructor
L1GtPackUnpackAnalyzer::~L1GtPackUnpackAnalyzer() {
  // empty
}

// member functions

// method called once each job just before starting event loop
void L1GtPackUnpackAnalyzer::beginJob() {
  // empty
}

// GT comparison
void L1GtPackUnpackAnalyzer::analyzeGT(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  // define an output stream to print into
  // it can then be directed to whatever log level is desired
  std::ostringstream myCoutStream;

  // get the initial L1GlobalTriggerReadoutRecord
  edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecordInitial;
  iEvent.getByLabel(m_initialDaqGtInputTag, gtReadoutRecordInitial);

  if (!gtReadoutRecordInitial.isValid()) {
    edm::LogError("L1GtTrigReport") << "Initial L1GlobalTriggerReadoutRecord with input tag \n  "
                                    << m_initialDaqGtInputTag << " not found.\n\n"
                                    << std::endl;
    return;
  }

  // get the final L1GlobalTriggerReadoutRecord
  edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecordFinal;
  iEvent.getByLabel(m_finalGtGmtInputTag, gtReadoutRecordFinal);

  if (!gtReadoutRecordFinal.isValid()) {
    edm::LogError("L1GtTrigReport") << "Final L1GlobalTriggerReadoutRecord with input tag \n  " << m_finalGtGmtInputTag
                                    << " not found.\n\n"
                                    << std::endl;
    return;
  }

  // compare GTFE
  const L1GtfeWord& gtfeWordInitial = gtReadoutRecordInitial->gtfeWord();
  const L1GtfeWord& gtfeWordFinal = gtReadoutRecordFinal->gtfeWord();

  if (gtfeWordInitial == gtfeWordFinal) {
    myCoutStream << "\nInitial and final GTFE blocks: identical.\n";
    gtfeWordInitial.print(myCoutStream);
  } else {
    myCoutStream << "\nInitial and final GTFE blocks: different.\n";

    myCoutStream << "\nInitial GTFE block\n";
    gtfeWordInitial.print(myCoutStream);

    myCoutStream << "\nFinal GTFE block\n";
    gtfeWordFinal.print(myCoutStream);
  }

  edm::LogInfo("L1GtPackUnpackAnalyzer") << myCoutStream.str() << std::endl;

  myCoutStream.str("");
  myCoutStream.clear();

  // FDL comparison
  const std::vector<L1GtFdlWord>& gtFdlVectorInitial = gtReadoutRecordInitial->gtFdlVector();
  const std::vector<L1GtFdlWord>& gtFdlVectorFinal = gtReadoutRecordFinal->gtFdlVector();

  int gtFdlVectorInitialSize = gtFdlVectorInitial.size();
  int gtFdlVectorFinalSize = gtFdlVectorFinal.size();

  if (gtFdlVectorInitialSize == gtFdlVectorFinalSize) {
    myCoutStream << "\nInitial and final FDL vector size: identical.\n";
    myCoutStream << "  Size: " << gtFdlVectorInitialSize << std::endl;

    for (int iFdl = 0; iFdl < gtFdlVectorInitialSize; ++iFdl) {
      const L1GtFdlWord& fdlWordInitial = gtFdlVectorInitial[iFdl];
      const L1GtFdlWord& fdlWordFinal = gtFdlVectorFinal[iFdl];

      if (fdlWordInitial == fdlWordFinal) {
        myCoutStream << "\nInitial and final FDL blocks: identical.\n";
        fdlWordInitial.print(myCoutStream);

      } else {
        myCoutStream << "\nInitial and final FDL blocks: different.\n";

        myCoutStream << "\nInitial FDL block\n";
        fdlWordInitial.print(myCoutStream);

        myCoutStream << "\nFinal FDL block\n";
        fdlWordFinal.print(myCoutStream);
      }
    }
  } else {
    myCoutStream << "\nInitial and final FDL vector size: different.\n";
    myCoutStream << "  Initial size: " << gtFdlVectorInitialSize << std::endl;
    myCoutStream << "  Final size: " << gtFdlVectorFinalSize << std::endl;
  }

  edm::LogInfo("L1GtPackUnpackAnalyzer") << myCoutStream.str() << std::endl;

  myCoutStream.str("");
  myCoutStream.clear();

  // PSB comparison
  const std::vector<L1GtPsbWord>& gtPsbVectorInitial = gtReadoutRecordInitial->gtPsbVector();
  const std::vector<L1GtPsbWord>& gtPsbVectorFinal = gtReadoutRecordFinal->gtPsbVector();

  int gtPsbVectorInitialSize = gtPsbVectorInitial.size();
  int gtPsbVectorFinalSize = gtPsbVectorFinal.size();

  if (gtPsbVectorInitialSize == gtPsbVectorFinalSize) {
    myCoutStream << "\nInitial and final PSB vector size: identical.\n";
    myCoutStream << "  Size: " << gtPsbVectorInitialSize << std::endl;

    // the order of PSB block in the gtPsbVector is different in emulator and unpacker
    // TODO can be fixed?
    for (int iPsb = 0; iPsb < gtPsbVectorInitialSize; ++iPsb) {
      const L1GtPsbWord& psbWordInitial = gtPsbVectorInitial[iPsb];
      const uint16_t boardIdInitial = psbWordInitial.boardId();
      const int bxInEventInitial = psbWordInitial.bxInEvent();

      // search the corresponding PSB in the final record using the
      // BoardId and the BxInEvent

      bool foundPSB = false;

      for (int iPsbF = 0; iPsbF < gtPsbVectorFinalSize; ++iPsbF) {
        const L1GtPsbWord& psbWordFinal = gtPsbVectorFinal[iPsbF];
        const uint16_t boardIdFinal = psbWordFinal.boardId();
        const int bxInEventFinal = psbWordFinal.bxInEvent();

        if ((boardIdFinal == boardIdInitial) && (bxInEventInitial == bxInEventFinal)) {
          foundPSB = true;

          // compare the boards
          if (psbWordInitial == psbWordFinal) {
            myCoutStream << "\nInitial and final PSB blocks: identical.\n";
            psbWordInitial.print(myCoutStream);

          } else {
            myCoutStream << "\nInitial and final PSB blocks: different.\n";

            myCoutStream << "\nInitial PSB block\n";
            psbWordInitial.print(myCoutStream);

            myCoutStream << "\nFinal PSB block\n";
            psbWordFinal.print(myCoutStream);
          }
        }
      }

      if (!foundPSB) {
        myCoutStream << "\nNo final PSB with boardID = " << boardIdInitial << " and BxINEvent = " << bxInEventInitial
                     << " was found"
                     << "\nInitial and final PSB vectors: different";
      }
    }
  } else {
    myCoutStream << "\nInitial and final PSB vector size: different.\n";
    myCoutStream << "  Initial size: " << gtPsbVectorInitialSize << std::endl;
    myCoutStream << "  Final size: " << gtPsbVectorFinalSize << std::endl;
  }

  edm::LogInfo("L1GtPackUnpackAnalyzer") << myCoutStream.str() << std::endl;

  myCoutStream.str("");
  myCoutStream.clear();

  // get reference to muon collection
  const edm::RefProd<L1MuGMTReadoutCollection> muCollRefProdInitial = gtReadoutRecordInitial->muCollectionRefProd();

  const edm::RefProd<L1MuGMTReadoutCollection> muCollRefProdFinal = gtReadoutRecordFinal->muCollectionRefProd();

  if (muCollRefProdInitial == muCollRefProdFinal) {
    myCoutStream << "\nInitial and final RefProd<L1MuGMTReadoutCollection>: identical.\n";
  } else {
    myCoutStream << "\nInitial and final RefProd<L1MuGMTReadoutCollection>: different.\n";
  }

  edm::LogInfo("L1GtPackUnpackAnalyzer") << myCoutStream.str() << std::endl;

  myCoutStream.str("");
  myCoutStream.clear();
}

// GMT comparison
void L1GtPackUnpackAnalyzer::analyzeGMT(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  // define an output stream to print into
  // it can then be directed to whatever log level is desired
  std::ostringstream myCoutStream;

  // get initial L1MuGMTReadoutCollection
  edm::Handle<L1MuGMTReadoutCollection> gmtRcInitial;
  iEvent.getByLabel(m_initialMuGmtInputTag, gmtRcInitial);

  if (!gmtRcInitial.isValid()) {
    edm::LogError("L1GtPackUnpackAnalyzer")
        << "Initial L1MuGMTReadoutCollection with input tag \n  " << m_initialMuGmtInputTag << " not found.\n\n"
        << std::endl;
    return;
  }

  std::vector<L1MuGMTReadoutRecord> muRecordsInitial = gmtRcInitial->getRecords();

  // get final L1MuGMTReadoutCollection
  edm::Handle<L1MuGMTReadoutCollection> gmtRcFinal;
  iEvent.getByLabel(m_finalGtGmtInputTag, gmtRcFinal);

  if (!gmtRcFinal.isValid()) {
    edm::LogError("L1GtPackUnpackAnalyzer")
        << "Final L1MuGMTReadoutCollection with input tag \n  " << m_finalGtGmtInputTag << " not found.\n\n"
        << std::endl;
    return;
  }

  std::vector<L1MuGMTReadoutRecord> muRecordsFinal = gmtRcFinal->getRecords();

  int muRecordsInitialSize = muRecordsInitial.size();
  int muRecordsFinalSize = muRecordsFinal.size();

  if (muRecordsInitialSize == muRecordsFinalSize) {
    myCoutStream << "\nInitial and final L1MuGMTReadoutCollection record size: identical.\n";
    myCoutStream << "  Size: " << muRecordsInitialSize << std::endl;
  } else {
    myCoutStream << "\nInitial and final  L1MuGMTReadoutCollection record size: different.\n";
    myCoutStream << "  Initial size: " << muRecordsInitialSize << std::endl;
    myCoutStream << "  Final size: " << muRecordsFinalSize << std::endl;
  }

  edm::LogInfo("L1GtPackUnpackAnalyzer") << myCoutStream.str() << std::endl;

  myCoutStream.str("");
  myCoutStream.clear();
}

// analyze each event: event loop
void L1GtPackUnpackAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  // GT DAQ comparison
  analyzeGT(iEvent, evSetup);

  // GMT comparison
  analyzeGMT(iEvent, evSetup);
}

// method called once each job just after ending the event loop
void L1GtPackUnpackAnalyzer::endJob() {
  // empty
}
