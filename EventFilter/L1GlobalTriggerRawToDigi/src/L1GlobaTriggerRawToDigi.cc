/**
 * \class L1GlobalTriggerRawToDigi
 *
 *
 * Description: unpack raw data into digitized data.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna -  GT
 * \author: Ivan Mikulec       - HEPHY Vienna - GMT
 *
 *
 */

// this class header
#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GlobalTriggerRawToDigi.h"

// system include files
#include <iostream>
#include <iomanip>
#include <algorithm>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtBoard.h"

// constructor(s)
L1GlobalTriggerRawToDigi::L1GlobalTriggerRawToDigi(const edm::ParameterSet& pSet)
    :

      // input tag for DAQ GT record
      m_daqGtInputTag(pSet.getParameter<edm::InputTag>("DaqGtInputTag")),

      // FED Id for GT DAQ record
      // default value defined in DataFormats/FEDRawData/src/FEDNumbering.cc
      // default value: assume the DAQ record is the last GT record
      m_daqGtFedId(pSet.getUntrackedParameter<int>("DaqGtFedId", FEDNumbering::MAXTriggerGTPFEDID)),

      // mask for active boards
      m_activeBoardsMaskGt(pSet.getParameter<unsigned int>("ActiveBoardsMask")),

      // EventSetup Tokens
      m_trigScalesToken(esConsumes<L1MuTriggerScales, L1MuTriggerScalesRcd>()),
      m_trigPtScaleToken(esConsumes<L1MuTriggerPtScale, L1MuTriggerPtScaleRcd>()),
      m_l1GtBMToken(esConsumes<L1GtBoardMaps, L1GtBoardMapsRcd>()),

      // number of bunch crossing to be unpacked
      m_unpackBxInEvent(pSet.getParameter<int>("UnpackBxInEvent")),

      // create GTFE, FDL, PSB cards once per producer
      // content will be reset whenever needed

      m_lowSkipBxInEvent(0),
      m_uppSkipBxInEvent(0),

      m_recordLength0(0),
      m_recordLength1(0),

      m_totalBxInEvent(0),

      m_verbosity(pSet.getUntrackedParameter<int>("Verbosity", 0)),

      m_isDebugEnabled(edm::isDebugEnabled())

{
  produces<L1GlobalTriggerReadoutRecord>();
  produces<L1MuGMTReadoutCollection>();

  produces<std::vector<L1MuRegionalCand> >("DT");
  produces<std::vector<L1MuRegionalCand> >("CSC");
  produces<std::vector<L1MuRegionalCand> >("RPCb");
  produces<std::vector<L1MuRegionalCand> >("RPCf");
  produces<std::vector<L1MuGMTCand> >();
  consumes<FEDRawDataCollection>(m_daqGtInputTag);

  // create GTFE, FDL, PSB cards once per producer
  // content will be reset whenever needed
  m_gtfeWord = new L1GtfeWord();
  m_gtFdlWord = new L1GtFdlWord();
  m_gtPsbWord = new L1GtPsbWord();

  if (m_verbosity && m_isDebugEnabled) {
    LogDebug("L1GlobalTriggerRawToDigi") << "\nInput tag for DAQ GT record:             " << m_daqGtInputTag
                                         << "\nFED Id for DAQ GT record:                " << m_daqGtFedId
                                         << "\nMask for active boards (hex format):     " << std::hex
                                         << std::setw(sizeof(m_activeBoardsMaskGt) * 2) << std::setfill('0')
                                         << m_activeBoardsMaskGt << std::dec << std::setfill(' ')
                                         << "\nNumber of bunch crossing to be unpacked: " << m_unpackBxInEvent << "\n"
                                         << std::endl;
  }

  if ((m_unpackBxInEvent > 0) && ((m_unpackBxInEvent % 2) == 0)) {
    m_unpackBxInEvent = m_unpackBxInEvent - 1;

    if (m_verbosity) {
      edm::LogInfo("L1GlobalTriggerRawToDigi")
          << "\nWARNING: Number of bunch crossing to be unpacked rounded to: " << m_unpackBxInEvent
          << "\n         The number must be an odd number!\n"
          << std::endl;
    }
  }
}

// destructor
L1GlobalTriggerRawToDigi::~L1GlobalTriggerRawToDigi() {
  delete m_gtfeWord;
  delete m_gtFdlWord;
  delete m_gtPsbWord;
}

void L1GlobalTriggerRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  static const char* const kComm1 =
      "# input tag for GT readout collection: \n"
      "#     source = hardware record, \n"
      "#     l1GtPack = GT packer (DigiToRaw)";
  desc.add<edm::InputTag>("DaqGtInputTag", edm::InputTag("l1GtPack"))->setComment(kComm1);
  static const char* const kComm2 =
      "# FED Id for GT DAQ record \n"
      "# default value defined in DataFormats/FEDRawData/src/FEDNumbering.cc";
  desc.addUntracked<int>("DaqGtFedId", FEDNumbering::MAXTriggerGTPFEDID)->setComment(kComm2);
  static const char* const kComm3 =
      "# mask for active boards (actually 16 bits) \n"
      "#      if bit is zero, the corresponding board will not be unpacked \n"
      "#      default: no board masked";
  desc.add<unsigned int>("ActiveBoardsMask", 0xFFFF)->setComment(kComm3);
  static const char* const kComm4 =
      "# number of 'bunch crossing in the event' (bxInEvent) to be unpacked \n"
      "# symmetric around L1Accept (bxInEvent = 0): \n"
      "#    1 (bxInEvent = 0); 3 (F 0 1) (standard record); 5 (E F 0 1 2) (debug record) \n"
      "# even numbers (except 0) 'rounded' to the nearest lower odd number \n"
      "# negative value: unpack all available bxInEvent \n"
      "# if more bxInEvent than available are required, unpack what exists and write a warning";
  desc.add<int>("UnpackBxInEvent", -1)->setComment(kComm4);
  desc.addUntracked<int>("Verbosity", 0);
  descriptions.add("l1GlobalTriggerRawToDigi", desc);
}

// member functions

// method called to produce the data
void L1GlobalTriggerRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& evSetup) {
  // get records from EventSetup

  //  muon trigger scales
  edm::ESHandle<L1MuTriggerScales> trigscales_h = evSetup.getHandle(m_trigScalesToken);
  m_TriggerScales = trigscales_h.product();

  edm::ESHandle<L1MuTriggerPtScale> trigptscale_h = evSetup.getHandle(m_trigPtScaleToken);
  m_TriggerPtScale = trigptscale_h.product();

  //  board maps
  edm::ESHandle<L1GtBoardMaps> l1GtBM = evSetup.getHandle(m_l1GtBMToken);

  const std::vector<L1GtBoard> boardMaps = l1GtBM->gtBoardMaps();
  int boardMapsSize = boardMaps.size();

  typedef std::vector<L1GtBoard>::const_iterator CItBoardMaps;

  // create an ordered vector for the GT DAQ record
  // header (pos 0 in record) and trailer (last position in record)
  // not included, as they are not in board list
  std::vector<L1GtBoard> gtRecordMap;
  gtRecordMap.reserve(boardMapsSize);

  for (int iPos = 0; iPos < boardMapsSize; ++iPos) {
    for (CItBoardMaps itBoard = boardMaps.begin(); itBoard != boardMaps.end(); ++itBoard) {
      if (itBoard->gtPositionDaqRecord() == iPos) {
        gtRecordMap.push_back(*itBoard);
        break;
      }
    }
  }

  // raw collection

  edm::Handle<FEDRawDataCollection> fedHandle;
  iEvent.getByLabel(m_daqGtInputTag, fedHandle);

  if (!fedHandle.isValid()) {
    if (m_verbosity) {
      edm::LogWarning("L1GlobalTriggerRawToDigi")
          << "\nWarning: FEDRawDataCollection with input tag " << m_daqGtInputTag
          << "\nrequested in configuration, but not found in the event."
          << "\nQuit unpacking this event" << std::endl;
    }

    produceEmptyProducts(iEvent);

    return;
  }

  // retrieve data for Global Trigger FED (GT + GMT)
  const FEDRawData& raw = (fedHandle.product())->FEDData(m_daqGtFedId);

  int gtSize = raw.size();

  // get a const pointer to the beginning of the data buffer
  const unsigned char* ptrGt = raw.data();

  // get a const pointer to the end of the data buffer
  const unsigned char* endPtrGt = ptrGt + gtSize;

  //
  if (m_verbosity && m_isDebugEnabled) {
    LogTrace("L1GlobalTriggerRawToDigi") << "\n Size of raw data: " << gtSize << "\n" << std::endl;

    std::ostringstream myCoutStream;
    dumpFedRawData(ptrGt, gtSize, myCoutStream);

    LogTrace("L1GlobalTriggerRawToDigi") << "\n Dump FEDRawData\n" << myCoutStream.str() << "\n" << std::endl;
  }

  // unpack header (we have one header only)
  int headerSize = 8;

  if ((ptrGt + headerSize) > endPtrGt) {
    // a common error - no need to print an error anymore
    produceEmptyProducts(iEvent);

    return;
  }

  FEDHeader cmsHeader(ptrGt);
  FEDTrailer cmsTrailer(ptrGt + gtSize - headerSize);

  unpackHeader(ptrGt, cmsHeader);
  ptrGt += headerSize;  // advance with header size

  // unpack first GTFE to find the length of the record and the active boards
  // here GTFE assumed immediately after the header

  // if pointer after GTFE payload is greater than pointer at
  // the end of GT payload, produce empty products and quit unpacking
  if ((ptrGt + m_gtfeWord->getSize()) > endPtrGt) {
    edm::LogError("L1GlobalTriggerRawToDigi") << "\nError: Pointer after GTFE greater than end pointer."
                                              << "\n Put empty products in the event!"
                                              << "\n Quit unpacking this event." << std::endl;

    produceEmptyProducts(iEvent);

    return;
  }

  bool gtfeUnpacked = false;

  for (CItBoardMaps itBoard = boardMaps.begin(); itBoard != boardMaps.end(); ++itBoard) {
    if (itBoard->gtBoardType() == GTFE) {
      // unpack GTFE
      if (itBoard->gtPositionDaqRecord() == 1) {
        m_gtfeWord->unpack(ptrGt);
        ptrGt += m_gtfeWord->getSize();  // advance with GTFE block size
        gtfeUnpacked = true;

        if (m_verbosity && m_isDebugEnabled) {
          std::ostringstream myCoutStream;
          m_gtfeWord->print(myCoutStream);
          LogTrace("L1GlobalTriggerRawToDigi") << myCoutStream.str() << "\n" << std::endl;
        }

        // break the loop - GTFE was found
        break;

      } else {
        if (m_verbosity) {
          edm::LogWarning("L1GlobalTriggerRawToDigi")
              << "\nWarning: GTFE block found in raw data does not follow header."
              << "\nAssumed start position of the block is wrong!"
              << "\nQuit unpacking this event" << std::endl;
        }

        produceEmptyProducts(iEvent);

        return;
      }
    }
  }

  // quit if no GTFE found
  if (!gtfeUnpacked) {
    if (m_verbosity) {
      edm::LogWarning("L1GlobalTriggerRawToDigi")
          << "\nWarning: no GTFE block found in raw data."
          << "\nCan not find the record length (BxInEvent) and the active boards!"
          << "\nQuit unpacking this event" << std::endl;
    }

    produceEmptyProducts(iEvent);

    return;
  }

  // life normal here, GTFE found

  // get list of active blocks
  // blocks not active are not written to the record
  cms_uint16_t activeBoardsGtInitial = m_gtfeWord->activeBoards();
  cms_uint16_t altNrBxBoardInitial = m_gtfeWord->altNrBxBoard();

  // mask some boards, if needed
  cms_uint16_t activeBoardsGt = activeBoardsGtInitial & m_activeBoardsMaskGt;
  m_gtfeWord->setActiveBoards(activeBoardsGt);

  if (m_verbosity) {
    LogDebug("L1GlobalTriggerRawToDigi") << "\nActive boards before masking: 0x" << std::hex
                                         << std::setw(sizeof(activeBoardsGtInitial) * 2) << std::setfill('0')
                                         << activeBoardsGtInitial << std::dec << std::setfill(' ')
                                         << "\nActive boards after masking:  0x" << std::hex
                                         << std::setw(sizeof(activeBoardsGt) * 2) << std::setfill('0') << activeBoardsGt
                                         << std::dec << std::setfill(' ') << " \n"
                                         << std::endl;
  }

  // loop over other blocks in the raw record, count them if they are active

  int numberFdlBoards = 0;
  int numberPsbBoards = 0;

  for (CItBoardMaps itBoard = boardMaps.begin(); itBoard != boardMaps.end(); ++itBoard) {
    int iActiveBit = itBoard->gtBitDaqActiveBoards();
    bool activeBoardToUnpack = false;

    if (iActiveBit >= 0) {
      activeBoardToUnpack = activeBoardsGt & (1 << iActiveBit);
    } else {
      // board not in the ActiveBoards for the record
      continue;
    }

    if (activeBoardToUnpack) {
      switch (itBoard->gtBoardType()) {
        case GTFE:
          break;
        case FDL: {
          numberFdlBoards++;
        }

        break;
        case PSB: {
          numberPsbBoards++;
        }

        break;
        case GMT:
          break;
        case TCS:
          break;
        case TIM:
          break;
        default: {
          // do nothing, all blocks are given in GtBoardType enum
          if (m_verbosity) {
            LogDebug("L1GlobalTriggerRawToDigi")
                << "\nBoard of type " << itBoard->gtBoardType() << " not expected  in record.\n"
                << std::endl;
          }
        }

        break;
      }
    }
  }

  // produce the L1GlobalTriggerReadoutRecord now, after we found the maximum number of
  // BxInEvent the record has and how many boards are active (it is just reserving space
  // for vectors)
  //LogDebug("L1GlobalTriggerRawToDigi")
  //<< "\nL1GlobalTriggerRawToDigi: producing L1GlobalTriggerReadoutRecord\n"
  //<< "\nL1GlobalTriggerRawToDigi: producing L1MuGMTReadoutCollection;\n"
  //<< std::endl;

  // get number of Bx in the event from GTFE block corresponding to alternative 0 and 1 in
  m_recordLength0 = m_gtfeWord->recordLength();
  m_recordLength1 = m_gtfeWord->recordLength1();

  int maxBxInEvent = std::max(m_recordLength0, m_recordLength1);

  std::unique_ptr<L1GlobalTriggerReadoutRecord> gtReadoutRecord(
      new L1GlobalTriggerReadoutRecord(maxBxInEvent, numberFdlBoards, numberPsbBoards));

  // produce also the GMT readout collection and set the reference in GT record
  std::unique_ptr<L1MuGMTReadoutCollection> gmtrc(new L1MuGMTReadoutCollection(maxBxInEvent));

  //edm::RefProd<L1MuGMTReadoutCollection> refProdMuGMT = iEvent.getRefBeforePut<
  //        L1MuGMTReadoutCollection> ();

  //if (m_verbosity) {
  //    LogDebug("L1GlobalTriggerRawToDigi")
  //            << "\nL1GlobalTriggerRawToDigi: set L1MuGMTReadoutCollection RefProd"
  //            << " in L1GlobalTriggerReadoutRecord.\n" << std::endl;
  //}
  //gtReadoutRecord->setMuCollectionRefProd(refProdMuGMT);

  // ... then unpack modules other than GTFE, if requested

  for (CItBoardMaps itBoard = gtRecordMap.begin(); itBoard != gtRecordMap.end(); ++itBoard) {
    int iActiveBit = itBoard->gtBitDaqActiveBoards();

    bool activeBoardToUnpack = false;
    bool activeBoardInitial = false;

    int altNrBxBoardVal = -1;

    if (iActiveBit >= 0) {
      activeBoardInitial = activeBoardsGtInitial & (1 << iActiveBit);
      activeBoardToUnpack = activeBoardsGt & (1 << iActiveBit);

      altNrBxBoardVal = (altNrBxBoardInitial & (1 << iActiveBit)) >> iActiveBit;

      if (altNrBxBoardVal == 1) {
        m_totalBxInEvent = m_recordLength1;
      } else if (altNrBxBoardVal == 0) {
        m_totalBxInEvent = m_recordLength0;
      } else {
        if (m_verbosity) {
          edm::LogWarning("L1GlobalTriggerRawToDigi")
              << "\n\nWARNING: Wrong value altNrBxBoardVal = " << altNrBxBoardVal << " for board " << std::hex
              << (itBoard->gtBoardId()) << std::dec << "\n  iActiveBit =            " << iActiveBit
              << "\n  altNrBxBoardInitial = 0x" << std::hex << altNrBxBoardInitial << std::dec
              << "\n  activeBoardsGt =      0x" << std::hex << activeBoardsGt << std::dec
              << "\n  activeBoardInitial =    " << activeBoardInitial
              << "\n  activeBoardToUnpack =   " << activeBoardToUnpack << "\n Set altNrBxBoardVal tentatively to "
              << m_recordLength0 << "\n Job may crash or produce wrong results!\n\n"
              << std::endl;
        }

        m_totalBxInEvent = m_recordLength0;
      }

      // number of BX required to be unpacked

      if (m_unpackBxInEvent > m_totalBxInEvent) {
        if (m_verbosity) {
          LogDebug("L1GlobalTriggerRawToDigi")
              << "\nWARNING: Number of available bunch crosses for board" << (itBoard->gtBoardId())
              << " in the record ( " << m_totalBxInEvent
              << " ) \n is smaller than the number of bunch crosses requested to be unpacked (" << m_unpackBxInEvent
              << " )!!! \n         Unpacking only " << m_totalBxInEvent << " bunch crosses.\n"
              << std::endl;
        }

        m_lowSkipBxInEvent = 0;
        m_uppSkipBxInEvent = m_totalBxInEvent;

      } else if (m_unpackBxInEvent < 0) {
        m_lowSkipBxInEvent = 0;
        m_uppSkipBxInEvent = m_totalBxInEvent;

        if (m_verbosity) {
          LogDebug("L1GlobalTriggerRawToDigi") << "\nUnpacking all " << m_totalBxInEvent << " bunch crosses available."
                                               << "\n"
                                               << std::endl;
        }

      } else if (m_unpackBxInEvent == 0) {
        m_lowSkipBxInEvent = m_totalBxInEvent;
        m_uppSkipBxInEvent = m_totalBxInEvent;

        if (m_verbosity) {
          LogDebug("L1GlobalTriggerRawToDigi")
              << "\nNo bxInEvent required to be unpacked from " << m_totalBxInEvent << " bunch crosses available."
              << "\n"
              << std::endl;
        }

        // change RecordLength
        // cast int to cms_uint16_t (there are normally 3 or 5 BxInEvent)
        m_gtfeWord->setRecordLength(static_cast<cms_uint16_t>(m_unpackBxInEvent));
        m_gtfeWord->setRecordLength1(static_cast<cms_uint16_t>(m_unpackBxInEvent));

      } else {
        m_lowSkipBxInEvent = (m_totalBxInEvent - m_unpackBxInEvent) / 2;
        m_uppSkipBxInEvent = m_totalBxInEvent - m_lowSkipBxInEvent;

        if (m_verbosity) {
          LogDebug("L1GlobalTriggerRawToDigi") << "\nUnpacking " << m_unpackBxInEvent << " bunch crosses from "
                                               << m_totalBxInEvent << " bunch crosses available."
                                               << "\n"
                                               << std::endl;
        }

        // change RecordLength
        // cast int to cms_uint16_t (there are normally 3 or 5 BxInEvent)
        m_gtfeWord->setRecordLength(static_cast<cms_uint16_t>(m_unpackBxInEvent));
        m_gtfeWord->setRecordLength1(static_cast<cms_uint16_t>(m_unpackBxInEvent));
      }

    } else {
      // board not in the ActiveBoards for the record
      continue;
    }

    if (!activeBoardInitial) {
      if (m_verbosity) {
        LogDebug("L1GlobalTriggerRawToDigi") << "\nBoard of type " << itBoard->gtBoardName() << " with index "
                                             << itBoard->gtBoardIndex() << " not active initially in raw data.\n"
                                             << std::endl;
      }
      continue;
    }

    // active board initially, could unpack it
    switch (itBoard->gtBoardType()) {
      case FDL: {
        for (int iFdl = 0; iFdl < m_totalBxInEvent; ++iFdl) {
          // if pointer after FDL payload is greater than pointer at
          // the end of GT payload, produce empty products and quit unpacking
          if ((ptrGt + m_gtFdlWord->getSize()) > endPtrGt) {
            edm::LogError("L1GlobalTriggerRawToDigi")
                << "\nError: Pointer after FDL " << iFdl << " greater than end pointer."
                << "\n Put empty products in the event!"
                << "\n Quit unpacking this event." << std::endl;

            produceEmptyProducts(iEvent);

            return;
          }

          // unpack only if requested, otherwise skip it
          if (activeBoardToUnpack) {
            // unpack only bxInEvent requested, otherwise skip it
            if ((iFdl >= m_lowSkipBxInEvent) && (iFdl < m_uppSkipBxInEvent)) {
              m_gtFdlWord->unpack(ptrGt);

              // add 1 to the GT luminosity number to use the same convention as
              // offline, where LS number starts with 1;
              // in GT hardware, LS starts with 0
              cms_uint16_t lsNr = m_gtFdlWord->lumiSegmentNr() + 1;
              m_gtFdlWord->setLumiSegmentNr(lsNr);

              // add FDL block to GT readout record
              gtReadoutRecord->setGtFdlWord(*m_gtFdlWord);

              if (m_verbosity && m_isDebugEnabled) {
                std::ostringstream myCoutStream;
                m_gtFdlWord->print(myCoutStream);
                LogTrace("L1GlobalTriggerRawToDigi") << myCoutStream.str() << "\n" << std::endl;
              }

              // ... and reset it
              m_gtFdlWord->reset();
            }
          }

          ptrGt += m_gtFdlWord->getSize();  // advance with FDL block size
        }
      }

      break;
      case PSB: {
        for (int iPsb = 0; iPsb < m_totalBxInEvent; ++iPsb) {
          // if pointer after PSB payload is greater than pointer at
          // the end of GT payload, produce empty products and quit unpacking
          if ((ptrGt + m_gtPsbWord->getSize()) > endPtrGt) {
            edm::LogError("L1GlobalTriggerRawToDigi")
                << "\nError: Pointer after PSB " << iPsb << " greater than end pointer."
                << "\n Put empty products in the event!"
                << "\n Quit unpacking this event." << std::endl;

            produceEmptyProducts(iEvent);

            return;
          }

          // unpack only if requested, otherwise skip it
          if (activeBoardToUnpack) {
            // unpack only bxInEvent requested, otherwise skip it
            if ((iPsb >= m_lowSkipBxInEvent) && (iPsb < m_uppSkipBxInEvent)) {
              unpackPSB(evSetup, ptrGt, *m_gtPsbWord);

              // add PSB block to GT readout record
              gtReadoutRecord->setGtPsbWord(*m_gtPsbWord);

              if (m_verbosity && m_isDebugEnabled) {
                std::ostringstream myCoutStream;
                m_gtPsbWord->print(myCoutStream);
                LogTrace("L1GlobalTriggerRawToDigi") << myCoutStream.str() << "\n" << std::endl;
              }

              // ... and reset it
              m_gtPsbWord->reset();
            }
          }

          ptrGt += m_gtPsbWord->getSize();  // advance with PSB block size
        }
      } break;
      case GMT: {
        // 17*64/8 TODO FIXME ask Ivan for a getSize() function for GMT record
        unsigned int gmtRecordSize = 136;
        unsigned int gmtCollSize = m_totalBxInEvent * gmtRecordSize;

        // if pointer after GMT payload is greater than pointer at
        // the end of GT payload, produce empty products and quit unpacking
        if ((ptrGt + gmtCollSize) > endPtrGt) {
          edm::LogError("L1GlobalTriggerRawToDigi") << "\nError: Pointer after GMT "
                                                    << " greater than end pointer."
                                                    << "\n Put empty products in the event!"
                                                    << "\n Quit unpacking this event." << std::endl;

          produceEmptyProducts(iEvent);

          return;
        }

        // unpack only if requested, otherwise skip it
        if (activeBoardToUnpack) {
          unpackGMT(ptrGt, gmtrc, iEvent);
        }

        ptrGt += gmtCollSize;  // advance with GMT block size
      } break;
      default: {
        // do nothing, all blocks are given in GtBoardType enum
        if (m_verbosity) {
          LogDebug("L1GlobalTriggerRawToDigi")
              << "\nBoard of type " << itBoard->gtBoardType() << " not expected  in record.\n"
              << std::endl;
        }
      } break;
    }
  }

  // add GTFE block to GT readout record, after updating active boards and record length

  gtReadoutRecord->setGtfeWord(*m_gtfeWord);

  // ... and reset it
  m_gtfeWord->reset();

  // unpack trailer

  int trailerSize = 8;

  // if pointer after trailer is greater than pointer at
  // the end of GT payload, produce empty products and quit unpacking
  if ((ptrGt + trailerSize) > endPtrGt) {
    edm::LogError("L1GlobalTriggerRawToDigi") << "\nError: Pointer after trailer "
                                              << " greater than end pointer."
                                              << "\n Put empty products in the event!"
                                              << "\n Quit unpacking this event." << std::endl;

    produceEmptyProducts(iEvent);

    return;
  }

  unpackTrailer(ptrGt, cmsTrailer);

  //
  if (m_verbosity && m_isDebugEnabled) {
    std::ostringstream myCoutStream;
    gtReadoutRecord->print(myCoutStream);
    LogTrace("L1GlobalTriggerRawToDigi") << "\n The following L1 GT DAQ readout record was unpacked.\n"
                                         << myCoutStream.str() << "\n"
                                         << std::endl;
  }

  // put records into event

  iEvent.put(std::move(gmtrc));
  iEvent.put(std::move(gtReadoutRecord));
}

// unpack header
void L1GlobalTriggerRawToDigi::unpackHeader(const unsigned char* gtPtr, FEDHeader& cmsHeader) {
  // TODO  if needed in another format

  // print the header info
  if (m_verbosity && m_isDebugEnabled) {
    const cms_uint64_t* payload = reinterpret_cast<cms_uint64_t*>(const_cast<unsigned char*>(gtPtr));

    std::ostringstream myCoutStream;

    // one word only
    int iWord = 0;

    myCoutStream << std::setw(4) << iWord << "  " << std::hex << std::setfill('0') << std::setw(16) << payload[iWord]
                 << std::dec << std::setfill(' ') << "\n"
                 << std::endl;

    myCoutStream << "  Event_type:  " << std::hex << " hex: "
                 << "     " << std::setw(1) << std::setfill('0') << cmsHeader.triggerType() << std::setfill(' ')
                 << std::dec << " dec: " << cmsHeader.triggerType() << std::endl;

    myCoutStream << "  LVL1_Id:     " << std::hex << " hex: "
                 << "" << std::setw(6) << std::setfill('0') << cmsHeader.lvl1ID() << std::setfill(' ') << std::dec
                 << " dec: " << cmsHeader.lvl1ID() << std::endl;

    myCoutStream << "  BX_Id:       " << std::hex << " hex: "
                 << "   " << std::setw(3) << std::setfill('0') << cmsHeader.bxID() << std::setfill(' ') << std::dec
                 << " dec: " << cmsHeader.bxID() << std::endl;

    myCoutStream << "  Source_Id:   " << std::hex << " hex: "
                 << "   " << std::setw(3) << std::setfill('0') << cmsHeader.sourceID() << std::setfill(' ') << std::dec
                 << " dec: " << cmsHeader.sourceID() << std::endl;

    myCoutStream << "  FOV:         " << std::hex << " hex: "
                 << "     " << std::setw(1) << std::setfill('0') << cmsHeader.version() << std::setfill(' ') << std::dec
                 << " dec: " << cmsHeader.version() << std::endl;

    myCoutStream << "  H:           " << std::hex << " hex: "
                 << "     " << std::setw(1) << std::setfill('0') << cmsHeader.moreHeaders() << std::setfill(' ')
                 << std::dec << " dec: " << cmsHeader.moreHeaders() << std::endl;

    LogDebug("L1GlobalTriggerRawToDigi") << "\n CMS Header \n" << myCoutStream.str() << "\n" << std::endl;
  }
}

// unpack PSB records
// psbPtr pointer to the beginning of the each PSB block obtained from gtPtr
void L1GlobalTriggerRawToDigi::unpackPSB(const edm::EventSetup& evSetup,
                                         const unsigned char* psbPtr,
                                         L1GtPsbWord& psbWord) {
  //LogDebug("L1GlobalTriggerRawToDigi")
  //<< "\nUnpacking PSB block.\n"
  //<< std::endl;

  int uLength = L1GlobalTriggerReadoutSetup::UnitLength;

  int psbSize = psbWord.getSize();
  int psbWords = psbSize / uLength;

  const cms_uint64_t* payload = reinterpret_cast<cms_uint64_t*>(const_cast<unsigned char*>(psbPtr));

  for (int iWord = 0; iWord < psbWords; ++iWord) {
    // fill PSB
    // the second argument must match the word index defined in L1GtPsbWord class

    psbWord.setBoardId(payload[iWord], iWord);
    psbWord.setBxInEvent(payload[iWord], iWord);
    psbWord.setBxNr(payload[iWord], iWord);
    psbWord.setEventNr(payload[iWord], iWord);

    psbWord.setAData(payload[iWord], iWord);
    psbWord.setBData(payload[iWord], iWord);

    psbWord.setLocalBxNr(payload[iWord], iWord);

    LogTrace("L1GlobalTriggerRawToDigi") << std::setw(4) << iWord << "  " << std::hex << std::setfill('0')
                                         << std::setw(16) << payload[iWord] << std::dec << std::setfill(' ')
                                         << std::endl;
  }
}

// unpack the GMT record
void L1GlobalTriggerRawToDigi::unpackGMT(const unsigned char* chp,
                                         std::unique_ptr<L1MuGMTReadoutCollection>& gmtrc,
                                         edm::Event& iEvent) {
  //LogDebug("L1GlobalTriggerRawToDigi")
  //<< "\nUnpacking GMT collection.\n"
  //<< std::endl;

  // 17*64/2 TODO FIXME ask Ivan for a getSize() function for GMT record
  const unsigned int gmtRecordSize32 = 34;

  std::unique_ptr<std::vector<L1MuRegionalCand> > DTCands(new std::vector<L1MuRegionalCand>);
  std::unique_ptr<std::vector<L1MuRegionalCand> > CSCCands(new std::vector<L1MuRegionalCand>);
  std::unique_ptr<std::vector<L1MuRegionalCand> > RPCbCands(new std::vector<L1MuRegionalCand>);
  std::unique_ptr<std::vector<L1MuRegionalCand> > RPCfCands(new std::vector<L1MuRegionalCand>);
  std::unique_ptr<std::vector<L1MuGMTCand> > GMTCands(new std::vector<L1MuGMTCand>);

  const unsigned* p = (const unsigned*)chp;

  // min Bx's in the event, computed after m_totalBxInEvent is obtained from GTFE block
  // assume symmetrical number of BX around L1Accept
  int iBxInEvent = (m_totalBxInEvent + 1) / 2 - m_totalBxInEvent;

  for (int iGmtRec = 0; iGmtRec < m_totalBxInEvent; ++iGmtRec) {
    // unpack only bxInEvent requested, otherwise skip it
    if ((iGmtRec >= m_lowSkipBxInEvent) && (iGmtRec < m_uppSkipBxInEvent)) {
      // Dump the block
      const cms_uint64_t* bp = reinterpret_cast<cms_uint64_t*>(const_cast<unsigned*>(p));
      for (int iWord = 0; iWord < 17; iWord++) {
        LogTrace("L1GlobalTriggerRawToDigi") << std::setw(4) << iWord << "  " << std::hex << std::setfill('0')
                                             << std::setw(16) << *bp++ << std::dec << std::setfill(' ') << std::endl;
      }

      L1MuGMTReadoutRecord gmtrr(iBxInEvent);

      gmtrr.setEvNr((*p) & 0xffffff);
      gmtrr.setBCERR(((*p) >> 24) & 0xff);
      p++;

      gmtrr.setBxNr((*p) & 0xfff);
      if (((*p) >> 15) & 1) {
        gmtrr.setBxInEvent((((*p) >> 12) & 7) - 8);
      } else {
        gmtrr.setBxInEvent((((*p) >> 12) & 7));
      }
      // to do: check here the block length and the board id
      p++;

      for (int im = 0; im < 16; im++) {
        // flip the pt and quality bits -- this should better be done by GMT input chips
        unsigned waux = *p++;
        waux = (waux & 0xffff00ff) | ((~waux) & 0x0000ff00);
        L1MuRegionalCand cand(waux, iBxInEvent);
        // fix the type assignment (csc=2, rpcb=1) -- should be done by GMT input chips
        if (im >= 4 && im < 8)
          cand.setType(1);
        if (im >= 8 && im < 12)
          cand.setType(2);
        cand.setPhiValue(m_TriggerScales->getPhiScale()->getLowEdge(cand.phi_packed()));
        cand.setEtaValue(m_TriggerScales->getRegionalEtaScale(cand.type_idx())->getCenter(cand.eta_packed()));
        cand.setPtValue(m_TriggerPtScale->getPtScale()->getLowEdge(cand.pt_packed()));
        gmtrr.setInputCand(im, cand);
        if (!cand.empty()) {
          if (im < 4)
            DTCands->push_back(cand);
          if (im >= 4 && im < 8)
            RPCbCands->push_back(cand);
          if (im >= 8 && im < 12)
            CSCCands->push_back(cand);
          if (im >= 12)
            RPCfCands->push_back(cand);
        }
      }

      unsigned char* prank = (unsigned char*)(p + 12);

      for (int im = 0; im < 12; im++) {
        unsigned waux = *p++;
        unsigned raux = im < 8 ? *prank++ : 0;  // only fwd and brl cands have valid rank
        L1MuGMTExtendedCand cand(waux, raux, iBxInEvent);
        cand.setPhiValue(m_TriggerScales->getPhiScale()->getLowEdge(cand.phiIndex()));
        cand.setEtaValue(m_TriggerScales->getGMTEtaScale()->getCenter(cand.etaIndex()));
        cand.setPtValue(m_TriggerPtScale->getPtScale()->getLowEdge(cand.ptIndex()));
        if (im < 4)
          gmtrr.setGMTBrlCand(im, cand);
        else if (im < 8)
          gmtrr.setGMTFwdCand(im - 4, cand);
        else {
          gmtrr.setGMTCand(im - 8, cand);
          if (!cand.empty())
            GMTCands->push_back(cand);
        }
      }

      // skip the two sort rank words and two chip BX words
      p += 4;

      gmtrc->addRecord(gmtrr);

    } else {
      // increase the pointer with the GMT record size
      p += gmtRecordSize32;
    }

    // increase the BxInEvent number
    iBxInEvent++;
  }

  iEvent.put(std::move(DTCands), "DT");
  iEvent.put(std::move(CSCCands), "CSC");
  iEvent.put(std::move(RPCbCands), "RPCb");
  iEvent.put(std::move(RPCfCands), "RPCf");
  iEvent.put(std::move(GMTCands));
}

// unpack trailer word
// trPtr pointer to the beginning of trailer obtained from gtPtr
void L1GlobalTriggerRawToDigi::unpackTrailer(const unsigned char* trlPtr, FEDTrailer& cmsTrailer) {
  // TODO  if needed in another format

  // print the trailer info
  if (m_verbosity && m_isDebugEnabled) {
    const cms_uint64_t* payload = reinterpret_cast<cms_uint64_t*>(const_cast<unsigned char*>(trlPtr));

    std::ostringstream myCoutStream;

    // one word only
    int iWord = 0;

    myCoutStream << std::setw(4) << iWord << "  " << std::hex << std::setfill('0') << std::setw(16) << payload[iWord]
                 << std::dec << std::setfill(' ') << "\n"
                 << std::endl;

    myCoutStream << "  Event_length:  " << std::hex << " hex: "
                 << "" << std::setw(6) << std::setfill('0') << cmsTrailer.fragmentLength() << std::setfill(' ')
                 << std::dec << " dec: " << cmsTrailer.fragmentLength() << std::endl;

    myCoutStream << "  CRC:           " << std::hex << " hex: "
                 << "  " << std::setw(4) << std::setfill('0') << cmsTrailer.crc() << std::setfill(' ') << std::dec
                 << " dec: " << cmsTrailer.crc() << std::endl;

    myCoutStream << "  Event_status:  " << std::hex << " hex: "
                 << "    " << std::setw(2) << std::setfill('0') << cmsTrailer.evtStatus() << std::setfill(' ')
                 << std::dec << " dec: " << cmsTrailer.evtStatus() << std::endl;

    myCoutStream << "  TTS_bits:      " << std::hex << " hex: "
                 << "     " << std::setw(1) << std::setfill('0') << cmsTrailer.ttsBits() << std::setfill(' ')
                 << std::dec << " dec: " << cmsTrailer.ttsBits() << std::endl;

    myCoutStream << "  More trailers: " << std::hex << " hex: "
                 << "     " << std::setw(1) << std::setfill('0') << cmsTrailer.moreTrailers() << std::setfill(' ')
                 << std::dec << " dec: " << cmsTrailer.moreTrailers() << std::endl;

    LogDebug("L1GlobalTriggerRawToDigi") << "\n CMS Trailer \n" << myCoutStream.str() << "\n" << std::endl;
  }
}

// produce empty products in case of problems
void L1GlobalTriggerRawToDigi::produceEmptyProducts(edm::Event& iEvent) {
  std::unique_ptr<L1GlobalTriggerReadoutRecord> gtReadoutRecord(new L1GlobalTriggerReadoutRecord());

  std::unique_ptr<L1MuGMTReadoutCollection> gmtrc(new L1MuGMTReadoutCollection());

  std::unique_ptr<std::vector<L1MuRegionalCand> > DTCands(new std::vector<L1MuRegionalCand>);
  std::unique_ptr<std::vector<L1MuRegionalCand> > CSCCands(new std::vector<L1MuRegionalCand>);
  std::unique_ptr<std::vector<L1MuRegionalCand> > RPCbCands(new std::vector<L1MuRegionalCand>);
  std::unique_ptr<std::vector<L1MuRegionalCand> > RPCfCands(new std::vector<L1MuRegionalCand>);
  std::unique_ptr<std::vector<L1MuGMTCand> > GMTCands(new std::vector<L1MuGMTCand>);

  // put empty records into event

  iEvent.put(std::move(gmtrc));
  iEvent.put(std::move(gtReadoutRecord));

  iEvent.put(std::move(DTCands), "DT");
  iEvent.put(std::move(CSCCands), "CSC");
  iEvent.put(std::move(RPCbCands), "RPCb");
  iEvent.put(std::move(RPCfCands), "RPCf");
  iEvent.put(std::move(GMTCands));
}

// dump FED raw data
void L1GlobalTriggerRawToDigi::dumpFedRawData(const unsigned char* gtPtr, int gtSize, std::ostream& myCout) {
  LogDebug("L1GlobalTriggerRawToDigi") << "\nDump FED raw data.\n" << std::endl;

  int wLength = L1GlobalTriggerReadoutSetup::WordLength;
  int uLength = L1GlobalTriggerReadoutSetup::UnitLength;

  int gtWords = gtSize / uLength;
  LogTrace("L1GlobalTriggerRawToDigi") << "\nFED GT words (" << wLength << " bits):" << gtWords << "\n" << std::endl;

  const cms_uint64_t* payload = reinterpret_cast<cms_uint64_t*>(const_cast<unsigned char*>(gtPtr));

  for (unsigned int i = 0; i < gtSize / sizeof(cms_uint64_t); i++) {
    myCout << std::setw(4) << i << "  " << std::hex << std::setfill('0') << std::setw(16) << payload[i] << std::dec
           << std::setfill(' ') << std::endl;
  }
}

// static class members
