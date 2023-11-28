/**
 * \class L1GlobalTriggerEvmRawToDigi
 *
 *
 * Description: unpack raw data into digitized data.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// this class header
#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GlobalTriggerEvmRawToDigi.h"

// system include files
#include <iostream>
#include <iomanip>
#include <algorithm>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeExtWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1TcsWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtBoard.h"

// constructor(s)
L1GlobalTriggerEvmRawToDigi::L1GlobalTriggerEvmRawToDigi(const edm::ParameterSet& pSet)
    :

      // input tag for EVM GT record
      m_evmGtInputTag(pSet.getParameter<edm::InputTag>("EvmGtInputTag")),

      // FED Id for GT EVM record
      // default value defined in DataFormats/FEDRawData/src/FEDNumbering.cc
      // default value: assume the EVM record is the first GT record
      m_evmGtFedId(pSet.getUntrackedParameter<int>("EvmGtFedId", FEDNumbering::MINTriggerGTPFEDID)),

      /// EventSetup Token for L1GtBoardMaps
      m_l1GtBMToken(esConsumes<L1GtBoardMaps, L1GtBoardMapsRcd>()),

      // mask for active boards
      m_activeBoardsMaskGt(pSet.getParameter<unsigned int>("ActiveBoardsMask")),

      // number of bunch crossing to be unpacked
      m_unpackBxInEvent(pSet.getParameter<int>("UnpackBxInEvent")),

      m_lowSkipBxInEvent(0),
      m_uppSkipBxInEvent(0),

      m_recordLength0(0),
      m_recordLength1(0),

      m_totalBxInEvent(0),

      // length of BST record (in bytes)
      m_bstLengthBytes(pSet.getParameter<int>("BstLengthBytes")),

      m_verbosity(pSet.getUntrackedParameter<int>("Verbosity", 0)),

      m_isDebugEnabled(edm::isDebugEnabled())

{
  produces<L1GlobalTriggerEvmReadoutRecord>();

  if (m_verbosity && m_isDebugEnabled) {
    LogDebug("L1GlobalTriggerEvmRawToDigi")
        << "\nInput tag for EVM GT record:             " << m_evmGtInputTag
        << "\nFED Id for EVM GT record:                " << m_evmGtFedId
        << "\nMask for active boards (hex format):     " << std::hex << std::setw(sizeof(m_activeBoardsMaskGt) * 2)
        << std::setfill('0') << m_activeBoardsMaskGt << std::dec << std::setfill(' ')
        << "\nNumber of bunch crossing to be unpacked: " << m_unpackBxInEvent
        << "\nLength of BST message [bytes]:           " << m_bstLengthBytes << "\n"
        << std::endl;
  }

  if ((m_unpackBxInEvent > 0) && ((m_unpackBxInEvent % 2) == 0)) {
    m_unpackBxInEvent = m_unpackBxInEvent - 1;

    if (m_verbosity) {
      edm::LogInfo("L1GlobalTriggerEvmRawToDigi")
          << "\nWARNING: Number of bunch crossing to be unpacked rounded to: " << m_unpackBxInEvent
          << "\n         The number must be an odd number!\n"
          << std::endl;
    }
  }

  // create GTFE, TCS, FDL cards once per analyzer
  // content will be reset whenever needed

  m_gtfeWord = new L1GtfeExtWord();
  m_tcsWord = new L1TcsWord();
  m_gtFdlWord = new L1GtFdlWord();
  consumes<FEDRawDataCollection>(m_evmGtInputTag);

  /// EventSetup Token for L1GtParameters
  if (m_bstLengthBytes < 0) {
    m_l1GtParamToken = esConsumes<L1GtParameters, L1GtParametersRcd>();
  }
}

// destructor
L1GlobalTriggerEvmRawToDigi::~L1GlobalTriggerEvmRawToDigi() {
  delete m_gtfeWord;
  delete m_tcsWord;
  delete m_gtFdlWord;
}

// member functions

// method called to produce the data
void L1GlobalTriggerEvmRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& evSetup) {
  // get records from EventSetup

  //  board maps
  edm::ESHandle<L1GtBoardMaps> l1GtBM = evSetup.getHandle(m_l1GtBMToken);

  const std::vector<L1GtBoard> boardMaps = l1GtBM->gtBoardMaps();
  int boardMapsSize = boardMaps.size();

  typedef std::vector<L1GtBoard>::const_iterator CItBoardMaps;

  // create an ordered vector for the GT EVM record
  // header (pos 0 in record) and trailer (last position in record)
  // not included, as they are not in board list
  std::vector<L1GtBoard> gtRecordMap;
  gtRecordMap.reserve(boardMapsSize);

  for (int iPos = 0; iPos < boardMapsSize; ++iPos) {
    for (CItBoardMaps itBoard = boardMaps.begin(); itBoard != boardMaps.end(); ++itBoard) {
      if (itBoard->gtPositionEvmRecord() == iPos) {
        gtRecordMap.push_back(*itBoard);
        break;
      }
    }
  }

  // raw collection

  edm::Handle<FEDRawDataCollection> fedHandle;
  iEvent.getByLabel(m_evmGtInputTag, fedHandle);

  if (!fedHandle.isValid()) {
    if (m_verbosity) {
      edm::LogWarning("L1GlobalTriggerEvmRawToDigi")
          << "\nWarning: FEDRawDataCollection with input tag " << m_evmGtInputTag
          << "\nrequested in configuration, but not found in the event."
          << "\nQuit unpacking this event" << std::endl;
    }

    produceEmptyProducts(iEvent);

    return;
  }

  // retrieve data for Global Trigger EVM FED
  const FEDRawData& raw = (fedHandle.product())->FEDData(m_evmGtFedId);

  int gtSize = raw.size();

  // get a const pointer to the beginning of the data buffer
  const unsigned char* ptrGt = raw.data();

  // get a const pointer to the end of the data buffer
  const unsigned char* endPtrGt = ptrGt + gtSize;

  //
  if (m_verbosity && m_isDebugEnabled) {
    LogTrace("L1GlobalTriggerEvmRawToDigi") << "\n Size of raw data: " << gtSize << "\n" << std::endl;

    std::ostringstream myCoutStream;
    dumpFedRawData(ptrGt, gtSize, myCoutStream);

    LogTrace("L1GlobalTriggerEvmRawToDigi") << "\n Dump FEDRawData\n" << myCoutStream.str() << "\n" << std::endl;
  }

  // unpack header
  int headerSize = 8;

  if ((ptrGt + headerSize) > endPtrGt) {
    edm::LogError("L1GlobalTriggerEvmRawToDigi") << "\nError: Pointer after header greater than end pointer."
                                                 << "\n Put empty products in the event!"
                                                 << "\n Quit unpacking this event." << std::endl;

    produceEmptyProducts(iEvent);

    return;
  }

  FEDHeader cmsHeader(ptrGt);
  FEDTrailer cmsTrailer(ptrGt + gtSize - headerSize);

  unpackHeader(ptrGt, cmsHeader);
  ptrGt += headerSize;  // advance with header size

  // unpack first GTFE to find the length of the record and the active boards
  // here GTFE assumed immediately after the header

  bool gtfeUnpacked = false;

  // get the length of the BST message from parameter set or from event setup

  int bstLengthBytes = 0;

  if (m_bstLengthBytes < 0) {
    // length from event setup // TODO cache it, if too slow

    edm::ESHandle<L1GtParameters> l1GtPar = evSetup.getHandle(m_l1GtParamToken);
    const L1GtParameters* m_l1GtPar = l1GtPar.product();

    bstLengthBytes = static_cast<int>(m_l1GtPar->gtBstLengthBytes());

  } else {
    // length from parameter set
    bstLengthBytes = m_bstLengthBytes;
  }

  if (m_verbosity) {
    LogTrace("L1GlobalTriggerEvmRawToDigi") << "\n Length of BST message (in bytes): " << bstLengthBytes << "\n"
                                            << std::endl;
  }

  for (CItBoardMaps itBoard = boardMaps.begin(); itBoard != boardMaps.end(); ++itBoard) {
    if (itBoard->gtBoardType() == GTFE) {
      // unpack GTFE
      if (itBoard->gtPositionEvmRecord() == 1) {
        // resize to the right size before unapacking
        m_gtfeWord->resize(bstLengthBytes);

        m_gtfeWord->unpack(ptrGt);
        ptrGt += m_gtfeWord->getSize();  // advance with GTFE block size
        gtfeUnpacked = true;

        if (m_verbosity && m_isDebugEnabled) {
          std::ostringstream myCoutStream;
          m_gtfeWord->print(myCoutStream);
          LogTrace("L1GlobalTriggerEvmRawToDigi") << myCoutStream.str() << "\n" << std::endl;
        }

        // break the loop - GTFE was found
        break;

      } else {
        if (m_verbosity) {
          edm::LogWarning("L1GlobalTriggerEvmRawToDigi")
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
      edm::LogWarning("L1GlobalTriggerEvmRawToDigi")
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
    LogDebug("L1GlobalTriggerEvmRawToDigi")
        << "\nActive boards before masking(hex format): " << std::hex << std::setw(sizeof(activeBoardsGtInitial) * 2)
        << std::setfill('0') << activeBoardsGtInitial << std::dec << std::setfill(' ')
        << "\nActive boards after masking(hex format):  " << std::hex << std::setw(sizeof(activeBoardsGt) * 2)
        << std::setfill('0') << activeBoardsGt << std::dec << std::setfill(' ') << " \n"
        << std::endl;
  }

  // loop over other blocks in the raw record, count them if they are active

  int numberFdlBoards = 0;

  for (CItBoardMaps itBoard = boardMaps.begin(); itBoard != boardMaps.end(); ++itBoard) {
    int iActiveBit = itBoard->gtBitEvmActiveBoards();
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
        case PSB:
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
            LogDebug("L1GlobalTriggerEvmRawToDigi")
                << "\nBoard of type " << itBoard->gtBoardType() << " not expected  in record.\n"
                << std::endl;
          }

        }

        break;
      }
    }
  }

  // produce the L1GlobalTriggerEvmReadoutRecord now, after we found how many
  // BxInEvent the record has and how many boards are active
  //LogDebug("L1GlobalTriggerEvmRawToDigi")
  //<< "\nL1GlobalTriggerEvmRawToDigi: producing L1GlobalTriggerEvmReadoutRecord\n"
  //<< std::endl;

  // get number of Bx in the event from GTFE block corresponding to alternative 0 and 1 in
  m_recordLength0 = m_gtfeWord->recordLength();
  m_recordLength1 = m_gtfeWord->recordLength1();

  int maxBxInEvent = std::max(m_recordLength0, m_recordLength1);

  std::unique_ptr<L1GlobalTriggerEvmReadoutRecord> gtReadoutRecord(
      new L1GlobalTriggerEvmReadoutRecord(maxBxInEvent, numberFdlBoards));

  // ... then unpack modules other than GTFE, if requested

  for (CItBoardMaps itBoard = gtRecordMap.begin(); itBoard != gtRecordMap.end(); ++itBoard) {
    int iActiveBit = itBoard->gtBitEvmActiveBoards();

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
          edm::LogWarning("L1GlobalTriggerEvmRawToDigi")
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
          LogDebug("L1GlobalTriggerEvmRawToDigi")
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
          LogDebug("L1GlobalTriggerEvmRawToDigi")
              << "\nUnpacking all " << m_totalBxInEvent << " bunch crosses available."
              << "\n"
              << std::endl;
        }

      } else if (m_unpackBxInEvent == 0) {
        m_lowSkipBxInEvent = m_totalBxInEvent;
        m_uppSkipBxInEvent = m_totalBxInEvent;

        if (m_verbosity) {
          LogDebug("L1GlobalTriggerEvmRawToDigi")
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
          LogDebug("L1GlobalTriggerEvmRawToDigi") << "\nUnpacking " << m_unpackBxInEvent << " bunch crosses from "
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
        LogDebug("L1GlobalTriggerEvmRawToDigi") << "\nBoard of type " << itBoard->gtBoardName() << " with index "
                                                << itBoard->gtBoardIndex() << " not active initially in raw data.\n"
                                                << std::endl;
      }
      continue;
    }

    // active board initially, could unpack it
    switch (itBoard->gtBoardType()) {
      case TCS: {
        // if pointer after TCS payload is greater than pointer at
        // the end of GT payload, produce empty products and quit unpacking
        if ((ptrGt + m_tcsWord->getSize()) > endPtrGt) {
          edm::LogError("L1GlobalTriggerEvmRawToDigi") << "\nError: Pointer after TCS "
                                                       << " greater than end pointer."
                                                       << "\n Put empty products in the event!"
                                                       << "\n Quit unpacking this event." << std::endl;

          produceEmptyProducts(iEvent);

          return;
        }

        // unpack only if requested, otherwise skip it
        if (activeBoardToUnpack) {
          m_tcsWord->unpack(ptrGt);

          // add 1 to the GT luminosity number to use the same convention as
          // offline, where LS number starts with 1;
          // in GT hardware, LS starts with 0
          cms_uint16_t lsNr = m_tcsWord->luminositySegmentNr() + 1;
          m_tcsWord->setLuminositySegmentNr(lsNr);

          // add TCS block to GT EVM readout record
          gtReadoutRecord->setTcsWord(*m_tcsWord);

          if (m_verbosity && m_isDebugEnabled) {
            std::ostringstream myCoutStream;
            m_tcsWord->print(myCoutStream);
            LogTrace("L1GlobalTriggerEvmRawToDigi") << myCoutStream.str() << "\n" << std::endl;
          }

          // ... and reset it
          m_tcsWord->reset();
        }

        ptrGt += m_tcsWord->getSize();  // advance with TCS block size

      } break;
      case FDL: {
        for (int iFdl = 0; iFdl < m_totalBxInEvent; ++iFdl) {
          // if pointer after FDL payload is greater than pointer at
          // the end of GT payload, produce empty products and quit unpacking
          if ((ptrGt + m_gtFdlWord->getSize()) > endPtrGt) {
            edm::LogError("L1GlobalTriggerEvmRawToDigi")
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
                LogTrace("L1GlobalTriggerEvmRawToDigi") << myCoutStream.str() << "\n" << std::endl;
              }

              // ... and reset it
              m_gtFdlWord->reset();
            }
          }

          ptrGt += m_gtFdlWord->getSize();  // advance with FDL block size
        }
      }

      break;
      default: {
        // do nothing, all blocks are given in GtBoardType enum
        if (m_verbosity) {
          LogDebug("L1GlobalTriggerEvmRawToDigi")
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
    edm::LogError("L1GlobalTriggerEvmRawToDigi") << "\nError: Pointer after trailer "
                                                 << " greater than end pointer."
                                                 << "\n Put empty products in the event!"
                                                 << "\n Quit unpacking this event." << std::endl;

    produceEmptyProducts(iEvent);

    return;
  }

  unpackTrailer(ptrGt, cmsTrailer);

  if (m_verbosity && m_isDebugEnabled) {
    std::ostringstream myCoutStream;
    gtReadoutRecord->print(myCoutStream);
    LogTrace("L1GlobalTriggerEvmRawToDigi") << "\n The following L1 GT EVM readout record was unpacked.\n"
                                            << myCoutStream.str() << "\n"
                                            << std::endl;
  }

  // put records into event
  iEvent.put(std::move(gtReadoutRecord));
}

// unpack header
void L1GlobalTriggerEvmRawToDigi::unpackHeader(const unsigned char* gtPtr, FEDHeader& cmsHeader) {
  // TODO  if needed in another format

  // print the header info
  if (edm::isDebugEnabled()) {
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

    LogDebug("L1GlobalTriggerEvmRawToDigi") << "\n CMS Header \n" << myCoutStream.str() << "\n" << std::endl;
  }
}

// unpack trailer word
// trPtr pointer to the beginning of trailer obtained from gtPtr
void L1GlobalTriggerEvmRawToDigi::unpackTrailer(const unsigned char* trlPtr, FEDTrailer& cmsTrailer) {
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

    LogDebug("L1GlobalTriggerEvmRawToDigi") << "\n CMS Trailer \n" << myCoutStream.str() << "\n" << std::endl;
  }
}

// produce empty products in case of problems
void L1GlobalTriggerEvmRawToDigi::produceEmptyProducts(edm::Event& iEvent) {
  std::unique_ptr<L1GlobalTriggerEvmReadoutRecord> gtReadoutRecord(new L1GlobalTriggerEvmReadoutRecord());

  // put empty records into event

  iEvent.put(std::move(gtReadoutRecord));
}

// dump FED raw data
void L1GlobalTriggerEvmRawToDigi::dumpFedRawData(const unsigned char* gtPtr, int gtSize, std::ostream& myCout) {
  LogDebug("L1GlobalTriggerEvmRawToDigi") << "\nDump FED raw data.\n" << std::endl;

  int wLength = L1GlobalTriggerReadoutSetup::WordLength;
  int uLength = L1GlobalTriggerReadoutSetup::UnitLength;

  int gtWords = gtSize / uLength;
  LogTrace("L1GlobalTriggerEvmRawToDigi") << "\nFED GT words (" << wLength << " bits):" << gtWords << "\n" << std::endl;

  const cms_uint64_t* payload = reinterpret_cast<cms_uint64_t*>(const_cast<unsigned char*>(gtPtr));

  for (unsigned int i = 0; i < gtSize / sizeof(cms_uint64_t); i++) {
    myCout << std::setw(4) << i << "  " << std::hex << std::setfill('0') << std::setw(16) << payload[i] << std::dec
           << std::setfill(' ') << std::endl;
  }
}

// static class members
