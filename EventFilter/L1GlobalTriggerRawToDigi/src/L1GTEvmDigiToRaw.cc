/**
 * \class L1GTEvmDigiToRaw
 *
 *
 * Description: generate raw data from digis.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// this class header
#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GTEvmDigiToRaw.h"

// system include files
#include <vector>
#include <iostream>
#include <iomanip>


// user include files
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "FWCore/Utilities/interface/CRC16.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeExtWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1TcsWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtBoard.h"

#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"
#include "CondFormats/DataRecord/interface/L1GtBoardMapsRcd.h"


// constructor(s)
L1GTEvmDigiToRaw::L1GTEvmDigiToRaw(const edm::ParameterSet& pSet) :
    m_evmGtFedId(pSet.getUntrackedParameter<int>("EvmGtFedId",
            FEDNumbering::MINTriggerGTPFEDID)),
    m_evmGtInputToken(consumes<L1GlobalTriggerEvmReadoutRecord>(pSet.getParameter<edm::InputTag>("EvmGtInputTag"))),
    m_evmGtInputTag(pSet.getParameter<edm::InputTag>("EvmGtInputTag")),
    m_activeBoardsMaskGt(pSet.getParameter<unsigned int>("ActiveBoardsMask")),
    m_totalBxInEvent(0),
    m_minBxInEvent(0), m_maxBxInEvent(),
    m_verbosity(pSet.getUntrackedParameter<int> ("Verbosity", 0)),
    m_isDebugEnabled(edm::isDebugEnabled())



{

    LogDebug("L1GTEvmDigiToRaw") << "\nMask for active boards (hex format): "
            << std::hex << std::setw(sizeof(m_activeBoardsMaskGt)*2)
            << std::setfill('0') << m_activeBoardsMaskGt << std::dec
            << std::setfill(' ') << "\nInput tag for EVM GT record: "
            << m_evmGtInputTag << "\nFED Id for EVM GT record: "
            << m_evmGtFedId << " \n" << std::endl;

    //
    produces<FEDRawDataCollection>();

}

// member functions

// method called to produce the data
void L1GTEvmDigiToRaw::produce(edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    // define new FEDRawDataCollection
    // it contains ALL FEDs in an event
    std::unique_ptr<FEDRawDataCollection> allFedRawData(new FEDRawDataCollection);

    FEDRawData& gtRawData = allFedRawData->FEDData(m_evmGtFedId);

    // get records from EventSetup

    //  board maps
    edm::ESHandle< L1GtBoardMaps > l1GtBM;
    evSetup.get< L1GtBoardMapsRcd >().get( l1GtBM );

    const std::vector<L1GtBoard> boardMaps = l1GtBM->gtBoardMaps();
    int boardMapsSize = boardMaps.size();

    typedef std::vector<L1GtBoard>::const_iterator CItBoardMaps;

    // create an ordered vector for the GT EVM record
    // header (pos 0 in record) and trailer (last position in record)
    // not included, as they are not in board list
    std::vector<L1GtBoard> gtRecordMap;
    gtRecordMap.reserve(boardMapsSize);

    for (int iPos = 0; iPos < boardMapsSize; ++iPos) {
        for (CItBoardMaps itBoard = boardMaps.begin(); itBoard
                != boardMaps.end(); ++itBoard) {

            if (itBoard->gtPositionEvmRecord() == iPos) {
                gtRecordMap.push_back(*itBoard);
                break;
            }

        }
    }



    // get L1GlobalTriggerEvmReadoutRecord
    edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtReadoutRecord;
    iEvent.getByToken(m_evmGtInputToken, gtReadoutRecord);

    if (!gtReadoutRecord.isValid()) {
        edm::LogWarning("L1GTEvmDigiToRaw")
                << "\nWarning: L1GlobalTriggerEvmReadoutRecord with input tag " << m_evmGtInputTag
                << "\nrequested in configuration, but not found in the event."
                << "\nQuit packing this event" << std::endl;

        // put the raw data in the event
        iEvent.put(std::move(allFedRawData));

        return;
    }

    if (m_verbosity && m_isDebugEnabled) {
        std::ostringstream myCoutStream;
        gtReadoutRecord->print(myCoutStream);
        LogTrace("L1GTEvmDigiToRaw")
                << "\n The following L1 GT EVM readout record will be packed.\n"
                << " Some boards could be disabled before packing,"
                << " see detailed board packing.\n"
                << myCoutStream.str() << "\n"
                << std::endl;
    }

    // get GTFE block
    L1GtfeExtWord gtfeBlock = gtReadoutRecord->gtfeWord();

    // get the number of Bx in the event for alternative 0 and alternative 1
    cms_uint16_t recordLength0 = gtfeBlock.recordLength();
    cms_uint16_t recordLength1 = gtfeBlock.recordLength1();

    // length of BST record (in bytes)
    m_bstLengthBytes= static_cast<int> (gtfeBlock.bstLengthBytes());

    // get list of active blocks from the GTFE block
    // and mask some blocks, if required
    // blocks not active are not written to the record

    cms_uint16_t activeBoardsGtInitial = gtfeBlock.activeBoards();
    cms_uint16_t altNrBxBoardInitial = gtfeBlock.altNrBxBoard();

    // mask some boards, if needed

    cms_uint16_t activeBoardsGt = activeBoardsGtInitial & m_activeBoardsMaskGt;

    if (m_verbosity && m_isDebugEnabled) {
        LogDebug("L1GTEvmDigiToRaw")
                << "\nActive boards before masking(hex format): " << std::hex
                << std::setw(sizeof ( activeBoardsGtInitial ) * 2) << std::setfill('0')
                << activeBoardsGtInitial << std::dec << std::setfill(' ')
                << "\nActive boards after masking(hex format):  " << std::hex
                << std::setw(sizeof ( activeBoardsGt ) * 2) << std::setfill('0')
                << activeBoardsGt << std::dec
                << std::setfill(' ') << " \n"
                << std::endl;
    }

    // get the size of the record

    unsigned int gtDataSize = 0;

    unsigned int headerSize = 8;
    gtDataSize += headerSize;

    for (CItBoardMaps
            itBoard = boardMaps.begin();
            itBoard != boardMaps.end(); ++itBoard) {

        if (itBoard->gtBoardType() == GTFE) {
            gtDataSize += gtfeBlock.getSize();
            continue;
        }


        int iActiveBit = itBoard->gtBitEvmActiveBoards();
        bool activeBoardToPack = false;

        int altNrBxBoardVal = -1;

        if (iActiveBit >= 0) {
            activeBoardToPack = activeBoardsGt & (1 << iActiveBit);

            altNrBxBoardVal = (altNrBxBoardInitial & ( 1 << iActiveBit )) >> iActiveBit;

            if (altNrBxBoardVal == 1) {
                m_totalBxInEvent = recordLength1;
            } else if (altNrBxBoardVal == 0) {
                m_totalBxInEvent = recordLength0;
            } else {
                if (m_verbosity) {
                    edm::LogWarning("L1GTEvmDigiToRaw")
                    << "\n\nWARNING: Wrong value altNrBxBoardVal = " << altNrBxBoardVal
                    << " for board " << std::hex << ( itBoard->gtBoardId() ) << std::dec
                    << "\n  iActiveBit =            " << iActiveBit
                    << "\n  altNrBxBoardInitial = 0x" << std::hex << altNrBxBoardInitial <<  std::dec
                    << "\n  activeBoardsGt =      0x" << std::hex << activeBoardsGt <<  std::dec
                    << "\n  activeBoardToPack =   " << activeBoardToPack
                    << "\n Set altNrBxBoardVal tentatively to "
                    << recordLength0 << "\n Job may crash or produce wrong results!\n\n"
                    << std::endl;
                }

                m_totalBxInEvent = recordLength0;
            }
        } else {
            // board not in the ActiveBoards for the record
            continue;
        }

        if (activeBoardToPack) {

            switch (itBoard->gtBoardType()) {
                case GTFE: {
                        // size already added;
                    }

                    break;
                case FDL: {
                        L1GtFdlWord fdlBlock;
                        gtDataSize += m_totalBxInEvent*fdlBlock.getSize();
                    }

                    break;
                case TCS: {
                        L1TcsWord tcsBlock;
                        gtDataSize += tcsBlock.getSize();
                    }

                    break;
                case TIM: {
                        // not considered
                    }

                    break;
                default: {
                        // do nothing, all blocks are given in GtBoardType enum
                    }

                    break;
            }
        }

    }


    unsigned int trailerSize = 8;
    gtDataSize += trailerSize;

    // resize, GT raw data record has variable length,
    // depending on active boards (read in GTFE)
    gtRawData.resize(gtDataSize);

    // ptrGt: pointer to the beginning of GT record in the raw data

    unsigned char* ptrGt = gtRawData.data();
    unsigned char* ptrGtBegin = gtRawData.data();

    if (m_verbosity && m_isDebugEnabled) {
        LogDebug("L1GTEvmDigiToRaw") << "\n Size of raw data: " << gtRawData.size() << "\n"
                << std::endl;
    }

    // ------- pack boards -------

    // pack header
    packHeader(ptrGt, iEvent);
    ptrGt += headerSize; // advance with header size

    // loop over other blocks in the raw record, if they are active

    for (CItBoardMaps
            itBoard = gtRecordMap.begin();
            itBoard != gtRecordMap.end(); ++itBoard) {

        if (itBoard->gtBoardType() == GTFE) {

            packGTFE(evSetup, ptrGt, gtfeBlock, activeBoardsGt);

            if (m_verbosity && m_isDebugEnabled) {

                std::ostringstream myCoutStream;
                gtfeBlock.print(myCoutStream);
                LogTrace("L1GTEvmDigiToRaw")
                        << myCoutStream.str() << "\n"
                        << std::endl;
            }

            ptrGt += gtfeBlock.getSize(); // advance with GTFE block size

            continue;
        }


        // pack modules other than GTFE if they are active

        int iActiveBit = itBoard->gtBitEvmActiveBoards();
        bool activeBoardToPack = false;

        int altNrBxBoardVal = -1;

        if (iActiveBit >= 0) {
            activeBoardToPack = activeBoardsGt & (1 << iActiveBit);

            altNrBxBoardVal = (altNrBxBoardInitial & ( 1 << iActiveBit )) >> iActiveBit;

            if (altNrBxBoardVal == 1) {
                m_totalBxInEvent = recordLength1;
            } else if (altNrBxBoardVal == 0) {
                m_totalBxInEvent = recordLength0;
            } else {
                if (m_verbosity) {
                    edm::LogWarning("L1GTEvmDigiToRaw")
                    << "\n\nWARNING: Wrong value altNrBxBoardVal = " << altNrBxBoardVal
                    << " for board " << std::hex << ( itBoard->gtBoardId() ) << std::dec
                    << "\n  iActiveBit =            " << iActiveBit
                    << "\n  altNrBxBoardInitial = 0x" << std::hex << altNrBxBoardInitial <<  std::dec
                    << "\n  activeBoardsGt =      0x" << std::hex << activeBoardsGt <<  std::dec
                    << "\n  activeBoardToPack =   " << activeBoardToPack
                    << "\n Set altNrBxBoardVal tentatively to "
                    << recordLength0 << "\n Job may crash or produce wrong results!\n\n"
                    << std::endl;
                }

                m_totalBxInEvent = recordLength0;
            }

            m_minBxInEvent = (m_totalBxInEvent + 1)/2 - m_totalBxInEvent;
            m_maxBxInEvent = (m_totalBxInEvent + 1)/2 - 1;

        } else {
            // board not in the ActiveBoards for the record
            continue;
        }

        if (activeBoardToPack) {

            if (m_verbosity && m_isDebugEnabled) {
                LogDebug("L1GTEvmDigiToRaw")
                        << "\nBoard " << std::hex << "0x" << ( itBoard->gtBoardId() ) << std::dec
                        << "\n  Number of bunch crosses in the record: " << m_totalBxInEvent
                        << " = " << "[" << m_minBxInEvent << ", " << m_maxBxInEvent
                        << "] BX\n"
                        << std::endl;
            }

            // active board, pack it
            switch (itBoard->gtBoardType()) {

                case TCS: {

                        L1TcsWord tcsBlock = gtReadoutRecord->tcsWord();
                        packTCS(evSetup, ptrGt, tcsBlock);

                        if (m_verbosity && m_isDebugEnabled) {

                            std::ostringstream myCoutStream;
                            tcsBlock.print(myCoutStream);
                            LogTrace("L1GTEvmDigiToRaw")
                                    << myCoutStream.str() << "\n"
                                    << std::endl;
                        }

                        ptrGt += tcsBlock.getSize(); // advance with TCS block size

                    }
                    break;
                case FDL: {

                        for (int iBxInEvent = m_minBxInEvent; iBxInEvent <= m_maxBxInEvent;
                                ++iBxInEvent) {

                            L1GtFdlWord fdlBlock = gtReadoutRecord->gtFdlWord(iBxInEvent);
                            packFDL(evSetup, ptrGt, fdlBlock);

                            if (m_verbosity && m_isDebugEnabled) {

                                std::ostringstream myCoutStream;
                                fdlBlock.print(myCoutStream);
                                LogTrace("L1GTEvmDigiToRaw")
                                        << myCoutStream.str() << "\n"
                                        << std::endl;
                            }

                            ptrGt += fdlBlock.getSize(); // advance with FDL block size
                        }

                    }
                    break;
                default: {

                        // do nothing, all blocks are given in GtBoardType enum
                        break;
                    }
            }

        }
    }

    // pack trailer
    packTrailer(ptrGt, ptrGtBegin, gtDataSize);

    // put the raw data in the event

    iEvent.put(std::move(allFedRawData));


}


// pack header
void L1GTEvmDigiToRaw::packHeader(unsigned char* ptrGt, edm::Event& iEvent)
{
    // TODO FIXME where from to get all numbers?

    // Event Trigger type identifier
    int triggerTypeVal = 0;

    // Level-1 event number generated by the TTC system
    int lvl1IdVal = iEvent.id().event();

    // The bunch crossing number
    int bxCross = iEvent.bunchCrossing();
    cms_uint16_t bxCrossHw = 0;
    if ((bxCross & 0xFFF) == bxCross) {
        bxCrossHw = static_cast<cms_uint16_t> (bxCross);
    }
    else {
        bxCrossHw = 0; // Bx number too large, set to 0!
        if (m_verbosity && m_isDebugEnabled) {
            LogDebug("L1GTEvmDigiToRaw")
                    << "\nBunch cross number [hex] = "
                    << std::hex << bxCross
                    << "\n  larger than 12 bits. Set to 0! \n"
                    << std::dec
                    << std::endl;
        }
    }
    int bxIdVal = bxCrossHw;

    // Identifier of the FED
    int sourceIdVal = m_evmGtFedId;

    // Version identifier of the FED data format
    int versionVal = 0;

    // 0 -> the current header word is the last one.
    // 1-> other header words can follow
    // (always 1 for ECAL)
    bool moreHeadersVal = false;


    FEDHeader gtFEDHeader(ptrGt);

    gtFEDHeader.set(ptrGt,
                    triggerTypeVal, lvl1IdVal, bxIdVal, sourceIdVal, versionVal,
                    moreHeadersVal);


}

// pack the GTFE block
void L1GTEvmDigiToRaw::packGTFE(
    const edm::EventSetup& evSetup,
    unsigned char* ptrGt,
    L1GtfeExtWord& gtfeBlock,
    cms_uint16_t activeBoardsGtValue)
{

    if (m_verbosity && m_isDebugEnabled) {
        LogDebug("L1GTEvmDigiToRaw") << "\nPacking GTFE \n" << std::endl;
    }

    int uLength = L1GlobalTriggerReadoutSetup::UnitLength;

    // initialize the required number of word64
    int nrWord64 = gtfeBlock.getSize()/uLength;
    std::vector<cms_uint64_t> tmpWord64;
    tmpWord64.resize(nrWord64);

    for (int iWord = 0; iWord < nrWord64; ++iWord) {
        tmpWord64[iWord] = 0x0000000000000000ULL;
    }

    // fill the values in the words
    for (int iWord = 0; iWord < nrWord64; ++iWord) {

        gtfeBlock.setBoardIdWord64(tmpWord64[iWord], iWord);
        gtfeBlock.setRecordLength1Word64(tmpWord64[iWord], iWord);
        gtfeBlock.setRecordLengthWord64(tmpWord64[iWord], iWord);
        gtfeBlock.setBxNrWord64(tmpWord64[iWord], iWord);
        gtfeBlock.setSetupVersionWord64(tmpWord64[iWord], iWord);
        gtfeBlock.setActiveBoardsWord64(tmpWord64[iWord], iWord, activeBoardsGtValue);
        gtfeBlock.setAltNrBxBoardWord64(tmpWord64[iWord], iWord);
        gtfeBlock.setTotalTriggerNrWord64(tmpWord64[iWord], iWord);

        for (int iBst = 0; iBst < m_bstLengthBytes; ++iBst) {
            gtfeBlock.setBstWord64(tmpWord64[iWord], iBst, iWord);
        }

    }

    // put the words in the FED record

    cms_uint64_t* pw =
        reinterpret_cast<cms_uint64_t*>(const_cast<unsigned char*>(ptrGt));

    for (int iWord = 0; iWord < nrWord64; ++iWord) {

        *pw++ = tmpWord64[iWord];

        if (m_verbosity && m_isDebugEnabled) {
            LogTrace("L1GTEvmDigiToRaw")
                    << std::setw(4) << iWord << "  "
                    << std::hex << std::setfill('0')
                    << std::setw(16) << tmpWord64[iWord]
                    << std::dec << std::setfill(' ')
                    << std::endl;
        }
    }


}

// pack the TCS block
void L1GTEvmDigiToRaw::packTCS(
    const edm::EventSetup& evSetup,
    unsigned char* ptrGt,
    L1TcsWord& tcsBlock)
{

    if (m_verbosity && m_isDebugEnabled) {
        LogDebug("L1GTEvmDigiToRaw") << "\nPacking TCS \n" << std::endl;
    }

    int uLength = L1GlobalTriggerReadoutSetup::UnitLength;

    // initialize the required number of word64
    int nrWord64 = tcsBlock.getSize()/uLength;
    std::vector<cms_uint64_t> tmpWord64;
    tmpWord64.resize(nrWord64);

    for (int iWord = 0; iWord < nrWord64; ++iWord) {
        tmpWord64[iWord] = 0x0000000000000000ULL;
    }

    // fill the values in the words
    for (int iWord = 0; iWord < nrWord64; ++iWord) {

        tcsBlock.setBoardIdWord64(tmpWord64[iWord], iWord);
        tcsBlock.setBxNrWord64(tmpWord64[iWord], iWord);
        tcsBlock.setDaqNrWord64(tmpWord64[iWord], iWord);
        tcsBlock.setTriggerTypeWord64(tmpWord64[iWord], iWord);
        tcsBlock.setStatusWord64(tmpWord64[iWord], iWord);
        tcsBlock.setLuminositySegmentNrWord64(tmpWord64[iWord], iWord);

        tcsBlock.setPartRunNrWord64(tmpWord64[iWord], iWord);
        tcsBlock.setAssignedPartitionsWord64(tmpWord64[iWord], iWord);

        tcsBlock.setPartTrigNrWord64(tmpWord64[iWord], iWord);
        tcsBlock.setEventNrWord64(tmpWord64[iWord], iWord);

        tcsBlock.setOrbitNrWord64(tmpWord64[iWord], iWord);

    }

    // put the words in the FED record

    cms_uint64_t* pw =
        reinterpret_cast<cms_uint64_t*>(const_cast<unsigned char*>(ptrGt));

    for (int iWord = 0; iWord < nrWord64; ++iWord) {

        *pw++ = tmpWord64[iWord];

        if (m_verbosity && m_isDebugEnabled) {
            LogTrace("L1GTEvmDigiToRaw")
                    << std::setw(4) << iWord << "  "
                    << std::hex << std::setfill('0')
                    << std::setw(16) << tmpWord64[iWord]
                    << std::dec << std::setfill(' ')
                    << std::endl;
        }
    }


}

// pack the FDL block
void L1GTEvmDigiToRaw::packFDL(
    const edm::EventSetup& evSetup,
    unsigned char* ptrGt,
    L1GtFdlWord& fdlBlock)
{

    if (m_verbosity && m_isDebugEnabled) {
        LogDebug("L1GTEvmDigiToRaw") << "\nPacking FDL \n" << std::endl;
    }

    int uLength = L1GlobalTriggerReadoutSetup::UnitLength;

    // initialize the required number of word64
    int nrWord64 = fdlBlock.getSize()/uLength;
    std::vector<cms_uint64_t> tmpWord64;
    tmpWord64.resize(nrWord64);

    for (int iWord = 0; iWord < nrWord64; ++iWord) {
        tmpWord64[iWord] = 0x0000000000000000ULL;
    }

    // fill the values in the words
    for (int iWord = 0; iWord < nrWord64; ++iWord) {

        fdlBlock.setBoardIdWord64(tmpWord64[iWord], iWord);
        fdlBlock.setBxInEventWord64(tmpWord64[iWord], iWord);
        fdlBlock.setBxNrWord64(tmpWord64[iWord], iWord);
        fdlBlock.setEventNrWord64(tmpWord64[iWord], iWord);

        fdlBlock.setGtTechnicalTriggerWordWord64(tmpWord64[iWord], iWord);

        fdlBlock.setGtDecisionWordAWord64(tmpWord64[iWord], iWord);
        fdlBlock.setGtDecisionWordBWord64(tmpWord64[iWord], iWord);

        fdlBlock.setGtDecisionWordExtendedWord64(tmpWord64[iWord], iWord);

        fdlBlock.setPhysicsDeclaredWord64(tmpWord64[iWord], iWord);
        fdlBlock.setGtPrescaleFactorIndexTechWord64(tmpWord64[iWord], iWord);
        fdlBlock.setGtPrescaleFactorIndexAlgoWord64(tmpWord64[iWord], iWord);
        fdlBlock.setNoAlgoWord64(tmpWord64[iWord], iWord);
        fdlBlock.setFinalORWord64(tmpWord64[iWord], iWord);

        fdlBlock.setOrbitNrWord64(tmpWord64[iWord], iWord);
        fdlBlock.setLumiSegmentNrWord64(tmpWord64[iWord], iWord);
        fdlBlock.setLocalBxNrWord64(tmpWord64[iWord], iWord);

    }

    // put the words in the FED record

    cms_uint64_t* pw =
        reinterpret_cast<cms_uint64_t*>(const_cast<unsigned char*>(ptrGt));

    for (int iWord = 0; iWord < nrWord64; ++iWord) {

        *pw++ = tmpWord64[iWord];

        if (m_verbosity && m_isDebugEnabled) {
            LogTrace("L1GTEvmDigiToRaw")
            << std::setw(4) << iWord << "  "
            << std::hex << std::setfill('0')
            << std::setw(16) << tmpWord64[iWord]
            << std::dec << std::setfill(' ')
            << std::endl;
        }
    }

}


// pack trailer
void L1GTEvmDigiToRaw::packTrailer(unsigned char* ptrGt,
                                   unsigned char* ptrGtBegin, int dataSize)
{

    // TODO FIXME where from to get all numbers?

    // The length of the event fragment counted in 64-bit words including header and trailer
    int lengthVal = dataSize/8;

    // Cyclic Redundancy Code of the event fragment including header and trailer
    int crcVal = evf::compute_crc(ptrGtBegin, dataSize);

    // Event fragment status information
    int evtStatusVal = 0;

    // Current value of the Trigger Throttling System bits.
    int ttsBitsVal = 0;

    // 0 -> the current trailer word is the last one.
    // 1-> other trailer words can follow
    // (always 0 for ECAL)
    bool moreTrailersVal = false;

    FEDTrailer gtFEDTrailer(ptrGt);
    gtFEDTrailer.set(ptrGt,
                     lengthVal, crcVal, evtStatusVal, ttsBitsVal,
                     moreTrailersVal);

}



// static class members
