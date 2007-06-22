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
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GlobalTriggerEvmRawToDigi.h"

// system include files
#include <boost/cstdint.hpp>
#include <iostream>
#include <iomanip>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
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
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"


// constructor(s)
L1GlobalTriggerEvmRawToDigi::L1GlobalTriggerEvmRawToDigi(const edm::ParameterSet& pSet)
{

    produces<L1GlobalTriggerEvmReadoutRecord>();

    // input tag for EVM GT record
    m_evmGtInputTag = pSet.getUntrackedParameter<edm::InputTag>(
                          "EvmGtInputTag", edm::InputTag("l1GtEvmPack"));

    LogDebug("L1GlobalTriggerEvmRawToDigi")
    << "\nInput tag for EVM GT record: "
    << m_evmGtInputTag.label() << " \n"
    << std::endl;

    // mask for active boards
    m_activeBoardsMaskGt = pSet.getParameter<unsigned int>("ActiveBoardsMask");

    LogDebug("L1GlobalTriggerEvmRawToDigi")
    << "\nMask for active boards (hex format): "
    << std::hex << std::setw(sizeof(m_activeBoardsMaskGt)*2) << std::setfill('0')
    << m_activeBoardsMaskGt
    << std::dec << std::setfill(' ') << " \n"
    << std::endl;

    // number of bunch crossing to be unpacked

    m_unpackBxInEvent = pSet.getParameter<int>("UnpackBxInEvent");

    LogDebug("L1GlobalTriggerEvmRawToDigi")
    << "\nNumber of bunch crossing to be unpacked: "
    << m_unpackBxInEvent << "\n"
    << std::endl;

    if ((m_unpackBxInEvent > 0)  && ( (m_unpackBxInEvent%2) == 0) ) {
        m_unpackBxInEvent = m_unpackBxInEvent - 1;

        edm::LogInfo("L1GlobalTriggerEvmRawToDigi")
        << "\nWARNING: Number of bunch crossing to be unpacked rounded to: "
        << m_unpackBxInEvent << "\n         The number must be an odd number!\n"
        << std::endl;
    }


    // create GTFE, TCS, FDL cards once per analyzer
    // content will be reset whenever needed

    m_gtfeWord = new L1GtfeExtWord();
    m_tcsWord = new L1TcsWord();
    m_gtFdlWord = new L1GtFdlWord();

    // total Bx's in the events will be set after reading GTFE block
    m_totalBxInEvent = 1;

    // loop range: int m_totalBxInEvent is normally even (L1A-1, L1A, L1A+1, with L1A = 0)

}

// destructor
L1GlobalTriggerEvmRawToDigi::~L1GlobalTriggerEvmRawToDigi()
{

    delete m_gtfeWord;
    delete m_tcsWord;
    delete m_gtFdlWord;

}


// member functions

void L1GlobalTriggerEvmRawToDigi::beginJob(const edm::EventSetup& evSetup)
{

    // empty now

}

// method called to produce the data
void L1GlobalTriggerEvmRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    edm::Handle<FEDRawDataCollection> fedHandle;
    iEvent.getByLabel(m_evmGtInputTag.label(), fedHandle);

    // retrieve data for Global Trigger EVM FED
    const FEDRawData& raw =
        (fedHandle.product())->FEDData(FEDNumbering::getTriggerGTPFEDIds().first);

    int gtSize = raw.size();
    LogDebug("L1GlobalTriggerEvmRawToDigi")
    << "\n Size of raw data: " << gtSize << "\n"
    << std::endl;

    // get a const pointer to the beginning of the data buffer
    const unsigned char* ptrGt = raw.data();

    //
    if ( edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;
        dumpFedRawData(ptrGt, gtSize, myCoutStream);
        LogDebug("L1GlobalTriggerEvmRawToDigi")
        << "\n Dump FEDRawData\n" << myCoutStream.str() << "\n"
        << std::endl;

    }

    // unpack header
    int headerSize = 8;

    FEDHeader cmsHeader(ptrGt);
    FEDTrailer cmsTrailer(ptrGt + gtSize - headerSize);

    unpackHeader(ptrGt, cmsHeader);
    ptrGt += headerSize; // advance with header size

    //
    L1GlobalTriggerReadoutSetup tmpGtSetup; // TODO FIXME temporary event setup
    std::map<int, L1GlobalTriggerReadoutSetup::GtBoard> recordMap =
        tmpGtSetup.GtEvmRecordMap;

    typedef std::map<int, L1GlobalTriggerReadoutSetup::GtBoard>::const_iterator CItRecord;

    // unpack first GTFE to find the length of the record and the active boards
    // here GTFE assumed immediately after the header, the loop is superfluous
    // one must have only fixed-size blocks before GTFE in the record
    int gtfeKey = -1; // negative integer for GTFE key

    for (CItRecord itRecord = recordMap.begin();
            itRecord != recordMap.end(); ++itRecord) {

        if (itRecord->second.boardType == GTFE) {

            // unpack GTFE
            unpackGTFE(evSetup, ptrGt, m_gtfeWord);
            gtfeKey = itRecord->first;

            break; // there is only one GTFE block

        }
    }

    // throw exception if no GTFE found (action for NotFound: SkipEvent)
    if (gtfeKey < 0 ) {

        throw cms::Exception("NotFound")
        << "\nError: no GTFE block found in raw data.\n"
        << "Can not find the record length (BxInEvent) and the active boards!\n"
        << std::endl;
    }

    // TODO check boardID for GTFE

    // life normal here, GTFE found

    // get number of Bx in the event from GTFE block
    m_totalBxInEvent = m_gtfeWord->recordLength();

    LogDebug("L1GlobalTriggerEvmRawToDigi")
    << "\nNumber of bunch crosses in the record: "
    <<  m_totalBxInEvent << " \n"
    << std::endl;

    // number of BX required to be unpacked

    if (m_unpackBxInEvent > m_totalBxInEvent) {
        edm::LogInfo("L1GlobalTriggerEvmRawToDigi")
        << "\nWARNING: Number of bunch crosses in the record ( "
        <<  m_totalBxInEvent
        << " ) is smaller than the number of bunch crosses requested to be unpacked ("
        << m_unpackBxInEvent  << " )!!! \n         Unpacking only " <<  m_totalBxInEvent
        << " bunch crosses.\n"
        << std::endl;

        m_lowSkipBxInEvent = 0;
        m_uppSkipBxInEvent = m_totalBxInEvent;

        // no need to change RecordLength

    } else if (m_unpackBxInEvent < 0) {

        m_lowSkipBxInEvent = 0;
        m_uppSkipBxInEvent = m_totalBxInEvent;

        LogDebug("L1GlobalTriggerEvmRawToDigi")
        << "\nUnpacking all "
        << m_totalBxInEvent  << " bunch crosses available." << "\n"
        << std::endl;

        // no need to change RecordLength

    } else if (m_unpackBxInEvent == 0) {

        m_lowSkipBxInEvent = m_totalBxInEvent;
        m_uppSkipBxInEvent = m_totalBxInEvent;

        LogDebug("L1GlobalTriggerEvmRawToDigi")
        << "\nNo bxInEvent required to be unpacked from "
        << m_totalBxInEvent  << " bunch crosses available." << "\n"
        << std::endl;

        // change RecordLength
        m_gtfeWord->setRecordLength(m_unpackBxInEvent);

    } else {

        m_lowSkipBxInEvent = (m_totalBxInEvent - m_unpackBxInEvent)/2;
        m_uppSkipBxInEvent = m_totalBxInEvent - m_lowSkipBxInEvent;

        LogDebug("L1GlobalTriggerEvmRawToDigi")
        << "\nUnpacking " <<  m_unpackBxInEvent
        << " bunch crosses from "
        << m_totalBxInEvent  << " bunch crosses available." << "\n"
        << std::endl;

        // change RecordLength
        m_gtfeWord->setRecordLength(m_unpackBxInEvent);

    }



    // get list of active blocks
    // blocks not active are not written to the record
    boost::uint16_t activeBoardsGtInitial = m_gtfeWord->activeBoards();

    LogDebug("L1GlobalTriggerEvmRawToDigi")
    << "\nActive boards before masking(hex format): "
    << std::hex << std::setw(sizeof(activeBoardsGtInitial)*2) << std::setfill('0')
    << activeBoardsGtInitial
    << std::dec << std::setfill(' ')
    << std::endl;

    // mask some boards, if needed
    boost::uint16_t activeBoardsGt = activeBoardsGtInitial & m_activeBoardsMaskGt;
    m_gtfeWord->setActiveBoards(activeBoardsGt);

    LogTrace("L1GlobalTriggerEvmRawToDigi")
    << "Active boards after masking(hex format):  "
    << std::hex << std::setw(sizeof(activeBoardsGt)*2) << std::setfill('0')
    << activeBoardsGt
    << std::dec << std::setfill(' ') << " \n"
    << std::endl;

    // produce the L1GlobalTriggerEvmReadoutRecord now, after we found how many Bx's it has
    LogDebug("L1GlobalTriggerEvmRawToDigi")
    << "\nL1GlobalTriggerEvmRawToDigi: producing L1GlobalTriggerEvmReadoutRecord\n"
    << "\nL1GlobalTriggerEvmRawToDigi: producing L1MuGMTReadoutCollection;\n"
    << std::endl;

    std::auto_ptr<L1GlobalTriggerEvmReadoutRecord> gtReadoutRecord(
        new L1GlobalTriggerEvmReadoutRecord(m_totalBxInEvent) );


    // add GTFE block to GT readout record, after updating active boards and record length

    gtReadoutRecord->setGtfeWord(*m_gtfeWord);

    // ... and reset it
    m_gtfeWord->reset();

    // loop over other blocks in the raw record, if they are active

    std::map<L1GlobalTriggerReadoutSetup::GtBoard, int> activeBoardsMap =
        tmpGtSetup.GtEvmActiveBoardsMap;
    typedef std::map<L1GlobalTriggerReadoutSetup::GtBoard, int>::const_iterator CItActive;

    for (CItRecord itRecord = recordMap.begin();
            itRecord != recordMap.end(); ++itRecord) {

        if (itRecord->first == gtfeKey) {
            ptrGt += m_gtfeWord->getSize(); // advance with GTFE block size

            LogDebug("L1GlobalTriggerEvmRawToDigi")
            << "\nSkip GTFE - already unpacked.\n"
            << std::endl;

            continue;
        }

        // unpack modules other than GTFE

        // skip if the board is not active
        bool activeBoardInitial = false;
        bool activeBoardToUnpack = false;

        CItActive itBoard = activeBoardsMap.find(itRecord->second);
        if (itBoard != activeBoardsMap.end()) {
            activeBoardInitial = activeBoardsGtInitial & (1 << (itBoard->second));
            activeBoardToUnpack = activeBoardsGt & (1 << (itBoard->second));
        } else {
            // board not found in the map

            LogDebug("L1GlobalTriggerEvmRawToDigi")
            << "\nBoard of type " << itRecord->second.boardType
            << " with index "  << itRecord->second.boardIndex
            << " not found in the activeBoardsMap\n"
            << std::endl;

            continue;
        }

        if ( !activeBoardInitial ) {
            LogDebug("L1GlobalTriggerEvmRawToDigi")
            << "\nBoard of type " << itRecord->second.boardType
            << " with index "  << itRecord->second.boardIndex
            << " not active in raw data (from activeBoardsMap)\n"
            << std::endl;

            continue;
        }

        // active board initially, unpack it
        switch (itRecord->second.boardType) {

            case TCS: {
                    // unpack only if requested, otherwise skip it
                    if (activeBoardToUnpack) {

                        // unpack only bxInEvent requested, otherwise skip it

                        unpackTCS(evSetup, ptrGt, *m_tcsWord);


                        // add TCS block to GT EVM readout record
                        gtReadoutRecord->setTcsWord(*m_tcsWord);

                        // ... and reset it
                        m_tcsWord->reset();
                    }


                    ptrGt += m_tcsWord->getSize(); // advance with TCS block size

                }
                break;
            case FDL: {
                    for (int iFdl = 0; iFdl < m_totalBxInEvent; ++iFdl) {

                        // unpack only if requested, otherwise skip it
                        if (activeBoardToUnpack) {

                            // unpack only bxInEvent requested, otherwise skip it
                            if (
                                (iFdl >= m_lowSkipBxInEvent) &&
                                (iFdl <  m_uppSkipBxInEvent) ) {

                                unpackFDL(evSetup, ptrGt, *m_gtFdlWord);

                                // get bxInEvent
                                const int iBxInEvent = m_gtFdlWord->bxInEvent();

                                // add FDL block to GT readout record
                                gtReadoutRecord->setGtFdlWord(*m_gtFdlWord, iBxInEvent);

                                // ... and reset it
                                m_gtFdlWord->reset();
                            }

                        }

                        ptrGt += m_gtFdlWord->getSize(); // advance with FDL block size

                    }
                }

                break;
            default: {
                    // do nothing, all blocks are given in GtBoardType enum

                    LogDebug("L1GlobalTriggerEvmRawToDigi")
                    << "\nBoard " << itRecord->second.boardType
                    << " asked to be unpacked, but not in GtBoardEnum!\n"
                    << std::endl;
                }
                break;

        }

    }

    // unpack trailer
    unpackTrailer(ptrGt, cmsTrailer);

    // put records into event

    iEvent.put( gtReadoutRecord );

}

// unpack header
void L1GlobalTriggerEvmRawToDigi::unpackHeader(
    const unsigned char* gtPtr, FEDHeader& cmsHeader)
{

    // TODO  if needed in another format

    // print the header info
    if ( edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;

        myCoutStream
        << "   Event_type (hex): " << std::hex << cmsHeader.triggerType()
        << std::dec << std::endl;
        myCoutStream << "   LVL1_Id:   " << cmsHeader.lvl1ID() << std::endl;
        myCoutStream << "   BX_Id:     " << cmsHeader.bxID() << std::endl;
        myCoutStream << "   Source_Id: " << cmsHeader.sourceID() << std::endl;
        myCoutStream << "   FOV:       " << cmsHeader.version() << std::endl;
        myCoutStream << "   H:         " << cmsHeader.moreHeaders() << std::endl;
        LogDebug("L1GlobalTriggerEvmRawToDigi")
        << "\n CMS Header \n" << myCoutStream.str() << "\n"
        << std::endl;

    }


}


// unpack the GTFE record
// gtfePtr pointer to the beginning of the GTFE block, obtained from gtPtr
void L1GlobalTriggerEvmRawToDigi::unpackGTFE(
    const edm::EventSetup& evSetup,
    const unsigned char* gtfePtr,
    L1GtfeExtWord* gtfeWord)
{

    LogDebug("L1GlobalTriggerEvmRawToDigi")
    << "\nUnpacking GTFE block.\n"
    << std::endl;

    int uLength = L1GlobalTriggerReadoutSetup::UnitLength;

    int gtfeSize = gtfeWord->getSize();
    int gtfeWords = gtfeSize/uLength;

    const boost::uint64_t* payload =
        reinterpret_cast<boost::uint64_t*>(const_cast<unsigned char*>(gtfePtr));

    for (int iWord = 0; iWord < gtfeWords; ++iWord) {

        // fill GTFE
        // the second argument must match the word index defined in L1GtfeExtWord class

        gtfeWord->setBoardId(payload[iWord], iWord);
        gtfeWord->setRecordLength(payload[iWord], iWord);
        gtfeWord->setBxNr(payload[iWord], iWord);
        gtfeWord->setSetupVersion(payload[iWord], iWord);
        gtfeWord->setActiveBoards(payload[iWord], iWord);
        gtfeWord->setTotalTriggerNr(payload[iWord], iWord);

        // TODO FIXME add BST blocks
        
        LogTrace("L1GlobalTriggerEvmRawToDigi")
        << std::setw(4) << iWord << "  "
        << std::hex << std::setfill('0')
        << std::setw(16) << payload[iWord]
        << std::dec << std::setfill(' ')
        << std::endl;

    }


}

// unpack the TCS record
// tcsPtr pointer to the beginning of the TCS block, obtained from gtPtr
void L1GlobalTriggerEvmRawToDigi::unpackTCS(
    const edm::EventSetup& evSetup,
    const unsigned char* tcsPtr,
    L1TcsWord& tcsWord)
{

    LogDebug("L1GlobalTriggerEvmRawToDigi")
    << "\nUnpacking TCS block.\n"
    << std::endl;

    int uLength = L1GlobalTriggerReadoutSetup::UnitLength;

    int tcsSize = tcsWord.getSize();
    int tcsWords = tcsSize/uLength;

    const boost::uint64_t* payload =
        reinterpret_cast<boost::uint64_t*>(const_cast<unsigned char*>(tcsPtr));

    for (int iWord = 0; iWord < tcsWords; ++iWord) {

        // fill TCS
        // the second argument must match the word index defined in L1TcsWord class

        tcsWord.setDaqNr(payload[iWord], iWord);
        tcsWord.setTriggerType(payload[iWord], iWord);
        tcsWord.setStatus(payload[iWord], iWord);
        tcsWord.setBxNr(payload[iWord], iWord);
        tcsWord.setPartTrigNr(payload[iWord], iWord);
        tcsWord.setEventNr(payload[iWord], iWord);
        tcsWord.setAssignedPartitions(payload[iWord], iWord);
        tcsWord.setPartRunNr(payload[iWord], iWord);
        tcsWord.setOrbitNr(payload[iWord], iWord);

        LogTrace("L1GlobalTriggerEvmRawToDigi")
        << std::setw(4) << iWord << "  "
        << std::hex << std::setfill('0')
        << std::setw(16) << payload[iWord]
        << std::dec << std::setfill(' ')
        << std::endl;

    }


}


// unpack FDL records for various bunch crosses
// fdlPtr pointer to the beginning of the each FDL block obtained from gtPtr
void L1GlobalTriggerEvmRawToDigi::unpackFDL(
    const edm::EventSetup& evSetup,
    const unsigned char* fdlPtr,
    L1GtFdlWord& fdlWord)
{

    LogDebug("L1GlobalTriggerEvmRawToDigi")
    << "\nUnpacking FDL block.\n"
    << std::endl;

    int uLength = L1GlobalTriggerReadoutSetup::UnitLength;

    int fdlSize = fdlWord.getSize();
    int fdlWords = fdlSize/uLength;

    const boost::uint64_t* payload =
        reinterpret_cast<boost::uint64_t*>(const_cast<unsigned char*>(fdlPtr));

    for (int iWord = 0; iWord < fdlWords; ++iWord) {

        // fill FDL
        // the second argument must match the word index defined in L1GtFdlWord class

        fdlWord.setBoardId(payload[iWord], iWord);
        fdlWord.setBxInEvent(payload[iWord], iWord);
        fdlWord.setBxNr(payload[iWord], iWord);
        fdlWord.setEventNr(payload[iWord], iWord);

        fdlWord.setGtTechnicalTriggerWord(payload[iWord], iWord);

        fdlWord.setGtDecisionWordA(payload[iWord], iWord);

        fdlWord.setGtDecisionWordB(payload[iWord], iWord);

        fdlWord.setGtDecisionWordExtended(payload[iWord], iWord);

        fdlWord.setN0Algo(payload[iWord], iWord);
        fdlWord.setFinalOR(payload[iWord], iWord);

        fdlWord.setLocalBxNr(payload[iWord], iWord);

        LogTrace("L1GlobalTriggerEvmRawToDigi")
        << std::setw(4) << iWord << "  "
        << std::hex << std::setfill('0')
        << std::setw(16) << payload[iWord]
        << std::dec << std::setfill(' ')
        << std::endl;
    }

}


// unpack trailer word
// trPtr pointer to the beginning of trailer obtained from gtPtr
void L1GlobalTriggerEvmRawToDigi::unpackTrailer(
    const unsigned char* trlPtr, FEDTrailer& cmsTrailer)
{

    // TODO  if needed in another format

    // print the trailer info
    if ( edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;

        myCoutStream << "   Event_length:  " << cmsTrailer.lenght() << std::endl;
        myCoutStream << "   CRC:           " << cmsTrailer.crc() << std::endl;
        myCoutStream << "   Event_stat:    " << cmsTrailer.evtStatus() << std::endl;
        myCoutStream << "   TTS_bits:      " << cmsTrailer.ttsBits() << std::endl;
        myCoutStream << "   More trailers: " << cmsTrailer.moreTrailers() << std::endl;
        LogDebug("L1GlobalTriggerEvmRawToDigi")
        << "\n CMS Trailer \n" << myCoutStream.str() << "\n"
        << std::endl;

    }

}

// dump FED raw data
void L1GlobalTriggerEvmRawToDigi::dumpFedRawData(
    const unsigned char* gtPtr,
    int gtSize,
    std::ostream& myCout)
{

    LogDebug("L1GlobalTriggerEvmRawToDigi")
    << "\nDump FED raw data.\n"
    << std::endl;

    int wLength = L1GlobalTriggerReadoutSetup::WordLength;
    int uLength = L1GlobalTriggerReadoutSetup::UnitLength;

    int gtWords = gtSize/uLength;
    LogTrace("L1GlobalTriggerEvmRawToDigi")
    << "\nFED GT words (" << wLength << " bits):" << gtWords << "\n"
    << std::endl;

    const boost::uint64_t* payload =
        reinterpret_cast<boost::uint64_t*>(const_cast<unsigned char*>(gtPtr));

    for (unsigned int i = 0; i < gtSize/sizeof(boost::uint64_t); i++) {
        myCout << std::setw(4) << i << "  "
        << std::hex << std::setfill('0')
        << std::setw(16) << payload[i]
        << std::dec << std::setfill(' ')
        << std::endl;
    }

}

//
void L1GlobalTriggerEvmRawToDigi::endJob()
{

    // empty now
}


// static class members

