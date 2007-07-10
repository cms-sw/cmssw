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

    // FED Id for GT EVM record
    // default value defined in DataFormats/FEDRawData/src/FEDNumbering.cc
    // default value: assume the EVM record is the first GT record 
    m_evmGtFedId = pSet.getUntrackedParameter<int>(
                       "EvmGtFedId", FEDNumbering::getTriggerGTPFEDIds().first);

    LogDebug("L1GlobalTriggerEvmRawToDigi")
    << "\nFED Id for EVM GT record: "
    << m_evmGtFedId << " \n"
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
        (fedHandle.product())->FEDData(m_evmGtFedId);

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
            m_gtfeWord->unpack(ptrGt);

            if ( edm::isDebugEnabled() ) {

                std::ostringstream myCoutStream;
                m_gtfeWord->print(myCoutStream);
                LogTrace("L1GlobalTriggerEvmRawToDigi")
                << myCoutStream.str() << "\n"
                << std::endl;
            }


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

        // no need to change RecordLength in GTFE,
        // but must change the number of BxInEvent
        // for the readout record

        m_unpackBxInEvent = m_totalBxInEvent;


    } else if (m_unpackBxInEvent < 0) {

        m_lowSkipBxInEvent = 0;
        m_uppSkipBxInEvent = m_totalBxInEvent;

        LogDebug("L1GlobalTriggerEvmRawToDigi")
        << "\nUnpacking all "
        << m_totalBxInEvent  << " bunch crosses available." << "\n"
        << std::endl;

        // no need to change RecordLength in GTFE,
        // but must change the number of BxInEvent
        // for the readout record

        m_unpackBxInEvent = m_totalBxInEvent;

    } else if (m_unpackBxInEvent == 0) {

        m_lowSkipBxInEvent = m_totalBxInEvent;
        m_uppSkipBxInEvent = m_totalBxInEvent;

        LogDebug("L1GlobalTriggerEvmRawToDigi")
        << "\nNo bxInEvent required to be unpacked from "
        << m_totalBxInEvent  << " bunch crosses available." << "\n"
        << std::endl;

        // change RecordLength
        // cast int to boost::uint16_t (there are normally 3 or 5 BxInEvent)
        m_gtfeWord->setRecordLength(static_cast<boost::uint16_t>(m_unpackBxInEvent));

    } else {

        m_lowSkipBxInEvent = (m_totalBxInEvent - m_unpackBxInEvent)/2;
        m_uppSkipBxInEvent = m_totalBxInEvent - m_lowSkipBxInEvent;

        LogDebug("L1GlobalTriggerEvmRawToDigi")
        << "\nUnpacking " <<  m_unpackBxInEvent
        << " bunch crosses from "
        << m_totalBxInEvent  << " bunch crosses available." << "\n"
        << std::endl;

        // change RecordLength
        // cast int to boost::uint16_t (there are normally 3 or 5 BxInEvent)
        m_gtfeWord->setRecordLength(static_cast<boost::uint16_t>(m_unpackBxInEvent));

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
    // m_unpackBxInEvent was resized, if it was needed
    LogDebug("L1GlobalTriggerEvmRawToDigi")
    << "\nL1GlobalTriggerEvmRawToDigi: producing L1GlobalTriggerEvmReadoutRecord\n"
    << std::endl;

    std::auto_ptr<L1GlobalTriggerEvmReadoutRecord> gtReadoutRecord(
        new L1GlobalTriggerEvmReadoutRecord(m_unpackBxInEvent) );


    // add GTFE block to GT readout record, after updating active boards and record length

    gtReadoutRecord->setGtfeWord(*m_gtfeWord);

    // ... and reset it
    m_gtfeWord->reset();

    // loop over other blocks in the raw record, if they are active

    std::map<L1GlobalTriggerReadoutSetup::GtBoard, int> activeBoardsMap =
        tmpGtSetup.GtEvmActiveBoardsMap;
    typedef std::map<L1GlobalTriggerReadoutSetup::GtBoard, int>::const_iterator CItActive;

    // first, clean the L1GlobalTriggerEvmReadoutRecord of un-necessary empty boards ...
    for (CItRecord itRecord = recordMap.begin();
            itRecord != recordMap.end(); ++itRecord) {

        if (itRecord->first == gtfeKey) {
            continue;
        }

        // find if the board is requested to be unpacked
        bool activeBoardToUnpack = false;

        CItActive itBoard = activeBoardsMap.find(itRecord->second);
        if (itBoard != activeBoardsMap.end()) {
            activeBoardToUnpack = activeBoardsGt & (1 << (itBoard->second));
        } else {
            // board not found in the map (pretty strange, throw exception? TODO)
            LogDebug("L1GlobalTriggerEvmRawToDigi")
            << "\nBoard of type " << itRecord->second.boardType
            << " with index "  << itRecord->second.boardIndex
            << " not found in the activeBoardsMap\n"
            << std::endl;

            continue;
        }

        if (activeBoardToUnpack) {
            LogDebug("L1GlobalTriggerEvmRawToDigi")
            << "\nBoard of type " << itRecord->second.boardType
            << " with index "  << itRecord->second.boardIndex
            << "\nexists in the raw data and is requested to be unpacked.\n"
            << std::endl;

        } else {
            // clean the L1GlobalTriggerEvmReadoutRecord of
            // un-necessary empty boards

            // all FDL boards are created in constructor
            // for the required number of BxInEvent at the beginning

            switch (itRecord->second.boardType) {

                case FDL: {

                        // remove m_unpackBxInEvent FDLs if FDL is not active
                        // it is not needed to check which BxInEvent is, as
                        // there are only m_unpackBxInEvent FDLs in the record
                        // constructor
                        for (int iBx = 0; iBx < m_unpackBxInEvent; ++iBx) {
                            gtReadoutRecord->gtFdlVector().pop_back();
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


    // ... then unpack modules other than GTFE, if requested
    for (CItRecord itRecord = recordMap.begin();
            itRecord != recordMap.end(); ++itRecord) {

        if (itRecord->first == gtfeKey) {
            ptrGt += m_gtfeWord->getSize(); // advance with GTFE block size

            continue;
        }

        // skip if the board is not active
        bool activeBoardInitial = false;
        bool activeBoardToUnpack = false;

        CItActive itBoard = activeBoardsMap.find(itRecord->second);
        if (itBoard != activeBoardsMap.end()) {
            activeBoardInitial = activeBoardsGtInitial & (1 << (itBoard->second));
            activeBoardToUnpack = activeBoardsGt & (1 << (itBoard->second));
        } else {
            // board not found in the map
            continue;
        }

        if ( !activeBoardInitial ) {
            LogDebug("L1GlobalTriggerEvmRawToDigi")
            << "\nBoard of type " << itRecord->second.boardType
            << " with index "  << itRecord->second.boardIndex
            << " not active initially in raw data (from activeBoardsMap)\n"
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

                        if ( edm::isDebugEnabled() ) {

                            std::ostringstream myCoutStream;
                            m_tcsWord->print(myCoutStream);
                            LogTrace("L1GlobalTriggerEvmRawToDigi")
                            << myCoutStream.str() << "\n"
                            << std::endl;
                        }

                        // ... and reset it
                        m_tcsWord->reset();
                    }

                    // FIXME TODO put it back
                    ptrGt += 2*64/8; // advance with TCS block size
                    //                    ptrGt += m_tcsWord->getSize(); // advance with TCS block size

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

                                m_gtFdlWord->unpack(ptrGt);

                                // get bxInEvent
                                const int iBxInEvent = m_gtFdlWord->bxInEvent();

                                // add FDL block to GT readout record
                                gtReadoutRecord->setGtFdlWord(*m_gtFdlWord, iBxInEvent);

                                if ( edm::isDebugEnabled() ) {

                                    std::ostringstream myCoutStream;
                                    m_gtFdlWord->print(myCoutStream);
                                    LogTrace("L1GlobalTriggerEvmRawToDigi")
                                    << myCoutStream.str() << "\n"
                                    << std::endl;
                                }

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

                }
                break;

        }

    }

    // unpack trailer
    unpackTrailer(ptrGt, cmsTrailer);

    if ( edm::isDebugEnabled() ) {
        std::ostringstream myCoutStream;
        gtReadoutRecord->print(myCoutStream);
        LogTrace("L1GlobalTriggerEvmRawToDigi")
        << "\n The following L1 GT EVM readout record was unpacked.\n"
        << myCoutStream.str() << "\n"
        << std::endl;
    }

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

        const boost::uint64_t* payload =
            reinterpret_cast<boost::uint64_t*>(const_cast<unsigned char*>(gtPtr));

        std::ostringstream myCoutStream;

        // one word only
        int iWord = 0;

        myCoutStream
        << std::setw(4) << iWord << "  "
        << std::hex << std::setfill('0')
        << std::setw(16) << payload[iWord]
        << std::dec << std::setfill(' ') << "\n"
        << std::endl;

        myCoutStream
        << "  Event_type:  "
        << std::hex << " hex: " << "     "
        << std::setw(1) << std::setfill('0') << cmsHeader.triggerType()
        << std::setfill(' ')
        << std::dec << " dec: "
        << cmsHeader.triggerType()
        << std::endl;

        myCoutStream
        << "  LVL1_Id:     "
        << std::hex << " hex: " << ""
        << std::setw(6) << std::setfill('0') << cmsHeader.lvl1ID()
        << std::setfill(' ')
        << std::dec << " dec: "
        << cmsHeader.lvl1ID()
        << std::endl;

        myCoutStream
        << "  BX_Id:       "
        << std::hex << " hex: " << "   "
        << std::setw(3) << std::setfill('0') << cmsHeader.bxID()
        << std::setfill(' ')
        << std::dec << " dec: "
        << cmsHeader.bxID()
        << std::endl;

        myCoutStream
        << "  Source_Id:   "
        << std::hex << " hex: " << "   "
        << std::setw(3) << std::setfill('0') << cmsHeader.sourceID()
        << std::setfill(' ')
        << std::dec << " dec: "
        << cmsHeader.sourceID()
        << std::endl;

        myCoutStream
        << "  FOV:         "
        << std::hex << " hex: " << "     "
        << std::setw(1) << std::setfill('0') << cmsHeader.version()
        << std::setfill(' ')
        << std::dec << " dec: "
        << cmsHeader.version()
        << std::endl;

        myCoutStream
        << "  H:           "
        << std::hex << " hex: " << "     "
        << std::setw(1) << std::setfill('0') << cmsHeader.moreHeaders()
        << std::setfill(' ')
        << std::dec << " dec: "
        << cmsHeader.moreHeaders()
        << std::endl;

        LogDebug("L1GlobalTriggerRawToDigi")
        << "\n CMS Header \n" << myCoutStream.str() << "\n"
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

// unpack trailer word
// trPtr pointer to the beginning of trailer obtained from gtPtr
void L1GlobalTriggerEvmRawToDigi::unpackTrailer(
    const unsigned char* trlPtr, FEDTrailer& cmsTrailer)
{

    // TODO  if needed in another format

    // print the trailer info
    if ( edm::isDebugEnabled() ) {

        const boost::uint64_t* payload =
            reinterpret_cast<boost::uint64_t*>(const_cast<unsigned char*>(trlPtr));

        std::ostringstream myCoutStream;

        // one word only
        int iWord = 0;

        myCoutStream
        << std::setw(4) << iWord << "  "
        << std::hex << std::setfill('0')
        << std::setw(16) << payload[iWord]
        << std::dec << std::setfill(' ') << "\n"
        << std::endl;

        myCoutStream
        << "  Event_length:  "
        << std::hex << " hex: " << ""
        << std::setw(6) << std::setfill('0') << cmsTrailer.lenght()
        << std::setfill(' ')
        << std::dec << " dec: "
        << cmsTrailer.lenght()
        << std::endl;

        myCoutStream
        << "  CRC:           "
        << std::hex << " hex: " << "  "
        << std::setw(4) << std::setfill('0') << cmsTrailer.crc()
        << std::setfill(' ')
        << std::dec << " dec: "
        << cmsTrailer.crc()
        << std::endl;

        myCoutStream
        << "  Event_status:  "
        << std::hex << " hex: " << "    "
        << std::setw(2) << std::setfill('0') << cmsTrailer.evtStatus()
        << std::setfill(' ')
        << std::dec << " dec: "
        << cmsTrailer.evtStatus()
        << std::endl;

        myCoutStream
        << "  TTS_bits:      "
        << std::hex << " hex: " << "     "
        << std::setw(1) << std::setfill('0') << cmsTrailer.ttsBits()
        << std::setfill(' ')
        << std::dec << " dec: "
        << cmsTrailer.ttsBits()
        << std::endl;

        myCoutStream
        << "  More trailers: "
        << std::hex << " hex: " << "     "
        << std::setw(1) << std::setfill('0') << cmsTrailer.moreTrailers()
        << std::setfill(' ')
        << std::dec << " dec: "
        << cmsTrailer.moreTrailers()
        << std::endl;

        LogDebug("L1GlobalTriggerRawToDigi")
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

