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
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GlobalTriggerRawToDigi.h"

// system include files
#include <boost/cstdint.hpp>
#include <iostream>
#include <iomanip>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
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

#include "DataFormats/Common/interface/RefProd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtBoard.h"

#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"
#include "CondFormats/DataRecord/interface/L1GtBoardMapsRcd.h"


// constructor(s)
L1GlobalTriggerRawToDigi::L1GlobalTriggerRawToDigi(const edm::ParameterSet& pSet)
{

    produces<L1GlobalTriggerReadoutRecord>();
    produces<L1MuGMTReadoutCollection>();

    produces<std::vector<L1MuRegionalCand> >("DT");
    produces<std::vector<L1MuRegionalCand> >("CSC");
    produces<std::vector<L1MuRegionalCand> >("RPCb");
    produces<std::vector<L1MuRegionalCand> >("RPCf");
    produces<std::vector<L1MuGMTCand> >();

    // input tag for DAQ GT record
    m_daqGtInputTag = pSet.getUntrackedParameter<edm::InputTag>(
                          "DaqGtInputTag", edm::InputTag("l1GtPack"));

    // FED Id for GT DAQ record
    // default value defined in DataFormats/FEDRawData/src/FEDNumbering.cc
    // default value: assume the DAQ record is the last GT record
    m_daqGtFedId = pSet.getUntrackedParameter<int>(
                       "DaqGtFedId", FEDNumbering::getTriggerGTPFEDIds().second);

    // mask for active boards
    m_activeBoardsMaskGt = pSet.getParameter<unsigned int>("ActiveBoardsMask");


    // number of bunch crossing to be unpacked

    m_unpackBxInEvent = pSet.getParameter<int>("UnpackBxInEvent");

    LogDebug("L1GlobalTriggerRawToDigi")
    << "\nInput tag for DAQ GT record:             "
    << m_daqGtInputTag.label()
    << "\nFED Id for DAQ GT record:                "
    << m_daqGtFedId
    << "\nMask for active boards (hex format):     "
    << std::hex << std::setw(sizeof(m_activeBoardsMaskGt)*2) << std::setfill('0')
    << m_activeBoardsMaskGt
    << std::dec << std::setfill(' ')
    << "\nNumber of bunch crossing to be unpacked: "
    << m_unpackBxInEvent << "\n"
    << std::endl;

    if ((m_unpackBxInEvent > 0)  && ( (m_unpackBxInEvent%2) == 0) ) {
        m_unpackBxInEvent = m_unpackBxInEvent - 1;

        edm::LogInfo("L1GlobalTriggerRawToDigi")
        << "\nWARNING: Number of bunch crossing to be unpacked rounded to: "
        << m_unpackBxInEvent << "\n         The number must be an odd number!\n"
        << std::endl;
    }


    // create GTFE, FDL, PSB cards once per analyzer
    // content will be reset whenever needed

    m_gtfeWord = new L1GtfeWord();
    m_gtFdlWord = new L1GtFdlWord();
    m_gtPsbWord = new L1GtPsbWord();

    // total Bx's in the events will be set after reading GTFE block
    m_totalBxInEvent = 1;

    // loop range: int m_totalBxInEvent is normally even (L1A-1, L1A, L1A+1, with L1A = 0)

}

// destructor
L1GlobalTriggerRawToDigi::~L1GlobalTriggerRawToDigi()
{

    delete m_gtfeWord;
    delete m_gtFdlWord;
    delete m_gtPsbWord;

}


// member functions

void L1GlobalTriggerRawToDigi::beginJob(const edm::EventSetup& evSetup)
{
    // empty
}

// method called to produce the data
void L1GlobalTriggerRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    // get records from EventSetup

    //  muon trigger scales
    edm::ESHandle< L1MuTriggerScales > trigscales_h;
    evSetup.get< L1MuTriggerScalesRcd >().get( trigscales_h );
    m_TriggerScales = trigscales_h.product();

    //  board maps
    edm::ESHandle< L1GtBoardMaps > l1GtBM;
    evSetup.get< L1GtBoardMapsRcd >().get( l1GtBM );

    const std::vector<L1GtBoard> boardMaps = l1GtBM->gtBoardMaps();
    int boardMapsSize = boardMaps.size();
    
    typedef std::vector<L1GtBoard>::const_iterator CItBoardMaps;
    
    // create an ordered vector for the GT DAQ record
    // header (pos 0 in record) and trailer (last position in record) 
    // not included, as they are not in board list 
    std::vector<L1GtBoard> gtRecordMap;
    gtRecordMap.reserve(boardMapsSize);
    
    for (int iPos = 0; iPos < boardMapsSize; ++iPos) {
        for (CItBoardMaps itBoard = boardMaps.begin(); itBoard
                != boardMaps.end(); ++itBoard) {

            if (itBoard->gtPositionDaqRecord() == iPos) {
                gtRecordMap.push_back(*itBoard);
                break;
            }
            
        }
    }
    
    // raw collection

    edm::Handle<FEDRawDataCollection> fedHandle;
    iEvent.getByLabel(m_daqGtInputTag.label(), fedHandle);

    // retrieve data for Global Trigger FED (GT + GMT)
    const FEDRawData& raw =
        (fedHandle.product())->FEDData(m_daqGtFedId);

    int gtSize = raw.size();

    // get a const pointer to the beginning of the data buffer
    const unsigned char* ptrGt = raw.data();

    //
    if ( edm::isDebugEnabled() ) {

        std::ostringstream myCoutStream;
        dumpFedRawData(ptrGt, gtSize, myCoutStream);
        LogTrace("L1GlobalTriggerRawToDigi")
        << "\n Size of raw data: " << gtSize << "\n"
        << "\n Dump FEDRawData\n" << myCoutStream.str() << "\n"
        << std::endl;

    }

    // unpack header
    int headerSize = 8;

    FEDHeader cmsHeader(ptrGt);
    FEDTrailer cmsTrailer(ptrGt + gtSize - headerSize);

    unpackHeader(ptrGt, cmsHeader);
    ptrGt += headerSize; // advance with header size

    // unpack first GTFE to find the length of the record and the active boards
    // here GTFE assumed immediately after the header

    bool gtfeUnpacked = false;

    for (CItBoardMaps
            itBoard = boardMaps.begin();
            itBoard != boardMaps.end(); ++itBoard) {

        if (itBoard->gtBoardType() == GTFE) {

            // unpack GTFE
            if (itBoard->gtPositionDaqRecord() == 1) {

                m_gtfeWord->unpack(ptrGt);
                ptrGt += m_gtfeWord->getSize(); // advance with GTFE block size
                gtfeUnpacked = true;

                if ( edm::isDebugEnabled() ) {

                    std::ostringstream myCoutStream;
                    m_gtfeWord->print(myCoutStream);
                    LogTrace("L1GlobalTriggerRawToDigi")
                    << myCoutStream.str() << "\n"
                    << std::endl;
                }

                // break the loop - GTFE was found
                break;

            } else {

                throw cms::Exception("Configuration")
                << "\nError: GTFE block found in raw data does not follow header.\n"
                << "Assumed start position of the block is wrong!\n"
                << std::endl;

            }

        }
    }

    // throw exception if no GTFE found (action for NotFound: SkipEvent)
    if ( ! gtfeUnpacked ) {

        throw cms::Exception("NotFound")
        << "\nError: no GTFE block found in raw data.\n"
        << "Can not find the record length (BxInEvent) and the active boards!\n"
        << std::endl;
    }

    // life normal here, GTFE found

    // get number of Bx in the event from GTFE block
    m_totalBxInEvent = m_gtfeWord->recordLength();

    // number of BX required to be unpacked

    if (m_unpackBxInEvent > m_totalBxInEvent) {
        edm::LogInfo("L1GlobalTriggerRawToDigi")
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

        LogDebug("L1GlobalTriggerRawToDigi")
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

        LogDebug("L1GlobalTriggerRawToDigi")
        << "\nNo bxInEvent required to be unpacked from "
        << m_totalBxInEvent  << " bunch crosses available." << "\n"
        << std::endl;

        // change RecordLength
        // cast int to boost::uint16_t (there are normally 3 or 5 BxInEvent)
        m_gtfeWord->setRecordLength(static_cast<boost::uint16_t>(m_unpackBxInEvent));

    } else {

        m_lowSkipBxInEvent = (m_totalBxInEvent - m_unpackBxInEvent)/2;
        m_uppSkipBxInEvent = m_totalBxInEvent - m_lowSkipBxInEvent;

        LogDebug("L1GlobalTriggerRawToDigi")
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

    // mask some boards, if needed
    boost::uint16_t activeBoardsGt = activeBoardsGtInitial & m_activeBoardsMaskGt;
    m_gtfeWord->setActiveBoards(activeBoardsGt);

    LogDebug("L1GlobalTriggerRawToDigi")
    << "\nActive boards before masking(hex format): "
    << std::hex << std::setw(sizeof(activeBoardsGtInitial)*2) << std::setfill('0')
    << activeBoardsGtInitial
    << std::dec << std::setfill(' ')
    << "\nActive boards after masking(hex format):  "
    << std::hex << std::setw(sizeof(activeBoardsGt)*2) << std::setfill('0')
    << activeBoardsGt
    << std::dec << std::setfill(' ') << " \n"
    << std::endl;

    // loop over other blocks in the raw record, count them if they are active

    int numberGtfeBoards = 0;
    int numberFdlBoards = 0;
    int numberPsbBoards = 0;
    int numberGmtBoards = 0;
    int numberTcsBoards = 0;
    int numberTimBoards = 0;

    for (CItBoardMaps
            itBoard = boardMaps.begin();
            itBoard != boardMaps.end(); ++itBoard) {

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
                case GTFE: {
                        numberGtfeBoards++;
                    }

                    break;
                case FDL: {
                        numberFdlBoards++;
                    }

                    break;
                case PSB: {
                        numberPsbBoards++;
                    }

                    break;
                case GMT: {
                        numberGmtBoards++;
                    }

                    break;
                case TCS: {
                        numberTcsBoards++;
                    }

                    break;
                case TIM: {
                        numberTimBoards++;
                    }

                    break;
                default: {
                        // do nothing, all blocks are given in GtBoardType enum
                    }

                    break;
            }
        }

    }

    // produce the L1GlobalTriggerReadoutRecord now, after we found how many
    // BxInEvent the record has and how many boards are active
    //LogDebug("L1GlobalTriggerRawToDigi")
    //<< "\nL1GlobalTriggerRawToDigi: producing L1GlobalTriggerReadoutRecord\n"
    //<< "\nL1GlobalTriggerRawToDigi: producing L1MuGMTReadoutCollection;\n"
    //<< std::endl;

    std::auto_ptr<L1GlobalTriggerReadoutRecord> gtReadoutRecord(
        new L1GlobalTriggerReadoutRecord(m_unpackBxInEvent, numberFdlBoards, numberPsbBoards) );

    // produce also the GMT readout collection and set the reference in GT record
    std::auto_ptr<L1MuGMTReadoutCollection> gmtrc(
        new L1MuGMTReadoutCollection(m_unpackBxInEvent) );

    // TODO FIXME fails at running...

    //    edm::RefProd<L1MuGMTReadoutCollection> refProdMuGMT =
    //        iEvent.getRefBeforePut<L1MuGMTReadoutCollection>();
    //
    //    LogDebug("L1GlobalTriggerRawToDigi")
    //    << "\nL1GlobalTriggerRawToDigi: set L1MuGMTReadoutCollection RefProd"
    //    << " in L1GlobalTriggerReadoutRecord.\n"
    //    << std::endl;
    //
    //    gtReadoutRecord->setMuCollectionRefProd(refProdMuGMT);

    // add GTFE block to GT readout record, after updating active boards and record length

    gtReadoutRecord->setGtfeWord(*m_gtfeWord);

    // ... and reset it
    m_gtfeWord->reset();

    // ... then unpack modules other than GTFE, if requested

    for (CItBoardMaps
            itBoard = gtRecordMap.begin();
            itBoard != gtRecordMap.end(); ++itBoard) {

        int iActiveBit = itBoard->gtBitDaqActiveBoards();

        bool activeBoardToUnpack = false;
        bool activeBoardInitial = false;

        if (iActiveBit >= 0) {
            activeBoardInitial = activeBoardsGtInitial & (1 << iActiveBit);
            activeBoardToUnpack = activeBoardsGt & (1 << iActiveBit);
        } else {
            // board not in the ActiveBoards for the record
            continue;
        }

        if ( !activeBoardInitial ) {
            LogDebug("L1GlobalTriggerRawToDigi")
            << "\nBoard of type " << itBoard->gtBoardName()
            << " with index "  << itBoard->gtBoardIndex()
            << " not active initially in raw data.\n"
            << std::endl;

            continue;
        }

        // active board initially, could unpack it
        switch (itBoard->gtBoardType()) {

            case FDL: {
                    for (int iFdl = 0; iFdl < m_totalBxInEvent; ++iFdl) {

                        // unpack only if requested, otherwise skip it
                        if (activeBoardToUnpack) {

                            // unpack only bxInEvent requested, otherwise skip it
                            if (
                                (iFdl >= m_lowSkipBxInEvent) &&
                                (iFdl <  m_uppSkipBxInEvent) ) {

                                m_gtFdlWord->unpack(ptrGt);

                                // add FDL block to GT readout record
                                gtReadoutRecord->setGtFdlWord(*m_gtFdlWord);

                                if ( edm::isDebugEnabled() ) {

                                    std::ostringstream myCoutStream;
                                    m_gtFdlWord->print(myCoutStream);
                                    LogTrace("L1GlobalTriggerRawToDigi")
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
            case PSB: {
                    for (int iPsb = 0; iPsb < m_totalBxInEvent; ++iPsb) {

                        // unpack only if requested, otherwise skip it
                        if (activeBoardToUnpack) {

                            // unpack only bxInEvent requested, otherwise skip it
                            if (
                                (iPsb >= m_lowSkipBxInEvent) &&
                                (iPsb <  m_uppSkipBxInEvent) ) {

                                unpackPSB(evSetup, ptrGt, *m_gtPsbWord);

                                // add PSB block to GT readout record
                                gtReadoutRecord->setGtPsbWord(*m_gtPsbWord);

                                if ( edm::isDebugEnabled() ) {

                                    std::ostringstream myCoutStream;
                                    m_gtPsbWord->print(myCoutStream);
                                    LogTrace("L1GlobalTriggerRawToDigi")
                                    << myCoutStream.str() << "\n"
                                    << std::endl;
                                }

                                // ... and reset it
                                m_gtPsbWord->reset();
                            }

                        }

                        ptrGt += m_gtPsbWord->getSize(); // advance with PSB block size

                    }
                }
                break;
            case GMT: {

                    // unpack only if requested, otherwise skip it
                    if (activeBoardToUnpack) {
                        unpackGMT(ptrGt,gmtrc,iEvent);
                    }

                    // 17*64/8 TODO FIXME ask Ivan for a getSize() function for GMT record
                    unsigned int gmtRecordSize = 136;
                    unsigned int gmtCollSize = m_totalBxInEvent*gmtRecordSize;

                    ptrGt += gmtCollSize; // advance with GMT block size
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
        LogTrace("L1GlobalTriggerRawToDigi")
        << "\n The following L1 GT DAQ readout record was unpacked.\n"
        << myCoutStream.str() << "\n"
        << std::endl;
    }

    // put records into event

    iEvent.put(gmtrc);
    iEvent.put( gtReadoutRecord );

}

// unpack header
void L1GlobalTriggerRawToDigi::unpackHeader(
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


// unpack PSB records
// psbPtr pointer to the beginning of the each PSB block obtained from gtPtr
void L1GlobalTriggerRawToDigi::unpackPSB(
    const edm::EventSetup& evSetup,
    const unsigned char* psbPtr,
    L1GtPsbWord& psbWord)
{

    //LogDebug("L1GlobalTriggerRawToDigi")
    //<< "\nUnpacking PSB block.\n"
    //<< std::endl;

    int uLength = L1GlobalTriggerReadoutSetup::UnitLength;

    int psbSize = psbWord.getSize();
    int psbWords = psbSize/uLength;

    const boost::uint64_t* payload =
        reinterpret_cast<boost::uint64_t*>(const_cast<unsigned char*>(psbPtr));

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

        LogTrace("L1GlobalTriggerRawToDigi")
        << std::setw(4) << iWord << "  "
        << std::hex << std::setfill('0')
        << std::setw(16) << payload[iWord]
        << std::dec << std::setfill(' ')
        << std::endl;

    }

}

// unpack the GMT record
void L1GlobalTriggerRawToDigi::unpackGMT(
    const unsigned char* chp,
    std::auto_ptr<L1MuGMTReadoutCollection>& gmtrc,
    edm::Event& iEvent)
{

    //LogDebug("L1GlobalTriggerRawToDigi")
    //<< "\nUnpacking GMT collection.\n"
    //<< std::endl;

    // 17*64/2 TODO FIXME ask Ivan for a getSize() function for GMT record
    const unsigned int gmtRecordSize32 = 34;

    std::auto_ptr<std::vector<L1MuRegionalCand> > DTCands( new std::vector<L1MuRegionalCand> );
    std::auto_ptr<std::vector<L1MuRegionalCand> > CSCCands( new std::vector<L1MuRegionalCand> );
    std::auto_ptr<std::vector<L1MuRegionalCand> > RPCbCands( new std::vector<L1MuRegionalCand> );
    std::auto_ptr<std::vector<L1MuRegionalCand> > RPCfCands( new std::vector<L1MuRegionalCand> );
    std::auto_ptr<std::vector<L1MuGMTCand> >      GMTCands( new std::vector<L1MuGMTCand> );

    const unsigned* p = (const unsigned*) chp;

    // min Bx's in the event, computed after m_totalBxInEvent is obtained from GTFE block
    // assume symmetrical number of BX around L1Accept
    int iBxInEvent = (m_totalBxInEvent + 1)/2 - m_totalBxInEvent;

    for (int iGmtRec = 0; iGmtRec < m_totalBxInEvent; ++iGmtRec) {

        // unpack only bxInEvent requested, otherwise skip it
        if (
            (iGmtRec >= m_lowSkipBxInEvent) &&
            (iGmtRec <  m_uppSkipBxInEvent) ) {

            // Dump the block
            const boost::uint64_t* bp =
                reinterpret_cast<boost::uint64_t*>(const_cast<unsigned*>(p));
            for(int iWord=0; iWord<17; iWord++) {
                LogTrace("L1GlobalTriggerRawToDigi")
                << std::setw(4) << iWord << "  "
                << std::hex << std::setfill('0')
                << std::setw(16) << *bp++
                << std::dec << std::setfill(' ')
                << std::endl;
            }

            L1MuGMTReadoutRecord gmtrr(iBxInEvent);

            gmtrr.setEvNr((*p)&0xffffff);
            gmtrr.setBCERR(((*p)>>24)&0xff);
            p++;

            gmtrr.setBxNr((*p)&0xfff);
            if(((*p)>>15)&1) {
                gmtrr.setBxInEvent((((*p)>>12)&7)-8);
            } else {
                gmtrr.setBxInEvent((((*p)>>12)&7));
            }
            // to do: check here the block length and the board id
            p++;

            for(int im=0; im<16; im++) {
                // flip the pt and quality bits -- this should better be done by GMT input chips
                unsigned waux = *p++;
                waux = (waux&0xffff00ff) | ((~waux)&0x0000ff00);
                L1MuRegionalCand cand(waux,iBxInEvent);
                // fix the type assignment (csc=2, rpcb=1) -- should be done by GMT input chips
                if(im>=4 && im<8)
                    cand.setType(1);
                if(im>=8 && im<12)
                    cand.setType(2);
                cand.setPhiValue( m_TriggerScales->getPhiScale()->getLowEdge(cand.phi_packed()) );
                cand.setEtaValue( m_TriggerScales->getRegionalEtaScale(cand.type_idx())->getCenter(cand.eta_packed()) );
                cand.setPtValue( m_TriggerScales->getPtScale()->getLowEdge(cand.pt_packed()) );
                gmtrr.setInputCand(im, cand);
                if(!cand.empty()) {
                    if(im<4)
                        DTCands->push_back(cand);
                    if(im>=4 && im<8)
                        RPCbCands->push_back(cand);
                    if(im>=8 && im<12)
                        CSCCands->push_back(cand);
                    if(im>=12)
                        RPCfCands->push_back(cand);
                }
            }

            unsigned char* prank = (unsigned char*) (p+12);

            for(int im=0; im<12; im++) {
                unsigned waux = *p++;
                unsigned raux = im<8 ? *prank++ : 0; // only fwd and brl cands have valid rank
                L1MuGMTExtendedCand cand(waux,raux,iBxInEvent);
                cand.setPhiValue( m_TriggerScales->getPhiScale()->getLowEdge( cand.phiIndex() ));
                cand.setEtaValue( m_TriggerScales->getGMTEtaScale()->getCenter( cand.etaIndex() ));
                cand.setPtValue( m_TriggerScales->getPtScale()->getLowEdge( cand.ptIndex() ));
                if(im<4)
                    gmtrr.setGMTBrlCand(im, cand);
                else if(im<8)
                    gmtrr.setGMTFwdCand(im-4, cand);
                else {
                    gmtrr.setGMTCand(im-8, cand);
                    if(!cand.empty())
                        GMTCands->push_back(cand);
                }
            }

            // skip the two sort rank words and two chip BX words
            p+=4;

            gmtrc->addRecord(gmtrr);

        } else {
            // increase the pointer with the GMT record size
            p += gmtRecordSize32;
        }

        // increase the BxInEvent number
        iBxInEvent++;

    }

    iEvent.put(DTCands,"DT");
    iEvent.put(CSCCands,"CSC");
    iEvent.put(RPCbCands,"RPCb");
    iEvent.put(RPCfCands,"RPCf");
    iEvent.put(GMTCands);

}

// unpack trailer word
// trPtr pointer to the beginning of trailer obtained from gtPtr
void L1GlobalTriggerRawToDigi::unpackTrailer(
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
void L1GlobalTriggerRawToDigi::dumpFedRawData(
    const unsigned char* gtPtr,
    int gtSize,
    std::ostream& myCout)
{

    LogDebug("L1GlobalTriggerRawToDigi")
    << "\nDump FED raw data.\n"
    << std::endl;

    int wLength = L1GlobalTriggerReadoutSetup::WordLength;
    int uLength = L1GlobalTriggerReadoutSetup::UnitLength;

    int gtWords = gtSize/uLength;
    LogTrace("L1GlobalTriggerRawToDigi")
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
void L1GlobalTriggerRawToDigi::endJob()
{

    // empty now
}


// static class members

