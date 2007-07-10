/**
 * \class L1GTDigiToRaw
 * 
 * 
 * Description: generate raw data from digis.  
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
#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GTDigiToRaw.h"

// system include files
#include <vector>
#include <iostream>
#include <iomanip>

#include <boost/cstdint.hpp>

// user include files
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/Common/interface/RefProd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"


// constructor(s)
L1GTDigiToRaw::L1GTDigiToRaw(const edm::ParameterSet& pSet)
{

    // FED Id for GT DAQ record
    // default value defined in DataFormats/FEDRawData/src/FEDNumbering.cc
    // default value: assume the DAQ record is the last GT record 
    m_daqGtFedId = pSet.getUntrackedParameter<int>(
                       "DaqGtFedId", FEDNumbering::getTriggerGTPFEDIds().second);

    LogDebug("L1GTDigiToRaw")
    << "\nFED Id for DAQ GT record: "
    << m_daqGtFedId << " \n"
    << std::endl;

    // input tag for DAQ GT record
    m_daqGtInputTag = pSet.getUntrackedParameter<edm::InputTag>(
                          "DaqGtInputTag", edm::InputTag("L1GtEmul"));

    LogDebug("L1GTDigiToRaw")
    << "\nInput tag for DAQ GT record: "
    << m_daqGtInputTag.label() << " \n"
    << std::endl;

    // input tag for GMT record
    m_muGmtInputTag = pSet.getUntrackedParameter<edm::InputTag>(
                          "MuGmtInputTag", edm::InputTag("gmt"));

    LogDebug("L1GTDigiToRaw")
    << "\nInput tag for GMT record: "
    << m_muGmtInputTag.label() << " \n"
    << std::endl;

    // mask for active boards
    m_activeBoardsMaskGt = pSet.getParameter<unsigned int>("ActiveBoardsMask");

    LogDebug("L1GTDigiToRaw")
    << "\nMask for active boards (hex format): "
    << std::hex << std::setw(sizeof(m_activeBoardsMaskGt)*2) << std::setfill('0')
    << m_activeBoardsMaskGt
    << std::dec << std::setfill(' ') << " \n"
    << std::endl;

    //
    produces<FEDRawDataCollection>();

}

// destructor
L1GTDigiToRaw::~L1GTDigiToRaw()
{

    // empty now

}

// member functions

// beginning of job stuff
void L1GTDigiToRaw::beginJob(const edm::EventSetup& evSetup)
{

    // empty now

}


// method called to produce the data
void L1GTDigiToRaw::produce(edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    // get L1GlobalTriggerReadoutRecord
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    iEvent.getByLabel(m_daqGtInputTag.label(), gtReadoutRecord);

    if ( edm::isDebugEnabled() ) {
        std::ostringstream myCoutStream;
        gtReadoutRecord->print(myCoutStream);
        LogTrace("L1GTDigiToRaw")
        << "\n The following L1 GT DAQ readout record will be packed.\n"
        << " Some boards could be disabled before packing,"
        << " see detailed board packing.\n"
        << myCoutStream.str() << "\n"
        << std::endl;
    }

    //
    L1GlobalTriggerReadoutSetup tmpGtSetup; // TODO FIXME temporary event setup
    std::map<int, L1GlobalTriggerReadoutSetup::GtBoard> recordMap =
        tmpGtSetup.GtDaqRecordMap;

    typedef std::map<int, L1GlobalTriggerReadoutSetup::GtBoard>::const_iterator CItRecord;

    // get GTFE block
    L1GtfeWord gtfeBlock = gtReadoutRecord->gtfeWord();

    // set the number of Bx in the event
    m_totalBxInEvent = gtfeBlock.recordLength();

    m_minBxInEvent = (m_totalBxInEvent + 1)/2 - m_totalBxInEvent;
    m_maxBxInEvent = (m_totalBxInEvent + 1)/2 - 1;

    LogDebug("L1GTDigiToRaw")
    << "\nNumber of bunch crosses in the record: "
    << m_totalBxInEvent << " = " << "["
    << m_minBxInEvent << ", " << m_maxBxInEvent << "] BX\n"
    << std::endl;

    // GTFE is not in the list of active boards, need separate treatment
    int gtfeKey = -1; // negative integer for GTFE key

    for (CItRecord itRecord = recordMap.begin();
            itRecord != recordMap.end(); ++itRecord) {

        if (itRecord->second.boardType == GTFE) {

            gtfeKey = itRecord->first;
            break; // there is only one GTFE block

        }
    }

    // throw exception if no GTFE found (action for NotFound: SkipEvent)
    if (gtfeKey < 0 ) {

        throw cms::Exception("NotFound")
        << "\nError: GTFE block not requested to be written in raw data.\n"
        << "Can not find afterwards the record length (BxInEvent) and the active boards!\n"
        << std::endl;
    }


    // get list of active blocks from the GTFE block
    // and mask some blocks, if required
    // blocks not active are not written to the record

    boost::uint16_t activeBoardsGtInitial = gtfeBlock.activeBoards();

    LogDebug("L1GTDigiToRaw")
    << "\nActive boards before masking(hex format): "
    << std::hex << std::setw(sizeof(activeBoardsGtInitial)*2) << std::setfill('0')
    << activeBoardsGtInitial
    << std::dec << std::setfill(' ')
    << std::endl;

    // mask some boards, if needed

    boost::uint16_t activeBoardsGt = activeBoardsGtInitial & m_activeBoardsMaskGt;

    LogTrace("L1GTDigiToRaw")
    << "Active boards after masking(hex format):  "
    << std::hex << std::setw(sizeof(activeBoardsGt)*2) << std::setfill('0')
    << activeBoardsGt
    << std::dec << std::setfill(' ') << " \n"
    << std::endl;

    std::map<L1GlobalTriggerReadoutSetup::GtBoard, int> activeBoardsMap =
        tmpGtSetup.GtDaqActiveBoardsMap;
    typedef std::map<L1GlobalTriggerReadoutSetup::GtBoard, int>::const_iterator CItActive;


    // get the size of the record

    unsigned int gtDataSize = 0;

    unsigned int headerSize = 8;
    gtDataSize += headerSize;

    for (CItRecord itRecord = recordMap.begin();
            itRecord != recordMap.end(); ++itRecord) {

        if (itRecord->first == gtfeKey) {
            gtDataSize += gtfeBlock.getSize();
            continue;
        }

        // size of modules other than GTFE

        // skip if the board is not active
        bool activeBoard = false;

        CItActive itBoard = activeBoardsMap.find(itRecord->second);
        if (itBoard != activeBoardsMap.end()) {
            activeBoard = activeBoardsGt & (1 << (itBoard->second));
        } else {

            // board not found in the map
            LogDebug("L1GTDigiToRaw")
            << "\nBoard of type " << itRecord->second.boardType
            << " with index "  << itRecord->second.boardIndex
            << " not found in the activeBoardsMap\n"
            << std::endl;

            continue;
        }

        if ( !activeBoard ) {

            LogDebug("L1GTDigiToRaw")
            << "\nBoard of type " << itRecord->second.boardType
            << " with index "  << itRecord->second.boardIndex
            << " not active (from activeBoardsMap)\n"
            << std::endl;

            continue;
        }

        // active board, add its size
        switch (itRecord->second.boardType) {

            case FDL: {
                    L1GtFdlWord fdlBlock;
                    gtDataSize += m_totalBxInEvent*fdlBlock.getSize();
                }
                break;
            case PSB: {
                    L1GtPsbWord psbBlock;
                    gtDataSize += m_totalBxInEvent*psbBlock.getSize();
                }
                break;
            case GMT: {
                    // 17*64/8 TODO FIXME ask Ivan for a getSize() function for GMT record
                    unsigned int gmtRecordSize = 136;
                    unsigned int gmtCollSize = m_totalBxInEvent*gmtRecordSize;
                    gtDataSize += gmtCollSize;
                }
                break;
            default: {

                    // do nothing, all blocks are given in GtBoardType enum
                }
                break;
        }

    }

    unsigned int trailerSize = 8;
    gtDataSize += trailerSize;

    // define new FEDRawDataCollection
    // it contains ALL FEDs in an event
    std::auto_ptr<FEDRawDataCollection> allFedRawData(new FEDRawDataCollection);

    // ptrGt: pointer to the beginning of GT record in the raw data

    FEDRawData& gtRawData = allFedRawData->FEDData(m_daqGtFedId);
    
    // resize, GT raw data record has variable length,
    // depending on active boards (read in GTFE)
    gtRawData.resize(gtDataSize);


    unsigned char* ptrGt = gtRawData.data();

    LogDebug("L1GTDigiToRaw")
    << "\n Size of raw data: " << gtRawData.size() << "\n"
    << std::endl;


    // ------- pack boards -------

    // pack header
    packHeader(ptrGt);
    ptrGt += headerSize; // advance with header size

    // loop over other blocks in the raw record, if they are active

    for (CItRecord itRecord = recordMap.begin();
            itRecord != recordMap.end(); ++itRecord) {

        if (itRecord->first == gtfeKey) {

            packGTFE(evSetup, ptrGt, gtfeBlock, activeBoardsGt);

            if ( edm::isDebugEnabled() ) {

                std::ostringstream myCoutStream;
                gtfeBlock.print(myCoutStream);
                LogTrace("L1GTDigiToRaw")
                << myCoutStream.str() << "\n"
                << std::endl;
            }

            ptrGt += gtfeBlock.getSize(); // advance with GTFE block size

            continue;
        }

        // pack modules other than GTFE

        // skip if the board is not active
        bool activeBoard = false;

        CItActive itBoard = activeBoardsMap.find(itRecord->second);
        if (itBoard != activeBoardsMap.end()) {
            activeBoard = activeBoardsGt & (1 << (itBoard->second));
        } else {
            // board not found in the map

            LogDebug("L1GTDigiToRaw")
            << "\nBoard of type " << itRecord->second.boardType
            << " with index "  << itRecord->second.boardIndex
            << " not found in the activeBoardsMap\n"
            << std::endl;

            continue;
        }

        if ( !activeBoard ) {

            LogDebug("L1GTDigiToRaw")
            << "\nBoard of type " << itRecord->second.boardType
            << " with index "  << itRecord->second.boardIndex
            << " not active (from activeBoardsMap)\n"
            << std::endl;

            continue;
        }

        // active board, pack it
        switch (itRecord->second.boardType) {

            case FDL: {

                    for (int iBxInEvent = m_minBxInEvent; iBxInEvent <= m_maxBxInEvent;
                            ++iBxInEvent) {

                        L1GtFdlWord fdlBlock = gtReadoutRecord->gtFdlWord(iBxInEvent);
                        packFDL(evSetup, ptrGt, fdlBlock);

                        if ( edm::isDebugEnabled() ) {

                            std::ostringstream myCoutStream;
                            fdlBlock.print(myCoutStream);
                            LogTrace("L1GTDigiToRaw")
                            << myCoutStream.str() << "\n"
                            << std::endl;
                        }

                        ptrGt += fdlBlock.getSize(); // advance with FDL block size
                    }

                }
                break;
            case PSB: {

                    boost::uint16_t boardIdValue = itRecord->second.boardId();

                    LogDebug("L1GTDigiToRaw")
                    << "\nBoard of type " << itRecord->second.boardType
                    << " with index "  << itRecord->second.boardIndex
                    << " has the boardId " << std::hex << boardIdValue << std::dec << "\n"
                    << std::endl;

                    for (int iBxInEvent = m_minBxInEvent; iBxInEvent <= m_maxBxInEvent;
                            ++iBxInEvent) {

                        L1GtPsbWord psbBlock =
                            gtReadoutRecord->gtPsbWord(boardIdValue, iBxInEvent);

                        packPSB(evSetup, ptrGt, psbBlock);

                        if ( edm::isDebugEnabled() ) {

                            std::ostringstream myCoutStream;
                            psbBlock.print(myCoutStream);
                            LogTrace("L1GTDigiToRaw")
                            << myCoutStream.str() << "\n"
                            << std::endl;
                        }


                        ptrGt += psbBlock.getSize(); // advance with PSB block size
                    }

                }
                break;
            case GMT: {

                    // get GMT record TODO separate GMT record or via RefProd from GT record
                    edm::Handle<L1MuGMTReadoutCollection> gmtrc_handle;
                    iEvent.getByLabel(m_muGmtInputTag.label(), gmtrc_handle);
                    L1MuGMTReadoutCollection const* gmtrc = gmtrc_handle.product();

                    // pack the GMT record

                    unsigned int gmtCollSize = 0;
                    gmtCollSize = packGmtCollection(ptrGt, gmtrc);
                    ptrGt += gmtCollSize; // advance with GMT collection size

                }
                break;
            default: {

                    // do nothing, all blocks are given in GtBoardType enum
                    break;
                }
        }

    }

    // pack trailer
    //    FEDTrailer cmsTrailerGt = gtReadoutRecord.cmsTrailer();
    packTrailer(ptrGt, gtDataSize);

    // put the raw data in the event

    iEvent.put(allFedRawData);
}


// pack header
void L1GTDigiToRaw::packHeader(unsigned char* ptrGt)
{
    // TODO FIXME where from to get all numbers?

    // Event Trigger type identifier
    int triggerTypeVal = 0;

    // Level-1 event number generated by the TTC system
    int lvl1IdVal = 0;

    // The bunch crossing number
    int bxIdVal = 0;

    // Identifier of the FED
    int sourceIdVal = m_daqGtFedId;

    // Version identifier of the FED data format
    int versionVal = 0;

    // 0 -> the current header word is the last one.
    // 1-> other header words can follow
    // (always 1 for ECAL)
    bool moreHeadersVal = false;


    FEDHeader::set(ptrGt,
                   triggerTypeVal, lvl1IdVal, bxIdVal, sourceIdVal, versionVal,
                   moreHeadersVal);

}

// pack the GTFE block
void L1GTDigiToRaw::packGTFE(
    const edm::EventSetup& evSetup,
    unsigned char* ptrGt,
    L1GtfeWord& gtfeBlock,
    boost::uint16_t activeBoardsGtValue)
{

    LogDebug("L1GTDigiToRaw")
    << "\nPacking GTFE \n"
    << std::endl;

    int uLength = L1GlobalTriggerReadoutSetup::UnitLength;

    // initialize the required number of word64
    int nrWord64 = gtfeBlock.getSize()/uLength;
    std::vector<boost::uint64_t> tmpWord64;
    tmpWord64.resize(nrWord64);

    for (int iWord = 0; iWord < nrWord64; ++iWord) {
        tmpWord64[iWord] = 0x0000000000000000ULL;
    }

    // fill the values in the words
    for (int iWord = 0; iWord < nrWord64; ++iWord) {

        gtfeBlock.setBoardIdWord64(tmpWord64[iWord], iWord);
        gtfeBlock.setRecordLengthWord64(tmpWord64[iWord], iWord);
        gtfeBlock.setBxNrWord64(tmpWord64[iWord], iWord);
        gtfeBlock.setSetupVersionWord64(tmpWord64[iWord], iWord);
        gtfeBlock.setActiveBoardsWord64(tmpWord64[iWord], iWord, activeBoardsGtValue);
        gtfeBlock.setTotalTriggerNrWord64(tmpWord64[iWord], iWord);

    }

    // put the words in the FED record

    boost::uint64_t* pw =
        reinterpret_cast<boost::uint64_t*>(const_cast<unsigned char*>(ptrGt));

    for (int iWord = 0; iWord < nrWord64; ++iWord) {

        *pw++ = tmpWord64[iWord];

        LogTrace("L1GTDigiToRaw")
        << std::setw(4) << iWord << "  "
        << std::hex << std::setfill('0')
        << std::setw(16) << tmpWord64[iWord]
        << std::dec << std::setfill(' ')
        << std::endl;
    }


}


// pack the FDL block
void L1GTDigiToRaw::packFDL(
    const edm::EventSetup& evSetup,
    unsigned char* ptrGt,
    L1GtFdlWord& fdlBlock)
{

    LogDebug("L1GTDigiToRaw")
    << "\nPacking FDL \n"
    << std::endl;

    int uLength = L1GlobalTriggerReadoutSetup::UnitLength;

    // initialize the required number of word64
    int nrWord64 = fdlBlock.getSize()/uLength;
    std::vector<boost::uint64_t> tmpWord64;
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

        fdlBlock.setNoAlgoWord64(tmpWord64[iWord], iWord);
        fdlBlock.setFinalORWord64(tmpWord64[iWord], iWord);

        fdlBlock.setLocalBxNrWord64(tmpWord64[iWord], iWord);

    }

    // put the words in the FED record

    boost::uint64_t* pw =
        reinterpret_cast<boost::uint64_t*>(const_cast<unsigned char*>(ptrGt));

    for (int iWord = 0; iWord < nrWord64; ++iWord) {

        *pw++ = tmpWord64[iWord];

        LogTrace("L1GTDigiToRaw")
        << std::setw(4) << iWord << "  "
        << std::hex << std::setfill('0')
        << std::setw(16) << tmpWord64[iWord]
        << std::dec << std::setfill(' ')
        << std::endl;
    }

}

// pack the PSB block
void L1GTDigiToRaw::packPSB(
    const edm::EventSetup& evSetup,
    unsigned char* ptrGt,
    L1GtPsbWord& psbBlock)
{

    LogDebug("L1GTDigiToRaw")
    << "\nPacking PSB \n"
    << std::endl;

    int uLength = L1GlobalTriggerReadoutSetup::UnitLength;

    // initialize the required number of word64
    int nrWord64 = psbBlock.getSize()/uLength;
    std::vector<boost::uint64_t> tmpWord64;
    tmpWord64.resize(nrWord64);

    for (int iWord = 0; iWord < nrWord64; ++iWord) {
        tmpWord64[iWord] = 0x0000000000000000ULL;
    }

    // fill the values in the words
    for (int iWord = 0; iWord < nrWord64; ++iWord) {

        psbBlock.setBoardIdWord64(tmpWord64[iWord], iWord);
        psbBlock.setBxInEventWord64(tmpWord64[iWord], iWord);
        psbBlock.setBxNrWord64(tmpWord64[iWord], iWord);
        psbBlock.setEventNrWord64(tmpWord64[iWord], iWord);

        psbBlock.setADataWord64(tmpWord64[iWord], iWord);
        psbBlock.setBDataWord64(tmpWord64[iWord], iWord);

        psbBlock.setLocalBxNrWord64(tmpWord64[iWord], iWord);

    }

    // put the words in the FED record

    boost::uint64_t* pw =
        reinterpret_cast<boost::uint64_t*>(const_cast<unsigned char*>(ptrGt));

    for (int iWord = 0; iWord < nrWord64; ++iWord) {

        *pw++ = tmpWord64[iWord];

        LogTrace("L1GTDigiToRaw")
        << std::setw(4) << iWord << "  "
        << std::hex << std::setfill('0')
        << std::setw(16) << tmpWord64[iWord]
        << std::dec << std::setfill(' ')
        << std::endl;
    }

}

// pack the GMT collection using packGMT (GMT record packing)
unsigned int L1GTDigiToRaw::packGmtCollection(
    unsigned char* ptrGt,
    L1MuGMTReadoutCollection const* digis)
{

    LogDebug("L1GTDigiToRaw")
    << "\nPacking GMT collection \n"
    << std::endl;

    unsigned gmtsize = 0;

    // loop range: int m_totalBxInEvent is normally even (L1A-1, L1A, L1A+1, with L1A = 0)
    for (int iBxInEvent = m_minBxInEvent; iBxInEvent <= m_maxBxInEvent;
            ++iBxInEvent) {
        L1MuGMTReadoutRecord const& gmtrr = digis->getRecord(iBxInEvent);
        gmtsize = packGMT(gmtrr, ptrGt);
        ptrGt += gmtsize;
    }

    return m_totalBxInEvent*gmtsize;

}

// pack a GMT record
unsigned L1GTDigiToRaw::packGMT(L1MuGMTReadoutRecord const& gmtrr, unsigned char* chp)
{

    const unsigned SIZE=136;
    memset(chp,0,SIZE);

    unsigned* p = (unsigned*) chp;

    // event number + bcerr
    *p++ = (gmtrr.getEvNr()&0xffffff) | ((gmtrr.getBCERR()&0xff)<<24);
    // bx number, bx in event, length(?), board-id(?)
    *p++ = (gmtrr.getBxNr()&0xfff) | ((gmtrr.getBxInEvent()&0xf)<<12);

    std::vector<L1MuRegionalCand> vrc;
    std::vector<L1MuRegionalCand>::const_iterator irc;
    unsigned* pp = p;

    vrc = gmtrr.getDTBXCands();
    pp = p;
    for(irc=vrc.begin(); irc!=vrc.end(); irc++) {
        *pp++ = (*irc).getDataWord();
    }
    p+=4;

    vrc = gmtrr.getBrlRPCCands();
    pp = p;
    for(irc=vrc.begin(); irc!=vrc.end(); irc++) {
        *pp++ = (*irc).getDataWord();
    }
    p+=4;

    vrc = gmtrr.getCSCCands();
    pp = p;
    for(irc=vrc.begin(); irc!=vrc.end(); irc++) {
        *pp++ = (*irc).getDataWord();
    }
    p+=4;

    vrc = gmtrr.getFwdRPCCands();
    pp = p;
    for(irc=vrc.begin(); irc!=vrc.end(); irc++) {
        *pp++ = (*irc).getDataWord();
    }
    p+=4;

    std::vector<L1MuGMTExtendedCand> vgc;
    std::vector<L1MuGMTExtendedCand>::const_iterator igc;

    vgc = gmtrr.getGMTBrlCands();
    pp = p;
    for(igc=vgc.begin(); igc!=vgc.end(); igc++) {
        *pp++ = (*igc).getDataWord();
    }
    p+=4;

    vgc = gmtrr.getGMTFwdCands();
    pp = p;
    for(igc=vgc.begin(); igc!=vgc.end(); igc++) {
        *pp++ = (*igc).getDataWord();
    }
    p+=4;

    vgc = gmtrr.getGMTCands();
    pp = p;
    for(igc=vgc.begin(); igc!=vgc.end(); igc++) {
        *pp++ = (*igc).getDataWord();
    }
    p+=4;

    unsigned char* chpp;

    vgc = gmtrr.getGMTBrlCands();
    chpp = (unsigned char*) p;
    for(igc=vgc.begin(); igc!=vgc.end(); igc++) {
        *chpp++ = (*igc).rank();
    }
    p++;

    vgc = gmtrr.getGMTFwdCands();
    chpp = (unsigned char*) p;
    for(igc=vgc.begin(); igc!=vgc.end(); igc++) {
        *chpp++ = (*igc).rank();
    }
    p++;

    return SIZE;
}

// pack trailer
void L1GTDigiToRaw::packTrailer(unsigned char* ptrGt, int dataSize)
{

    // TODO FIXME where from to get all numbers?

    // The length of the event fragment counted in 64-bit words including header and trailer
    int lengthVal = dataSize/8;

    // Cyclic Redundancy Code of the event fragment including header and trailer
    int crcVal = 0;

    // Event fragment status information
    int evtStatusVal = 0;

    // Current value of the Trigger Throttling System bits.
    int ttsBitsVal = 0;

    // 0 -> the current trailer word is the last one.
    // 1-> other trailer words can follow
    // (always 0 for ECAL)
    bool moreTrailersVal = false;


    FEDTrailer::set(ptrGt,
                    lengthVal, crcVal, evtStatusVal, ttsBitsVal,
                    moreTrailersVal);

}


//
void L1GTDigiToRaw::endJob()
{

    // empty now
}


// static class members
