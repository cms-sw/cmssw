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
 * $Date:$
 * $Revision:$
 *
 */

// this class header
#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GTDigiToRaw.h"

// system include files
#include <vector>

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

    // input tag for DAQ GT record
    m_daqGtInputTag = pSet.getUntrackedParameter<edm::InputTag>(
                          "DaqGtInputTag", edm::InputTag("L1GtEmul"));

    LogDebug("L1GTDigiToRaw")
    << "\nInput tag for DAQ GT record: "
    << m_daqGtInputTag.label() << " \n"
    << std::endl;

    // flag to keep or change the active boards
    m_keepActiveBoardsStatus = pSet.getParameter<bool>("KeepActiveBoardsStatus");

    LogDebug("L1GTDigiToRaw")
    << "\nKeepActiveBoardsStatus: "
    << m_keepActiveBoardsStatus << " \n"
    << std::endl;

    // list of active boards
    m_activeBoardsGt = pSet.getParameter<int>("ActiveBoards");

    LogDebug("L1GTDigiToRaw")
    << "\nActive boards: "
    << m_activeBoardsGt << " \n"
    << std::endl;

    //
    produces<FEDRawDataCollection>();

}

// destructor
L1GTDigiToRaw::~L1GTDigiToRaw()
{}

// member functions

// beginning of job stuff
void L1GTDigiToRaw::beginJob(const edm::EventSetup& evSetup)
{

    // nothing yet

}


// method called to produce the data
void L1GTDigiToRaw::produce(edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    // define new FEDRawDataCollection
    // it contains ALL FEDs in an event
    std::auto_ptr<FEDRawDataCollection> allFedRawData(new FEDRawDataCollection);

    // ptrGt: pointer to the beginning of GT record in the raw data

    FEDRawData& gtRawData = allFedRawData->FEDData(FEDNumbering::getTriggerGTPFEDIds().first);
    // no resize, GT raw data record has variable length,
    // depending on active boards (read in GTFE)

    unsigned char* ptrGt = gtRawData.data();

    // get L1GlobalTriggerReadoutRecord
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    iEvent.getByLabel(m_daqGtInputTag.label(), gtReadoutRecord);

    //
    L1GlobalTriggerReadoutSetup tmpGtSetup; // TODO FIXME temporary event setup
    std::map<int, L1GlobalTriggerReadoutSetup::GtBoard> recordMap =
        tmpGtSetup.GtDaqRecordMap;

    typedef std::map<int, L1GlobalTriggerReadoutSetup::GtBoard>::const_iterator CItRecord;

    // get GTFE block
    L1GtfeWord gtfeBlock = gtReadoutRecord->gtfeWord();

    // set the number of Bx in the event to the right number
    m_totalBxInEvent = gtfeBlock.recordLength();

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


    if (m_keepActiveBoardsStatus) {
        // get list of active blocks from the GTFE block
        // and replace the list initialized in constructor
        // blocks not active are not written to the record

        m_activeBoardsGt = gtfeBlock.activeBoards();

    }

    // ------- pack boards -------

    // pack header // FIXME

    //    FEDHeader cmsHeaderGt = gtReadoutRecord.cmsHeader();
    unsigned int headerSize = 8;

    //    packHeader(ptrGt, cmsHeaderGt);
    ptrGt += headerSize; // advance with header size

    // loop over other blocks in the raw record, if they are active

    std::map<L1GlobalTriggerReadoutSetup::GtBoard, int> activeBoardsMap =
        tmpGtSetup.GtDaqActiveBoardsMap;
    typedef std::map<L1GlobalTriggerReadoutSetup::GtBoard, int>::const_iterator CItActive;

    for (CItRecord itRecord = recordMap.begin();
            itRecord != recordMap.end(); ++itRecord) {

        if (itRecord->first == gtfeKey) {
            packGTFE(evSetup, ptrGt, gtfeBlock);
            ptrGt += gtfeBlock.getSize(); // advance with GTFE block size

            continue;
        }

        // pack modules other than GTFE

        // skip if the board is not active
        bool activeBoard = false;

        CItActive itBoard = activeBoardsMap.find(itRecord->second);
        if (itBoard != activeBoardsMap.end()) {
            activeBoard = m_activeBoardsGt & (1 << (itBoard->second));
        } else {
            // board not found in the map
            continue;
        }

        if ( !activeBoard ) {
            continue;
        }

        // active board, pack it
        switch (itRecord->second.boardType) {

            case FDL: {

                    for (int iBxInEvent = 0; iBxInEvent < m_totalBxInEvent; ++iBxInEvent) {
                        L1GtFdlWord fdlBlock = gtReadoutRecord->gtFdlWord(iBxInEvent);
                        packFDL(evSetup, ptrGt, fdlBlock);
                        ptrGt += fdlBlock.getSize(); // advance with FDL block size
                    }

                }
                break;
            case PSB: {

                    for (int iBxInEvent = 0; iBxInEvent < m_totalBxInEvent; ++iBxInEvent) {
                        L1GtPsbWord psbBlock = gtReadoutRecord->gtPsbWord(iBxInEvent);
                        packPSB(evSetup, ptrGt, psbBlock);
                        ptrGt += psbBlock.getSize(); // advance with PSB block size
                    }

                }
                break;
            case GMT: {

                    // get GMT record TODO separate GMT record or via RefProd from GT record
                    edm::Handle<L1MuGMTReadoutCollection> gmtrc_handle;
                    iEvent.getByType(gmtrc_handle);
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
    //    unsigned int trailerSize = 8;
    //
    //    packTrailer(ptrGt, cmsTrailerGt);


    // put the raw data in the event

    iEvent.put(allFedRawData);
}

// pack the GTFE block
void L1GTDigiToRaw::packGTFE(
    const edm::EventSetup& evSetup, unsigned char* ptrGt, L1GtfeWord& gtfeBlock)
{

    unsigned char* ptrGtV = ptrGt;

    // initialize the required number of word64
    int nrWord64 = gtfeBlock.getSize()/L1GlobalTriggerReadoutSetup::UnitLength;
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
        gtfeBlock.setActiveBoardsWord64(tmpWord64[iWord], iWord);
        gtfeBlock.setTotalTriggerNrWord64(tmpWord64[iWord], iWord);

    }

    // put the words in the FED record
    for (int iWord = 0; iWord < nrWord64; ++iWord) {
        *ptrGtV++ = tmpWord64[iWord];
    }



}


// pack the FDL block
void L1GTDigiToRaw::packFDL(
    const edm::EventSetup& evSetup, unsigned char* ptrGt, L1GtFdlWord& fdlBlock)
{

    //
}

// pack the PSB block
void L1GTDigiToRaw::packPSB(
    const edm::EventSetup& evSetup, unsigned char* ptrGt, L1GtPsbWord& psbBlock)
{

    //
}

// pack the GMT collection using packGMT (GMT record packing)
unsigned int L1GTDigiToRaw::packGmtCollection(
    unsigned char* ptrGt,
    L1MuGMTReadoutCollection const* digis)
{

    unsigned gmtsize = 0;

    // loop range: int m_totalBxInEvent is normally even (L1A-1, L1A, L1A+1, with L1A = 0)
    int bxMin = (m_totalBxInEvent + 1)/2 - m_totalBxInEvent; 
    int bxMax = (m_totalBxInEvent + 1)/2; 
    
    for(int ibx = bxMin; ibx < bxMax; ibx++) {
        L1MuGMTReadoutRecord const& gmtrr = digis->getRecord(ibx);
        gmtsize = packGMT(gmtrr, ptrGt);
        ptrGt += gmtsize;
    }

    return m_totalBxInEvent*gmtsize;

}

// pack a GMT record
unsigned L1GTDigiToRaw::packGMT(L1MuGMTReadoutRecord const& gmtrr, unsigned char* chp)
{

    const unsigned SIZE=128;
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

//
void L1GTDigiToRaw::endJob()
{

    // empty now
}


// static class members
