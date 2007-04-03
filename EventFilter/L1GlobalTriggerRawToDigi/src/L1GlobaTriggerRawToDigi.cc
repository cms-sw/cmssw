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
 * $Date:$
 * $Revision:$
 *
 */

// this class header
#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GlobalTriggerRawToDigi.h"

// system include files
#include <boost/cstdint.hpp>

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

#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"


// constructor(s)
L1GlobalTriggerRawToDigi::L1GlobalTriggerRawToDigi(const edm::ParameterSet& ps)
{

    produces<L1GlobalTriggerReadoutRecord>();
    produces<L1MuGMTReadoutCollection>();

    // create GTFE, FDL, PSB cards once per analyzer
    // content will be reset whenever needed

    m_gtfeWord = new L1GtfeWord();
    m_gtFdlWord = new L1GtFdlWord();
    m_gtPsbWord = new L1GtPsbWord();

    // total Bx's in the events will be set after reading GTFE block
    m_totalBxInEvent = 1;

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

    // empty now

}

// method called to produce the data
void L1GlobalTriggerRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    edm::Handle<FEDRawDataCollection> fedHandle;
    iEvent.getByType(fedHandle);

    // retrieve data for Global Trigger FED (GT + GMT)
    const FEDRawData& raw =
        (fedHandle.product())->FEDData(FEDNumbering::getTriggerGTPFEDIds().first);

    // get a const pointer to the beginning of the data buffer
    const unsigned char* ptrGt = raw.data();

    // unpack header TODO FIXME
    int headerSize = 8;
    ptrGt += headerSize; // advance with header size

    //
    L1GlobalTriggerReadoutSetup tmpGtSetup; // TODO FIXME temporary event setup
    std::map<int, L1GlobalTriggerReadoutSetup::GtBoard> recordMap =
        tmpGtSetup.GtDaqRecordMap;

    typedef std::map<int, L1GlobalTriggerReadoutSetup::GtBoard>::const_iterator CItRecord;

    // unpack first GTFE to find the length of the record and the active boards

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

    // life normal here, GTFE found

    // get number of Bx in the event from GTFE block
    m_totalBxInEvent = m_gtfeWord->recordLength();

    // get list of active blocks
    // blocks not active are not written to the record
    boost::uint16_t activeBoardsGt = m_gtfeWord->activeBoards();

    // produce the L1GlobalTriggerReadoutRecord now, after we find how many Bx's it has
    LogDebug("L1GlobalTriggerRawToDigi")
    << "\nL1GlobalTriggerRawToDigi: producing L1GlobalTriggerReadoutRecord\n"
    << std::endl;

    std::auto_ptr<L1GlobalTriggerReadoutRecord> gtReadoutRecord(
        new L1GlobalTriggerReadoutRecord(m_totalBxInEvent) );

    // produce also the GMT readout collection
    std::auto_ptr<L1MuGMTReadoutCollection> gmtrc(new L1MuGMTReadoutCollection);


    // add GTFE block to GT readout record
    gtReadoutRecord->setGtfeWord(*m_gtfeWord);

    // ... and reset it
    m_gtfeWord->reset();

    // loop over other blocks in the raw record, if they are active

    std::map<L1GlobalTriggerReadoutSetup::GtBoard, int> activeBoardsMap =
        tmpGtSetup.GtDaqActiveBoardsMap;
    typedef std::map<L1GlobalTriggerReadoutSetup::GtBoard, int>::const_iterator CItActive;

    for (CItRecord itRecord = recordMap.begin();
            itRecord != recordMap.end(); ++itRecord) {

        if (itRecord->first == gtfeKey) {
            ptrGt += m_gtfeWord->getSize(); // advance with GTFE block size
            continue;
        }

        // unpack modules other than GTFE

        // skip if the board is not active
        bool activeBoard = false;

        CItActive itBoard = activeBoardsMap.find(itRecord->second);
        if (itBoard != activeBoardsMap.end()) {
            activeBoard = activeBoardsGt & (1 << (itBoard->second));
        } else {
            // board not found in the map
            continue;
        }

        if ( !activeBoard ) {
            continue;
        }

        // active board, unpack it
        switch (itRecord->second.boardType) {

            case FDL:

                for (int iBxInEvent = 0; iBxInEvent < m_totalBxInEvent; ++iBxInEvent) {
                    unpackFDL(evSetup, ptrGt, *m_gtFdlWord);
                    ptrGt += m_gtFdlWord->getSize(); // advance with FDL block size

                    // add FDL block to GT readout record
                    gtReadoutRecord->setGtFdlWord(*m_gtFdlWord, iBxInEvent);

                    // ... and reset it
                    m_gtFdlWord->reset();

                }

                break;
            case PSB:
                for (int iBxInEvent = 0; iBxInEvent < m_totalBxInEvent; ++iBxInEvent) {
                    unpackPSB(evSetup, ptrGt, *m_gtPsbWord);
                    ptrGt += m_gtPsbWord->getSize(); // advance with PSB block size

                    // add PSB block to GT readout record
                    gtReadoutRecord->setGtPsbWord(*m_gtPsbWord, iBxInEvent);

                    // ... and reset it
                    m_gtPsbWord->reset();

                }

                break;
            case GMT:
                for (int iBxInEvent = 0; iBxInEvent < m_totalBxInEvent; ++iBxInEvent) {
                    unpackGMT(ptrGt,gmtrc);
                    // advance with GMT block size is done in unpackGMT
                }

    
                // TODO FIXME RefProd to GMT collection in GT DAQ readout record
//                edm::Handle<L1MuGMTReadoutCollection> gmtRcHandle;
//                iEvent.getByLabel(m_gtSetup->muGmtInputTag().label(), gmtRcHandle);
//
//                gtReadoutRecord->setMuCollectionRefProd(gmtRcHandle);
//
//                void setMuCollectionRefProd(edm::Handle<L1MuGMTReadoutCollection>&);

                break;
            default:
                // do nothing, all blocks are given in GtBoardType enum
                break;

        }

    }


    // put records into event

    iEvent.put(gmtrc);
    iEvent.put( gtReadoutRecord );

}

// unpack header
void L1GlobalTriggerRawToDigi::unpackHeader(const unsigned char* gtPtr)
{


    // FIXME

}


// unpack the GTFE record
// gtfePtr pointer to the beginning of the GTFE block, obtained from gtPtr
void L1GlobalTriggerRawToDigi::unpackGTFE(
    const edm::EventSetup& evSetup,
    const unsigned char* gtfePtr,
    L1GtfeWord* gtfeWord)
{

    int BlocksPerWord = L1GlobalTriggerReadoutSetup::WordLength/
                        L1GlobalTriggerReadoutSetup::UnitLength;
    int gtfeSize = gtfeWord->getSize();

    unsigned char* startPtr = const_cast<unsigned char*> (gtfePtr);

    for (int iWord = 0; iWord < gtfeSize; ++iWord) {

        // 64bit word
        boost::uint64_t iWordValue = 0ULL;
        unsigned char* endPtr = startPtr + BlocksPerWord;

        unsigned char* iPtr = startPtr;

        for (iPtr = startPtr; iPtr < endPtr; ++iPtr) {
            iWordValue = (iWordValue << L1GlobalTriggerReadoutSetup::UnitLength) | *iPtr;

        }

        startPtr = iPtr;

        // fill GTFE
        // the second argument must match the word index defined in L1GtfeWord class

        gtfeWord->setBoardId(iWordValue, iWord);
        gtfeWord->setRecordLength(iWordValue, iWord);
        gtfeWord->setBxNr(iWordValue, iWord);
        gtfeWord->setSetupVersion(iWordValue, iWord);
        gtfeWord->setActiveBoards(iWordValue, iWord);
        gtfeWord->setTotalTriggerNr(iWordValue, iWord);

    }


}

// unapck FDL records for various bunch crosses
// fdlPtr pointer to the beginning of the each FDL block obtained from gtPtr
void L1GlobalTriggerRawToDigi::unpackFDL(
    const edm::EventSetup& evSetup,
    const unsigned char* fdlPtr,
    L1GtFdlWord&)
{

    // FIXME

}

// unpack PSB records
// psbPtr pointer to the beginning of the each PSB block obtained from gtPtr
void L1GlobalTriggerRawToDigi::unpackPSB(
    const edm::EventSetup& evSetup,
    const unsigned char* psbPtr,
    L1GtPsbWord&)
{

    // FIXME

}

// unpack the GMT record
void L1GlobalTriggerRawToDigi::unpackGMT(
    const unsigned char* chp,
    std::auto_ptr<L1MuGMTReadoutCollection>& gmtrc)
{

    const unsigned* p = (const unsigned*) chp;

    // loop range: int m_totalBxInEvent is normally even (L1A-1, L1A, L1A+1, with L1A = 0)
    int bxMin = (m_totalBxInEvent + 1)/2 - m_totalBxInEvent; 
    int bxMax = (m_totalBxInEvent + 1)/2; 
    
    for(int ib = bxMin; ib < bxMax; ib++) {

        L1MuGMTReadoutRecord gmtrr(ib);

        gmtrr.setEvNr((*p)&0xffffff);
        gmtrr.setBCERR(((*p)>>24)&0xff);
        p++;

        gmtrr.setBxNr((*p)&0xfff);
        gmtrr.setBxInEvent(((*p)>>12)&0xf);
        // to do: check here the block length and the board id
        p++;

        for(int im=0; im<16; im++) {
            gmtrr.setInputCand(im,*p++);
        }

        unsigned char* prank = (unsigned char*) (p+12);

        for(int im=0; im<4; im++) {
            gmtrr.setGMTBrlCand(im, *p++, *prank++);
        }

        for(int im=0; im<4; im++) {
            gmtrr.setGMTFwdCand(im, *p++, *prank++);
        }

        for(int im=0; im<4; im++) {
            gmtrr.setGMTCand(im, *p++);
        }

        // skip the two sort rank words
        p+=2;

        gmtrc->addRecord(gmtrr);

    }
}

// unpack trailer word
// trPtr pointer to the beginning of the each FDL block obtained from gtPtr
void L1GlobalTriggerRawToDigi::unpackTrailer(const unsigned char* trPtr)
{

    // empty now

}

//
void L1GlobalTriggerRawToDigi::endJob()
{

    // empty now
}


// static class members

