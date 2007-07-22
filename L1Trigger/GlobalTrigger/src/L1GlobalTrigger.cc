/**
 * \class L1GlobalTrigger
 * 
 * 
 * Description: see header file.  
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
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"

// system include files
#include <memory>
#include <iostream>
#include <bitset>

#include <boost/cstdint.hpp>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConfig.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFDL.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeExtWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1TcsWord.h"

#include "DataFormats/Common/interface/RefProd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"



// constructors

L1GlobalTrigger::L1GlobalTrigger(const edm::ParameterSet& iConfig)
{

    LogDebug ("Trace") << "Entering L1GlobalTriger constructor";

    // register products
    produces<L1GlobalTriggerReadoutRecord>();
    produces<L1GlobalTriggerEvmReadoutRecord>();
    produces<L1GlobalTriggerObjectMapRecord>();

    // set configuration parameters
    if(!m_gtSetup) {
        m_gtSetup = new L1GlobalTriggerSetup(*this, iConfig);
    }

    std::string emptyString;
    m_gtSetup->setTriggerMenu(emptyString);

    // create new PSBs
    LogDebug("L1GlobalTrigger") << "\n Creating GT PSBs" << std::endl;
    m_gtPSB = new L1GlobalTriggerPSB(*this);

    // create new GTL
    LogDebug("L1GlobalTrigger") << "\n Creating GT GTL" << std::endl;
    m_gtGTL = new L1GlobalTriggerGTL(*this);

    // create new FDL
    LogDebug("L1GlobalTrigger") << "\n Creating GT FDL" << std::endl;
    m_gtFDL = new L1GlobalTriggerFDL(*this);

    // set the total number of bunch crosses in the GT readout records

    m_totalBxInEvent = m_gtSetup->getParameterSet()->getParameter<int>("totalBxInEvent");

    if (m_totalBxInEvent > 0) {
        if ( (m_totalBxInEvent%2) == 0 ) {
            m_totalBxInEvent = m_totalBxInEvent - 1;

            edm::LogInfo("L1GlobalTrigger")
            << "\nWARNING: Number of bunch crossing in event rounded to: "
            << m_totalBxInEvent << "\n         The number must be an odd number!\n"
            << std::endl;
        }
    } else {

        edm::LogInfo("L1GlobalTrigger")
        << "\nWARNING: Number of bunch crossing in event must be a positive number!"
        << "\n  Requested value was: " << m_totalBxInEvent
        << "\n  Reset to 1 (L1Accept bunch only).\n"
        << std::endl;

        m_totalBxInEvent = 1;

    }

    m_minBxInEvent = (m_totalBxInEvent + 1)/2 - m_totalBxInEvent;
    m_maxBxInEvent = (m_totalBxInEvent + 1)/2 - 1;

    LogDebug("L1GlobalTrigger")
    << "\nTotal number of bunch crosses to put in the GT readout record: "
    << m_totalBxInEvent << " = " << "["
    << m_minBxInEvent << ", " << m_maxBxInEvent << "] BX\n"
    << std::endl;

    // set the list of active boards

    m_activeBoards =
        static_cast<boost::uint16_t>(
            m_gtSetup->getParameterSet()->getParameter<unsigned int>("ActiveBoards"));
    LogDebug("L1GlobalTrigger")
    << "\nActive boards in L1 GT: "
    << m_activeBoards
    << std::endl;


}

// destructor
L1GlobalTrigger::~L1GlobalTrigger()
{

    if(m_gtSetup)
        delete m_gtSetup;
    m_gtSetup = 0;

    delete m_gtPSB;
    delete m_gtGTL;
    delete m_gtFDL;
}

// member functions

// method called to produce the data
void L1GlobalTrigger::produce(edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    using namespace edm;

    // process event iEvent

    // * produce the L1GlobalTriggerReadoutRecord
    LogDebug("L1GlobalTrigger")
    << "\nL1GlobalTrigger : producing L1GlobalTriggerReadoutRecord\n"
    << std::endl;

    std::auto_ptr<L1GlobalTriggerReadoutRecord> gtReadoutRecord(
        new L1GlobalTriggerReadoutRecord(m_totalBxInEvent) );


    // * produce the L1GlobalTriggerEvmReadoutRecord
    LogDebug("L1GlobalTrigger")
    << "\nL1GlobalTrigger : producing L1GlobalTriggerEvmReadoutRecord\n"
    << std::endl;

    std::auto_ptr<L1GlobalTriggerEvmReadoutRecord> gtEvmReadoutRecord(
        new L1GlobalTriggerEvmReadoutRecord(m_totalBxInEvent) );

    // * produce the L1GlobalTriggerObjectMapRecord
    LogDebug("L1GlobalTrigger")
    << "\nL1GlobalTrigger : producing L1GlobalTriggerObjectMapRecord\n"
    << std::endl;

    std::auto_ptr<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord(
        new L1GlobalTriggerObjectMapRecord() );


    // TODO FIXME temporary event setup
    L1GlobalTriggerReadoutSetup tmpGtSetup;
    std::map<L1GlobalTriggerReadoutSetup::GtBoard, int> slotMap =
        tmpGtSetup.GtBoardSlotMap;

    typedef std::map<L1GlobalTriggerReadoutSetup::GtBoard, int>::const_iterator CItSlot;

    boost::uint16_t gtfeBoardId = 0;  //  GTFE:   8 bits board identifier
    boost::uint16_t tcsBoardId = 0;   //  other: 16 bits board identifier

    for (CItSlot itSlot = slotMap.begin(); itSlot != slotMap.end(); ++itSlot) {

        // active board, add its size
        switch (itSlot->first.boardType) {

            case GTFE: {
                    gtfeBoardId = itSlot->second;
                }
                break;
            case TCS: {
                    tcsBoardId = tcsBoardId*100 + itSlot->second; // FIXME
                }
                break;
            default: {

                    // do nothing here
                }
                break;
        }

    }

    // * create L1GtfeExtWord

    L1GtfeExtWord gtfeExtWordValue;
    gtfeExtWordValue.setBoardId(gtfeBoardId);
    // cast int to boost::uint16_t (there are normally 3 or 5 BxInEvent)
    gtfeExtWordValue.setRecordLength(static_cast<boost::uint16_t>(m_totalBxInEvent));

    // set the list of active boards
    gtfeExtWordValue.setActiveBoards(m_activeBoards);

    // ** fill L1GtfeWord in GT DAQ record

    L1GtfeWord& gtfeWordValue = dynamic_cast<L1GtfeExtWord&>(gtfeExtWordValue);
    gtReadoutRecord->setGtfeWord(gtfeWordValue);

    // ** fill L1GtfeExtWord in GT EVM record
    gtEvmReadoutRecord->setGtfeWord(gtfeExtWordValue);

    LogDebug("L1GlobalTrigger")
    << "\n  GTFE board " << gtReadoutRecord->gtfeWord().boardId() << "\n"
    << "    GTFE word: total number of bx in DAQ record = "
    << gtReadoutRecord->gtfeWord().recordLength()
    << "    GTFE word: total number of bx in EVM record = "
    << gtEvmReadoutRecord->gtfeWord().recordLength()
    << std::endl;

    // * create L1TcsWord

    L1TcsWord tcsWordValue;
    tcsWordValue.setTriggerType(0x5); // 0101 simulated event

    // ** fill L1TcsWord in the EVM record

    gtEvmReadoutRecord->setTcsWord(tcsWordValue);
    LogDebug("L1GlobalTrigger")
    << "\n  TCS word: trigger type = "
    << std::bitset<4>(gtEvmReadoutRecord->tcsWord().triggerType())
    << std::endl;

    // loop over bx in event
    for (int iBxInEvent = m_minBxInEvent; iBxInEvent <= m_maxBxInEvent;
            ++iBxInEvent) {

        // * receive GCT data via PSBs
        if ( m_gtPSB ) {
            LogDebug("L1GlobalTrigger")
            << "\nL1GlobalTrigger : running PSB for bx = " << iBxInEvent << "\n"
            << std::endl;
            m_gtPSB->receiveData(iEvent, iBxInEvent);
            //            m_gtPSB->receiveData(iEvent);
        }

        // * receive GMT data via GTL
        if ( m_gtGTL ) {
            LogDebug("L1GlobalTrigger")
            << "\nL1GlobalTrigger : receiving GMT data for bx = " << iBxInEvent << "\n"
            << std::endl;
            m_gtGTL->receiveData(iEvent, iBxInEvent);
        }

        // * run GTL
        if ( m_gtGTL ) {
            LogDebug("L1GlobalTrigger")
            << "\nL1GlobalTrigger : running GTL for bx = " << iBxInEvent << "\n"
            << std::endl;

            m_gtGTL->run(iBxInEvent);

            LogDebug("L1GlobalTrigger")
            << "\n AlgorithmOR\n" << m_gtGTL->getAlgorithmOR() << "\n"
            << std::endl;

            if (iBxInEvent == 0) { // TODO map only for BX = 0?

                const std::vector<L1GlobalTriggerObjectMap>* objMapVec = m_gtGTL->objectMap();

                gtObjectMapRecord->setGtObjectMap(*objMapVec);
                delete objMapVec;

            }

        }

        // * run FDL
        if ( m_gtFDL ) {
            LogDebug("L1GlobalTrigger")
            << "\nL1GlobalTrigger : running FDL for bx = " << iBxInEvent << "\n"
            << std::endl;

            m_gtFDL->run(iBxInEvent);

            std::ostringstream myCoutStream;
            if ( edm::isDebugEnabled() ) {
                m_gtFDL->gtFdlWord()->printGtDecisionWord(myCoutStream);
            }
            LogDebug("L1GlobalTrigger")
            << "\n FDL decision word\n" << myCoutStream.str() << "\n"
            << std::endl;

            gtReadoutRecord->setGtFdlWord( *(*m_gtFDL).gtFdlWord(), iBxInEvent );
            gtEvmReadoutRecord->setGtFdlWord( *(*m_gtFDL).gtFdlWord(), iBxInEvent );

        }

        // reset
        m_gtPSB->reset();
        m_gtGTL->reset();
        m_gtFDL->reset();

        LogDebug("L1GlobalTrigger") << "\n Reset PSB, GTL, FDL\n" << std::endl;

    }


    std::ostringstream myCoutStream;

    if ( edm::isDebugEnabled() ) {
        gtReadoutRecord->printGtDecision(myCoutStream);

        LogDebug("L1GlobalTrigger")
        << myCoutStream.str()
        << std::endl;
        myCoutStream.str("");
        myCoutStream.clear();
    }


    // print result for every bx in event
    if ( edm::isDebugEnabled() ) {
        for (int iBxInEvent = m_minBxInEvent; iBxInEvent <= m_maxBxInEvent;
                ++iBxInEvent) {

            gtReadoutRecord->printGtDecision(myCoutStream, iBxInEvent);
            LogDebug("L1GlobalTrigger")
            << myCoutStream.str()
            << std::endl;
            myCoutStream.str("");
            myCoutStream.clear();

            gtReadoutRecord->printTechnicalTrigger(myCoutStream, iBxInEvent);
            LogDebug("L1GlobalTrigger")
            << myCoutStream.str()
            << std::endl;
            myCoutStream.str("");
            myCoutStream.clear();

        }
    }

    if ( m_gtSetup->gtConfig()->getInputMask()[1] ) {

        LogDebug("L1GlobalTrigger")
        << "\n**** Global Muon input disabled! \n  inputMask[1] = "
        << m_gtSetup->gtConfig()->getInputMask()[1]
        << "\n  No persistent reference for L1MuGMTReadoutCollection."
        << "\n**** \n"
        << std::endl;
    } else {

        // ** set muons in L1GlobalTriggerReadoutRecord

        LogDebug("L1GlobalTrigger")
        << "\n**** "
        << "\n  Persistent reference for L1MuGMTReadoutCollection with input tag: "
        << m_gtSetup->muGmtInputTag().label()
        << "\n**** \n"
        << std::endl;

        // get L1MuGMTReadoutCollection reference and set it in GT record

        edm::Handle<L1MuGMTReadoutCollection> gmtRcHandle;
        iEvent.getByLabel(m_gtSetup->muGmtInputTag().label(), gmtRcHandle);

        gtReadoutRecord->setMuCollectionRefProd(gmtRcHandle);

    }

    // test muon part in L1GlobalTriggerReadoutRecord

    if ( edm::isDebugEnabled() && !( m_gtSetup->gtConfig()->getInputMask()[1] ) ) {

        LogTrace("L1GlobalTrigger")
        << "\n Test muon collection in the GT readout record"
        << std::endl;

        // get reference to muon collection
        const edm::RefProd<L1MuGMTReadoutCollection>
        muCollRefProd = gtReadoutRecord->muCollectionRefProd();

        if (muCollRefProd.isNull()) {
            LogTrace("L1GlobalTrigger")
            << "Null reference for L1MuGMTReadoutCollection"
            << std::endl;

        } else {
            LogTrace("L1GlobalTrigger")
            << "RefProd address = " << &muCollRefProd
            << std::endl;

            // test all three variants to get muon index 0 in BXInEvent = 0
            unsigned int indexCand = 0;
            int bxInEvent = 0;

            // test first if the record has the required number of candidates
            if ((*muCollRefProd).getRecord(bxInEvent).getGMTCands().size() > indexCand) {

                LogTrace("L1GlobalTrigger")
                << "Three variants to get muon index 0 in BXInEvent = 0"
                << "\n via RefProd, muonCand(indexCand, bxInEvent), muonCand(indexCand)"
                << std::endl;

                L1MuGMTExtendedCand mu00 = (*muCollRefProd).getRecord(bxInEvent).getGMTCands()[indexCand];
                mu00.print();

                L1MuGMTExtendedCand mu00A = gtReadoutRecord->muonCand(indexCand, bxInEvent);
                mu00A.print();

                L1MuGMTExtendedCand mu00B = gtReadoutRecord->muonCand(indexCand);
                mu00B.print();

            }

            // test methods to get GMT records
            std::vector<L1MuGMTReadoutRecord> muRecords = (*muCollRefProd).getRecords();
            LogTrace("L1GlobalTrigger")
            << "\nNumber of records in the GMT RefProd readout collection = "
            << muRecords.size()
            << std::endl;

            for (std::vector<L1MuGMTReadoutRecord>::const_iterator itMu = muRecords.begin();
                    itMu < muRecords.end(); ++itMu) {

                std::vector<L1MuGMTExtendedCand>
                ::const_iterator gmt_iter;
                std::vector<L1MuGMTExtendedCand> exc = itMu->getGMTCands();
                for(gmt_iter = exc.begin(); gmt_iter != exc.end(); gmt_iter++) {
                    (*gmt_iter).print();
                }

            }

            // test GMT record for BxInEvent = 0  (default argument)
            std::vector<L1MuGMTExtendedCand> muRecord0 = gtReadoutRecord->muonCands();
            LogTrace("L1GlobalTrigger")
            << "\nRecord for BxInEvent = 0 using default argument"
            << std::endl;

            for(std::vector<L1MuGMTExtendedCand>::const_iterator gmt_iter = muRecord0.begin();
                    gmt_iter != muRecord0.end(); gmt_iter++) {
                (*gmt_iter).print();
            }

            // test GMT record for BxInEvent = 1
            std::vector<L1MuGMTExtendedCand> muRecord1 = gtReadoutRecord->muonCands(1);
            LogTrace("L1GlobalTrigger")
            << "\nRecord for BxInEvent = 1 using BxInEvent argument"
            << std::endl;

            for(std::vector<L1MuGMTExtendedCand>::const_iterator gmt_iter = muRecord1.begin();
                    gmt_iter != muRecord1.end(); gmt_iter++) {
                (*gmt_iter).print();
            }
        }

    }

    if ( edm::isDebugEnabled() ) {

        const std::vector<L1GlobalTriggerObjectMap>&
        objMapVec = gtObjectMapRecord->gtObjectMap();

        for (std::vector<L1GlobalTriggerObjectMap>::const_iterator it = objMapVec.begin();
                it != objMapVec.end(); ++it) {
            (*it).print(myCoutStream);
        }

        //const CombinationsInCond*
        //comb = gtObjectMapRecord->getCombinationsInCond("Algo_ComplexSyntax", "Mu_60");

        //if (comb != 0) {
        //    myCoutStream << "\n  Number of combinations passing (Algo_ComplexSyntax, Mu_60): "
        //    << comb->size();
        //}


        //bool result = gtObjectMapRecord->getConditionResult("Algo_ComplexSyntax", "Mu_60");

        //myCoutStream << "\n  Result for condition Mu_60 in Algo_ComplexSyntax: "
        //<< result;

        LogDebug("L1GlobalTrigger")
        << "Test gtObjectMapRecord in L1GlobalTrigger \n\n" << myCoutStream.str() << "\n\n"
        << std::endl;
        myCoutStream.str("");
        myCoutStream.clear();

    }

    // **
    iEvent.put( gtReadoutRecord );
    iEvent.put( gtEvmReadoutRecord );
    iEvent.put( gtObjectMapRecord );

}

// static data members

L1GlobalTriggerSetup* L1GlobalTrigger::m_gtSetup = 0;
