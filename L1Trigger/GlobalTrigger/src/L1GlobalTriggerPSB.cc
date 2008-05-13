/**
 * \class L1GlobalTriggerPSB
 * 
 * 
 * Description: Pipelined Synchronising Buffer, see header file for details.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: M. Fierro            - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

// system include files
#include <bitset>
#include <iostream>
#include <iomanip>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

// forward declarations

// constructor
L1GlobalTriggerPSB::L1GlobalTriggerPSB()
        :
        m_candL1NoIsoEG ( new std::vector<const L1GctCand*>),
        m_candL1IsoEG   ( new std::vector<const L1GctCand*>),
        m_candL1CenJet  ( new std::vector<const L1GctCand*>),
        m_candL1ForJet  ( new std::vector<const L1GctCand*>),
        m_candL1TauJet  ( new std::vector<const L1GctCand*>),
        m_candETM(0),
        m_candETT(0),
        m_candHTT(0),
        m_candJetCounts(0),
        m_techTrigSelector(edm::Selector( edm::ModuleLabelSelector("")))
{

    // empty
    
}

L1GlobalTriggerPSB::L1GlobalTriggerPSB(const std::string selLabel)
        :
        m_candL1NoIsoEG( new std::vector<const L1GctCand*> ), 
        m_candL1IsoEG  ( new std::vector<const L1GctCand*>),
        m_candL1CenJet ( new std::vector<const L1GctCand*>),
        m_candL1ForJet ( new std::vector<const L1GctCand*>),
        m_candL1TauJet ( new std::vector<const L1GctCand*>),
        m_candETM(0),
        m_candETT(0),
        m_candHTT(0),
        m_candJetCounts(0),
        m_techTrigSelector(edm::Selector( edm::ModuleLabelSelector(selLabel)))
{

    // empty
    
}

// destructor
L1GlobalTriggerPSB::~L1GlobalTriggerPSB()
{

    reset();
    
    delete m_candL1NoIsoEG;
    delete m_candL1IsoEG;
    delete m_candL1CenJet;
    delete m_candL1ForJet;
    delete m_candL1TauJet;

}

// operations
void L1GlobalTriggerPSB::init(const int nrL1NoIsoEG, const int nrL1IsoEG, 
        const int nrL1CenJet, const int nrL1ForJet, const int nrL1TauJet,
        const int numberTechnicalTriggers)
{

    m_candL1NoIsoEG->reserve(nrL1NoIsoEG);
    m_candL1IsoEG->reserve(nrL1IsoEG);
    m_candL1CenJet->reserve(nrL1CenJet);
    m_candL1ForJet->reserve(nrL1ForJet);
    m_candL1TauJet->reserve(nrL1TauJet);

    m_gtTechnicalTriggers.reserve(numberTechnicalTriggers);
    m_gtTechnicalTriggers.assign(numberTechnicalTriggers, false);
    
}

// receive input data

void L1GlobalTriggerPSB::receiveGctObjectData(
    edm::Event& iEvent,
    const edm::InputTag& caloGctInputTag, const int iBxInEvent,
    const bool receiveNoIsoEG, const int nrL1NoIsoEG,
    const bool receiveIsoEG, const int nrL1IsoEG,
    const bool receiveCenJet, const int nrL1CenJet,
    const bool receiveForJet, const int nrL1ForJet,
    const bool receiveTauJet, const int nrL1TauJet,
    const bool receiveETM, const bool receiveETT, const bool receiveHTT,
    const bool receiveJetCounts)
{

    LogDebug("L1GlobalTriggerPSB")
            << "\n**** L1GlobalTriggerPSB receiving calorimeter data for BxInEvent = "
            << iBxInEvent << "\n     from " << caloGctInputTag << "\n"
            << std::endl;

    reset();

    if (receiveNoIsoEG) {

        // get GCT NoIsoEG
        edm::Handle<L1GctEmCandCollection> emCands;
        iEvent.getByLabel(caloGctInputTag.label(), "nonIsoEm", emCands);

        if (!emCands.isValid()) {
            throw cms::Exception("ProductNotFound")
            << "\nError: L1GctEmCandCollection with input label " << caloGctInputTag.label()
            << " and instance \"nonIsoEm\" \n"
            << "requested in configuration, but not found in the event.\n"
            << std::endl;            
        }

        for (L1GctEmCandCollection::const_iterator
                it = emCands->begin(); it != emCands->end(); it++) {
            
            if ((*it).bx() == iBxInEvent) {
                
                (*m_candL1NoIsoEG).push_back(&(*it));
                //LogTrace("L1GlobalTriggerPSB") << "NoIsoEG:  " << (*it) << std::endl;

            }
        }
        
        size_t numberObjects = m_candL1NoIsoEG->size();
        if (numberObjects < nrL1NoIsoEG) {
            throw cms::Exception("EventCorruption")
            << "\nError: L1GctEmCandCollection contains for BxInEvent = " << iBxInEvent
            << " only " << numberObjects << " NoIsoEG objects instead of expected " 
            << nrL1NoIsoEG << " objects.\n"
            << std::endl;            
        }

    }

    if (receiveIsoEG) {

        // get GCT IsoEG
        edm::Handle<L1GctEmCandCollection> isoEmCands;
        iEvent.getByLabel(caloGctInputTag.label(), "isoEm",    isoEmCands);

        if (!isoEmCands.isValid()) {
            throw cms::Exception("ProductNotFound")
            << "\nError: L1GctEmCandCollection with input label " << caloGctInputTag.label()
            << " and instance \"isoEm\" \n"
            << "requested in configuration, but not found in the event.\n"
            << std::endl;            
        }

        for (L1GctEmCandCollection::const_iterator
                it = isoEmCands->begin(); it != isoEmCands->end(); it++) {
            
            if ((*it).bx() == iBxInEvent) {
                
                (*m_candL1IsoEG).push_back(&(*it));
                //LogTrace("L1GlobalTriggerPSB") << "IsoEG:    " <<  (*it) << std::endl;

            }
        }

        size_t numberObjects = m_candL1IsoEG->size();
        if (numberObjects < nrL1IsoEG) {
            throw cms::Exception("EventCorruption")
            << "\nError: L1GctEmCandCollection contains for BxInEvent = " << iBxInEvent
            << " only " << numberObjects << " IsoEG objects instead of expected " 
            << nrL1IsoEG << " objects.\n"
            << std::endl;            
        }

    }
    

    if (receiveCenJet) {

        // get GCT CenJet
        edm::Handle<L1GctJetCandCollection> cenJets;
        iEvent.getByLabel(caloGctInputTag.label(), "cenJets", cenJets);

        if (!cenJets.isValid()) {
            throw cms::Exception("ProductNotFound")
            << "\nError: L1GctJetCandCollection with input label " << caloGctInputTag.label()
            << " and instance \"cenJets\" \n"
            << "requested in configuration, but not found in the event.\n"
            << std::endl;            
        }

        for (L1GctJetCandCollection::const_iterator
                it = cenJets->begin(); it != cenJets->end(); it++) {
            
            if ((*it).bx() == iBxInEvent) {
                
                (*m_candL1CenJet).push_back(&(*it));
                //LogTrace("L1GlobalTriggerPSB") << "CenJet    " <<  (*it) << std::endl;

            }
        }

        size_t numberObjects = m_candL1CenJet->size();
        if (numberObjects < nrL1CenJet) {
            throw cms::Exception("EventCorruption")
            << "\nError: L1GctJetCandCollection contains for BxInEvent = " << iBxInEvent
            << " only " << numberObjects << " CenJet objects instead of expected " 
            << nrL1CenJet << " objects.\n"
            << std::endl;            
        }
    }
        
    if (receiveForJet) {

        // get GCT ForJet
        edm::Handle<L1GctJetCandCollection> forJets;
        iEvent.getByLabel(caloGctInputTag.label(), "forJets", forJets);

        if (!forJets.isValid()) {
            throw cms::Exception("ProductNotFound")
            << "\nError: L1GctJetCandCollection with input label " << caloGctInputTag.label()
            << " and instance \"forJets\" \n"
            << "requested in configuration, but not found in the event.\n"
            << std::endl;            
        }

        for (L1GctJetCandCollection::const_iterator
                it = forJets->begin(); it != forJets->end(); it++) {
            
            if ((*it).bx() == iBxInEvent) {
                
                (*m_candL1ForJet).push_back(&(*it));
                //LogTrace("L1GlobalTriggerPSB") << "ForJet    " <<  (*it) << std::endl;

            }
        }

        size_t numberObjects = m_candL1ForJet->size();
        if (numberObjects < nrL1ForJet) {
            throw cms::Exception("EventCorruption")
            << "\nError: L1GctJetCandCollection contains for BxInEvent = " << iBxInEvent
            << " only " << numberObjects << " ForJet objects instead of expected " 
            << nrL1ForJet << " objects.\n"
            << std::endl;            
        }
    }

    if (receiveTauJet) {

        // get GCT TauJet
        edm::Handle<L1GctJetCandCollection> tauJets;
        iEvent.getByLabel(caloGctInputTag.label(), "tauJets", tauJets);

        if (!tauJets.isValid()) {
            throw cms::Exception("ProductNotFound")
            << "\nError: L1GctJetCandCollection with input label " << caloGctInputTag.label()
            << " and instance \"tauJets\" \n"
            << "requested in configuration, but not found in the event.\n"
            << std::endl;            
        }

        for (L1GctJetCandCollection::const_iterator
                it = tauJets->begin(); it != tauJets->end(); it++) {
            
            if ((*it).bx() == iBxInEvent) {
                
                (*m_candL1TauJet).push_back(&(*it));
                //LogTrace("L1GlobalTriggerPSB") << "TauJet    " <<  (*it) << std::endl;

            }
        }

        size_t numberObjects = m_candL1TauJet->size();
        if (numberObjects < nrL1TauJet) {
            throw cms::Exception("EventCorruption")
            << "\nError: L1GctJetCandCollection contains for BxInEvent = " << iBxInEvent
            << " only " << numberObjects << " TauJet objects instead of expected " 
            << nrL1TauJet << " objects.\n"
            << std::endl;            
        }

    }

    // get GCT ETM
    if (receiveETM) {

        edm::Handle<L1GctEtMissCollection> missEtColl;
        iEvent.getByLabel(caloGctInputTag, missEtColl) ;

        if (!missEtColl.isValid()) {
            throw cms::Exception("ProductNotFound")
            << "\nError: L1GctEtMissCollection with input tag " << caloGctInputTag
            << "\nrequested in configuration, but not found in the event.\n"
            << std::endl;            
        }

        for (L1GctEtMissCollection::const_iterator it = missEtColl->begin(); it
                != missEtColl->end(); it++) {

            if ((*it).bx() == iBxInEvent) {

                m_candETM = &(*it);
                //LogTrace("L1GlobalTriggerPSB") << "ETM      " << (*it) << std::endl;

            }
        }
        
        if ( m_candETM == 0) {
            throw cms::Exception("EventCorruption")
            << "\nError: L1GctEtMissCollection contains for BxInEvent = " << iBxInEvent
            << " no ETM object.\n"
            << std::endl;            
        }

    }

    // get GCT ETT
    if (receiveETT) {

        edm::Handle<L1GctEtTotalCollection> sumEtColl;
        iEvent.getByLabel(caloGctInputTag, sumEtColl) ;

        if (!sumEtColl.isValid()) {
            throw cms::Exception("ProductNotFound")
            << "\nError: L1GctEtTotalCollection with input tag " << caloGctInputTag
            << "\nrequested in configuration, but not found in the event.\n"
            << std::endl;            
        }

        for (L1GctEtTotalCollection::const_iterator it = sumEtColl->begin(); it
                != sumEtColl->end(); it++) {

            if ((*it).bx() == iBxInEvent) {

                m_candETT = &(*it);
                //LogTrace("L1GlobalTriggerPSB") << "ETT      " << (*it) << std::endl;

            }
        }

        if ( m_candETT == 0) {
            throw cms::Exception("EventCorruption")
            << "\nError: L1GctEtTotalCollection contains for BxInEvent = " << iBxInEvent
            << " no ETT object.\n"
            << std::endl;            
        }
    }

    // get GCT HTT
    if (receiveHTT) {

        edm::Handle<L1GctEtHadCollection> sumHtColl;
        iEvent.getByLabel(caloGctInputTag, sumHtColl) ;

        if (!sumHtColl.isValid()) {
            throw cms::Exception("ProductNotFound")
            << "\nError: L1GctEtHadCollection with input tag " << caloGctInputTag
            << "\nrequested in configuration, but not found in the event.\n"
            << std::endl;            
        }

        for (L1GctEtHadCollection::const_iterator it = sumHtColl->begin(); it
                != sumHtColl->end(); it++) {

            if ((*it).bx() == iBxInEvent) {

                m_candHTT = &(*it);
                //LogTrace("L1GlobalTriggerPSB") << "HTT      "  << (*it) << std::endl;

            }
        }

        if ( m_candHTT == 0) {
            throw cms::Exception("EventCorruption")
            << "\nError: L1GctEtHadCollection contains for BxInEvent = " << iBxInEvent
            << " no HTT object.\n"
            << std::endl;            
        }
    }

    // get GCT JetCounts
    if (receiveJetCounts) {

        edm::Handle<L1GctJetCountsCollection> jetCountColl;
        iEvent.getByLabel(caloGctInputTag, jetCountColl) ;

        if (!jetCountColl.isValid()) {
            throw cms::Exception("ProductNotFound")
            << "\nError: L1GctJetCountsCollection with input tag " << caloGctInputTag
            << "\nrequested in configuration, but not found in the event.\n"
            << std::endl;            
        }

        for (L1GctJetCountsCollection::const_iterator it =
                jetCountColl->begin(); it != jetCountColl->end(); it++) {

            if ((*it).bx() == iBxInEvent) {

                m_candJetCounts = &(*it);
                //LogTrace("L1GlobalTriggerPSB") << (*it) << std::endl;

            }
        }

        if ( m_candJetCounts == 0) {
            throw cms::Exception("EventCorruption")
            << "\nError: L1GctJetCountsCollection contains for BxInEvent = " << iBxInEvent
            << " no JetCounts object.\n"
            << std::endl;            
        }
    }
    
    
    if ( edm::isDebugEnabled() ) {
        LogDebug("L1GlobalTriggerPSB")
        << "**** L1GlobalTriggerPSB received calorimeter data from input tag "
        << caloGctInputTag
        << std::endl;

        printGctObjectData(iBxInEvent);
    }


}

// receive technical trigger
void L1GlobalTriggerPSB::receiveTechnicalTriggers(edm::Event& iEvent,
    const edm::InputTag& technicalTriggersInputTag, const int iBxInEvent,
    const bool receiveTechTr, const int nrL1TechTr) {
    
    // reset the technical trigger bits
    m_gtTechnicalTriggers = std::vector<bool>(nrL1TechTr, false);


    if (receiveTechTr) {

        // get the technical trigger bits, change the values
        iEvent.getMany(m_techTrigSelector, m_techTrigRecords);

        size_t recordsSize = m_techTrigRecords.size();
        for (size_t iRec = 0; iRec < recordsSize; ++iRec) {

            const L1GtTechnicalTriggerRecord& ttRecord = *m_techTrigRecords[iRec];
            const std::vector<L1GtTechnicalTrigger>& ttVec = ttRecord.gtTechnicalTrigger(); 
            size_t ttVecSize = ttVec.size();

            for (size_t iTT = 0; iTT < ttVecSize; ++iTT) {

                const L1GtTechnicalTrigger& ttBxRecord = ttVec[iTT];
                int ttBxInEvent = ttBxRecord.bxInEvent();

                if (ttBxInEvent == iBxInEvent) {
                    int ttBitNumber = ttBxRecord.gtTechnicalTriggerBitNumber();
                    bool ttResult = ttBxRecord.gtTechnicalTriggerResult();

                    m_gtTechnicalTriggers.at(ttBitNumber) = ttResult;

                    LogTrace("L1GlobalTriggerPSB")
                        << "\n Add technical trigger with bit number " << ttBitNumber
                        << " and result " << ttResult
                        << std::endl;

                    break;
                    
                }

            }

        }

    }    

    if ( edm::isDebugEnabled() ) {
        LogDebug("L1GlobalTriggerPSB")
            << "\n**** L1GlobalTriggerPSB receiving technical triggers from input tag "
            << technicalTriggersInputTag 
            << "\n**** Technical triggers (bitset style): "
            << std::endl;

        int sizeW64 = 64; // 64 bits words
        int iBit = 0;
        
        std::ostringstream myCout;

        for (std::vector<bool>::reverse_iterator ritBit = m_gtTechnicalTriggers.rbegin();
                ritBit != m_gtTechnicalTriggers.rend(); ++ritBit) {

            myCout << (*ritBit ? '1' : '0');

            if ( (((iBit + 1)%16) == (sizeW64%16)) && (iBit != 63) ) {
                myCout << " ";
            }

            iBit++;
        }

        LogTrace("L1GlobalTriggerPSB")
        << myCout.str() << "\n"
        << std::endl;

    
    }

}


// fill the content of active PSB boards
void L1GlobalTriggerPSB::fillPsbBlock(
    edm::Event& iEvent,
    const boost::uint16_t& activeBoardsGtDaq,
    const std::vector<L1GtBoard>& boardMaps,
    const int iBxInEvent,
    std::auto_ptr<L1GlobalTriggerReadoutRecord>& gtDaqReadoutRecord)
{

    typedef std::vector<L1GtBoard>::const_iterator CItBoardMaps;

    // loop over PSB blocks in the GT DAQ record and fill them
    // with the content of the object list

    for (CItBoardMaps
            itBoard = boardMaps.begin();
            itBoard != boardMaps.end(); ++itBoard) {

        int iPosition = itBoard->gtPositionDaqRecord();
        if (iPosition > 0) {

            int iActiveBit = itBoard->gtBitDaqActiveBoards();
            bool activeBoard = false;

            if (iActiveBit >= 0) {
                activeBoard = activeBoardsGtDaq & (1 << iActiveBit);
            }

            if (activeBoard && (itBoard->gtBoardType() == PSB)) {

                L1GtPsbWord psbWordValue;

                // set board ID
                psbWordValue.setBoardId(itBoard->gtBoardId());

                // set bunch cross in the GT event record
                psbWordValue.setBxInEvent(iBxInEvent);

                // set bunch cross number of the actual bx
                boost::uint16_t bxNrValue = 0; // FIXME
                psbWordValue.setBxNr(bxNrValue);


                // set event number since last L1 reset generated in PSB
                psbWordValue.setEventNr(
                    static_cast<boost::uint32_t>(iEvent.id().event()) );

                // set local bunch cross number of the actual bx
                boost::uint16_t localBxNrValue = 0; // FIXME
                psbWordValue.setLocalBxNr(localBxNrValue);

                // get the objects coming to this PSB and the quadruplet index 
                
                // two objects writen one after another from the same quadruplet
                int nrObjRow = 2;
                
                std::vector<L1GtPsbQuad> quadInPsb = itBoard->gtQuadInPsb();
                int nrCables = quadInPsb.size();
                
                boost::uint16_t aDataVal = 0;
                boost::uint16_t bDataVal = 0;
                
                int iCable = -1;
                for (std::vector<L1GtPsbQuad>::const_iterator
                        itQuad = quadInPsb.begin();
                        itQuad != quadInPsb.end(); ++itQuad) {
                    
                    iCable++;
                    
                    int iAB = (nrCables - iCable - 1)*nrObjRow;

                    switch (*itQuad) {

                        case TechTr: {

                            //LogTrace("L1GlobalTriggerPSB")
                            //<< "\nL1GlobalTriggerPSB: write TechTr"
                            //<< std::endl;

                            // order: 16-bit words
                            int bitsPerWord = 16;

                            //
                            int iPair = 0;
                            aDataVal = 0;

                            int iBit = 0;
                            boost::uint16_t bitVal = 0;

                            for (int i = 0; i < bitsPerWord; ++i) {
                                if (m_gtTechnicalTriggers[iBit]) {
                                    bitVal = 1;
                                }
                                else {
                                    bitVal = 0;
                                }

                                aDataVal = aDataVal | (bitVal << i);
                                iBit++;
                            }
                            psbWordValue.setAData(aDataVal, iAB + iPair);

                            //
                            bDataVal = 0;
                            
                            for (int i = 0; i < bitsPerWord; ++i) {
                                if (m_gtTechnicalTriggers[iBit]) {
                                    bitVal = 1;
                                }
                                else {
                                    bitVal = 0;
                                }

                                bDataVal = bDataVal | (bitVal << i);
                                iBit++;
                            }
                            psbWordValue.setBData(bDataVal, iAB + iPair);

                            //
                            iPair = 1;
                            aDataVal = 0;

                            for (int i = 0; i < bitsPerWord; ++i) {
                                if (m_gtTechnicalTriggers[iBit]) {
                                    bitVal = 1;
                                }
                                else {
                                    bitVal = 0;
                                }

                                aDataVal = aDataVal | (bitVal << i);
                                iBit++;
                            }
                            psbWordValue.setAData(aDataVal, iAB + iPair);

                            bDataVal = 0;

                            for (int i = 0; i < bitsPerWord; ++i) {
                                if (m_gtTechnicalTriggers[iBit]) {
                                    bitVal = 1;
                                }
                                else {
                                    bitVal = 0;
                                }

                                bDataVal = bDataVal | (bitVal << i);
                                iBit++;
                            }
                            psbWordValue.setBData(bDataVal, iAB + iPair);
                        }

                            break;
                        case NoIsoEGQ: {

                            //LogTrace("L1GlobalTriggerPSB")
                            //<< "\nL1GlobalTriggerPSB: write NoIsoEGQ"
                            //<< std::endl;

                            int recL1NoIsoEG = m_candL1NoIsoEG->size();
                            for (int iPair = 0; iPair < nrObjRow; ++iPair) {
                                if (iPair < recL1NoIsoEG) {
                                    aDataVal = 
                                        (static_cast<const L1GctEmCand*> ((*m_candL1NoIsoEG)[iPair]))->raw();
                                }
                                else {
                                    aDataVal = 0;
                                }
                                psbWordValue.setAData(aDataVal, iAB + iPair);

                                if ((iPair + nrObjRow) < recL1NoIsoEG) {
                                    bDataVal =
                                        (static_cast<const L1GctEmCand*> ((*m_candL1NoIsoEG)[iPair + nrObjRow]))->raw();
                                }
                                else {
                                    bDataVal = 0;
                                }
                                psbWordValue.setBData(bDataVal, iAB + iPair);

                            }
                        }

                            break;
                        case IsoEGQ: {
                            
                            //LogTrace("L1GlobalTriggerPSB")
                            //<< "\nL1GlobalTriggerPSB: write IsoEGQ"
                            //<< std::endl;

                            int recL1IsoEG = m_candL1IsoEG->size();
                            for (int iPair = 0; iPair < nrObjRow; ++iPair) {
                                if (iPair < recL1IsoEG) {
                                    aDataVal = 
                                        (static_cast<const L1GctEmCand*> ((*m_candL1IsoEG)[iPair]))->raw();
                                }
                                else {
                                    aDataVal = 0;
                                }
                                psbWordValue.setAData(aDataVal, iAB + iPair);

                                if ((iPair + nrObjRow) < recL1IsoEG) {
                                    bDataVal =
                                        (static_cast<const L1GctEmCand*> ((*m_candL1IsoEG)[iPair + nrObjRow]))->raw();
                                }
                                else {
                                    bDataVal = 0;
                                }
                                psbWordValue.setBData(bDataVal, iAB + iPair);

                            }

                        }

                            break;
                        case CenJetQ: {

                            //LogTrace("L1GlobalTriggerPSB")
                            //<< "\nL1GlobalTriggerPSB: write CenJetQ"
                            //<< std::endl;
                            
                            int recL1CenJet = m_candL1CenJet->size();
                            for (int iPair = 0; iPair < nrObjRow; ++iPair) {
                                if (iPair < recL1CenJet) {
                                    aDataVal = 
                                        (static_cast<const L1GctJetCand*> ((*m_candL1CenJet)[iPair]))->raw();
                                }
                                else {
                                    aDataVal = 0;
                                }
                                psbWordValue.setAData(aDataVal, iAB + iPair);

                                if ((iPair + nrObjRow) < recL1CenJet) {
                                    bDataVal = 
                                        (static_cast<const L1GctJetCand*> ((*m_candL1CenJet)[iPair + nrObjRow]))->raw();
                                }
                                else {
                                    bDataVal = 0;
                                }
                                psbWordValue.setBData(bDataVal, iAB + iPair);

                            }
                        }

                            break;
                        case ForJetQ: {
                            
                            //LogTrace("L1GlobalTriggerPSB")
                            //<< "\nL1GlobalTriggerPSB: write ForJetQ"
                            //<< std::endl;

                            int recL1ForJet = m_candL1ForJet->size();
                            for (int iPair = 0; iPair < nrObjRow; ++iPair) {
                                if (iPair < recL1ForJet) {
                                    aDataVal = 
                                        (static_cast<const L1GctJetCand*> ((*m_candL1ForJet)[iPair]))->raw();
                                }
                                else {
                                    aDataVal = 0;
                                }
                                psbWordValue.setAData(aDataVal, iAB + iPair);

                                if ((iPair + nrObjRow) < recL1ForJet) {
                                    bDataVal = 
                                        (static_cast<const L1GctJetCand*> ((*m_candL1ForJet)[iPair + nrObjRow]))->raw();
                                }
                                else {
                                    bDataVal = 0;
                                }
                                psbWordValue.setBData(bDataVal, iAB + iPair);

                            }

                        }

                            break;
                        case TauJetQ: {

                            //LogTrace("L1GlobalTriggerPSB")
                            //<< "\nL1GlobalTriggerPSB: write TauJetQ"
                            //<< std::endl;
                            
                            int recL1TauJet = m_candL1TauJet->size();
                            for (int iPair = 0; iPair < nrObjRow; ++iPair) {
                                if (iPair < recL1TauJet) {
                                    aDataVal = 
                                        (static_cast<const L1GctJetCand*> ((*m_candL1TauJet)[iPair]))->raw();
                                }
                                else {
                                    aDataVal = 0;
                                }
                                psbWordValue.setAData(aDataVal, iAB + iPair);

                                if ((iPair + nrObjRow) < recL1TauJet) {
                                    bDataVal = 
                                        (static_cast<const L1GctJetCand*> ((*m_candL1TauJet)[iPair + nrObjRow]))->raw();
                                }
                                else {
                                    bDataVal = 0;
                                }
                                psbWordValue.setBData(bDataVal, iAB + iPair);

                            }

                        }

                            break;
                        case ESumsQ: {
                            
                            //LogTrace("L1GlobalTriggerPSB")
                            //<< "\nL1GlobalTriggerPSB: write ESumsQ"
                            //<< std::endl;

                            // order: ETT, ETM et, HTT, ETM phi... hardcoded here
                            int iPair = 0;
                            
                            if (m_candETT) {
                                aDataVal = m_candETT->raw();
                            }
                            else {
                                aDataVal = 0;
                            }
                            psbWordValue.setAData(aDataVal, iAB + iPair);

                            if (m_candHTT) {
                                bDataVal = m_candHTT->raw();
                            }
                            else {
                                bDataVal = 0;
                            }
                            psbWordValue.setBData(bDataVal, iAB + iPair);

                            //
                            iPair = 1;
                            if (m_candETM) {
                                aDataVal = m_candETM->et();
                            }
                            else {
                                aDataVal = 0;
                            }
                            psbWordValue.setAData(aDataVal, iAB + iPair);

                            if (m_candETM) {
                                bDataVal = m_candETM->phi();
                            }
                            else {
                                bDataVal = 0;
                            }
                            psbWordValue.setBData(bDataVal, iAB + iPair);
                            
                            
                        }

                            break;
                        case JetCountsQ: {

                            //LogTrace("L1GlobalTriggerPSB")
                            //<< "\nL1GlobalTriggerPSB: write JetCountsQ"
                            //<< std::endl;
                            
                            // order: 3 JetCounts per 16-bits word ... hardcoded here
                            int jetCountsBits = 5; // FIXME get it from event setup
                            int countsPerWord = 3;
                            
                            //
                            int iPair = 0;
                            aDataVal = 0;
                            bDataVal = 0;
                            
                            int iCount = 0;
                            
                            if (m_candJetCounts) {

                                for (int i = 0; i < countsPerWord; ++i) {
                                    aDataVal = aDataVal
                                            | ((m_candJetCounts->count(iCount))
                                                    << (jetCountsBits*i));
                                    iCount++;
                                }

                                //

                                for (int i = 0; i < countsPerWord; ++i) {
                                    bDataVal = bDataVal
                                            | ((m_candJetCounts->count(iCount))
                                                    << (jetCountsBits*i));
                                    iCount++;
                                }

                            }

                            psbWordValue.setAData(aDataVal, iAB + iPair);
                            psbWordValue.setBData(bDataVal, iAB + iPair);

                            //
                            iPair = 1;
                            aDataVal = 0;
                            bDataVal = 0;

                            if (m_candJetCounts) {
                                for (int i = 0; i < countsPerWord; ++i) {
                                    aDataVal = aDataVal
                                            | ((m_candJetCounts->count(iCount))
                                                    << (jetCountsBits*i));
                                    iCount++;
                                }

                                //

                                for (int i = 0; i < countsPerWord; ++i) {
                                    bDataVal = bDataVal
                                            | ((m_candJetCounts->count(iCount))
                                                    << (jetCountsBits*i));
                                    iCount++;
                                }

                            }
                            
                            psbWordValue.setAData(aDataVal, iAB + iPair);
                            psbWordValue.setBData(bDataVal, iAB + iPair);
                        }

                            break;
                            // FIXME add MIP/Iso bits
                        default: {
                            // do nothing
                        }

                            break;
                    } // end switch (*itQuad)

                } // end for: (itQuad)

                // ** fill L1PsbWord in GT DAQ record

                //LogTrace("L1GlobalTriggerPSB")
                //<< "\nL1GlobalTriggerPSB: write psbWordValue"
                //<< std::endl;
                
                gtDaqReadoutRecord->setGtPsbWord(psbWordValue);

            } // end if (active && PSB)

        } // end if (iPosition)

    } // end for (itBoard


}

// clear PSB

void L1GlobalTriggerPSB::reset() {

    m_candL1NoIsoEG->clear();
    m_candL1IsoEG->clear();
    m_candL1CenJet->clear();
    m_candL1ForJet->clear();
    m_candL1TauJet->clear();

    // no reset() available...
    m_candETM = 0;
    m_candETT = 0;
    m_candHTT = 0;

    m_candJetCounts = 0;

}

// print Global Calorimeter Trigger data
// use int to bitset conversion to print
void L1GlobalTriggerPSB::printGctObjectData(const int iBxInEvent) const
{

    LogTrace("L1GlobalTriggerPSB")
            << "\nL1GlobalTrigger: GCT data [hex] received by PSBs for BxInEvent = "
            << iBxInEvent << "\n" << std::endl;
    
    std::vector<const L1GctCand*>::const_iterator iterConst;

    LogTrace("L1GlobalTriggerPSB") << "   GCT NoIsoEG " << std::endl;
    for ( iterConst = m_candL1NoIsoEG->begin(); iterConst != m_candL1NoIsoEG->end(); iterConst++ ) {

        LogTrace("L1GlobalTriggerPSB")
        << std::hex
        << "Rank = " << (*iterConst)->rank()
        << " Eta index = " << (*iterConst)->etaIndex()
        << " Phi index = " << (*iterConst)->phiIndex()
        << std::dec
        << std::endl;
    }

    LogTrace("L1GlobalTriggerPSB") << "   GCT IsoEG " << std::endl;
    for ( iterConst = m_candL1IsoEG->begin(); iterConst != m_candL1IsoEG->end(); iterConst++ ) {
        LogTrace("L1GlobalTriggerPSB")
        << std::hex
        << "Rank = " << (*iterConst)->rank()
        << " Eta index = " << (*iterConst)->etaIndex()
        << " Phi index = " << (*iterConst)->phiIndex()
        << std::dec
        << std::endl;
    }

    LogTrace("L1GlobalTriggerPSB") << "   GCT CenJet " << std::endl;
    for ( iterConst = m_candL1CenJet->begin(); iterConst != m_candL1CenJet->end(); iterConst++ ) {
        LogTrace("L1GlobalTriggerPSB")
        << std::hex
        << "Rank = " << (*iterConst)->rank()
        << " Eta index = " << (*iterConst)->etaIndex()
        << " Phi index = " << (*iterConst)->phiIndex()
        << std::dec
        << std::endl;
    }

    LogTrace("L1GlobalTriggerPSB") << "   GCT ForJet " << std::endl;
    for ( iterConst = m_candL1ForJet->begin(); iterConst != m_candL1ForJet->end(); iterConst++ ) {
        LogTrace("L1GlobalTriggerPSB")
        << std::hex
        << "Rank = " << (*iterConst)->rank()
        << " Eta index = " << (*iterConst)->etaIndex()
        << " Phi index = " << (*iterConst)->phiIndex()
        << std::dec
        << std::endl;
    }

    LogTrace("L1GlobalTriggerPSB") << "   GCT TauJet " << std::endl;
    for ( iterConst = m_candL1TauJet->begin(); iterConst != m_candL1TauJet->end(); iterConst++ ) {
        LogTrace("L1GlobalTriggerPSB")
        << std::hex
        << "Rank = " << (*iterConst)->rank()
        << " Eta index = " << (*iterConst)->etaIndex()
        << " Phi index = " << (*iterConst)->phiIndex()
        << std::dec
        << std::endl;
    }

    LogTrace("L1GlobalTriggerPSB") << "   GCT ETM " << std::endl;
    if ( m_candETM ) {
        LogTrace("L1GlobalTriggerPSB")         
        << std::hex
        << "ET  = " << m_candETM->et() 
        << std::dec
        << std::endl;
        
        LogTrace("L1GlobalTriggerPSB") 
        << std::hex
        << "phi = " << m_candETM->phi() 
        << std::dec
        << std::endl;
    }

    LogTrace("L1GlobalTriggerPSB") << "   GCT ETT " << std::endl;
    if ( m_candETT )   {
        LogTrace("L1GlobalTriggerPSB") 
        << std::hex
        <<  "ET  = " << m_candETT->et() 
        << std::dec
        << std::endl;
    }

    LogTrace("L1GlobalTriggerPSB") << "   GCT HTT " << std::endl;
    if ( m_candHTT )   {
        LogTrace("L1GlobalTriggerPSB") 
        << std::hex
        <<  "ET  = " << m_candHTT->et() 
        << std::dec
        << std::endl;
    }

    LogTrace("L1GlobalTriggerPSB") << "   GCT JetCounts " << std::endl;
    if ( m_candJetCounts ) {
        LogTrace("L1GlobalTriggerPSB") << (*m_candJetCounts) << std::endl;
    }


}



// static data members
