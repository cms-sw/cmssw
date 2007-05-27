/**
 * \class L1GlobalTriggerReadoutRecord
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
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

// system include files
#include <iostream>
#include <iomanip>
#include <bitset>
#include <boost/cstdint.hpp>


// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructors
L1GlobalTriggerReadoutRecord::L1GlobalTriggerReadoutRecord()
{

    m_gtfeWord = L1GtfeWord();

    // reserve just one L1GtFdlWord, set bunch cross 0
    m_gtFdlWord.reserve(1);
    m_gtFdlWord.assign(1, L1GtFdlWord());

    int iBx = 0; // not really necessary, default bxInEvent in L1GtFdlWord() is zero
    m_gtFdlWord[iBx].setBxInEvent(iBx);

    // reserve totalNumberPsb L1GtPsbWord, set bunch cross 0
    int numberPsb = L1GlobalTriggerReadoutSetup::NumberPsbBoards;
    int totalNumberPsb = numberPsb;

    m_gtPsbWord.reserve(totalNumberPsb);
    m_gtPsbWord.assign(totalNumberPsb, L1GtPsbWord());

}

L1GlobalTriggerReadoutRecord::L1GlobalTriggerReadoutRecord(int numberBxInEvent)
{

    m_gtfeWord = L1GtfeWord();

    m_gtFdlWord.reserve(numberBxInEvent);
    m_gtFdlWord.assign(numberBxInEvent, L1GtFdlWord());

    // min value of bxInEvent
    int minBxInEvent = (numberBxInEvent + 1)/2 - numberBxInEvent;
    //int maxBxInEvent = (numberBxInEvent + 1)/2 - 1; // not needed

    // matrix index [0, numberBxInEvent) -> bxInEvent [minBxInEvent, maxBxInEvent]
    // warning: matrix index != bxInEvent
    for (int iFdl = 0; iFdl < numberBxInEvent; ++iFdl) {
        int iBxInEvent = minBxInEvent + iFdl;
        m_gtFdlWord[iFdl].setBxInEvent(iBxInEvent);
    }

    // PSBs
    int numberPsb = L1GlobalTriggerReadoutSetup::NumberPsbBoards;
    int totalNumberPsb = numberPsb*numberBxInEvent;

    m_gtPsbWord.reserve(totalNumberPsb);
    m_gtPsbWord.assign(totalNumberPsb, L1GtPsbWord());


}

// copy constructor
L1GlobalTriggerReadoutRecord::L1GlobalTriggerReadoutRecord(
    const L1GlobalTriggerReadoutRecord& result)
{

    m_gtfeWord = result.m_gtfeWord;
    m_gtFdlWord = result.m_gtFdlWord;
    m_gtPsbWord = result.m_gtPsbWord;

    m_muCollRefProd = result.m_muCollRefProd;


}

// destructor
L1GlobalTriggerReadoutRecord::~L1GlobalTriggerReadoutRecord()
{

    // empty now

}

// assignment operator
L1GlobalTriggerReadoutRecord& L1GlobalTriggerReadoutRecord::operator=(
    const L1GlobalTriggerReadoutRecord& result)
{

    if ( this != &result ) {

        m_gtfeWord = result.m_gtfeWord;
        m_gtFdlWord  = result.m_gtFdlWord;
        m_gtPsbWord  = result.m_gtPsbWord;

        m_muCollRefProd = result.m_muCollRefProd;

    }

    return *this;

}

// equal operator
bool L1GlobalTriggerReadoutRecord::operator==(
    const L1GlobalTriggerReadoutRecord& result) const
{

    if (m_gtfeWord != result.m_gtfeWord) {
        return false;
    }

    if (m_gtFdlWord  != result.m_gtFdlWord) {
        return false;
    }

    if (m_gtPsbWord  != result.m_gtPsbWord) {
        return false;
    }

    if (m_muCollRefProd != result.m_muCollRefProd) {
        return false;
    }

    // all members identical
    return true;

}

// unequal operator
bool L1GlobalTriggerReadoutRecord::operator!=(
    const L1GlobalTriggerReadoutRecord& result) const
{

    return !( result == *this);

}
// methods


// get Global Trigger decision
//    general bxInEvent
const bool L1GlobalTriggerReadoutRecord::decision(int bxInEventValue) const
{

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            return (*itBx).finalOR();
        }
    }

    // if bunch cross not found, throw exception (action: SkipEvent)
    // TODO re-evaluate action

    throw cms::Exception("NotFound")
    << "\nError: requested GtFdlWord for bxInEvent = " << bxInEventValue
    << " does not exist.\n"
    << "Can not return global decision for this bx!\n"
    << std::endl;

    return false;
}

//    bxInEvent = 0
const bool L1GlobalTriggerReadoutRecord::decision() const
{

    int bxInEventL1Accept = 0;
    return decision(bxInEventL1Accept);
}

// get Global Trigger decision word

const DecisionWord
L1GlobalTriggerReadoutRecord::decisionWord(int bxInEventValue) const
{

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            return (*itBx).gtDecisionWord();
        }
    }

    // if bunch cross not found, throw exception (action: SkipEvent)
    // TODO re-evaluate action

    throw cms::Exception("NotFound")
    << "\nError: requested GtFdlWord for bxInEvent = " << bxInEventValue
    << " does not exist.\n"
    << "Can not return decision word for this bx!\n"
    << std::endl;

    DecisionWord dW; // empty; it does not arrive here
    return dW;
}

const DecisionWord
L1GlobalTriggerReadoutRecord::decisionWord() const
{

    int bxInEventL1Accept = 0;
    return decisionWord(bxInEventL1Accept);
}


// set global decision
//    general
void L1GlobalTriggerReadoutRecord::setDecision(const bool& t, int bxInEventValue)
{

    for (std::vector<L1GtFdlWord>::iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {

            // TODO FIXME when manipulating partitions
            (*itBx).setFinalOR(static_cast<uint16_t> (t));
            return;
        }
    }

    // if bunch cross not found, throw exception (action: SkipEvent)

    throw cms::Exception("NotFound")
    << "\nError: requested GtFdlWord for bxInEvent = " << bxInEventValue
    << " does not exist.\n"
    << "Can not set global decision for this bx!\n"
    << std::endl;

}

//    bxInEvent = 0
void L1GlobalTriggerReadoutRecord::setDecision(const bool& t)
{

    int bxInEventL1Accept = 0;
    setDecision(t, bxInEventL1Accept);
}

// set decision word
void L1GlobalTriggerReadoutRecord::setDecisionWord(
    const DecisionWord& decisionWordValue,
    int bxInEventValue)
{

    for (std::vector<L1GtFdlWord>::iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {

            (*itBx).setGtDecisionWord (decisionWordValue);
            return;
        }
    }

    // if bunch cross not found, throw exception (action: SkipEvent)

    throw cms::Exception("NotFound")
    << "\nError: requested GtFdlWord for bxInEvent = " << bxInEventValue
    << " does not exist.\n"
    << "Can not set decision word for this bx!\n"
    << std::endl;

}

void L1GlobalTriggerReadoutRecord::setDecisionWord(
    const DecisionWord& decisionWordValue)
{

    int bxInEventL1Accept = 0;
    setDecisionWord(decisionWordValue, bxInEventL1Accept);

}

// print global decision and algorithm decision word
void L1GlobalTriggerReadoutRecord::printGtDecision(
    std::ostream& myCout, int bxInEventValue) const
{

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {

            myCout << "\nL1 Global Trigger Record: " << std::endl;

            myCout << "  Bunch cross " << bxInEventValue
            << std::endl
            << "  Global Decision = " << std::setw(5) << (*itBx).globalDecision()
            << std::endl
            << "  Decision word (bitset style)\n  ";

            (*itBx).printGtDecisionWord(myCout);

        }
    }

    myCout << std::endl;

}

void L1GlobalTriggerReadoutRecord::printGtDecision(std::ostream& myCout) const
{

    int bxInEventL1Accept = 0;
    printGtDecision(myCout, bxInEventL1Accept);

}

// print technical trigger word (reverse order for vector<bool>)
void L1GlobalTriggerReadoutRecord::printTechnicalTrigger(
    std::ostream& myCout, int bxInEventValue
) const
{

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {

            myCout << "\nL1 Global Trigger Record: " << std::endl;

            myCout << "  Bunch cross " << bxInEventValue
            << std::endl
            << "  Technical Trigger word (bitset style)\n  ";

            (*itBx).printGtTechnicalTriggerWord(myCout);
        }
    }

    myCout << std::endl;

}

void L1GlobalTriggerReadoutRecord::printTechnicalTrigger(std::ostream& myCout) const
{

    int bxInEventL1Accept = 0;
    printTechnicalTrigger(myCout, bxInEventL1Accept);

}


// get/set physical candidates

// muons

// get / set reference to L1MuGMTReadoutCollection
const edm::RefProd<L1MuGMTReadoutCollection>
L1GlobalTriggerReadoutRecord::muCollectionRefProd() const
{

    return m_muCollRefProd;
}

void L1GlobalTriggerReadoutRecord::setMuCollectionRefProd(
    edm::Handle<L1MuGMTReadoutCollection>& muHandle)
{

    m_muCollRefProd = edm::RefProd<L1MuGMTReadoutCollection>(muHandle);

}

void L1GlobalTriggerReadoutRecord::setMuCollectionRefProd(
    const edm::RefProd<L1MuGMTReadoutCollection>& refProdMuGMT)
{

    m_muCollRefProd = refProdMuGMT;

}

// return muon candidate(s)

const L1MuGMTExtendedCand L1GlobalTriggerReadoutRecord::muonCand(
    unsigned int indexCand, int bxInEvent) const
{

    assert( indexCand >= 0 && indexCand < L1GlobalTriggerReadoutSetup::NumberL1Muons );
    return (*m_muCollRefProd).getRecord(bxInEvent).getGMTCands()[indexCand];

}

const L1MuGMTExtendedCand L1GlobalTriggerReadoutRecord::muonCand(
    unsigned int indexCand) const
{

    int bxInEventL1Accept = 0;
    return muonCand(indexCand, bxInEventL1Accept);

}

std::vector<L1MuGMTExtendedCand> L1GlobalTriggerReadoutRecord::muonCands(
    int bxInEventValue) const
{

    return (*m_muCollRefProd).getRecord(bxInEventValue).getGMTCands();

}

std::vector<L1MuGMTExtendedCand> L1GlobalTriggerReadoutRecord::muonCands() const
{

    int bxInEventL1Accept = 0;
    return muonCands(bxInEventL1Accept);

}

// electron

const L1GctEmCand L1GlobalTriggerReadoutRecord::electronCand(
    unsigned int indexCand, int bxInEventValue) const
{

    // TODO FIXME bxInEvent dependence, retrieval from PSB
    // now, it return an empty object
    assert( indexCand >= 0 && indexCand < L1GlobalTriggerReadoutSetup::NumberL1Electrons );

    bool electronIso = false;
    CaloDataWord dataword = 0;

    return L1GctEmCand( dataword, electronIso );

}

const L1GctEmCand L1GlobalTriggerReadoutRecord::electronCand(
    unsigned int indexCand) const
{

    int bxInEventL1Accept = 0;
    return electronCand(indexCand, bxInEventL1Accept);

}

std::vector<L1GctEmCand> L1GlobalTriggerReadoutRecord::electronCands(
    int bxInEventValue) const
{

    std::vector<L1GctEmCand> result;

    for ( unsigned int  i = 0; i != L1GlobalTriggerReadoutSetup::NumberL1Electrons; i++ ) {
        L1GctEmCand cand(electronCand(i, bxInEventValue));
        if ( !cand.empty() )
            result.push_back(cand);
    }

    return result;

}

std::vector<L1GctEmCand> L1GlobalTriggerReadoutRecord::electronCands() const
{

    int bxInEventL1Accept = 0;
    return electronCands(bxInEventL1Accept);

}

// isolated electron
const L1GctEmCand L1GlobalTriggerReadoutRecord::isolatedElectronCand(
    unsigned int indexCand, int bxInEventValue) const
{

    // TODO FIXME bxInEvent dependence, retrieval from PSB
    // now, it return an empty object
    assert( indexCand >= 0 && indexCand < L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons );

    bool electronIso = true;
    CaloDataWord dataword = 0;

    return L1GctEmCand( dataword, electronIso );

}
const L1GctEmCand L1GlobalTriggerReadoutRecord::isolatedElectronCand(
    unsigned int indexCand) const
{

    int bxInEventL1Accept = 0;
    return isolatedElectronCand(indexCand, bxInEventL1Accept);

}

std::vector<L1GctEmCand> L1GlobalTriggerReadoutRecord::isolatedElectronCands(
    int bxInEventValue) const
{

    std::vector<L1GctEmCand> result;

    for ( unsigned int  i = 0; i != L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons; i++ ) {
        L1GctEmCand cand(isolatedElectronCand(i, bxInEventValue));
        if ( !cand.empty() )
            result.push_back(cand);
    }

    return result;

}

std::vector<L1GctEmCand> L1GlobalTriggerReadoutRecord::isolatedElectronCands() const
{

    int bxInEventL1Accept = 0;
    return isolatedElectronCands(bxInEventL1Accept);

}

// central jet
const L1GctJetCand L1GlobalTriggerReadoutRecord::centralJetCand(
    unsigned int indexCand, int bxInEventValue) const
{

    // TODO FIXME bxInEvent dependence, retrieval from PSB
    // now, it return an empty object
    assert( indexCand >= 0 && indexCand < L1GlobalTriggerReadoutSetup::NumberL1CentralJets );

    bool isTau = false;
    bool isFor = false;
    CaloDataWord dataword = 0;

    return L1GctJetCand( dataword, isTau, isFor );

}

const L1GctJetCand L1GlobalTriggerReadoutRecord::centralJetCand(
    unsigned int indexCand) const
{

    int bxInEventL1Accept = 0;
    return centralJetCand(indexCand, bxInEventL1Accept);

}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::centralJetCands(
    int bxInEventValue) const
{

    std::vector<L1GctJetCand> result;

    for ( unsigned int  i = 0; i != L1GlobalTriggerReadoutSetup::NumberL1CentralJets; i++ ) {
        L1GctJetCand cand(centralJetCand(i, bxInEventValue));
        if ( !cand.empty() )
            result.push_back(cand);
    }

    return result;

}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::centralJetCands() const
{

    int bxInEventL1Accept = 0;
    return centralJetCands(bxInEventL1Accept);

}

// forward jet
const L1GctJetCand L1GlobalTriggerReadoutRecord::forwardJetCand(
    unsigned int indexCand, int bxInEventValue) const
{

    // TODO FIXME bxInEvent dependence, retrieval from PSB
    // now, it return an empty object
    assert( indexCand >= 0 && indexCand < L1GlobalTriggerReadoutSetup::NumberL1ForwardJets );

    bool isTau = false;
    bool isFor = true;
    CaloDataWord dataword = 0;

    return L1GctJetCand( dataword, isTau, isFor );

}

const L1GctJetCand L1GlobalTriggerReadoutRecord::forwardJetCand(
    unsigned int indexCand) const
{

    int bxInEventL1Accept = 0;
    return forwardJetCand(indexCand, bxInEventL1Accept);

}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::forwardJetCands(
    int bxInEventValue) const
{

    std::vector<L1GctJetCand> result;

    for ( unsigned int  i = 0; i != L1GlobalTriggerReadoutSetup::NumberL1ForwardJets; i++ ) {
        L1GctJetCand cand(forwardJetCand(i, bxInEventValue));
        if ( !cand.empty() )
            result.push_back(cand);
    }

    return result;

}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::forwardJetCands() const
{

    int bxInEventL1Accept = 0;
    return forwardJetCands(bxInEventL1Accept);

}

// tau jet
const L1GctJetCand L1GlobalTriggerReadoutRecord::tauJetCand(
    unsigned int indexCand, int bxInEventValue) const
{

    // TODO FIXME bxInEvent dependence, retrieval from PSB
    // now, it return an empty object
    assert( indexCand >= 0 && indexCand < L1GlobalTriggerReadoutSetup::NumberL1TauJets );

    bool isTau = true;
    bool isFor = false;
    CaloDataWord dataword = 0;

    return L1GctJetCand( dataword, isTau, isFor );

}

const L1GctJetCand L1GlobalTriggerReadoutRecord::tauJetCand(
    unsigned int indexCand) const
{

    int bxInEventL1Accept = 0;
    return tauJetCand(indexCand, bxInEventL1Accept);


}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::tauJetCands(
    int bxInEventValue) const
{

    std::vector<L1GctJetCand> result;

    for ( unsigned int  i = 0; i != L1GlobalTriggerReadoutSetup::NumberL1TauJets; i++ ) {
        L1GctJetCand cand(tauJetCand(i, bxInEventValue));
        if ( !cand.empty() )
            result.push_back(cand);
    }

    return result;

}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::tauJetCands() const
{

    int bxInEventL1Accept = 0;
    return tauJetCands(bxInEventL1Accept);

}

// missing Et
const L1GctEtMiss L1GlobalTriggerReadoutRecord::missingEt(
    int bxInEventValue) const
{

    // TODO FIXME bxInEvent dependence, retrieval from PSB
    // now, it return an empty object
    return L1GctEtMiss();

}

const L1GctEtMiss L1GlobalTriggerReadoutRecord::missingEt() const
{

    int bxInEventL1Accept = 0;
    return missingEt(bxInEventL1Accept);

}

// total Et
const L1GctEtTotal L1GlobalTriggerReadoutRecord::totalEt(
    int bxInEventValue) const
{

    // TODO FIXME bxInEvent dependence, retrieval from PSB
    // now, it return an empty object
    return L1GctEtTotal();

}

const L1GctEtTotal L1GlobalTriggerReadoutRecord::totalEt() const
{

    int bxInEventL1Accept = 0;
    return totalEt(bxInEventL1Accept);

}

// total calibrated Et in jets
const L1GctEtHad L1GlobalTriggerReadoutRecord::totalHt(
    int bxInEventValue) const
{

    // TODO FIXME bxInEvent dependence, retrieval from PSB
    // now, it return an empty object
    return L1GctEtHad();

}

const L1GctEtHad L1GlobalTriggerReadoutRecord::totalHt() const
{

    int bxInEventL1Accept = 0;
    return totalHt(bxInEventL1Accept);

}

// jet counts
const L1GctJetCounts L1GlobalTriggerReadoutRecord::jetCounts(
    int bxInEventValue) const
{

    // TODO FIXME bxInEvent dependence, retrieval from PSB
    // now, it return an empty object
    return L1GctJetCounts();

}

const L1GctJetCounts L1GlobalTriggerReadoutRecord::jetCounts() const
{

    int bxInEventL1Accept = 0;
    return jetCounts(bxInEventL1Accept);

}

// set candidate data words (all non-empty candidates)

// muon

void L1GlobalTriggerReadoutRecord::setMuons(
    const std::vector<MuonDataWord>& vec, int bxInEventValue)
{

    // TODO bxInEvent dependence
    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1Muons ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1Muons;
    for (int i = 0; i < max; i++) {
        //        m_gtMuon[i] = vec[i]; // FIXME
    }

}

void L1GlobalTriggerReadoutRecord::setMuons(
    const std::vector<MuonDataWord>& vec)
{

    int bxInEventL1Accept = 0;
    setMuons(vec, bxInEventL1Accept);

}

// electron
void L1GlobalTriggerReadoutRecord::setElectrons(
    const std::vector<CaloDataWord>& vec, int bxInEventValue)
{

    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1Electrons ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1Electrons;

    // TODO FIXME set the objects in the corresponding PSB

}

void L1GlobalTriggerReadoutRecord::setElectrons(
    const std::vector<CaloDataWord>& vec)
{

    int bxInEventL1Accept = 0;
    setElectrons(vec, bxInEventL1Accept);

}

// isolated electron
void L1GlobalTriggerReadoutRecord::setIsolatedElectrons(
    const std::vector<CaloDataWord>& vec, int bxInEventValue)
{

    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons;
    // TODO FIXME set the objects in the corresponding PSB

}

void L1GlobalTriggerReadoutRecord::setIsolatedElectrons(
    const std::vector<CaloDataWord>& vec)
{

    int bxInEventL1Accept = 0;
    setIsolatedElectrons(vec, bxInEventL1Accept);

}

// central jets
void L1GlobalTriggerReadoutRecord::setCentralJets(
    const std::vector<CaloDataWord>& vec, int bxInEventValue)
{

    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1CentralJets ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1CentralJets;
    // TODO FIXME set the objects in the corresponding PSB

}

void L1GlobalTriggerReadoutRecord::setCentralJets(
    const std::vector<CaloDataWord>& vec)
{

    int bxInEventL1Accept = 0;
    setCentralJets(vec, bxInEventL1Accept);

}

// forward jets
void L1GlobalTriggerReadoutRecord::setForwardJets(
    const std::vector<CaloDataWord>& vec, int bxInEventValue)
{

    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1ForwardJets ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1ForwardJets;
    // TODO FIXME set the objects in the corresponding PSB

}
void L1GlobalTriggerReadoutRecord::setForwardJets(
    const std::vector<CaloDataWord>& vec)
{

    int bxInEventL1Accept = 0;
    setForwardJets(vec, bxInEventL1Accept);

}

// tau jets
void L1GlobalTriggerReadoutRecord::setTauJets(
    const std::vector<CaloDataWord>& vec, int bxInEventValue)
{

    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1TauJets ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1TauJets;
    // TODO FIXME set the objects in the corresponding PSB

}

void L1GlobalTriggerReadoutRecord::setTauJets(
    const std::vector<CaloDataWord>& vec)
{

    int bxInEventL1Accept = 0;
    setTauJets(vec, bxInEventL1Accept);

}

// missing Et
void L1GlobalTriggerReadoutRecord::setMissingEt(
    const CaloMissingEtWord& met, int bxInEventValue)
{

    // TODO FIXME set the objects in the corresponding PSB

}

void L1GlobalTriggerReadoutRecord::setMissingEt(
    const CaloMissingEtWord& met)
{

    int bxInEventL1Accept = 0;
    setMissingEt(met, bxInEventL1Accept);

}

// total Et
void L1GlobalTriggerReadoutRecord::setTotalEt(
    const CaloDataWord& ett, int bxInEventValue)
{

    // TODO FIXME set the objects in the corresponding PSB

}

void L1GlobalTriggerReadoutRecord::setTotalEt(
    const CaloDataWord& ett)
{

    int bxInEventL1Accept = 0;
    setTotalEt(ett, bxInEventL1Accept);

}

// total calibrated Et
void L1GlobalTriggerReadoutRecord::setTotalHt(
    const CaloDataWord& htt, int bxInEventValue)
{

    // TODO FIXME set the objects in the corresponding PSB

}

void L1GlobalTriggerReadoutRecord::setTotalHt(
    const CaloDataWord& htt)
{

    int bxInEventL1Accept = 0;
    setTotalHt(htt, bxInEventL1Accept);

}

// jet count
void L1GlobalTriggerReadoutRecord::setJetCounts(
    const CaloJetCountsWord& jnr, int bxInEventValue)
{

    // TODO FIXME set the objects in the corresponding PSB

}

void L1GlobalTriggerReadoutRecord::setJetCounts(
    const CaloJetCountsWord& jnr)
{

    int bxInEventL1Accept = 0;
    setJetCounts(jnr, bxInEventL1Accept);

}

// print all L1 Trigger Objects (use int to bitset conversion)
void L1GlobalTriggerReadoutRecord::printL1Objects(
    std::ostream& myCout, int bxInEventValue) const
{

    myCout << "\nL1GlobalTriggerReadoutRecord: L1 Trigger Objects \n" << std::endl;

    myCout << "   GMT Muons " << std::endl;
    L1MuGMTReadoutRecord muRecord = (*m_muCollRefProd).getRecord(bxInEventValue);

    std::vector<L1MuGMTExtendedCand> exc = muRecord.getGMTCands();
    for(std::vector<L1MuGMTExtendedCand>::const_iterator muRecIt = exc.begin();
            muRecIt != exc.end(); muRecIt++) {

        myCout << *muRecIt << std::endl;

    }

    myCout << "   GCT Non Isolated Electrons " << std::endl;

    std::vector<L1GctEmCand> gtElectrons = electronCands(bxInEventValue);
    for (unsigned int i = 0; i < L1GlobalTriggerReadoutSetup::NumberL1Electrons; i++) {
        myCout << gtElectrons.at(i) << std::endl;
    }

    myCout << "   GCT Isolated Electrons " << std::endl;

    std::vector<L1GctEmCand> gtIsoElectrons = isolatedElectronCands(bxInEventValue);
    for ( unsigned int  i = 0; i < L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons; i++) {
        myCout << gtIsoElectrons.at(i) << std::endl;
    }

    myCout << "   GCT Central Jets " << std::endl;

    std::vector<L1GctJetCand> gtCJet = centralJetCands(bxInEventValue);
    for ( unsigned int  i = 0; i < L1GlobalTriggerReadoutSetup::NumberL1CentralJets; i++) {
        myCout << gtCJet.at(i) << std::endl;
    }

    myCout << "   GCT Forward Jets " << std::endl;

    std::vector<L1GctJetCand> gtFJet = forwardJetCands(bxInEventValue);
    for ( unsigned int  i = 0; i < L1GlobalTriggerReadoutSetup::NumberL1ForwardJets; i++) {
        myCout << gtFJet.at(i) << std::endl;
    }

    myCout << "   GCT Tau Jets " << std::endl;

    std::vector<L1GctJetCand> gtTJet = tauJetCands(bxInEventValue);
    for ( unsigned int  i = 0; i < L1GlobalTriggerReadoutSetup::NumberL1TauJets; i++) {
        myCout << gtTJet.at(i) << std::endl;
    }

    myCout << "   GCT Missing Transverse Energy " << std::endl;

    const L1GctEtMiss gtMissingEt = missingEt(bxInEventValue);
    //    myCout << gtMissingEt << std::endl;

    CaloMissingEtWord rawMet = gtMissingEt.raw();
    myCout
    << std::bitset<L1GlobalTriggerReadoutSetup::NumberMissingEtBits>(rawMet)
    << std::endl;

    myCout << "   GCT Total Transverse Energy " << std::endl;

    const L1GctEtTotal gtTotalEt = totalEt(bxInEventValue);
    //    myCout << gtTotalEt << std::endl;

    CaloDataWord rawEtt = gtTotalEt.raw();
    myCout
    << std::bitset<L1GlobalTriggerReadoutSetup::NumberCaloBits>(rawEtt)
    << std::endl;

    myCout << "   GCT Total Hadron Transverse Energy " << std::endl;

    const L1GctEtHad gtTotalHt = totalHt(bxInEventValue);
    //    myCout << gtTotalHt << std::endl;

    CaloDataWord rawHtt = gtTotalHt.raw();
    myCout
    << std::bitset<L1GlobalTriggerReadoutSetup::NumberCaloBits>(rawHtt)
    << std::endl;

    myCout << "   GCT Jet Counts " << std::endl;

    const L1GctJetCounts gtJetCounts = jetCounts(bxInEventValue);
    myCout << gtJetCounts << std::endl;

    myCout << std::endl;

}

void L1GlobalTriggerReadoutRecord::printL1Objects(std::ostream& myCout) const
{

    int bxInEventL1Accept = 0;
    printL1Objects(myCout, bxInEventL1Accept);

}

// get/set hardware-related words

// get / set GTFE word (record) in the GT readout record
const L1GtfeWord L1GlobalTriggerReadoutRecord::gtfeWord() const
{

    return m_gtfeWord;

}

void L1GlobalTriggerReadoutRecord::setGtfeWord(const L1GtfeWord& gtfeWordValue)
{

    m_gtfeWord = gtfeWordValue;

}

// get / set FDL word (record) in the GT readout record
const L1GtFdlWord L1GlobalTriggerReadoutRecord::gtFdlWord(int bxInEventValue) const
{

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            return (*itBx);
        }
    }

    // if bunch cross not found, throw exception (action: SkipEvent)

    throw cms::Exception("NotFound")
    << "\nError: requested L1GtFdlWord for bxInEvent = " << bxInEventValue
    << " does not exist.\n"
    << std::endl;

    // return empty record - actually does not arrive here
    return L1GtFdlWord();

}

const L1GtFdlWord L1GlobalTriggerReadoutRecord::gtFdlWord() const
{

    int bxInEventL1Accept = 0;
    return gtFdlWord(bxInEventL1Accept);
}

void L1GlobalTriggerReadoutRecord::setGtFdlWord(
    const L1GtFdlWord& gtFdlWordValue, int bxInEventValue)
{

    // if a L1GtFdlWord exists for bxInEventValue, replace it
    for (std::vector<L1GtFdlWord>::iterator itBx = m_gtFdlWord.begin();
            itBx != m_gtFdlWord.end(); ++itBx) {

        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            *itBx = gtFdlWordValue;
            LogDebug("L1GlobalTriggerReadoutRecord")
            << "Replacing L1GtFdlWord for bunch bxInEvent = " << bxInEventValue << "\n"
            << std::endl;
            return;
        }
    }

    // if bunch cross not found, throw exception (action: SkipEvent)
    // all L1GtFdlWord are created in the record constructor for allowed bunch crosses

    throw cms::Exception("NotFound")
    << "\nError: Cannot set L1GtFdlWord for bxInEvent = " << bxInEventValue
    << std::endl;

}

void L1GlobalTriggerReadoutRecord::setGtFdlWord(const L1GtFdlWord& gtFdlWordValue)
{

    int bxInEventL1Accept = 0;
    setGtFdlWord(gtFdlWordValue, bxInEventL1Accept);

}


// get / set PSB word (record) in the GT readout record
const L1GtPsbWord L1GlobalTriggerReadoutRecord::gtPsbWord(
    boost::uint16_t boardIdValue, int bxInEventValue) const
{

    for (std::vector<L1GtPsbWord>::const_iterator itBx = m_gtPsbWord.begin();
            itBx != m_gtPsbWord.end(); ++itBx) {

        if (
            ((*itBx).bxInEvent() == bxInEventValue) &&
            ((*itBx).boardId() == boardIdValue)) {

            return (*itBx);

        }
    }

    // if bunch cross or boardId not found, throw exception (action: SkipEvent)

    throw cms::Exception("NotFound")
    << "\nError: requested L1GtPsbWord for boardId = "
    << std::hex << boardIdValue << std::dec
    << " and bunch bxInEvent = " << bxInEventValue
    << " does not exist.\n"
    << std::endl;

    // return empty record - actually does not arrive here
    return L1GtPsbWord();

}

const L1GtPsbWord L1GlobalTriggerReadoutRecord::gtPsbWord(boost::uint16_t boardIdValue) const
{

    int bxInEventL1Accept = 0;
    return gtPsbWord(boardIdValue, bxInEventL1Accept);
}

void L1GlobalTriggerReadoutRecord::setGtPsbWord(
    const L1GtPsbWord& gtPsbWordValue, boost::uint16_t boardIdValue, int bxInEventValue)
{

    // if a L1GtPsbWord with the same bxInEventValue and boardIdValue exists, replace it
    for (std::vector<L1GtPsbWord>::iterator itBx = m_gtPsbWord.begin();
            itBx != m_gtPsbWord.end(); ++itBx) {

        if (
            ((*itBx).bxInEvent() == bxInEventValue) &&
            ((*itBx).boardId() == boardIdValue)) {

            *itBx = gtPsbWordValue;

            LogDebug("L1GlobalTriggerReadoutRecord")
            << "\nReplacing L1GtPsbWord with boardId = "
            << std::hex << boardIdValue << std::dec
            << " and bunch bxInEvent = " << bxInEventValue
            << "\n"
            << std::endl;
            return;
        }
    }

    // otherwise, write in the first empty PSB
    // empty means: PSB with bxInEvent = 0, boardId = 0

    for (std::vector<L1GtPsbWord>::iterator itBx = m_gtPsbWord.begin();
            itBx != m_gtPsbWord.end(); ++itBx) {

        if (
            ((*itBx).bxInEvent() == 0) &&
            ((*itBx).boardId() == 0)) {

            *itBx = gtPsbWordValue;

            LogDebug("L1GlobalTriggerReadoutRecord")
            << "\nFilling an empty L1GtPsbWord for PSB with boardId = "
            << std::hex << boardIdValue << std::dec
            << " and bunch bxInEvent = " << bxInEventValue
            << "\n"
            << std::endl;
            return;
        }
    }

    // no PSB to replace, no empty PSB: throw exception (action: SkipEvent)
    // all L1GtPsbWord are created in the record constructor

    throw cms::Exception("NotFound")
    << "\nError: Cannot set L1GtPsbWord for PSB with boardId = "
    << std::hex << boardIdValue << std::dec
    << " and bunch bxInEvent = " << bxInEventValue
    << "\n  No PSB to replace and no empty PSB found!\n"
    << std::endl;


}

void L1GlobalTriggerReadoutRecord::setGtPsbWord(
    const L1GtPsbWord& gtPsbWordValue, boost::uint16_t boardIdValue)
{

    int bxInEventL1Accept = 0;
    setGtPsbWord(gtPsbWordValue, boardIdValue, bxInEventL1Accept);

}

// other methods

// clear the record
// it resets the content of the members only!
void L1GlobalTriggerReadoutRecord::reset()
{

    m_gtfeWord.reset();

    for (std::vector<L1GtFdlWord>::iterator itFdl = m_gtFdlWord.begin();
            itFdl != m_gtFdlWord.end(); ++itFdl) {

        itFdl->reset();

    }

    for (std::vector<L1GtPsbWord>::iterator itPsb = m_gtPsbWord.begin();
            itPsb != m_gtPsbWord.end(); ++itPsb) {

        itPsb->reset();

    }

    // TODO FIXME reset m_muCollRefProd

}

// output stream operator
std::ostream& operator<<(std::ostream& s, const L1GlobalTriggerReadoutRecord& result)
{
    // TODO FIXME put together all prints
    s << "Not available yet - sorry";

    return s;

}
