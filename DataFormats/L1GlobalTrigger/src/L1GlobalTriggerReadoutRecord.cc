/**
 * \class L1GlobalTriggerReadoutRecord
 * 
 * 
 * 
 * Description: see header file 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: N. Neumeister        - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
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

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructors
L1GlobalTriggerReadoutRecord::L1GlobalTriggerReadoutRecord() {

    m_gtfeWord = L1GtfeWord();
    
    // reserve just one L1GtFdlWord, set bunch cross 0
    m_gtFdlWord.reserve(1);
    m_gtFdlWord.assign(1, L1GtFdlWord());

    int iBx = 0; // not really necessary, default bxInEvent in L1GtFdlWord() is zero   
    m_gtFdlWord[iBx].setBxInEvent(iBx);
    
    // TODO FIXME RefProd m_muCollRefProd ?     

    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1Electrons; ++indexCand) {
        m_gtElectron[indexCand] = 0;        
    }  
    
    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons; ++indexCand) {
        m_gtIsoElectron[indexCand] = 0;        
    }  

    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1CentralJets; ++indexCand) {
        m_gtCJet[indexCand] = 0;        
    }  

    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1ForwardJets; ++indexCand) {
        m_gtFJet[indexCand] = 0;        
    }  
    
    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1TauJets; ++indexCand) {
        m_gtTJet[indexCand] = 0;        
    }  

    m_gtMissingEt = 0;
    m_gtTotalEt = 0;    
    m_gtTotalHt = 0;
  
    m_gtJetNr.reserve(L1GlobalTriggerReadoutSetup::NumberL1JetCounts);
    m_gtJetNr.assign(L1GlobalTriggerReadoutSetup::NumberL1JetCounts, 0);
                  
}

L1GlobalTriggerReadoutRecord::L1GlobalTriggerReadoutRecord(int NumberBxInEvent) {

    m_gtfeWord = L1GtfeWord();
    
    m_gtFdlWord.reserve(NumberBxInEvent);
    m_gtFdlWord.assign(NumberBxInEvent, L1GtFdlWord());
    
    for (int iBx = 0; iBx < NumberBxInEvent; ++iBx) {  // TODO review here after hw discussion
        m_gtFdlWord[iBx].setBxInEvent(iBx);
    }        
         
    // TODO FIXME RefProd m_muCollRefProd ?     

    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1Electrons; ++indexCand) {
        m_gtElectron[indexCand] = 0;        
    }  
    
    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons; ++indexCand) {
        m_gtIsoElectron[indexCand] = 0;        
    }  

    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1CentralJets; ++indexCand) {
        m_gtCJet[indexCand] = 0;        
    }  

    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1ForwardJets; ++indexCand) {
        m_gtFJet[indexCand] = 0;        
    }  
    
    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1TauJets; ++indexCand) {
        m_gtTJet[indexCand] = 0;        
    }  

    m_gtMissingEt = 0;
    m_gtTotalEt = 0;    
    m_gtTotalHt = 0;
  
    m_gtJetNr.reserve(L1GlobalTriggerReadoutSetup::NumberL1JetCounts);
    m_gtJetNr.assign(L1GlobalTriggerReadoutSetup::NumberL1JetCounts, 0);
                  
}

// copy constructor
L1GlobalTriggerReadoutRecord::L1GlobalTriggerReadoutRecord(
    const L1GlobalTriggerReadoutRecord& result) {

    m_gtfeWord = result.m_gtfeWord;
    m_gtFdlWord = result.m_gtFdlWord;
        
    m_muCollRefProd = result.m_muCollRefProd;
    
    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1Electrons; ++indexCand) {
        m_gtElectron[indexCand] = result.m_gtElectron[indexCand];  
    }
    
    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons; ++indexCand) {
        m_gtIsoElectron[indexCand] = result.m_gtIsoElectron[indexCand];
    }
      
    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1CentralJets; ++indexCand) {
        m_gtCJet[indexCand] = result.m_gtCJet[indexCand];  
    }
    
    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1ForwardJets; ++indexCand) {
        m_gtFJet[indexCand] = result.m_gtFJet[indexCand];  
    }
    
    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1TauJets; ++indexCand) {
        m_gtTJet[indexCand] = result.m_gtTJet[indexCand];
    }
  
    m_gtMissingEt = result.m_gtMissingEt;
    m_gtTotalEt = result.m_gtTotalEt;    
    m_gtTotalHt = result.m_gtTotalHt;
  
    m_gtJetNr = result.m_gtJetNr;

 }

// destructor
L1GlobalTriggerReadoutRecord::~L1GlobalTriggerReadoutRecord() {
    
    
}
      
// assignment operator
L1GlobalTriggerReadoutRecord& L1GlobalTriggerReadoutRecord::operator=(
    const L1GlobalTriggerReadoutRecord& result) {

    if ( this != &result ) {

        m_gtfeWord = result.m_gtfeWord;
        m_gtFdlWord  = result.m_gtFdlWord;
        
        m_muCollRefProd = result.m_muCollRefProd;
    
        for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::L1GlobalTriggerReadoutSetup::NumberL1Electrons; ++indexCand) {
            m_gtElectron[indexCand] = result.m_gtElectron[indexCand];  
        }
    
        for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons; ++indexCand) {
            m_gtIsoElectron[indexCand] = result.m_gtIsoElectron[indexCand];
        }
      
        for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1CentralJets; ++indexCand) {
            m_gtCJet[indexCand] = result.m_gtCJet[indexCand];  
        }
    
        for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1ForwardJets; ++indexCand) {
            m_gtFJet[indexCand] = result.m_gtFJet[indexCand];  
        }
    
        for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1TauJets; ++indexCand) {
            m_gtTJet[indexCand] = result.m_gtTJet[indexCand];
        }
  
        m_gtMissingEt = result.m_gtMissingEt;
        m_gtTotalEt = result.m_gtTotalEt;    
        m_gtTotalHt = result.m_gtTotalHt;
  
        m_gtJetNr = result.m_gtJetNr;
    }
    
    return *this;
    
 }
  
// equal operator
bool L1GlobalTriggerReadoutRecord::operator==(
    const L1GlobalTriggerReadoutRecord& result) const {

    if (m_gtfeWord != result.m_gtfeWord) return false;
    
    if (m_gtFdlWord  != result.m_gtFdlWord)  return false;

    if (m_muCollRefProd != result.m_muCollRefProd) return false;
  
    if (m_gtElectron    != result.m_gtElectron) return false;  
    if (m_gtIsoElectron != result.m_gtIsoElectron) return false;
  
    if (m_gtCJet != result.m_gtCJet) return false;  
    if (m_gtFJet != result.m_gtFJet) return false;  
    if (m_gtTJet != result.m_gtTJet) return false;
  
    if (m_gtMissingEt != result.m_gtMissingEt) return false;
    if (m_gtTotalEt   != result.m_gtTotalEt) return false;    
    if (m_gtTotalHt   != result.m_gtTotalHt) return false;
  
    if (m_gtJetNr != result.m_gtJetNr) return false;

    // all members identical
    return true;
    
}

// unequal operator
bool L1GlobalTriggerReadoutRecord::operator!=(
    const L1GlobalTriggerReadoutRecord& result) const{
    
    return !( result == *this);
    
}
// methods


// get Global Trigger decision
//    general bxInEvent 
const bool L1GlobalTriggerReadoutRecord::decision(unsigned int bxInEventValue) const {

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin(); 
        itBx != m_gtFdlWord.end(); ++itBx) {
        
        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            return (*itBx).finalOR();
        }               
    }
    
    // if bunch cross not found, throw exception (action: SkipEvent) // TODO re-evaluate action 
    
    throw cms::Exception("NotFound")
        << "\nError: requested GtFdlWord for bx = " << bxInEventValue << " does not exist.\n"
        << "Can not return global decision for this bx!\n"
        << std::endl;

    return false;          
}

//    bxInEvent = 0  
const bool L1GlobalTriggerReadoutRecord::decision() const {
    
    unsigned int bxInEventL1Accept = 0;
    return decision(bxInEventL1Accept);
}

// get Global Trigger decision word

const L1GlobalTriggerReadoutRecord::DecisionWord 
    L1GlobalTriggerReadoutRecord::decisionWord(unsigned int bxInEventValue) const {

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin(); 
        itBx != m_gtFdlWord.end(); ++itBx) {
        
        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            return (*itBx).gtDecisionWord();
        }               
    }
    
    // if bunch cross not found, throw exception (action: SkipEvent) // TODO re-evaluate action
    
    throw cms::Exception("NotFound")
        << "\nError: requested GtFdlWord for bx = " << bxInEventValue << " does not exist.\n"
        << "Can not return decision word for this bx!\n"
        << std::endl;

    DecisionWord dW; // empty; it does not arrive here
    return dW;          
}

const L1GlobalTriggerReadoutRecord::DecisionWord 
    L1GlobalTriggerReadoutRecord::decisionWord() const { 
    
    unsigned int bxInEventL1Accept = 0;
    return decisionWord(bxInEventL1Accept);
}
  

// set global decision
//    general 
void L1GlobalTriggerReadoutRecord::setDecision(const bool& t, unsigned int bxInEventValue) { 

    for (std::vector<L1GtFdlWord>::iterator itBx = m_gtFdlWord.begin(); 
        itBx != m_gtFdlWord.end(); ++itBx) {
        
        if ( (*itBx).bxInEvent() == bxInEventValue ) {

            (*itBx).setFinalOR(static_cast<uint16_t> (t)); // TODO FIXME when manipulating partitions
            return;            
        }               
    }
    
    // if bunch cross not found, throw exception (action: SkipEvent)
    
    throw cms::Exception("NotFound")
        << "\nError: requested GtFdlWord for bx = " << bxInEventValue << " does not exist.\n"
        << "Can not set global decision for this bx!\n"
        << std::endl;

}

//    bxInEvent = 0  
void L1GlobalTriggerReadoutRecord::setDecision(const bool& t) { 

    unsigned int bxInEventL1Accept = 0;
    setDecision(t, bxInEventL1Accept);
}

// set decision word
void L1GlobalTriggerReadoutRecord::setDecisionWord(
    const L1GlobalTriggerReadoutRecord::DecisionWord& decisionWordValue, 
    unsigned int bxInEventValue) {

    for (std::vector<L1GtFdlWord>::iterator itBx = m_gtFdlWord.begin(); 
        itBx != m_gtFdlWord.end(); ++itBx) {
        
        if ( (*itBx).bxInEvent() == bxInEventValue ) {

            (*itBx).setGtDecisionWord (decisionWordValue);
            return;
        }               
    }
    
    // if bunch cross not found, throw exception (action: SkipEvent)
    
    throw cms::Exception("NotFound")
        << "\nError: requested GtFdlWord for bx = " << bxInEventValue << " does not exist.\n"
        << "Can not set decision word for this bx!\n"
        << std::endl;

}

void L1GlobalTriggerReadoutRecord::setDecisionWord(
    const L1GlobalTriggerReadoutRecord::DecisionWord& decisionWordValue) {

    unsigned int bxInEventL1Accept = 0;
    setDecisionWord(decisionWordValue, bxInEventL1Accept);

}

// print global decision and algorithm decision word
void L1GlobalTriggerReadoutRecord::printGtDecision(
    std::ostream& myCout, unsigned int bxInEventValue
    ) const {
        
    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin(); 
        itBx != m_gtFdlWord.end(); ++itBx) {
        
        if ( (*itBx).bxInEvent() == bxInEventValue ) {

            myCout << "\nL1 Global Trigger Record: " << std::endl;
        
            myCout << "\t Bunch cross " << bxInEventValue 
                << std::endl
                << "\t Global Decision = " << std::setw(5) << (*itBx).globalDecision()
                 << std::endl 
                << "\t Decision word (bitset style) = "; 
            
            (*itBx).printGtDecisionWord(myCout);
        }               
    }      

    myCout << std::endl;
    
}

void L1GlobalTriggerReadoutRecord::printGtDecision(std::ostream& myCout) const {
    
    unsigned int bxInEventL1Accept = 0;
    printGtDecision(myCout, bxInEventL1Accept);
        
}

// print technical trigger word (reverse order for vector<bool>)
void L1GlobalTriggerReadoutRecord::printTechnicalTrigger(
    std::ostream& myCout, unsigned int bxInEventValue
    ) const {
        
    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin(); 
        itBx != m_gtFdlWord.end(); ++itBx) {
        
        if ( (*itBx).bxInEvent() == bxInEventValue ) {

            myCout << "\nL1 Global Trigger Record: " << std::endl;
        
            myCout << "\t Bunch cross " << bxInEventValue 
                << std::endl
                << "\t Technical Trigger word (bitset style) = "; 
            
            (*itBx).printGtTechnicalTriggerWord(myCout);
        }               
    }      

    myCout << std::endl;
    
}

void L1GlobalTriggerReadoutRecord::printTechnicalTrigger(std::ostream& myCout) const {
    
    unsigned int bxInEventL1Accept = 0;
    printTechnicalTrigger(myCout, bxInEventL1Accept);
        
}


// get/set physical candidates 

// muons

// get / set reference to L1MuGMTReadoutCollection
const edm::RefProd<L1MuGMTReadoutCollection> 
    L1GlobalTriggerReadoutRecord::muCollectionRefProd() const {
        
    return m_muCollRefProd; 
}

void L1GlobalTriggerReadoutRecord::setMuCollectionRefProd(
    edm::Handle<L1MuGMTReadoutCollection>& muHandle
    ) {

    m_muCollRefProd = edm::RefProd<L1MuGMTReadoutCollection>(muHandle);; 

} 

// return muon candidate(s)

const L1MuGMTExtendedCand L1GlobalTriggerReadoutRecord::muonCand(
    unsigned int indexCand, unsigned int bxInEvent) const {
        
    assert( indexCand >= 0 && indexCand < L1GlobalTriggerReadoutSetup::NumberL1Muons );
    return (*m_muCollRefProd).getRecord(bxInEvent).getGMTCands()[indexCand];
    
}

const L1MuGMTExtendedCand L1GlobalTriggerReadoutRecord::muonCand(
    unsigned int indexCand) const {
        
    unsigned int bxInEventL1Accept = 0;
    return muonCand(indexCand, bxInEventL1Accept);
    
}

std::vector<L1MuGMTExtendedCand> L1GlobalTriggerReadoutRecord::muonCands(
    unsigned int bxInEvent) const {

    return (*m_muCollRefProd).getRecord(bxInEvent).getGMTCands();
    
}

std::vector<L1MuGMTExtendedCand> L1GlobalTriggerReadoutRecord::muonCands() const {

    unsigned int bxInEventL1Accept = 0;
    return muonCands(bxInEventL1Accept);

}

// electron

const L1GctEmCand L1GlobalTriggerReadoutRecord::electronCand(
    unsigned int indexCand, unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < L1GlobalTriggerReadoutSetup::NumberL1Electrons );

    bool electronIso = false;
    return L1GctEmCand( m_gtElectron[indexCand], electronIso );

}

const L1GctEmCand L1GlobalTriggerReadoutRecord::electronCand(
    unsigned int indexCand) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < L1GlobalTriggerReadoutSetup::NumberL1Electrons );

    bool electronIso = false;
    return L1GctEmCand( m_gtElectron[indexCand], electronIso );

}

std::vector<L1GctEmCand> L1GlobalTriggerReadoutRecord::electronCands(
    unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    vector<L1GctEmCand> result;

    for ( unsigned int  i = 0; i != L1GlobalTriggerReadoutSetup::NumberL1Electrons; i++ ) {
        L1GctEmCand cand(electronCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;

}

std::vector<L1GctEmCand> L1GlobalTriggerReadoutRecord::electronCands() const {

    // TODO bxInEvent dependence
    vector<L1GctEmCand> result;

    for ( unsigned int  i = 0; i != L1GlobalTriggerReadoutSetup::NumberL1Electrons; i++ ) {
        L1GctEmCand cand(electronCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;
    
}

// isolated electron
const L1GctEmCand L1GlobalTriggerReadoutRecord::isolatedElectronCand(
    unsigned int indexCand, unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons );

    bool electronIso = true;
    return L1GctEmCand( m_gtIsoElectron[indexCand], electronIso );
    
}
const L1GctEmCand L1GlobalTriggerReadoutRecord::isolatedElectronCand(
    unsigned int indexCand) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons );

    bool electronIso = true;
    return L1GctEmCand( m_gtIsoElectron[indexCand], electronIso );

}

std::vector<L1GctEmCand> L1GlobalTriggerReadoutRecord::isolatedElectronCands(
    unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    vector<L1GctEmCand> result;

    for ( unsigned int  i = 0; i != L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons; i++ ) {
        L1GctEmCand cand(isolatedElectronCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;

}

std::vector<L1GctEmCand> L1GlobalTriggerReadoutRecord::isolatedElectronCands() const {

    // TODO bxInEvent dependence
    vector<L1GctEmCand> result;

    for ( unsigned int  i = 0; i != L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons; i++ ) {
        L1GctEmCand cand(isolatedElectronCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;
    
}

// central jet
const L1GctJetCand L1GlobalTriggerReadoutRecord::centralJetCand(
    unsigned int indexCand, unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < L1GlobalTriggerReadoutSetup::NumberL1CentralJets );

    bool isTau = false;
    bool isFor = false;
    return L1GctJetCand( m_gtCJet[indexCand], isTau, isFor );
    
}

const L1GctJetCand L1GlobalTriggerReadoutRecord::centralJetCand(
    unsigned int indexCand) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < L1GlobalTriggerReadoutSetup::NumberL1CentralJets );

    bool isTau = false;
    bool isFor = false;
    return L1GctJetCand( m_gtCJet[indexCand], isTau, isFor );
        
}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::centralJetCands(
    unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    vector<L1GctJetCand> result;

    for ( unsigned int  i = 0; i != L1GlobalTriggerReadoutSetup::NumberL1CentralJets; i++ ) {
        L1GctJetCand cand(centralJetCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;

}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::centralJetCands() const {

    // TODO bxInEvent dependence
    vector<L1GctJetCand> result;

    for ( unsigned int  i = 0; i != L1GlobalTriggerReadoutSetup::NumberL1CentralJets; i++ ) {
        L1GctJetCand cand(centralJetCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;

}

// forward jet
const L1GctJetCand L1GlobalTriggerReadoutRecord::forwardJetCand(
    unsigned int indexCand, unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < L1GlobalTriggerReadoutSetup::NumberL1ForwardJets );

    bool isTau = false;
    bool isFor = true;
    return L1GctJetCand( m_gtFJet[indexCand], isTau, isFor );
        
}

const L1GctJetCand L1GlobalTriggerReadoutRecord::forwardJetCand(
    unsigned int indexCand) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < L1GlobalTriggerReadoutSetup::NumberL1ForwardJets );

    bool isTau = false;
    bool isFor = true;
    return L1GctJetCand( m_gtFJet[indexCand], isTau, isFor );

}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::forwardJetCands(
    unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    vector<L1GctJetCand> result;

    for ( unsigned int  i = 0; i != L1GlobalTriggerReadoutSetup::NumberL1ForwardJets; i++ ) {
        L1GctJetCand cand(forwardJetCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;

}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::forwardJetCands() const {

    // TODO bxInEvent dependence
    vector<L1GctJetCand> result;

    for ( unsigned int  i = 0; i != L1GlobalTriggerReadoutSetup::NumberL1ForwardJets; i++ ) {
        L1GctJetCand cand(forwardJetCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;
    
}

// tau jet
const L1GctJetCand L1GlobalTriggerReadoutRecord::tauJetCand(
    unsigned int indexCand, unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < L1GlobalTriggerReadoutSetup::NumberL1TauJets );

    bool isTau = true;
    bool isFor = false;
    return L1GctJetCand( m_gtTJet[indexCand], isTau, isFor );
        
}

const L1GctJetCand L1GlobalTriggerReadoutRecord::tauJetCand(
    unsigned int indexCand) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < L1GlobalTriggerReadoutSetup::NumberL1TauJets );

    bool isTau = true;
    bool isFor = false;
    return L1GctJetCand( m_gtTJet[indexCand], isTau, isFor );

}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::tauJetCands(
    unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    vector<L1GctJetCand> result;

    for ( unsigned int  i = 0; i != L1GlobalTriggerReadoutSetup::NumberL1TauJets; i++ ) {
        L1GctJetCand cand(tauJetCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;
    
}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::tauJetCands() const {

    // TODO bxInEvent dependence
    vector<L1GctJetCand> result;

    for ( unsigned int  i = 0; i != L1GlobalTriggerReadoutSetup::NumberL1TauJets; i++ ) {
        L1GctJetCand cand(tauJetCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;
    
}

// missing Et
const L1GctEtMiss L1GlobalTriggerReadoutRecord::missingEt(
    unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    return L1GctEtMiss( m_gtMissingEt );
    
}

const L1GctEtMiss L1GlobalTriggerReadoutRecord::missingEt() const {

    // TODO bxInEvent dependence
    return L1GctEtMiss( m_gtMissingEt );

}

// total Et
const L1GctEtTotal L1GlobalTriggerReadoutRecord::totalEt(
    unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    return L1GctEtTotal( m_gtTotalEt );

}

const L1GctEtTotal L1GlobalTriggerReadoutRecord::totalEt() const {

    // TODO bxInEvent dependence
    return L1GctEtTotal( m_gtTotalEt );
    
}

// total calibrated Et in jets
const L1GctEtHad L1GlobalTriggerReadoutRecord::totalHt(
    unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    return L1GctEtHad( m_gtTotalHt );

}

const L1GctEtHad L1GlobalTriggerReadoutRecord::totalHt() const {

    // TODO bxInEvent dependence
    return L1GctEtHad( m_gtTotalHt );

}

// jet counts
const L1GctJetCounts L1GlobalTriggerReadoutRecord::jetCounts(
    unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    return L1GctJetCounts( m_gtJetNr );
        
}

const L1GctJetCounts L1GlobalTriggerReadoutRecord::jetCounts() const {

    // TODO bxInEvent dependence
    return L1GctJetCounts( m_gtJetNr );
    
}
  
// set candidate data words (all non-empty candidates)
  
// muon

void L1GlobalTriggerReadoutRecord::setMuons(
    const std::vector<MuonDataWord>& vec, unsigned int bxInEvent) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1Muons ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1Muons;
    for (int i = 0; i < max; i++) {
//        m_gtMuon[i] = vec[i]; // FIXME
    }

}

void L1GlobalTriggerReadoutRecord::setMuons(
    const std::vector<MuonDataWord>& vec) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1Muons ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1Muons;
    for (int i = 0; i < max; i++) {
//        m_gtMuon[i] = vec[i]; // FIXME
    }
        
}

// electron
void L1GlobalTriggerReadoutRecord::setElectrons(
    const std::vector<CaloDataWord>& vec, unsigned int bxInEvent) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1Electrons ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1Electrons;
    for (int i = 0; i < max; i++) {
        m_gtElectron[i] = vec[i];
    }
        
}

void L1GlobalTriggerReadoutRecord::setElectrons(
    const std::vector<CaloDataWord>& vec) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1Electrons ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1Electrons;
    for (int i = 0; i < max; i++) {
        m_gtElectron[i] = vec[i];
    }
        
}
  
// isolated electron
void L1GlobalTriggerReadoutRecord::setIsolatedElectrons(
    const std::vector<CaloDataWord>& vec, unsigned int bxInEvent) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons;
    for (int i = 0; i < max; i++) {
        m_gtIsoElectron[i] = vec[i];
    }

} 

void L1GlobalTriggerReadoutRecord::setIsolatedElectrons(
    const std::vector<CaloDataWord>& vec) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons;
    for (int i = 0; i < max; i++) {
        m_gtIsoElectron[i] = vec[i];
    }
    
}
  
// central jets
void L1GlobalTriggerReadoutRecord::setCentralJets(
    const std::vector<CaloDataWord>& vec, unsigned int bxInEvent) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1CentralJets ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1CentralJets;
    for (int i = 0; i < max; i++) {
        m_gtCJet[i] = vec[i];
    }
    
}

void L1GlobalTriggerReadoutRecord::setCentralJets(
    const std::vector<CaloDataWord>& vec) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1CentralJets ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1CentralJets;
    for (int i = 0; i < max; i++) {
        m_gtCJet[i] = vec[i];
    }

    
}
  
// forward jets
void L1GlobalTriggerReadoutRecord::setForwardJets(
    const std::vector<CaloDataWord>& vec, unsigned int bxInEvent) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1ForwardJets ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1ForwardJets;
    for (int i = 0; i < max; i++) {
        m_gtFJet[i] = vec[i];
    }
    
}
void L1GlobalTriggerReadoutRecord::setForwardJets(
    const std::vector<CaloDataWord>& vec) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1ForwardJets ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1ForwardJets;
    for (int i = 0; i < max; i++) {
        m_gtFJet[i] = vec[i];
    }
        
        
}
  
// tau jets
void L1GlobalTriggerReadoutRecord::setTauJets(
    const std::vector<CaloDataWord>& vec, unsigned int bxInEvent) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1TauJets ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1TauJets;
    for (int i = 0; i < max; i++) {
        m_gtTJet[i] = vec[i];
    }
    
}

void L1GlobalTriggerReadoutRecord::setTauJets(
    const std::vector<CaloDataWord>& vec) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= L1GlobalTriggerReadoutSetup::NumberL1TauJets ) ? vec.size() : L1GlobalTriggerReadoutSetup::NumberL1TauJets;
    for (int i = 0; i < max; i++) {
        m_gtTJet[i] = vec[i];
    }
        
}
  
// missing Et
void L1GlobalTriggerReadoutRecord::setMissingEt(
    const CaloMissingEtWord& met, unsigned int bxInEvent) {

    // TODO bxInEvent dependence
    m_gtMissingEt = met;
    
}

void L1GlobalTriggerReadoutRecord::setMissingEt(
    const CaloMissingEtWord& met) {

    // TODO bxInEvent dependence
    m_gtMissingEt = met;

}

// total Et
void L1GlobalTriggerReadoutRecord::setTotalEt(
    const CaloDataWord& ett, unsigned int bxInEvent) {

    // TODO bxInEvent dependence
    m_gtTotalEt = ett;

}

void L1GlobalTriggerReadoutRecord::setTotalEt(
    const CaloDataWord& ett) {

    // TODO bxInEvent dependence
    m_gtTotalEt = ett;

}

// total calibrated Et
void L1GlobalTriggerReadoutRecord::setTotalHt(
    const CaloDataWord& htt, unsigned int bxInEvent) {

    // TODO bxInEvent dependence
    m_gtTotalHt = htt;

}

void L1GlobalTriggerReadoutRecord::setTotalHt(
    const CaloDataWord& htt) {

    // TODO bxInEvent dependence
    m_gtTotalHt = htt;

}
  
// jet count
void L1GlobalTriggerReadoutRecord::setJetCounts(
    const CaloJetCountsWord& jnr, unsigned int bxInEvent) {

    // TODO bxInEvent dependence
    m_gtJetNr = jnr;

}

void L1GlobalTriggerReadoutRecord::setJetCounts(
    const CaloJetCountsWord& jnr) {

    // TODO bxInEvent dependence
    m_gtJetNr = jnr;
        
}

// print all L1 Trigger Objects (use int to bitset conversion) 
void L1GlobalTriggerReadoutRecord::printL1Objects(
    std::ostream& myCout, unsigned int bxInEventValue) const {

    // TODO FIXME add bunch cross dependence
    myCout << "\nL1GlobalTriggerReadoutRecord: L1 Trigger Objects \n" << std::endl;
     
    myCout << "   GMT Muons " << std::endl;
    L1MuGMTReadoutRecord muRecord = (*m_muCollRefProd).getRecord(bxInEventValue);

    std::vector<L1MuGMTExtendedCand> exc = muRecord.getGMTCands();
    for(std::vector<L1MuGMTExtendedCand>::const_iterator muRecIt = exc.begin(); 
        muRecIt != exc.end(); muRecIt++) {
        
        myCout << *muRecIt << std::endl;
    
    }
    
    myCout << "   GCT Non Isolated Electrons " << std::endl;
    for (unsigned int i = 0; i < L1GlobalTriggerReadoutSetup::NumberL1Electrons; i++) { 
        myCout << std::bitset<L1GlobalTriggerReadoutSetup::NumberCaloBits>(m_gtElectron[i]) << std::endl;
    }
    
    myCout << "   GCT Isolated Electrons " << std::endl;
    for ( unsigned int  i = 0; i < L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons; i++) {
        myCout << std::bitset<L1GlobalTriggerReadoutSetup::NumberCaloBits>(m_gtIsoElectron[i]) << std::endl;
    }
    
    myCout << "   GCT Central Jets " << std::endl;
    for ( unsigned int  i = 0; i < L1GlobalTriggerReadoutSetup::NumberL1CentralJets; i++) {
        myCout << std::bitset<L1GlobalTriggerReadoutSetup::NumberCaloBits>(m_gtCJet[i]) << std::endl;
    }
    
    myCout << "   GCT Forward Jets " << std::endl;
    for ( unsigned int  i = 0; i < L1GlobalTriggerReadoutSetup::NumberL1ForwardJets; i++) {
        myCout << std::bitset<L1GlobalTriggerReadoutSetup::NumberCaloBits>(m_gtFJet[i]) << std::endl;
    }
    
    myCout << "   GCT Tau Jets " << std::endl;
    for ( unsigned int  i = 0; i < L1GlobalTriggerReadoutSetup::NumberL1TauJets; i++) {
        myCout << std::bitset<L1GlobalTriggerReadoutSetup::NumberCaloBits>(m_gtTJet[i]) << std::endl;
    }
    
    myCout << "   GCT Missing Transverse Energy " << std::endl;
    myCout << std::bitset<L1GlobalTriggerReadoutSetup::NumberMissingEtBits>(m_gtMissingEt) << std::endl;

    myCout << "   GCT Total Transverse Energy " << std::endl;
    myCout << std::bitset<L1GlobalTriggerReadoutSetup::NumberCaloBits>(m_gtTotalEt) << std::endl;

    myCout << "   GCT Total Hadron Transverse Energy " << std::endl;
    myCout << std::bitset<L1GlobalTriggerReadoutSetup::NumberCaloBits>(m_gtTotalHt) << std::endl;

    myCout << "   GCT Jet Counts " << std::endl;
    myCout << "To be done" << std::endl; // TODO fix the way to print jet counts
        
    myCout << std::endl;
        
}

void L1GlobalTriggerReadoutRecord::printL1Objects(std::ostream& myCout) const {

    unsigned int bxInEventL1Accept = 0;
    printL1Objects(myCout, bxInEventL1Accept);
    
}
    
// get/set hardware-related words

// get / set GTFE word (record) in the GT readout record
const L1GtfeWord L1GlobalTriggerReadoutRecord::gtfeWord() const {

    return m_gtfeWord;
    
}

void L1GlobalTriggerReadoutRecord::setGtfeWord(const L1GtfeWord& gtfeWordValue) {

    m_gtfeWord = gtfeWordValue;

}

// get / set FDL word (record) in the GT readout record
const L1GtFdlWord L1GlobalTriggerReadoutRecord::gtFdlWord(unsigned int bxInEventValue) const {

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin(); 
        itBx != m_gtFdlWord.end(); ++itBx) {
        
        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            return (*itBx);
        }               
    }      
    
    // if bunch cross not found, throw exception (action: SkipEvent) 
    
    throw cms::Exception("NotFound")
        << "\nError: requested L1GtFdlWord for bx = " << bxInEventValue << " does not exist.\n"
        << std::endl;

    // return empty record - actually does not arrive here 
    return L1GtFdlWord();          
    
}

const L1GtFdlWord L1GlobalTriggerReadoutRecord::gtFdlWord() const {

    unsigned int bxInEventL1Accept = 0;
    return gtFdlWord(bxInEventL1Accept);
}

void L1GlobalTriggerReadoutRecord::setGtFdlWord(
    const L1GtFdlWord& gtFdlWordValue, unsigned int bxInEventValue) {

    // if a L1GtFdlWord exists for bxInEventValue, replace it
    for (std::vector<L1GtFdlWord>::iterator itBx = m_gtFdlWord.begin(); 
        itBx != m_gtFdlWord.end(); ++itBx) {
        
        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            *itBx = gtFdlWordValue;
            LogDebug("L1GlobalTriggerReadoutRecord") 
                << "Replacing L1GtFdlWord for bunch bx = " << bxInEventValue << "\n" 
                << std::endl;
            return;
        }               
    }
    
    // if bunch cross not found, throw exception (action: SkipEvent) 
    // all L1GtFdlWord are created in the record constructor for allowed bunch crosses
    
    throw cms::Exception("NotFound")
        << "\nError: Cannot set L1GtFdlWord for bx = " << bxInEventValue 
        << std::endl;

}

void L1GlobalTriggerReadoutRecord::setGtFdlWord(const L1GtFdlWord& gtFdlWordValue) {

    unsigned int bxInEventL1Accept = 0;
    setGtFdlWord(gtFdlWordValue, bxInEventL1Accept);
    
}


// other methods

// clear the record
void L1GlobalTriggerReadoutRecord::reset() {

    // TODO FIXME clear GTFE, FDL?
    
    // TODO FIXME reset GMT collection?
    
    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1Electrons; ++indexCand) {
        m_gtElectron[indexCand] = 0;        
    }  
    
    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons; ++indexCand) {
        m_gtIsoElectron[indexCand] = 0;        
    }  

    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1CentralJets; ++indexCand) {
        m_gtCJet[indexCand] = 0;        
    }  

    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1ForwardJets; ++indexCand) {
        m_gtFJet[indexCand] = 0;        
    }  
    
    for (unsigned int indexCand = 0; indexCand < L1GlobalTriggerReadoutSetup::NumberL1TauJets; ++indexCand) {
        m_gtTJet[indexCand] = 0;        
    }  

    m_gtMissingEt = 0;
    m_gtTotalEt = 0;    
    m_gtTotalHt = 0;
  
    CaloJetCountsWord tmpJetNr(L1GlobalTriggerReadoutSetup::NumberL1JetCounts);
    m_gtJetNr = tmpJetNr;
                  
    
} 
  
// output stream operator
std::ostream& operator<<(std::ostream& s, const L1GlobalTriggerReadoutRecord& result) {
    // TODO FIXME put together all prints
    s << "Not available yet - sorry";
        
  return s;
    
}
