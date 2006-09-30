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

// user include files
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructors
L1GlobalTriggerReadoutRecord::L1GlobalTriggerReadoutRecord() 
    : m_gtBxId(0), m_bxInEvent(0), m_gtGlobalDecision(false) {

    // decision word  std::vector<bool>      
    m_gtDecision.reserve(NumberPhysTriggers);
    m_gtDecision.assign(NumberPhysTriggers, false);

    // technical triggers
    m_gtTechnicalTrigger.reserve(NumberTechnicalTriggers);
    m_gtTechnicalTrigger.assign(NumberTechnicalTriggers, false);
    
    
    for (unsigned int indexCand = 0; indexCand < NumberL1Muons; ++indexCand) {
        m_gtMuon[indexCand] = 0;		
	}  

    for (unsigned int indexCand = 0; indexCand < NumberL1Electrons; ++indexCand) {
        m_gtElectron[indexCand] = 0;        
    }  
    
    for (unsigned int indexCand = 0; indexCand < NumberL1IsolatedElectrons; ++indexCand) {
        m_gtIsoElectron[indexCand] = 0;        
    }  

    for (unsigned int indexCand = 0; indexCand < NumberL1CentralJets; ++indexCand) {
        m_gtCJet[indexCand] = 0;        
    }  

    for (unsigned int indexCand = 0; indexCand < NumberL1ForwardJets; ++indexCand) {
        m_gtFJet[indexCand] = 0;        
    }  
    
    for (unsigned int indexCand = 0; indexCand < NumberL1TauJets; ++indexCand) {
        m_gtTJet[indexCand] = 0;        
    }  

    m_gtMissingEt = 0;
    m_gtTotalEt = 0;    
    m_gtTotalHt = 0;
  
    m_gtJetNr.reserve(NumberL1JetCounts);
    m_gtJetNr.assign(NumberL1JetCounts, 0);
                  
}

// copy constructor
L1GlobalTriggerReadoutRecord::L1GlobalTriggerReadoutRecord(
    const L1GlobalTriggerReadoutRecord& result) {

    m_gtBxId    = result.m_gtBxId;
    m_bxInEvent = result.m_bxInEvent;
        
    m_gtDecision       = result.m_gtDecision;
    m_gtGlobalDecision = result.m_gtGlobalDecision;

    m_gtTechnicalTrigger = result.m_gtTechnicalTrigger;

    for (unsigned int indexCand = 0; indexCand < NumberL1Muons; ++indexCand) {  
        m_gtMuon[indexCand] = result.m_gtMuon[indexCand];
    }
    
    for (unsigned int indexCand = 0; indexCand < NumberL1Electrons; ++indexCand) {
        m_gtElectron[indexCand] = result.m_gtElectron[indexCand];  
    }
    
    for (unsigned int indexCand = 0; indexCand < NumberL1IsolatedElectrons; ++indexCand) {
        m_gtIsoElectron[indexCand] = result.m_gtIsoElectron[indexCand];
    }
      
    for (unsigned int indexCand = 0; indexCand < NumberL1CentralJets; ++indexCand) {
        m_gtCJet[indexCand] = result.m_gtCJet[indexCand];  
    }
    
    for (unsigned int indexCand = 0; indexCand < NumberL1ForwardJets; ++indexCand) {
        m_gtFJet[indexCand] = result.m_gtFJet[indexCand];  
    }
    
    for (unsigned int indexCand = 0; indexCand < NumberL1TauJets; ++indexCand) {
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
        m_gtBxId    = result.m_gtBxId;
        m_bxInEvent = result.m_bxInEvent;
        
        m_gtDecision       = result.m_gtDecision;
        m_gtGlobalDecision = result.m_gtGlobalDecision;

        m_gtTechnicalTrigger = result.m_gtTechnicalTrigger;
        
        for (unsigned int indexCand = 0; indexCand < NumberL1Muons; ++indexCand) {  
            m_gtMuon[indexCand] = result.m_gtMuon[indexCand];
        }
    
        for (unsigned int indexCand = 0; indexCand < NumberL1Electrons; ++indexCand) {
            m_gtElectron[indexCand] = result.m_gtElectron[indexCand];  
        }
    
        for (unsigned int indexCand = 0; indexCand < NumberL1IsolatedElectrons; ++indexCand) {
            m_gtIsoElectron[indexCand] = result.m_gtIsoElectron[indexCand];
        }
      
        for (unsigned int indexCand = 0; indexCand < NumberL1CentralJets; ++indexCand) {
            m_gtCJet[indexCand] = result.m_gtCJet[indexCand];  
        }
    
        for (unsigned int indexCand = 0; indexCand < NumberL1ForwardJets; ++indexCand) {
            m_gtFJet[indexCand] = result.m_gtFJet[indexCand];  
        }
    
        for (unsigned int indexCand = 0; indexCand < NumberL1TauJets; ++indexCand) {
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

    if (m_gtBxId    != result.m_gtBxId) return false;
    if (m_bxInEvent != result.m_bxInEvent) return false;
        
    if (m_gtDecision       != result.m_gtDecision) return false;
    if (m_gtGlobalDecision != result.m_gtGlobalDecision) return false;  

    if (m_gtTechnicalTrigger != result.m_gtTechnicalTrigger) return false;

    if (m_gtMuon != result.m_gtMuon) return false;
  
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

// get/set physical candidates 

// return muon candidate(s)

const L1MuGMTCand L1GlobalTriggerReadoutRecord::muonCand(
    unsigned int indexCand, unsigned int bxInEvent) const {
        
    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < NumberL1Muons );
    return L1MuGMTCand( m_gtMuon[indexCand] );
    
}

const L1MuGMTCand L1GlobalTriggerReadoutRecord::muonCand(
    unsigned int indexCand) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < NumberL1Muons );
    return L1MuGMTCand( m_gtMuon[indexCand] );

}        

std::vector<L1MuGMTCand> L1GlobalTriggerReadoutRecord::muonCands(
    unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    vector<L1MuGMTCand> result;

    for ( unsigned int i = 0; i != NumberL1Muons; i++ ) {
        L1MuGMTCand cand(muonCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;

}

std::vector<L1MuGMTCand> L1GlobalTriggerReadoutRecord::muonCands() const {

    // TODO bxInEvent dependence
    vector<L1MuGMTCand> result;

    for ( unsigned int i = 0; i != NumberL1Muons; i++ ) {
        L1MuGMTCand cand(muonCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;

}

// electron

const L1GctEmCand L1GlobalTriggerReadoutRecord::electronCand(
    unsigned int indexCand, unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < NumberL1Electrons );

    bool electronIso = false;
    return L1GctEmCand( m_gtElectron[indexCand], electronIso );

}

const L1GctEmCand L1GlobalTriggerReadoutRecord::electronCand(
    unsigned int indexCand) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < NumberL1Electrons );

    bool electronIso = false;
    return L1GctEmCand( m_gtElectron[indexCand], electronIso );

}

std::vector<L1GctEmCand> L1GlobalTriggerReadoutRecord::electronCands(
    unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    vector<L1GctEmCand> result;

    for ( unsigned int  i = 0; i != NumberL1Electrons; i++ ) {
        L1GctEmCand cand(electronCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;

}

std::vector<L1GctEmCand> L1GlobalTriggerReadoutRecord::electronCands() const {

    // TODO bxInEvent dependence
    vector<L1GctEmCand> result;

    for ( unsigned int  i = 0; i != NumberL1Electrons; i++ ) {
        L1GctEmCand cand(electronCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;
    
}

// isolated electron
const L1GctEmCand L1GlobalTriggerReadoutRecord::isolatedElectronCand(
    unsigned int indexCand, unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < NumberL1IsolatedElectrons );

    bool electronIso = true;
    return L1GctEmCand( m_gtIsoElectron[indexCand], electronIso );
    
}
const L1GctEmCand L1GlobalTriggerReadoutRecord::isolatedElectronCand(
    unsigned int indexCand) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < NumberL1IsolatedElectrons );

    bool electronIso = true;
    return L1GctEmCand( m_gtIsoElectron[indexCand], electronIso );

}

std::vector<L1GctEmCand> L1GlobalTriggerReadoutRecord::isolatedElectronCands(
    unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    vector<L1GctEmCand> result;

    for ( unsigned int  i = 0; i != NumberL1IsolatedElectrons; i++ ) {
        L1GctEmCand cand(isolatedElectronCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;

}

std::vector<L1GctEmCand> L1GlobalTriggerReadoutRecord::isolatedElectronCands() const {

    // TODO bxInEvent dependence
    vector<L1GctEmCand> result;

    for ( unsigned int  i = 0; i != NumberL1IsolatedElectrons; i++ ) {
        L1GctEmCand cand(isolatedElectronCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;
    
}

// central jet
const L1GctJetCand L1GlobalTriggerReadoutRecord::centralJetCand(
    unsigned int indexCand, unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < NumberL1CentralJets );

    bool isTau = false;
    bool isFor = false;
    return L1GctJetCand( m_gtCJet[indexCand], isTau, isFor );
    
}

const L1GctJetCand L1GlobalTriggerReadoutRecord::centralJetCand(
    unsigned int indexCand) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < NumberL1CentralJets );

    bool isTau = false;
    bool isFor = false;
    return L1GctJetCand( m_gtCJet[indexCand], isTau, isFor );
        
}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::centralJetCands(
    unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    vector<L1GctJetCand> result;

    for ( unsigned int  i = 0; i != NumberL1CentralJets; i++ ) {
        L1GctJetCand cand(centralJetCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;

}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::centralJetCands() const {

    // TODO bxInEvent dependence
    vector<L1GctJetCand> result;

    for ( unsigned int  i = 0; i != NumberL1CentralJets; i++ ) {
        L1GctJetCand cand(centralJetCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;

}

// forward jet
const L1GctJetCand L1GlobalTriggerReadoutRecord::forwardJetCand(
    unsigned int indexCand, unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < NumberL1ForwardJets );

    bool isTau = false;
    bool isFor = true;
    return L1GctJetCand( m_gtFJet[indexCand], isTau, isFor );
        
}

const L1GctJetCand L1GlobalTriggerReadoutRecord::forwardJetCand(
    unsigned int indexCand) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < NumberL1ForwardJets );

    bool isTau = false;
    bool isFor = true;
    return L1GctJetCand( m_gtFJet[indexCand], isTau, isFor );

}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::forwardJetCands(
    unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    vector<L1GctJetCand> result;

    for ( unsigned int  i = 0; i != NumberL1ForwardJets; i++ ) {
        L1GctJetCand cand(forwardJetCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;

}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::forwardJetCands() const {

    // TODO bxInEvent dependence
    vector<L1GctJetCand> result;

    for ( unsigned int  i = 0; i != NumberL1ForwardJets; i++ ) {
        L1GctJetCand cand(forwardJetCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;
    
}

// tau jet
const L1GctJetCand L1GlobalTriggerReadoutRecord::tauJetCand(
    unsigned int indexCand, unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < NumberL1TauJets );

    bool isTau = true;
    bool isFor = false;
    return L1GctJetCand( m_gtTJet[indexCand], isTau, isFor );
        
}

const L1GctJetCand L1GlobalTriggerReadoutRecord::tauJetCand(
    unsigned int indexCand) const {

    // TODO bxInEvent dependence
    assert( indexCand >= 0 && indexCand < NumberL1TauJets );

    bool isTau = true;
    bool isFor = false;
    return L1GctJetCand( m_gtTJet[indexCand], isTau, isFor );

}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::tauJetCands(
    unsigned int bxInEvent) const {

    // TODO bxInEvent dependence
    vector<L1GctJetCand> result;

    for ( unsigned int  i = 0; i != NumberL1TauJets; i++ ) {
        L1GctJetCand cand(tauJetCand(i));
        if ( !cand.empty() ) result.push_back(cand);
    }

    return result;
    
}

std::vector<L1GctJetCand> L1GlobalTriggerReadoutRecord::tauJetCands() const {

    // TODO bxInEvent dependence
    vector<L1GctJetCand> result;

    for ( unsigned int  i = 0; i != NumberL1TauJets; i++ ) {
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
    int max = ( vec.size() <= NumberL1Muons ) ? vec.size() : NumberL1Muons;
    for (int i = 0; i < max; i++) {
        m_gtMuon[i] = vec[i];
    }

}

void L1GlobalTriggerReadoutRecord::setMuons(
    const std::vector<MuonDataWord>& vec) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= NumberL1Muons ) ? vec.size() : NumberL1Muons;
    for (int i = 0; i < max; i++) {
        m_gtMuon[i] = vec[i];
    }
        
}

// electron
void L1GlobalTriggerReadoutRecord::setElectrons(
    const std::vector<CaloDataWord>& vec, unsigned int bxInEvent) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= NumberL1Electrons ) ? vec.size() : NumberL1Electrons;
    for (int i = 0; i < max; i++) {
        m_gtElectron[i] = vec[i];
    }
        
}

void L1GlobalTriggerReadoutRecord::setElectrons(
    const std::vector<CaloDataWord>& vec) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= NumberL1Electrons ) ? vec.size() : NumberL1Electrons;
    for (int i = 0; i < max; i++) {
        m_gtElectron[i] = vec[i];
    }
        
}
  
// isolated electron
void L1GlobalTriggerReadoutRecord::setIsolatedElectrons(
    const std::vector<CaloDataWord>& vec, unsigned int bxInEvent) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= NumberL1IsolatedElectrons ) ? vec.size() : NumberL1IsolatedElectrons;
    for (int i = 0; i < max; i++) {
        m_gtIsoElectron[i] = vec[i];
    }

} 

void L1GlobalTriggerReadoutRecord::setIsolatedElectrons(
    const std::vector<CaloDataWord>& vec) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= NumberL1IsolatedElectrons ) ? vec.size() : NumberL1IsolatedElectrons;
    for (int i = 0; i < max; i++) {
        m_gtIsoElectron[i] = vec[i];
    }
    
}
  
// central jets
void L1GlobalTriggerReadoutRecord::setCentralJets(
    const std::vector<CaloDataWord>& vec, unsigned int bxInEvent) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= NumberL1CentralJets ) ? vec.size() : NumberL1CentralJets;
    for (int i = 0; i < max; i++) {
        m_gtCJet[i] = vec[i];
    }
    
}

void L1GlobalTriggerReadoutRecord::setCentralJets(
    const std::vector<CaloDataWord>& vec) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= NumberL1CentralJets ) ? vec.size() : NumberL1CentralJets;
    for (int i = 0; i < max; i++) {
        m_gtCJet[i] = vec[i];
    }

    
}
  
// forward jets
void L1GlobalTriggerReadoutRecord::setForwardJets(
    const std::vector<CaloDataWord>& vec, unsigned int bxInEvent) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= NumberL1ForwardJets ) ? vec.size() : NumberL1ForwardJets;
    for (int i = 0; i < max; i++) {
        m_gtFJet[i] = vec[i];
    }
    
}
void L1GlobalTriggerReadoutRecord::setForwardJets(
    const std::vector<CaloDataWord>& vec) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= NumberL1ForwardJets ) ? vec.size() : NumberL1ForwardJets;
    for (int i = 0; i < max; i++) {
        m_gtFJet[i] = vec[i];
    }
        
        
}
  
// tau jets
void L1GlobalTriggerReadoutRecord::setTauJets(
    const std::vector<CaloDataWord>& vec, unsigned int bxInEvent) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= NumberL1TauJets ) ? vec.size() : NumberL1TauJets;
    for (int i = 0; i < max; i++) {
        m_gtTJet[i] = vec[i];
    }
    
}

void L1GlobalTriggerReadoutRecord::setTauJets(
    const std::vector<CaloDataWord>& vec) {

    // TODO bxInEvent dependence
    int max = ( vec.size() <= NumberL1TauJets ) ? vec.size() : NumberL1TauJets;
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

// print global decision and algorithm decision word
void L1GlobalTriggerReadoutRecord::print() const {

    std::cout << "\nL1 Global Trigger Record : " << std::endl
        << "\t Global Decision = " << std::setw(5) << m_gtGlobalDecision << std::endl 
        << "\t Decision word (bitset style) = "; 

    for (std::vector<bool>::const_reverse_iterator ritBit = m_gtDecision.rbegin(); 
        ritBit != m_gtDecision.rend(); ++ritBit) {
        
        std::cout << (*ritBit ? '1' : '0');
                
    }      

    std::cout << std::endl;
    
}

// print technical trigger word (reverse order for vector<bool>)
void L1GlobalTriggerReadoutRecord::printTechnicalTrigger() const {

    std::cout << "\nL1 Global Trigger Record : " << std::endl
        << "\t Technical Trigger word (bitset style) = "; 

    for (std::vector<bool>::const_reverse_iterator ritBit = m_gtTechnicalTrigger.rbegin(); 
        ritBit != m_gtTechnicalTrigger.rend(); ++ritBit) {
        
        std::cout << (*ritBit ? '1' : '0');
                
    }      

    std::cout << std::endl;
    
}



// print all L1 Trigger Objects (use int to bitset conversion) 
void L1GlobalTriggerReadoutRecord::printL1Objects() const {

    std::cout << "\nL1GlobalTriggerReadoutRecord: L1 Trigger Objects \n" << std::endl;
     
    std::cout << "   GMT Muons " << std::endl;
    for (unsigned int i = 0; i < NumberL1Muons; i++) { 
        std::cout << std::bitset<NumberMuonBits>(m_gtMuon[i]) << std::endl;
    }
    
    std::cout << "   GCT Non Isolated Electrons " << std::endl;
    for (unsigned int i = 0; i < NumberL1Electrons; i++) { 
        std::cout << std::bitset<NumberCaloBits>(m_gtElectron[i]) << std::endl;
    }
    
    std::cout << "   GCT Isolated Electrons " << std::endl;
    for ( unsigned int  i = 0; i < NumberL1IsolatedElectrons; i++) {
        std::cout << std::bitset<NumberCaloBits>(m_gtIsoElectron[i]) << std::endl;
    }
    
    std::cout << "   GCT Central Jets " << std::endl;
    for ( unsigned int  i = 0; i < NumberL1CentralJets; i++) {
        std::cout << std::bitset<NumberCaloBits>(m_gtCJet[i]) << std::endl;
    }
    
    std::cout << "   GCT Forward Jets " << std::endl;
    for ( unsigned int  i = 0; i < NumberL1ForwardJets; i++) {
        std::cout << std::bitset<NumberCaloBits>(m_gtFJet[i]) << std::endl;
    }
    
    std::cout << "   GCT Tau Jets " << std::endl;
    for ( unsigned int  i = 0; i < NumberL1TauJets; i++) {
        std::cout << std::bitset<NumberCaloBits>(m_gtTJet[i]) << std::endl;
    }
    
    std::cout << "   GCT Missing Transverse Energy " << std::endl;
    std::cout << std::bitset<NumberMissingEtBits>(m_gtMissingEt) << std::endl;

    std::cout << "   GCT Total Transverse Energy " << std::endl;
    std::cout << std::bitset<NumberCaloBits>(m_gtTotalEt) << std::endl;

    std::cout << "   GCT Total Hadron Transverse Energy " << std::endl;
    std::cout << std::bitset<NumberCaloBits>(m_gtTotalHt) << std::endl;

    std::cout << "   GCT Jet Counts " << std::endl;
    std::cout << "To be done" << std::endl; // TODO fix the way to print jet counts
        
    std::cout << std::endl;
        
}


// get/set hardware-related words

// get/set DAQ and EVM words. Beware setting the words: they are correlated!

// get / set DAQ readout record
const L1GlobalTriggerReadoutRecord::L1GlobalTriggerDaqWord 
    L1GlobalTriggerReadoutRecord::daqWord() const {

    // TODO edit it when defined
    L1GlobalTriggerDaqWord voidValue;
    return voidValue;
}

void L1GlobalTriggerReadoutRecord::setDaqWord(const L1GlobalTriggerDaqWord& daqWordValue) {

    // TODO edit it when defined
    
}

// get / set EVM readout record
const L1GlobalTriggerReadoutRecord::L1GlobalTriggerEvmWord 
    L1GlobalTriggerReadoutRecord::evmWord() const {

    // TODO edit it when defined
    L1GlobalTriggerEvmWord voidValue;
    return voidValue;

}

void L1GlobalTriggerReadoutRecord::setEvmWord(const L1GlobalTriggerEvmWord& evmWordValue) {

    // TODO edit it when defined
        
}

// TODO: type for boardId; temporary unsigned int 

// get word for PSB board 
const L1GlobalTriggerReadoutRecord::L1GlobalTriggerPsbWord 
    L1GlobalTriggerReadoutRecord::psbWord(
    unsigned int boardId, unsigned int bxInEvent) const {

    // TODO edit it when defined
    L1GlobalTriggerPsbWord voidValue;
    return voidValue;
    
}

const L1GlobalTriggerReadoutRecord::L1GlobalTriggerPsbWord 
    L1GlobalTriggerReadoutRecord::psbWord(
    unsigned int boardId) const {

    // TODO edit it when defined
    L1GlobalTriggerPsbWord voidValue;
    return voidValue;
    
}

// set word for PSB board:
void L1GlobalTriggerReadoutRecord::setPsbWord(
    const L1GlobalTriggerPsbWord& psbWordValue, unsigned int bxInEvent) {

    // TODO edit it when defined
        
}

void L1GlobalTriggerReadoutRecord::setPsbWord(
    const L1GlobalTriggerPsbWord& psbWordValue) {

    // TODO edit it when defined

}

// other methods

// clear the record
void L1GlobalTriggerReadoutRecord::reset() {

    m_gtBxId = 0;
    m_bxInEvent = 0;
        
    m_gtGlobalDecision = false;

    for (unsigned int iBit = 0; iBit < m_gtDecision.size(); ++iBit) {
		m_gtDecision[iBit] = false;
	}

    for (unsigned int iBit = 0; iBit < m_gtTechnicalTrigger.size(); ++iBit) {
        m_gtTechnicalTrigger[iBit] = false;
    }

    for (unsigned int indexCand = 0; indexCand < NumberL1Muons; ++indexCand) {
        m_gtMuon[indexCand] = 0;        
    }  

    for (unsigned int indexCand = 0; indexCand < NumberL1Electrons; ++indexCand) {
        m_gtElectron[indexCand] = 0;        
    }  
    
    for (unsigned int indexCand = 0; indexCand < NumberL1IsolatedElectrons; ++indexCand) {
        m_gtIsoElectron[indexCand] = 0;        
    }  

    for (unsigned int indexCand = 0; indexCand < NumberL1CentralJets; ++indexCand) {
        m_gtCJet[indexCand] = 0;        
    }  

    for (unsigned int indexCand = 0; indexCand < NumberL1ForwardJets; ++indexCand) {
        m_gtFJet[indexCand] = 0;        
    }  
    
    for (unsigned int indexCand = 0; indexCand < NumberL1TauJets; ++indexCand) {
        m_gtTJet[indexCand] = 0;        
    }  

    m_gtMissingEt = 0;
    m_gtTotalEt = 0;    
    m_gtTotalHt = 0;
  
    CaloJetCountsWord tmpJetNr(NumberL1JetCounts);
    m_gtJetNr = tmpJetNr;
                  
    
} 
  
// output stream operator
std::ostream& operator<<(std::ostream& s, const L1GlobalTriggerReadoutRecord& result) {
    s << "Global Decision = " << std::setw(5) << result.decision() << std::endl
      << "Decision = ";
          
    for (std::vector<bool>::const_iterator itBit = result.m_gtDecision.begin(); 
        itBit != result.m_gtDecision.end(); ++itBit) {
        
        s << (*itBit ? '1' : '0');
        		
	}      
    
  return s;
    
}
