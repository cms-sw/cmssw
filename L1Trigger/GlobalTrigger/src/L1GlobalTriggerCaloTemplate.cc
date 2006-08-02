/**
 * \class L1GlobalTriggerCaloTemplate
 * 
 * 
 * 
 * Description: Single particle chip - description for calo conditions
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: M.Eder, H. Rohringer - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date:$
 * $Revision:$
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerCaloTemplate.h"

// system include files
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1TriggerObject.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
//#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


// constructor
L1GlobalTriggerCaloTemplate::L1GlobalTriggerCaloTemplate( 
    const std::string &name)
    : L1GlobalTriggerConditions(name) {

    LogDebug ("Trace") 
        << "****Entering " << __PRETTY_FUNCTION__ << " name= " << p_name << std::endl;

}

// copy constructor
void L1GlobalTriggerCaloTemplate::copy(const L1GlobalTriggerCaloTemplate &cp) {
    p_name = cp.getName();
    p_number = cp.getNumberParticles();
    p_wsc = cp.getWsc();
    setGeEq(cp.getGeEq());

    //TODO: macro instead of 4
    memcpy(p_particleparameter, cp.getParticleParameter(), sizeof(ParticleParameter)*4); 
    memcpy(&p_conditionparameter, cp.getConditionParameter(), sizeof(ConditionParameter));
}


L1GlobalTriggerCaloTemplate::L1GlobalTriggerCaloTemplate(
    const L1GlobalTriggerCaloTemplate& cp) 
    : L1GlobalTriggerConditions(cp.p_name) {

    copy(cp);
}

// destructor
L1GlobalTriggerCaloTemplate::~L1GlobalTriggerCaloTemplate() {

}

// equal operator
L1GlobalTriggerCaloTemplate& L1GlobalTriggerCaloTemplate::operator= (
    const L1GlobalTriggerCaloTemplate& cp) {

    copy(cp);
    return *this;
}


/**
 * setConditionParameter - set the parameters of the condition
 *
 * @param numparticles Number of particle conditions.
 * @param particlep Pointer to particle parameters.
 * @param conditionp Pointer to condition parameters.
 * @param pType Type of the particles in this condition. (eg, jet or ieg)
 * @wsc Indicates if this condition uses wsc.
 *
 */

void L1GlobalTriggerCaloTemplate::setConditionParameter(
    unsigned int numparticles, const ParticleParameter *particlep,
    const ConditionParameter* conditionp, ParticleType pType, bool wsc) {
    
    p_number = numparticles;
    if (p_number > 4) p_number = 4; // TODO: macro instead of 4
    p_wsc = wsc;
    p_particletype = pType;
   
    memcpy(p_particleparameter, particlep, sizeof(ParticleParameter)*p_number);
    memcpy(&p_conditionparameter, conditionp, sizeof(ConditionParameter));
}

/**
 * blockCondition - Try all permutations of particles and check correlation
 *
 * @return boolean result of the check
 *
 */

const bool L1GlobalTriggerCaloTemplate::blockCondition() const {

   // the candidates
    L1GctCand (*v[4]) = { (getCandidate(0)),
                             (getCandidate(1)),
                             (getCandidate(2)),
                             (getCandidate(3)) };

    // indices
    int index[4] = {0, 1, 2, 3};
  
    bool tmpResult;
  
    // first check if there is a permutation that matches
    do {
        tmpResult = true;
        for (int i = 0; i < (int) p_number; i++) {
            tmpResult &= checkParticle(i, *(v)[index[i]] );
        }
    
        if (tmpResult) break; 
    } while (std::next_permutation(index, index + p_number) );

    if (tmpResult == false) {
        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
            << " CALO checking for 4 failed total " 
            << std::endl;
        return false;
    }

    unsigned int deltaeta;
    unsigned int deltaphi;  
  
    //TODO: signs and correlations!!!! this code is wrong!

    if (p_wsc) { 
        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
            << " wsc check " 
            << std::endl;
        
        if (v[0]->etaIndex() > v[1]->etaIndex()) {
             deltaeta = v[0]->etaIndex() - v[1]->etaIndex();
        } else {
            deltaeta = v[1]->etaIndex() - v[0]->etaIndex();
        }
        
        if (v[0]->phiIndex() > v[1]->phiIndex()) {
            deltaphi = v[0]->phiIndex() - v[1]->phiIndex();
        } else {
            deltaphi = v[1]->phiIndex() - v[0]->phiIndex();
        }
    
        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
            << " deltaeta " << deltaeta << " "  
            << v[0]-> etaIndex() << " " 
            << v[1]-> etaIndex() 
            << std::endl;
            
        if (!checkBit(p_conditionparameter.delta_eta, deltaeta)) {
            return false;
        }
    
        //deltaphi needs to be checked if its >180
        if (deltaphi > p_conditionparameter.delta_phi_maxbits) {
            edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
                << "  phi 180 " << deltaphi 
                << " p_conditionparameter.delta_phi_maxbits " 
                << std::endl;
            //if deltaphi > 180 take 360 - deltaphi
            deltaphi = (p_conditionparameter.delta_phi_maxbits-1)*2 - deltaphi;   
            edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
                << " deltaphi change " <<  deltaphi 
                << std::endl;
        }
     
    
        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
            <<  std::dec << " deltaphi " << deltaphi << " " 
            << v[0]->phiIndex() << " " 
            << v[1]->phiIndex() 
            << std::endl;

        if ( deltaphi >= 10 ) {
            edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
                << " deltaphi big range " << deltaphi 
                << std::endl;
            deltaphi = 18 - deltaphi;
        }
    
        if (!checkBit(p_conditionparameter.delta_phi, deltaphi)) {
            return false;
        }
    }
    
    return tmpResult;
}

void L1GlobalTriggerCaloTemplate::printThresholds() const {

    std::cout << "L1GlobalTriggerCaloTemplate: Threshold values " << std::endl;
    std::cout << "Condition Name: " << getName() << std::endl;

    std::cout << "Particle: ";
    switch (p_particletype) {
        case EG:        
            std::cout << "eg";
            break;
        case IEG:       
            std::cout << "ieg";
            break;
        case JET:       
        case FWDJET:    
        case TAU:      
            std::cout << "jet";
            break;
        default:
            // write nothing - should not arrive here
            break;
    }
    std::cout << std::endl;

    std::cout << "Greater equal bit: " << p_ge_eq << std::endl;

    for(unsigned int i = 0; i < p_number; i++) {

        std::cout << "\nTEMPLATE " << i << std::endl;
        std::cout << "et_threshold          " 
            <<  std::hex << p_particleparameter[i].et_threshold 
            << std::endl;
        std::cout << "eta                   " 
            <<  std::hex << p_particleparameter[i].eta 
            << std::endl;
        std::cout << "phi                   " 
        <<  std::hex << p_particleparameter[i].phi << std::endl;
    }

    if (p_wsc) {
        std::cout << "///CORRELATION PARAMETERS" << std::endl;
        std::cout << "delta_eta             " << 
            std::hex << p_conditionparameter.delta_eta << std::endl; 
        std::cout << "delta_eta_maxbits     " 
            << std::dec << p_conditionparameter.delta_eta_maxbits << std::endl;
        std::cout << "delta_phi 	           " 
            << std::hex << p_conditionparameter.delta_phi << std::endl; 
        std::cout << "delta_phi_maxbits     " 
            << std::dec << p_conditionparameter.delta_phi_maxbits << std::endl;
    }

    //reset to decimal output
    std::cout << std::dec << std::endl;
    
}

/** 
 * getCandidate - decides what candidate to get using the particletype
 *
 * 
 * @param indexCand The number of the candidate
 * @return A reference to the candidate
 *
 */

L1GctCand* L1GlobalTriggerCaloTemplate::getCandidate (int indexCand) const {

// TODO uncomment
    switch (p_particletype) {
        case EG:        
//            return (*PSB->getElectronList())[indexCand];      
            return 0;
            break;
        case IEG:       
//            return (*PSB->getIsolElectronList())[indexCand];
            return 0;
            break;
        case JET:       
//            return (*PSB->getCentralJetList())[indexCand];
            return 0;
            break;
        case FWDJET:    
//            return (*PSB->getForwardJetList())[indexCand];
            return 0;
            break;
        case TAU:      
//        return (*PSB->getTauJetList())[indexCand];
            return 0;
            break;
        default:
            return 0;
            break;
    }
}

/**
 * checkParticle - Compare a single particle with a numbered condition.
 *
 * @param ncondition The number of the condition.
 * @param cand The candidate to compare.
 *
 * @return The result of the comparison (false if a condition does not exist).
 */

const bool L1GlobalTriggerCaloTemplate::checkParticle(
    int ncondition, L1GctCand &cand) const {

    if (ncondition >= (int) p_number || ncondition < 0) return false;

    // check threshold
    int ADHOC_SCALING_FACTOR = 5; // TODO why one needs such a hack -> look where the problem is!!!
    if (!checkThreshold(p_particleparameter[ncondition].et_threshold, cand.rank()/ADHOC_SCALING_FACTOR)) {
        return false;
    }
    
    // check eta
    edm::LogVerbatim("L1GlobalTriggerCaloTemplate") << " CALO ETA CHECK " << std::endl;
    // TODO  eta = 0 !!!

    // NENTCHEV UNITS 
    
    int etanentch = 0;
    if (cand.etaIndex() == 0) {
        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") << " zero eta region !!! " << std::endl;

        etanentch = etaconvvtt(cand.etaIndex());
        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") << "eta conversion " 
            << cand.etaIndex() << " " 
            << etanentch << " " 
            << std::endl;

        if (!checkBit(p_particleparameter[ncondition].eta, etanentch)) {
            return false;
        }
    }
    
    if (cand.etaIndex() != 0 ) {

        etanentch = etaconvvtt(cand.etaIndex());
        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") << "eta conversion " 
            << cand.etaIndex() << " " 
            << etanentch << " " 
            << std::endl;

        if (!checkBit(p_particleparameter[ncondition].eta, etanentch)) return false;
    }
    
    // check phi 
    edm::LogVerbatim("L1GlobalTriggerCaloTemplate") << " CALO PHI CHECK " << std::endl;
    // TODO  phi = 0 !!!
    if (cand.phiIndex() == 0 ) { 
        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") << " phi region zero " << std::endl;
        
        // ???? until solved, go to <= case
    }
    
    if (cand.phiIndex() > 8 ) {
        
        int phichange = 18 - cand.phiIndex();
        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") << " phi region is gt 8 " << " reformatted " 
            << phichange << std::endl;
        if (!checkBit(p_particleparameter[ncondition].phi, phichange )) {
            return false;
        }
    }
     
    if (cand.phiIndex() <= 8 ) {
        if (!checkBit(p_particleparameter[ncondition].phi, cand.phiIndex())) {
            return false;
        }
    }

    return true;
}

// eta conversions
// TODO sort out this stuff

//  convert ORCA input to FIERRO/TAUROCK schema  for etadifferences
const int L1GlobalTriggerCaloTemplate::etaconv(int abs_eta) const {

    int etac = -300;
    if ( abs_eta == 0)  etac    =  8;
    if ( abs_eta == 1)  etac    =  6;
    if ( abs_eta == 2)  etac    =  5;
    if ( abs_eta == 3)  etac    =  4;
    if ( abs_eta == 4)  etac    =  3;
    if ( abs_eta == 5)  etac    =  2;
    if ( abs_eta == 6)  etac    =  1;

    // eta value negative
    if ( abs_eta == 7)  etac    =  1;
    if ( abs_eta == 8)  etac    =  2;
    if ( abs_eta == 9)  etac    =  3;
    if ( abs_eta == 10) etac    =  4;
    if ( abs_eta == 11) etac    =  5;
    if ( abs_eta == 12) etac    =  6;
    if ( abs_eta == 13) etac    =  8;
    if ( abs_eta == 14) etac    =  10;
    if ( abs_eta == 15) etac    =  11;

    if ( abs_eta == 16) etac    =  13;

    return etac;

}

//  convert ORCA input to FIERRO/TAUROCK schema  for etadifferences
const int L1GlobalTriggerCaloTemplate::etaconvwind(int abs_eta) const {
    
    int abswind = -300;
    
    if ( abs_eta == 0)  abswind = 12;
    if ( abs_eta == 1)  abswind = 10;
    if ( abs_eta == 2)  abswind =  9;
    if ( abs_eta == 3)  abswind =  8;
    if ( abs_eta == 4)  abswind =  7;
    if ( abs_eta == 5)  abswind =  6;
    if ( abs_eta == 6)  abswind =  5;
    if ( abs_eta == 7)  abswind =  5;
    if ( abs_eta == 8)  abswind =  6;
    if ( abs_eta == 9)  abswind =  7;
    if ( abs_eta == 10) abswind =  8;
    if ( abs_eta == 11) abswind =  9;
    if ( abs_eta == 12) abswind = 10;
    if ( abs_eta == 13) abswind = 12;

    return abswind;
}

//  convert ORCA input to TAUROCK schema  for eta
const  int L1GlobalTriggerCaloTemplate::etaconvv(int abs_eta) const {

    int eta_taur = -300;
    // eta_taur in agrred units -7 < etahardware < + 7
    // abs_eta = eta - 4 von Hidas PAL conversion

    if ( abs_eta == 0)  eta_taur= -6;
    if ( abs_eta == 1)  eta_taur= -5;
    if ( abs_eta == 2)  eta_taur= -4;
    if ( abs_eta == 3)  eta_taur= -3;
    if ( abs_eta == 4)  eta_taur= -2;
    if ( abs_eta == 5)  eta_taur= -1;
    if ( abs_eta == 6)  eta_taur=  -0;
    if ( abs_eta == 7)  eta_taur=  +0;
    if ( abs_eta == 8)  eta_taur=  1;
    if ( abs_eta == 9)  eta_taur=  2;
    if ( abs_eta == 10) eta_taur=  3;
    if ( abs_eta == 11) eta_taur=  4;
    if ( abs_eta == 12) eta_taur=  5;
    if ( abs_eta == 13) eta_taur=  6;

    return eta_taur;
}

//  convert ORCA input to TAUROCK schema  for eta
const int L1GlobalTriggerCaloTemplate::etaconvvtt(int abs_eta) const { 

    int eta_taus = -300;

    if ( abs_eta == 0)  eta_taus= 14;
    if ( abs_eta == 1)  eta_taus= 13;
    if ( abs_eta == 2)  eta_taus= 12;
    if ( abs_eta == 3)  eta_taus= 11;
    if ( abs_eta == 4)  eta_taus= 10;
    if ( abs_eta == 5)  eta_taus=  9;
    if ( abs_eta == 6)  eta_taus=  8;
    if ( abs_eta == 7)  eta_taus=  7;
    if ( abs_eta == 8)  eta_taus=  6;
    if ( abs_eta == 9)  eta_taus=  5;
    if ( abs_eta == 10) eta_taus=  4;
    if ( abs_eta == 11) eta_taus=  3;
    if ( abs_eta == 12) eta_taus=  2;
    if ( abs_eta == 13) eta_taus=  1;

    return eta_taus;
}
