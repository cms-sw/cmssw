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
 * $Date$
 * $Revision$
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
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


// constructor
L1GlobalTriggerCaloTemplate::L1GlobalTriggerCaloTemplate( 
    const L1GlobalTrigger& gt,
    const std::string& name)
    : L1GlobalTriggerConditions(gt, name) {

//    LogDebug ("Trace") 
//        << "****Entering " << __PRETTY_FUNCTION__ << " name= " << p_name << std::endl;

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
    : L1GlobalTriggerConditions(cp.m_GT, cp.p_name) {

//    m_GT = cp.m_GT; // TODO uncomment ???
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

    // maximum number of candidates 
    // TODO take it from L1GlobalTriggerReadoutRecord, depending on particle type??    
    const int maxNumberCands = 4;

   // the candidates
    L1GctCand* v[maxNumberCands];
    int index[maxNumberCands];
    
    for (int i = 0; i < maxNumberCands; ++i) {
	   v[i] = getCandidate(i);
       index[i] = i;	
	}
      
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
        LogTrace("L1GlobalTriggerCaloTemplate") 
            << "  L1GlobalTriggerCaloTemplate: no permutation match for all four calo candidates.\n" 
            << std::endl;
        return false;
    }

    //TODO: check signs and correlations for p_wsc
    
    // compute spatial correlations between all candidate pairs
    if (p_wsc) { 
        
        LogTrace("L1GlobalTriggerCaloTemplate") 
            << " spatial correlations: check " 
            << std::endl;

        unsigned int deltaeta;
        unsigned int deltaphi;  

        for (int i = 0; i < maxNumberCands; ++i) {
            for (int k = i + 1; k < maxNumberCands; ++k) {
                
                // check deltaeta

                // absolute value of eta difference (... could use abs() ) 
                if (v[i]->etaIndex() > v[k]->etaIndex()) {
                    deltaeta = v[i]->etaIndex() - v[k]->etaIndex();
                } else {
                    deltaeta = v[k]->etaIndex() - v[i]->etaIndex();
                }
                                
                LogTrace("L1GlobalTriggerCaloTemplate") 
                    << "  deltaeta = " << deltaeta << " for "  
                    << "v[" << i <<"]->etaIndex() = " << v[i]->etaIndex() << ", " 
                    << "v[" << k <<"]->etaIndex() = " << v[k]->etaIndex()  
                    << std::endl;
                
                if (!checkBit(p_conditionparameter.delta_eta, deltaeta)) {
                    return false;
                }
                
                // check deltaphi

                // absolute value of phi difference 
                if (v[i]->phiIndex() > v[k]->phiIndex()) {
                    deltaphi = v[i]->phiIndex() - v[k]->phiIndex();
                } else {
                    deltaphi = v[k]->phiIndex() - v[i]->phiIndex();
                }

                LogTrace("L1GlobalTriggerCaloTemplate") 
                    << std::dec << "  deltaphi = " << deltaphi << " for " 
                    << "v[" << i <<"]->phiIndex() = " << v[i]->phiIndex() << ", " 
                    << "v[" << k <<"]->phiIndex() = " << v[k]->phiIndex()  
                    << std::endl;

                // check if deltaphi > 180 (aka delta_phi_maxbits) 
                if (deltaphi > p_conditionparameter.delta_phi_maxbits) {
                    LogTrace("L1GlobalTriggerCaloTemplate") 
                        << "  deltaphi = " << deltaphi 
                        << " > p_conditionparameter.delta_phi_maxbits ==> needs re-scaling" 
                        << std::endl;
                        
                    // deltaphi > 180 ==> take 360 - deltaphi
                    deltaphi = (p_conditionparameter.delta_phi_maxbits - 1)*2 - deltaphi;   
                    LogTrace("L1GlobalTriggerCaloTemplate") 
                        << "  deltaphi changed to: " <<  deltaphi 
                        << std::endl;
                }
                
                // TODO FIXME review phi range and get rid of hard-coded numbers 10, 18                
                if ( deltaphi >= 10 ) {
                    LogTrace("L1GlobalTriggerCaloTemplate") 
                        << "  big range deltaphi = " << deltaphi 
                        << std::endl;

                    deltaphi = 18 - deltaphi;

                    LogTrace("L1GlobalTriggerCaloTemplate") 
                        << "  re-scaled to deltaphi = " << deltaphi 
                        << std::endl;
                }
            
                if (!checkBit(p_conditionparameter.delta_phi, deltaphi)) {
                    return false;
                }			
            }			            
        }         
    } // end if(p_wsc)
    
    return tmpResult;
}

void L1GlobalTriggerCaloTemplate::printThresholds() const {

    edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
        << "L1GlobalTriggerCaloTemplate: Threshold values " << std::endl;
    edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
        << "Condition Name: " << getName() << std::endl;

    switch (p_particletype) {
        case EG:        
            edm::LogVerbatim("L1GlobalTriggerCaloTemplate") << "Particle: " << "eg";
            break;
        case IEG:       
            edm::LogVerbatim("L1GlobalTriggerCaloTemplate") << "Particle: " << "ieg";
            break;
        case CJET:       
        case FJET:    
        case TJET:      
            edm::LogVerbatim("L1GlobalTriggerCaloTemplate") << "Particle: " << "jet";
            break;
        default:
            // write nothing - should not arrive here
            break;
    }

    edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
        << "\ngreater or equal bit: " << p_ge_eq << std::endl;

    for(unsigned int i = 0; i < p_number; i++) {

        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") << "\n  TEMPLATE " << i 
            << std::endl;
        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
            << "    et_threshold          " 
            << std::hex << p_particleparameter[i].et_threshold 
            << std::endl;
        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
            << "    eta                   " 
            <<  std::hex << p_particleparameter[i].eta 
            << std::endl;
        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
            << "    phi                   " 
            <<  std::hex << p_particleparameter[i].phi << std::endl;
    }

    if (p_wsc) {
        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
            << "    Correlation parameters:" << std::endl;
        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
            << "    delta_eta             " 
            << std::hex << p_conditionparameter.delta_eta << std::endl; 
        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
            << "    delta_eta_maxbits     " 
            << std::dec << p_conditionparameter.delta_eta_maxbits << std::endl;
        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
            << "    delta_phi             " 
            << std::hex << p_conditionparameter.delta_phi << std::endl; 
        edm::LogVerbatim("L1GlobalTriggerCaloTemplate") 
            << "    delta_phi_maxbits     " 
            << std::dec << p_conditionparameter.delta_phi_maxbits << std::endl;
    }

    //reset to decimal output
    edm::LogVerbatim("L1GlobalTriggerCaloTemplate") << std::dec << std::endl;
    
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

    switch (p_particletype) {
        case EG:        
            return (*m_GT.gtPSB()->getElectronList())[indexCand];      
            break;
        case IEG:       
            return (*m_GT.gtPSB()->getIsolatedElectronList())[indexCand];
            break;
        case CJET:       
            return (*m_GT.gtPSB()->getCentralJetList())[indexCand];
            break;
        case FJET:    
            return (*m_GT.gtPSB()->getForwardJetList())[indexCand];
            break;
        case TJET:      
        return (*m_GT.gtPSB()->getTauJetList())[indexCand];
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

    std::string checkFalse = "\n  ==> checkParticle = false ";        

    if (ncondition >= (int) p_number || ncondition < 0) {
        LogTrace("L1GlobalTriggerCaloTemplate")
            << "L1GlobalTriggerCaloTemplate:" 
            << "  Number of condition outside [0, p_number) interval." 
            << "  checkParticle = false "
            << std::endl;
        return false;
    }

    // empty candidates can not be compared
    if (cand.empty()) {
        LogTrace("L1GlobalTriggerCaloTemplate") 
            << "  Empty calo candidate (" << &cand << ")." 
            << "  checkParticle = false "
            << std::endl;
        return false;
    }

    LogTrace("L1GlobalTriggerCaloTemplate") 
        << "\n  Non-empty calorimeter object: checkParticle starting" 
        << std::endl;

    // check threshold
    if ( !checkThreshold(p_particleparameter[ncondition].et_threshold, cand.rank()) ) {
        LogTrace("L1GlobalTriggerCaloTemplate") 
            << "  calo et_threshold: failed" 
            << checkFalse 
            << std::endl;
        return false;
    }

    // check eta
    LogTrace("L1GlobalTriggerCaloTemplate") 
        << "  L1GlobalTriggerCaloTemplate: checking eta " 
        << std::endl;

    // TODO check this stuff: eta in hw, eta in CMSSW
    
    // TODO  eta = +-0 !!!

    if (cand.etaIndex() == 0) {
        LogTrace("L1GlobalTriggerCaloTemplate") << "  zero eta region !!! " << std::endl;

        LogTrace("L1GlobalTriggerCaloTemplate") 
            << "  etaIndex() = "
            << cand.etaIndex() 
            << std::endl;

        if ( !checkBit(p_particleparameter[ncondition].eta, cand.etaIndex()) ) {
            return false;
        }
    } else {

        LogTrace("L1GlobalTriggerCaloTemplate") 
            << "  etaIndex() = "
            << cand.etaIndex() 
            << std::endl;

        if (!checkBit(p_particleparameter[ncondition].eta, cand.etaIndex())) return false;
    }
    
    // check phi 
    LogTrace("L1GlobalTriggerCaloTemplate") 
        << " L1GlobalTriggerCaloTemplate: checking phi " 
        << std::endl;

    if (cand.phiIndex() == 0 ) { 
        LogTrace("L1GlobalTriggerCaloTemplate") << "  phi region zero " << std::endl;
        // TODO  phi = 0 !!!        
    }
    
    if (cand.phiIndex() > 8 ) {
        
        int phichange = 18 - cand.phiIndex();
        LogTrace("L1GlobalTriggerCaloTemplate") << "  phi region is gt 8 " << " reformatted " 
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

    // particle matches if we get here
    LogTrace("L1GlobalTriggerCaloTemplate") 
        << "  checkParticle: calorimeter object OK, passes all requirements\n" 
        << std::endl;

    return true;
}

