/**
 * \class L1GlobalTriggerMuonTemplate
 * 
 * 
 * 
 * Description: Implementation of L1GlobalTriggerMuonTemplate
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
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerMuonTemplate.h"

// system include files
#include <algorithm>
#include <string>

#include <iostream>
#include <iomanip>

// user include files
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
 
// constructor
L1GlobalTriggerMuonTemplate::L1GlobalTriggerMuonTemplate(
    const L1GlobalTrigger& gt,
    const std::string& name) 
    : L1GlobalTriggerConditions(gt, name) {

//    LogDebug ("Trace") << "Entering " << __PRETTY_FUNCTION__ 
//        << " name= " << p_name << std::endl;    

}

// copy constructor
void L1GlobalTriggerMuonTemplate::copy(const L1GlobalTriggerMuonTemplate &cp) {
    p_name = cp.getName();
    p_number = cp.getNumberParticles();
    p_wsc = cp.getWsc();
    setGeEq(cp.getGeEq());

    // TODO: macro instead of 4
    memcpy(p_particleparameter, cp.getParticleParameter(), sizeof(ParticleParameter)*4); 
    memcpy(&p_conditionparameter, cp.getConditionParameter(), sizeof(ConditionParameter));
  
}


L1GlobalTriggerMuonTemplate::L1GlobalTriggerMuonTemplate(const L1GlobalTriggerMuonTemplate& cp)
    : L1GlobalTriggerConditions(cp.m_GT, cp.p_name) 
    {
        
    copy(cp);
}

// destructor
L1GlobalTriggerMuonTemplate::~L1GlobalTriggerMuonTemplate() {

}

// equal operator
L1GlobalTriggerMuonTemplate& 
    L1GlobalTriggerMuonTemplate::operator= (const L1GlobalTriggerMuonTemplate& cp) {

//    m_GT = cp.m_GT; // TODO uncomment ???
    copy(cp);

    return *this;
}

L1MuGMTCand* L1GlobalTriggerMuonTemplate::getCandidate( int indexCand ) const {
        
    return (*m_GT.gtGTL()->getMuonCandidates())[indexCand];

}

/**
 * setConditionParameter - set the parameters of the condition
 *
 * @param numparticles Number of particle conditions.
 * @param particlep Pointer to particle parameters.
 * @param conditionp Pointer to condition parameters.
 * @wsc Indicates if this condition uses wsc.
 *
 */

void L1GlobalTriggerMuonTemplate::setConditionParameter(
    unsigned int numparticles, const ParticleParameter *particlep,
    const ConditionParameter* conditionp, bool wsc) {
    
    p_number = numparticles;
    if (p_number > 4) p_number = 4; //TODO: macro instead of 4
    p_wsc= wsc;

    memcpy(p_particleparameter, particlep, sizeof(ParticleParameter)*p_number);
    memcpy(&p_conditionparameter, conditionp, sizeof(ConditionParameter));
  
}


/**
 * blockCondition - Try all permutations of particles and check correlation
 *
 * @return boolean result of the check
 *
 */

const bool L1GlobalTriggerMuonTemplate::blockCondition() const {

    // maximum number of candidates 
    // TODO take it from L1GlobalTriggerReadoutRecord?    
    const int maxNumberCands = 4;

   // the candidates
    L1MuGMTCand* v[maxNumberCands];
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
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
            << "  L1GlobalTriggerMuonTemplate: no permutation match for all four muon candidates.\n" 
            << std::endl;
        return false;
    } 


    // charge_correlation consists of 3 relevant bits (D2, D1, D0)
    if (p_conditionparameter.charge_correlation & 1 == 0) { // charge ignore bit (D0) not set?
        for (int i = 0; i < (int) p_number; i++) {
            // we can't check invalid charge            
            if ( !( v[i]->charge_valid() ) ) return false; 
        }

        if (p_number == 1) { // one particle condition
    
          // D2..enable pos, D1..enable neg
          if ( ! ( ((p_conditionparameter.charge_correlation & 4) != 0 && v[0]->charge() > 0) ||
    	           ((p_conditionparameter.charge_correlation & 2) != 0 && v[0]->charge() < 0)    ) ) {
            return false;
          }

        } else {
    
            bool signsequal = true;	
            for (int i = 0; i< (int) p_number-1; i++) {
                if (v[i]->charge() != v[i+1]->charge()) { // find out if signs are equal
                    signsequal = false;
    	            break;
                }
            }
            
            // two or three particle condition
            if (p_number == 2 || p_number == 3) {
                // D2..enable equal, D1..enable not equal
                if ( ! ( ((p_conditionparameter.charge_correlation & 4) != 0 &&  signsequal) ||
       	                 ((p_conditionparameter.charge_correlation & 2) != 0 && !signsequal)   ) ) {
                    return false;
                }
            }
    
            // four particle condition
            if (p_number == 4) {
                //counter to count positive charges to determine if there are pairs
                unsigned int poscount=0;  
                
                for (int i = 0; i < (int) p_number; i++) {
                    if ( v[i]->charge() > 0 ) poscount++;
                }
                
                // D2..enable equal, D1..enable pairs
                if ( ! ( ((p_conditionparameter.charge_correlation & 4) != 0 && signsequal   ) ||
    	                 ((p_conditionparameter.charge_correlation & 2) != 0 && poscount == 2)    ) ) {
                    return false;
                } 
            }
        }
    } // end signchecks
 
    if (p_wsc) {

        unsigned int delta_eta;
        unsigned int delta_phi;

        for (int i = 0; i < maxNumberCands; ++i) {
            for (int k = i + 1; k < maxNumberCands; ++k) {

                // check delta_eta
                
                // absolute value of eta difference (... could use abs() ) 
                if (v[i]->etaIndex() > v[k]->etaIndex()) { 
                    delta_eta = v[i]->etaIndex() - v[k]->etaIndex();
                } else {
                    delta_eta = v[k]->etaIndex() - v[i]->etaIndex();
                }
                
                LogTrace("L1GlobalTriggerMuonTemplate") 
                    << "  delta_eta = " << delta_eta << " for "  
                    << "v[" << i <<"]->etaIndex() = " << v[i]->etaIndex() << ", " 
                    << "v[" << k <<"]->etaIndex() = " << v[k]->etaIndex()  
                    << std::endl;
            
                if ( !checkBit(p_conditionparameter.delta_eta, delta_eta) ) {
                    return false;
                }
    
                // check delta_phi
                
                // calculate absolute value of delta_phi
                if (v[i]->phiIndex() > v[k]->phiIndex()) { 
                    delta_phi = v[i]->phiIndex() - v[k]->phiIndex();
                } else {
                    delta_phi = v[k]->phiIndex() - v[i]->phiIndex();
                }
                LogTrace("L1GlobalTriggerMuonTemplate") 
                    << std::dec << "  delta_phi = " << delta_phi << " for " 
                    << "v[" << i <<"]->phiIndex() = " << v[i]->phiIndex() << ", " 
                    << "v[" << k <<"]->phiIndex() = " << v[k]->phiIndex()  
                    << std::endl;
    
                // check if delta_phi > 180 (aka delta_phi_maxbits) 
                // delta_phi contains bits for 0..180 (0 and 180 included)
                // usualy delta_phi_maxbits == 73
                if (delta_phi > p_conditionparameter.delta_phi_maxbits) {
                    LogTrace("L1GlobalTriggerMuonTemplate") 
                        << "  delta_phi = " << delta_phi 
                        << " > p_conditionparameter.delta_phi_maxbits ==> needs re-scaling" 
                        << std::endl;
                        
                    // delta_phi > 180 ==> take 360 - delta_phi
                    delta_phi = (p_conditionparameter.delta_phi_maxbits - 1)*2 - delta_phi;   
                    LogTrace("L1GlobalTriggerMuonTemplate") 
                        << "  delta_phi changed to: " <<  delta_phi 
                        << std::endl;
                }
            
                // delta_phi bitmask is saved in two u_int64_t 
                if (delta_phi < 64) {
                    if (!checkBit(p_conditionparameter.delta_phil, delta_phi) ) {
                        return false;
                    }
                } else {
                    if (!checkBit(p_conditionparameter.delta_phih, delta_phi - 64)) {
                        return false;
                    }
                }
            
            }                       
        }         
    
    } // end wsc check

    // if we get here all checks were successfull
    return true;
    
}

void L1GlobalTriggerMuonTemplate::printThresholds() const {

    edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
        << "L1GlobalTriggerMuonTemplate: threshold values " << std::endl;
    edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
        << "Condition Name: " << getName() << std::endl;
    edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
        << "\ngreater or equal bit: " << p_ge_eq << std::endl;

    for (unsigned int i = 0; i < p_number; i++) {
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") << std::endl;
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") << "  TEMPLATE " << i << std::endl;
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
            << "    pt_h_threshold        " 
            <<  std::hex << p_particleparameter[i].pt_h_threshold << std::endl;
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
            << "    pt_l_threshold        " 
            <<  std::hex << p_particleparameter[i].pt_l_threshold << std::endl;
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
            << "    enable mip            " 
            <<  p_particleparameter[i].en_mip << std::endl;          
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
            << "    enable iso            " 
            <<  p_particleparameter[i].en_iso << std::endl;        
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
            << "    quality               " 
            <<  std::hex << p_particleparameter[i].quality << std::endl;
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
            << "    eta                   " 
            <<  std::hex << p_particleparameter[i].eta << std::endl;
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
            << "    phi_h                 " 
            <<  std::hex << p_particleparameter[i].phi_h << std::endl;
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
            << "    phi_l                 " 
            <<  std::hex << p_particleparameter[i].phi_l << std::endl;
    }

    edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
        << "    Correlation parameters:" <<  std::endl;
    edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
        << "    charge_correlation    " 
        << std::hex << p_conditionparameter.charge_correlation << std::endl; 
    if (p_wsc) {
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
            << "    delta_eta             " 
            << std::hex << p_conditionparameter.delta_eta << std::endl; 
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
            << "    delta_eta_maxbits     " 
            << std::dec << p_conditionparameter.delta_eta_maxbits << std::endl;
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
            << "    delta_phih            " 
            << std::hex << p_conditionparameter.delta_phih << std::endl; 
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
            << "    delta_phil            " 
            << std::hex << p_conditionparameter.delta_phil << std::endl;
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") 
            << "    delta_phi_maxbits     " 
            << std::dec << p_conditionparameter.delta_phi_maxbits << std::endl;
    }

    // reset to decimal output
    edm::LogVerbatim("L1GlobalTriggerMuonTemplate") << std::dec << std::endl;
}


/**
 * checkParticle - compare a single particle with a numbered condition
 * 
 * @param ncondition The number of the condition
 * @param cand The candidate to compare 
 *
 * @return The result of the comparison false if a condition does not exist
 * 
 */

const bool L1GlobalTriggerMuonTemplate::checkParticle(
    int ncondition, L1MuGMTCand &cand) const {
  
    std::string checkFalse = "\n  ==> checkParticle = false ";        
    
    if (ncondition >= (int) p_number || ncondition < 0) {
        LogTrace("L1GlobalTriggerMuonTemplate")
            << "L1GlobalTriggerMuonTemplate:"
            << "  Number of condition outside [0, p_number) interval." 
            << "  checkParticle = false "
            << std::endl;
        return false;
    }

    // empty muons can not be compared
    if (cand.empty()) {
        LogTrace("L1GlobalTriggerMuonTemplate") 
            << "  Empty muon candidate (" << &cand << ")." 
            << "  checkParticle = false "
            << std::endl;
        return false;
    }

    LogTrace("L1GlobalTriggerMuonTemplate") << "  Non-empty muon: checkParticle starting" 
        << std::endl;
  
    // check thresholds: 
    //   value > high pt threshold: 
    //       OK, trigger
    //   low pt threshold < value < high pt threshold: 
    //      check isolation if "enable isolation" set and if it is an isolated candidate
    
    // checkThreshold always check ">=" or ">" 
    // ==> checkThreshold = false for high pt threshold: check isolation  
    if (!checkThreshold(p_particleparameter[ncondition].pt_h_threshold, cand.ptIndex())) {
    
        LogTrace("L1GlobalTriggerMuonTemplate") 
            << "  muon: value < high pt threshold ==> check low pt threshold" 
            << std::endl;
            
        // check isolated muon

        // using the table from GTL-6U-module.pdf at top of page 35 TODO FIXME  
        // the exact behavior is not jet decided
    
        // false if lower than low threshold
        if ( !checkThreshold( p_particleparameter[ncondition].pt_l_threshold, cand.ptIndex() ) ) {
            LogTrace("L1GlobalTriggerMuonTemplate") << "  muon low pt threshold: failed" 
                << checkFalse 
                << std::endl;            
            return false;
        } else {
            LogTrace("L1GlobalTriggerMuonTemplate") << "  muon low pt threshold: passed" 
                << std::endl;
        }            
            
        // false if "enable isolation" set and muon not isolated    
        if ( !cand.isol() && p_particleparameter[ncondition].en_iso) {
            LogTrace("L1GlobalTriggerMuonTemplate") 
                << "  enable isolation set; muon not isolated" 
                << checkFalse 
                << std::endl;
            return false;
        } else {
            LogTrace("L1GlobalTriggerMuonTemplate") << "  muon isolation: passed" 
                << std::endl;
        }
    
    } else {
        LogTrace("L1GlobalTriggerMuonTemplate") << "  ===> muon high pt threshold: passed" 
            << std::endl;
    }
  
    // check eta
    if (!checkBit(p_particleparameter[ncondition].eta, cand.etaIndex())) {
        LogTrace("L1GlobalTriggerMuonTemplate") << "  muon eta: failed" 
            << checkFalse 
            << std::endl;
        return false;
    } else {
        LogTrace("L1GlobalTriggerMuonTemplate") << "  ==> muon eta: passed" << std::endl;
    }
  
    // check phi if it is in the range
    if ( p_particleparameter[ncondition].phi_h >= p_particleparameter[ncondition].phi_l) { 
        if (! ( p_particleparameter[ncondition].phi_l <= cand.phiIndex() &&
                cand.phiIndex() <= p_particleparameter[ncondition].phi_h    ) ) {

            LogTrace("L1GlobalTriggerMuonTemplate") << "  muon phi: failed" 
                << checkFalse 
                << std::endl;
            return false;
        } else {
            LogTrace("L1GlobalTriggerMuonTemplate") << "  muon phi: passed" << std::endl;
        }
    } else {
        if (! ( p_particleparameter[ncondition].phi_h <= cand.phiIndex() ||
                cand.phiIndex() <= p_particleparameter[ncondition].phi_h   ) ) {

            LogTrace("L1GlobalTriggerMuonTemplate") << "  muon phi: failed" 
                << checkFalse            
                << std::endl;
            return false;
        } else {
            LogTrace("L1GlobalTriggerMuonTemplate") << "  muon phi: passed" << std::endl;
        }         
    }    

  
    // check quality ( bit check )
    // if (!checkBit(p_particleparameter[ncondition].quality, cand.quality())) return false;
    if (!checkBitM(p_particleparameter[ncondition].quality, cand.quality())) {
        LogTrace("L1GlobalTriggerMuonTemplate") << "  muon quality: failed" 
            << checkFalse 
            << std::endl;
        return false;
    } else {
        LogTrace("L1GlobalTriggerMuonTemplate") << "  ==> muon quality: passed" << std::endl;
    }

    // check mip
    if (p_particleparameter[ncondition].en_mip) {
        if (!cand.mip()) {
    
            LogTrace("L1GlobalTriggerMuonTemplate") << "  muon mip: failed" 
                << checkFalse 
                << std::endl;
            return false;
        } else {
            LogTrace("L1GlobalTriggerMuonTemplate") << "  muon mip: passed" << std::endl;
        }
    }

    // particle matches if we get here
    LogTrace("L1GlobalTriggerMuonTemplate") 
        << "  checkParticle: muon OK, passes all requirements\n" 
        << std::endl;
        
    return true;
}
    
template<class Type1>
    const bool L1GlobalTriggerMuonTemplate::checkBitM(
    Type1 const &mask, unsigned int bitNumber) const {

    if (bitNumber >= 64) return false; // TODO 64 as static or parameter

    // do not accept muons with quality zero
    if ( bitNumber == 0) return false;
    
    if ( mask == 0) return true;
 
    u_int64_t oneBit = 1;
    oneBit <<= bitNumber;
      
    LogTrace("L1GlobalTriggerMuonTemplate") 
        << "  checkBit muon" 
        << "\n     mask address = " << &mask
        << std::dec  
        << "\n     dec: mask = " << mask << " oneBit = " << oneBit << " bitNumber = " << bitNumber 
        << std::hex 
        << "\n     hex: mask = " << mask << " oneBit = " << oneBit << " bitNumber = " << bitNumber
        << std::dec 
        << "\n     mask & oneBit result = " << bool ( mask & oneBit ) 
        << std::endl;
        
      
    return (mask & oneBit);
}

