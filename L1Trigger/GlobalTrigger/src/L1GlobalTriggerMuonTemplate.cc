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
 * $Date:$
 * $Revision:$
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

#include "FWCore/MessageLogger/interface/MessageLogger.h"
 
// constructor
L1GlobalTriggerMuonTemplate::L1GlobalTriggerMuonTemplate( 
    const std::string &name) 
    : L1GlobalTriggerConditions(name) {

    LogDebug ("Trace") << "Entering " << __PRETTY_FUNCTION__ 
        << " name= " << p_name << std::endl;    

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
    :
    L1GlobalTriggerConditions(cp.p_name) {
    copy(cp);
}

// destructor
L1GlobalTriggerMuonTemplate::~L1GlobalTriggerMuonTemplate() {

}

// equal operator
L1GlobalTriggerMuonTemplate& 
    L1GlobalTriggerMuonTemplate::operator= (const L1GlobalTriggerMuonTemplate& cp) {

    copy(cp);

    return *this;
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

    // the candidates
    L1MuGMTCand (*v[4]) = { 
        (getCandidate(0)),
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
    } while ( std::next_permutation(index, index + p_number) );

    if (tmpResult == false) return false;


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

        // check delta_eta
        unsigned int delta_eta;
        if (v[1]->etaRegionIndex() > v[0]->etaRegionIndex()) { 
            delta_eta = v[1]->etaRegionIndex() - v[0]->etaRegionIndex();
        } else {
            delta_eta = v[0]->etaRegionIndex() - v[1]->etaRegionIndex();
        }
        
        if ( !checkBit(p_conditionparameter.delta_eta, delta_eta) ) return false;

        // check delta_phi
        unsigned int delta_phi;
        unsigned int phisteps;				   // total steps for one full turn
        phisteps = (p_conditionparameter.delta_phi_maxbits-1)*2;  
        // deltaphi contains bits for 0..180 (0 and 180 included)
        // usualy delta_phi_maxbits == 73 ==> phisteps == 144


        // calculate absolute value of delta_phi
        if (v[1]->phiRegionIndex() > v[0]->phiRegionIndex()) { 
            delta_phi = v[1]->phiRegionIndex() - v[0]->phiRegionIndex();
        } else {
            delta_phi = v[0]->phiRegionIndex() - v[1]->phiRegionIndex();
        }

        // if delta_phi > 180
        if (delta_phi > p_conditionparameter.delta_phi_maxbits) {
            // lets hope that deltaphi is < phisteps.... TODO check here again!!!
            delta_phi = phisteps - delta_phi;		
        }
    
        // delta_phi bitmask is saved in two u_int64_t 
        if (delta_phi < 64) {
            if (!checkBit(p_conditionparameter.delta_phil, delta_phi) ) return false;
        } else {
            if (!checkBit(p_conditionparameter.delta_phih, delta_phi-64)) return false;
        }
    
    } // end wsc check

    // if we get here all checks were successfull
    return true;
    
}

void L1GlobalTriggerMuonTemplate::printThresholds() const {

    std::cout << "L1GlobalTriggerMuonTemplate: Threshold values " << std::endl;
    std::cout << "Condition Name: " << getName() << std::endl;
    std::cout << "greater equal Flag:	 " << p_ge_eq << std::endl;

    for (unsigned int i = 0; i < p_number; i++) {
        std::cout << std::endl;
        std::cout << "TEMPLATE " << i << std::endl;
        std::cout << "pt_h_threshold        " 
            <<  std::hex << p_particleparameter[i].pt_h_threshold << std::endl;
        std::cout << "pt_l_threshold        " 
            <<  std::hex << p_particleparameter[i].pt_l_threshold << std::endl;
        std::cout << "enable mip            " <<  p_particleparameter[i].en_mip << std::endl;          
        std::cout << "enable iso            " <<  p_particleparameter[i].en_iso << std::endl;        
        std::cout << "quality               " <<  std::hex << p_particleparameter[i].quality << std::endl;
        std::cout << "eta                   " <<  std::hex << p_particleparameter[i].eta << std::endl;
        std::cout << "phi_h                 " <<  std::hex << p_particleparameter[i].phi_h << std::endl;
        std::cout << "phi_l                 " <<  std::hex << p_particleparameter[i].phi_l << std::endl;
    }

    std::cout << "// CORRELATION TEMPLATE" <<  std::endl;
    std::cout << "charge_correlation    " 
        << std::hex << p_conditionparameter.charge_correlation << std::endl; 
    if (p_wsc) {
        std::cout << "delta_eta             " 
            << std::hex << p_conditionparameter.delta_eta << std::endl; 
        std::cout << "delta_eta_maxbits     " 
            << std::dec << p_conditionparameter.delta_eta_maxbits << std::endl;
        std::cout << "delta_phih            " 
            << std::hex << p_conditionparameter.delta_phih << std::endl; 
        std::cout << "delta_phil            " 
            << std::hex << p_conditionparameter.delta_phil << std::endl;
        std::cout << "delta_phi_maxbits     " 
            << std::dec << p_conditionparameter.delta_phi_maxbits << std::endl;
    }

    // reset to decimal output
    std::cout << std::dec << std::endl;
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
  
    // empty muons can't be compared
    if (ncondition >= (int) p_number || ncondition < 0 || cand.empty()) return false;
  
    // check threshold TODO
    edm::LogVerbatim("L1GlobalTriggerMuonTemplate") << " muon high pt thresh " << std::endl;
    if (!checkThreshold(p_particleparameter[ncondition].pt_h_threshold, cand.ptIndex())) {
        //check isolated muon if pt>pt_l_threshold and en_iso and an isolated candidate

        // using the table from GTL-6U-module.pdf at top of page 35 
        // the exact behavior is not jet decided
    
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") << " muon  low pt thresh " << std::endl;
        // false if lower than low threshold
        if ( !checkThreshold( p_particleparameter[ncondition].pt_l_threshold, cand.ptIndex() ) ) return false; 
     
        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") << " muon  for isolation " << std::endl;
        // false if enableisolation set and muon not isolated    
        if ( !cand.isol() && p_particleparameter[ncondition].en_iso) return false;

        edm::LogVerbatim("L1GlobalTriggerMuonTemplate") << " muon  after  isolation " << std::endl;
    
    }
  
    // check eta
    edm::LogVerbatim("L1GlobalTriggerMuonTemplate") << " muon eta  check     " << std::endl;
    if (!checkBit(p_particleparameter[ncondition].eta, cand.etaRegionIndex())) return false;
  
    // check phi if it is in the range
    edm::LogVerbatim("L1GlobalTriggerMuonTemplate") << " muon phi  check     " << std::endl;
    if ( p_particleparameter[ncondition].phi_h >= p_particleparameter[ncondition].phi_l) { 
        if (! ( p_particleparameter[ncondition].phi_l <= cand.phiRegionIndex() &&
                cand.phiRegionIndex() <= p_particleparameter[ncondition].phi_h    ) ) {
            return false;
        }
    } else {
        if (! ( p_particleparameter[ncondition].phi_h <= cand.phiRegionIndex() ||
                cand.phiRegionIndex() <= p_particleparameter[ncondition].phi_h   ) ) {
            return false;
        }         
    }    

  

    edm::LogVerbatim("L1GlobalTriggerMuonTemplate") << " muon qual check     " << std::endl;
    // check quality ( bit check )
    // if (!checkBit(p_particleparameter[ncondition].quality, cand.quality())) return false;
    if (!checkBitM(p_particleparameter[ncondition].quality, cand.quality())) return false;

    edm::LogVerbatim("L1GlobalTriggerMuonTemplate") << " muon mip  check     " << std::endl;

    // check mip
    if (p_particleparameter[ncondition].en_mip) {
        if (!cand.mip()) return false;
    }

    // particle matches if we get here
    edm::LogVerbatim("L1GlobalTriggerMuonTemplate") << " muon OAKY FINALLY   " << std::endl;
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
      
    bool mask1one = mask & oneBit;
    edm::LogVerbatim("L1GlobalTriggerMuonTemplate")  
        << " checkBitMUON  " 
        << &mask << " " << mask << hex << mask << dec << " " 
        << oneBit << " " << bitNumber 
        << " result " << mask1one 
        << std::endl;
      
    return (mask & oneBit);
}

