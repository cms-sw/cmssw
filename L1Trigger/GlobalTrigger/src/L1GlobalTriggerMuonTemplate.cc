/**
 * \class L1GlobalTriggerMuonTemplate
 * 
 * 
 * Description: Implementation of L1GlobalTriggerMuonTemplate.  
 *
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

#include <cmath>
#include <cstdlib>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerGTL.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFunctions.h"

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

    memcpy(p_particleparameter, cp.getParticleParameter(), 
        sizeof(ParticleParameter)*L1GlobalTriggerReadoutSetup::NumberL1Muons); 
    memcpy(&p_conditionparameter, cp.getConditionParameter(), 
        sizeof(ConditionParameter));
  
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
    if (p_number > L1GlobalTriggerReadoutSetup::NumberL1Muons) {
        p_number = L1GlobalTriggerReadoutSetup::NumberL1Muons;
    }
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
    const int maxNumberCands = L1GlobalTriggerReadoutSetup::NumberL1Muons;

    // the candidates
    L1MuGMTCand* v[maxNumberCands];
    int index[maxNumberCands];
    
    for (int i = 0; i < maxNumberCands; ++i) {
       v[i] = getCandidate(i);
       index[i] = i;    
    }
  
    int jumpIndex = 1;
    int jump = factorial(maxNumberCands - p_number);

    int totalLoops = 0;
    int passLoops = 0;
    
    // condition result condResult set to true if at least one permutation 
    //     passes all requirements
    // all possible permutations are checked 
    bool condResult = false;

    // store the indices of the muon objects 
    // from the combination evaluated in the condition     
    SingleCombInCond objectsInComb;

    // clear the p_combinationsInCond vector 
    (*p_combinationsInCond).clear();    

    // clear the p_objectsInCond vector 
    (*p_objectsInCond).clear();
        
    ObjectTypeInCond typeInCond;
    
    do {

        if (--jumpIndex) continue;

        LogTrace("L1GlobalTriggerMuonTemplate") 
            << "\n  Evaluating new trigger object permutation ... " << std::endl;
        
        jumpIndex = jump;        
        totalLoops++;
        
        // clean up the indices in the combination     
        objectsInComb.clear();
        typeInCond.clear();

        bool tmpResult = true;

        // check if there is a permutation that matches particle conditions
        for (int i = 0; i < static_cast<int> (p_number); i++) {
            LogTrace("L1GlobalTriggerMuonTemplate") 
                << "  Current condition index = " << i 
                << " < last index = " << p_number
                << ";  checking object with index = " << index[i]  
                << std::endl;
            
            tmpResult &= checkParticle(i, *(v)[index[i]] );
            objectsInComb.push_back(index[i]);
            
            typeInCond.push_back(Mu);
            
        }
    
        std::ostringstream myCout;                 
        std::copy(objectsInComb.begin(), objectsInComb.end(), 
            std::ostream_iterator<int> (myCout, " "));

        // if permutation does not match particle conditions
        // skip charge and spatial correlations
        if ( !tmpResult ) {

            LogTrace("L1GlobalTriggerMuonTemplate") 
                << "  Trigger object permutation ( " <<  myCout.str() 
                << ") fails single-particle requirements." 
                << std::endl;
            continue;
        } else {
            LogTrace("L1GlobalTriggerMuonTemplate") 
                << "  Trigger object permutation ( " <<  myCout.str() 
                << ") passes single-particle requirements." 
                << std::endl;
        }

        // charge_correlation consists of 3 relevant bits (D2, D1, D0)

        // charge ignore bit (D0) not set?
        if (p_conditionparameter.charge_correlation & 1 == 0) {

            LogTrace("L1GlobalTriggerMuonTemplate") 
                << "\n  Checking charge correlations: \n"
                << std::endl;

            for (int i = 0; i < static_cast<int> (p_number); i++) {
                // check valid charge - skip if invalid charge            
                bool chargeValid = v[index[i]]->charge_valid();
                tmpResult &= chargeValid; 

                if ( !chargeValid ) {
                    LogTrace("L1GlobalTriggerMuonTemplate") 
                        << "  Invalid charge for object " <<  index[i] 
                        << std::endl;
                    continue;
                }
            }
    
            if (! tmpResult) {
    
                LogTrace("L1GlobalTriggerMuonTemplate") 
                    << "  Trigger object permutation ( " <<  myCout.str() 
                    << ") fails charge validity requirements." 
                    << std::endl;
                continue;
            }
            
            if (p_number == 1) { // one particle condition
        
              // D2..enable pos, D1..enable neg
              if ( ! ( ( (p_conditionparameter.charge_correlation & 4) != 0 
                         && v[index[0]]->charge() > 0
                       ) ||
                       ( (p_conditionparameter.charge_correlation & 2) != 0 
                         && v[index[0]]->charge() < 0
                       )    
                     ) 
                 ) {
                    
                  continue;
              }
    
            } else {
        
                // find out if signs are equal
                bool equalSigns = true; 
                for (int i = 0; i < static_cast<int> (p_number)-1; i++) {
                    if (v[index[i]]->charge() != v[index[i+1]]->charge()) {
                        equalSigns = false;
                        break;
                    }
                }
                
                // two or three particle condition
                if (p_number == 2 || p_number == 3) {
                    // D2..enable equal, D1..enable not equal
                    if ( ! ( ( (p_conditionparameter.charge_correlation & 4) != 0 
                                &&  equalSigns
                             ) ||
                             ( (p_conditionparameter.charge_correlation & 2) != 0 
                                && !equalSigns
                             )   
                           ) 
                       ) {
                        
                        continue;
                    }
                }
        
                // four particle condition
                if (p_number == 4) {
                    //counter to count positive charges to determine if there are pairs
                    unsigned int posCount = 0;  
                    
                    for (int i = 0; i < static_cast<int> (p_number); i++) {
                        if ( v[index[i]]->charge() > 0 ) posCount++;
                    }
                    
                    // D2..enable equal, D1..enable pairs
                    if ( ! ( ( (p_conditionparameter.charge_correlation & 4) != 0 
                               && equalSigns   
                             ) ||
                             ( (p_conditionparameter.charge_correlation & 2) != 0 
                               && posCount == 2
                             )    
                           ) 
                       ) {
                        
                        continue;
                    } 
                }
            }
        } // end signchecks

        if (p_wsc) {
            
            LogTrace("L1GlobalTriggerMuonTemplate") 
                << "\n  Checking spatial correlations: \n"
                << std::endl;

            // wsc requirements have always p_number = 2
            // one can use directly index[0] and index[1] to compute 
            // eta and phi differences
            const int ObjInWscComb = 2;
            if (static_cast<int> (p_number) != ObjInWscComb) {
                edm::LogError("L1GlobalTriggerMuonTemplate") 
                    << "\n  ERROR: " 
                    << "number of particles in condition with spatial correlation = "
                    << p_number << "\n  it must be = " << ObjInWscComb 
                    << std::endl;
                // TODO Perhaps I should throw here an exception, 
                // since something is really wrong if p_number != ObjInWscComb (=2)
                continue;  
            }
    
            unsigned int delta_eta;
            unsigned int delta_phi;
        
            // check delta_eta
            
            // get eta index and the sign bit of the eta index (MSB is the sign)
            //   signedEta[i] is the signed eta index of v[index[i]]
            int signedEta[ObjInWscComb]; 
            int signBit[ObjInWscComb] = {0, 0}; 

            int etaBits = L1GlobalTriggerReadoutSetup::MuonEtaBits;
            
            int scaleEta = 1 << (etaBits - 1);
            LogTrace("L1GlobalTriggerMuonTemplate") 
                << "  scale factor for eta = " << scaleEta << " ( "  
                << etaBits << " bits for eta of the muon, MSB - sign bit), " 
                << std::endl;
            
            for (int i = 0; i < ObjInWscComb; ++i) {
                signBit[i] = (v[index[i]]->etaIndex() & scaleEta)>>(etaBits - 1);
                signedEta[i] = ( v[index[i]]->etaIndex() )%scaleEta;

                if (signBit[i] == 1) {
                    signedEta[i] = (-1)*signedEta[i];
                }                

            }

            // compute delta_eta - add 1 if signs are different (due to +0/-0 indices)            
            delta_eta = static_cast<int> (std::abs(signedEta[1] - signedEta[0])) 
                + static_cast<int> (signBit[1]^signBit[0]); 

            LogTrace("L1GlobalTriggerMuonTemplate") 
                << "  delta_eta = " << delta_eta << " for "  
                << "v[" << index[0] <<"]->etaIndex() = " << v[index[0]]->etaIndex()
                << " (scaled value " << signedEta[0] << "), " 
                << "v[" << index[1] <<"]->etaIndex() = " << v[index[1]]->etaIndex()  
                << " (scaled value " << signedEta[1] << "), " 
                << std::endl;
        
            if ( !checkBit(p_conditionparameter.delta_eta, delta_eta) ) {
                LogTrace("L1GlobalTriggerMuonTemplate") << "  muon delta_eta: failed" 
                    << std::endl;
                continue;
            } else {
                LogTrace("L1GlobalTriggerMuonTemplate") 
                    << "  ==> muon delta_eta: passed" 
                    << std::endl;
            }

            // check delta_phi
            
            // calculate absolute value of delta_phi
            if (v[index[0]]->phiIndex() > v[index[1]]->phiIndex()) { 
                delta_phi = v[index[0]]->phiIndex() - v[index[1]]->phiIndex();
            } else {
                delta_phi = v[index[1]]->phiIndex() - v[index[0]]->phiIndex();
            }
            LogTrace("L1GlobalTriggerMuonTemplate") 
                << std::dec << "  delta_phi = " << delta_phi << " for " 
                << "v[" << index[0] <<"]->phiIndex() = " << v[index[0]]->phiIndex() 
                << ", " 
                << "v[" << index[1] <<"]->phiIndex() = " << v[index[1]]->phiIndex()  
                << std::endl;

            // check if delta_phi > 180 (aka delta_phi_maxbits) 
            // delta_phi contains bits for 0..180 (0 and 180 included)
            // delta_phi_maxbits == 73 for a range of 2.5 deg (see scale file)
            while (delta_phi > p_conditionparameter.delta_phi_maxbits) {
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
        
            // delta_phi bitmask is saved in two u_int64_t (see parser) 
            if (delta_phi < 64) {
                if (!checkBit(p_conditionparameter.delta_phil, delta_phi) ) {
                    continue;
                }
            } else {
                if (!checkBit(p_conditionparameter.delta_phih, delta_phi - 64)) {
                    continue;
                }
            }
            
    
        } // end wsc check
        
        // if we get here all checks were successfull for this combination 
        // set the general result for blockCondition remains "true"

        condResult = true;
        passLoops++;
        (*p_combinationsInCond).push_back(objectsInComb);

        LogTrace("L1GlobalTriggerMuonTemplate") 
            << "\n  ... Trigger object permutation ( " <<  myCout.str() 
            << ") passes all requirements." 
            << std::endl;

                    
    } while ( std::next_permutation(index, index + maxNumberCands) );


   (*p_objectsInCond) = typeInCond;
    
    LogTrace("L1GlobalTriggerMuonTemplate") 
        << "\n  L1GlobalTriggerMuonTemplate: total number of permutations found:          " 
        << totalLoops 
        << "\n  L1GlobalTriggerMuonTemplate: number of permutations passing requirements: " 
        << passLoops << "\n" 
        << std::endl;    

        CombinationsInCond::const_iterator itVV;
        std::ostringstream myCout1;                 

        for(itVV  = (*p_combinationsInCond).begin(); 
            itVV != (*p_combinationsInCond).end(); itVV++) {
            
            myCout1 << "( ";
        
            std::copy((*itVV).begin(), (*itVV).end(), 
                std::ostream_iterator<int> (myCout1, " "));
            
            myCout1 << "); ";

        }


        LogTrace("L1GlobalTriggerMuonTemplate") 
            << "\n  List of combinations passing all requirements for this condition: \n  " 
            <<  myCout1.str() 
            << " \n" 
            << std::endl;

    return condResult;

}

void L1GlobalTriggerMuonTemplate::printThresholds(std::ostream& myCout) const {

    myCout << "L1GlobalTriggerMuonTemplate: threshold values " << std::endl;
    myCout << "Condition Name: " << getName() << std::endl;
    myCout << "\ngreater or equal bit: " << p_ge_eq << std::endl;

    for (unsigned int i = 0; i < p_number; i++) {
        myCout << std::endl;
        myCout << "  TEMPLATE " << i << std::endl;
        myCout << "    pt_h_threshold        " 
            <<  std::hex << p_particleparameter[i].pt_h_threshold << std::endl;
        myCout << "    pt_l_threshold        " 
            <<  std::hex << p_particleparameter[i].pt_l_threshold << std::endl;
        myCout << "    enable mip            " 
            <<  p_particleparameter[i].en_mip << std::endl;          
        myCout << "    enable iso            " 
            <<  p_particleparameter[i].en_iso << std::endl;        
        myCout << "    quality               " 
            <<  std::hex << p_particleparameter[i].quality << std::endl;
        myCout << "    eta                   " 
            <<  std::hex << p_particleparameter[i].eta << std::endl;
        myCout << "    phi_h                 " 
            <<  std::hex << p_particleparameter[i].phi_h << std::endl;
        myCout << "    phi_l                 " 
            <<  std::hex << p_particleparameter[i].phi_l << std::endl;
    }

    myCout << "    Correlation parameters:" <<  std::endl;
    myCout << "    charge_correlation    " 
        << std::hex << p_conditionparameter.charge_correlation << std::endl; 
    if (p_wsc) {
        myCout << "    delta_eta             " 
            << std::hex << p_conditionparameter.delta_eta << std::endl; 
        myCout << "    delta_eta_maxbits     " 
            << std::dec << p_conditionparameter.delta_eta_maxbits << std::endl;
        myCout << "    delta_phih            " 
            << std::hex << p_conditionparameter.delta_phih << std::endl; 
        myCout << "    delta_phil            " 
            << std::hex << p_conditionparameter.delta_phil << std::endl;
        myCout << "    delta_phi_maxbits     " 
            << std::dec << p_conditionparameter.delta_phi_maxbits << std::endl;
    }

    // reset to decimal output
    myCout << std::dec << std::endl;
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
    
    if (ncondition >= static_cast<int> (p_number) || ncondition < 0) {
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
    // if checkThreshold = false for high pt threshold: check isolation  
    if (!checkThreshold(p_particleparameter[ncondition].pt_h_threshold, 
        cand.ptIndex())) {
    
        LogTrace("L1GlobalTriggerMuonTemplate") 
            << "  muon: value < high pt threshold ==> check low pt threshold" 
            << std::endl;
            
        // check isolated muon

        // using the logic table from GTL-6U-module.pdf at top of page 35 
        // TODO FIXME change to the latest version implemented in hw  
    
        // false if lower than low threshold
        if ( !checkThreshold( p_particleparameter[ncondition].pt_l_threshold, 
            cand.ptIndex() ) ) {
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
        LogTrace("L1GlobalTriggerMuonTemplate") 
            << "  ===> muon high pt threshold: passed" 
            << std::endl;
    }
  
    // check eta
    LogTrace("L1GlobalTriggerMuonTemplate") << "  muon eta check"
        << " etaIndex() = " 
        << std::dec << cand.etaIndex() << " (dec) " 
        << std::hex << cand.etaIndex() << " (hex) "
        << std::dec
        << std::endl;
    
    if (!checkBit(p_particleparameter[ncondition].eta, cand.etaIndex())) {
        LogTrace("L1GlobalTriggerMuonTemplate") << "  muon eta: failed" 
            << checkFalse 
            << std::endl;
        return false;
    } else {
        LogTrace("L1GlobalTriggerMuonTemplate") << "  ==> muon eta: passed" << std::endl;
    }
  
    // check phi if it is in the range (no LUT used - LUT too big for hw chip)
    if (p_particleparameter[ncondition].phi_h >= 
        p_particleparameter[ncondition].phi_l) { 

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
    
    if ( cand.quality() == 0 ) {
        LogTrace("L1GlobalTriggerMuonTemplate") << "  muon with quality = 0 are rejected" 
            << checkFalse 
            << std::endl;
        return false;
    }
        
    if (p_particleparameter[ncondition].quality == 0) {
        LogTrace("L1GlobalTriggerMuonTemplate") << "  muon quality: mask = " 
            << p_particleparameter[ncondition].quality
            << " ==> quality not requested ==> passed "
            << std::endl;
    } else {
        if (!checkBit(p_particleparameter[ncondition].quality, cand.quality())) {
            LogTrace("L1GlobalTriggerMuonTemplate") << "  muon quality: failed" 
                << checkFalse 
                << std::endl;
            return false;
        } else {
            LogTrace("L1GlobalTriggerMuonTemplate") 
                << "  ==> muon quality: passed" 
                << std::endl;
        }
    }

    // check mip
    if (p_particleparameter[ncondition].en_mip) {
        if (!cand.mip()) {
    
            LogTrace("L1GlobalTriggerMuonTemplate") << "  muon mip: failed" 
                << checkFalse 
                << std::endl;
            return false;
        } else {
            LogTrace("L1GlobalTriggerMuonTemplate") << "  muon mip: passed" 
                << std::endl;
        }
    }

    // particle matches if we get here
    LogTrace("L1GlobalTriggerMuonTemplate") 
        << "  checkParticle: muon OK, passes all requirements\n" 
        << std::endl;
        
    return true;
}
    
