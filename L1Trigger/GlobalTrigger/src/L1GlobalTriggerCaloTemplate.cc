/**
 * \class L1GlobalTriggerCaloTemplate
 * 
 * 
 * Description: Single particle chip - description for calo conditions.  
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
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerCaloTemplate.h"

// system include files
#include <iostream>
#include <iomanip>

#include <string>

#include <algorithm>

#include <vector>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFunctions.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


// constructor
L1GlobalTriggerCaloTemplate::L1GlobalTriggerCaloTemplate(
    const L1GlobalTrigger& gt,
    const std::string& name)
        : L1GlobalTriggerConditions(gt, name)
{

    //    LogDebug ("Trace")
    //        << "****Entering " << __PRETTY_FUNCTION__ << " name= " << p_name << std::endl;

}

// copy constructor
void L1GlobalTriggerCaloTemplate::copy(const L1GlobalTriggerCaloTemplate &cp)
{
    p_name = cp.getName();
    p_number = cp.getNumberParticles();
    p_wsc = cp.getWsc();
    setGeEq(cp.getGeEq());

    // maximum number of candidates
    int maxNumberCands = 0;

    switch (p_particletype) {
        case EG:
            maxNumberCands = L1GlobalTriggerReadoutSetup::NumberL1Electrons;
            break;
        case IEG:
            maxNumberCands = L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons;
            break;
        case CJET:
            maxNumberCands = L1GlobalTriggerReadoutSetup::NumberL1CentralJets;
            break;
        case FJET:
            maxNumberCands = L1GlobalTriggerReadoutSetup::NumberL1ForwardJets;
            break;
        case TJET:
            maxNumberCands = L1GlobalTriggerReadoutSetup::NumberL1TauJets;
            break;
        default:
            // do nothing - should not arrive here
            break;
    }

    memcpy(p_particleparameter, cp.getParticleParameter(),
           sizeof(ParticleParameter)*maxNumberCands);
    memcpy(&p_conditionparameter, cp.getConditionParameter(), sizeof(ConditionParameter));
}


L1GlobalTriggerCaloTemplate::L1GlobalTriggerCaloTemplate(
    const L1GlobalTriggerCaloTemplate& cp)
        : L1GlobalTriggerConditions(cp.m_GT, cp.p_name)
{

    //    m_GT = cp.m_GT; // TODO uncomment ???
    copy(cp);
}

// destructor
L1GlobalTriggerCaloTemplate::~L1GlobalTriggerCaloTemplate()
{
    
    // empty
    
}

// equal operator
L1GlobalTriggerCaloTemplate& L1GlobalTriggerCaloTemplate::operator= (
    const L1GlobalTriggerCaloTemplate& cp)
{

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
    const ConditionParameter* conditionp, ParticleType pType, bool wsc)
{

    // maximum number of candidates
    int maxNumberCands = 0;

    switch (pType) {
        case EG:
            maxNumberCands = L1GlobalTriggerReadoutSetup::NumberL1Electrons;
            break;
        case IEG:
            maxNumberCands = L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons;
            break;
        case CJET:
            maxNumberCands = L1GlobalTriggerReadoutSetup::NumberL1CentralJets;
            break;
        case FJET:
            maxNumberCands = L1GlobalTriggerReadoutSetup::NumberL1ForwardJets;
            break;
        case TJET:
            maxNumberCands = L1GlobalTriggerReadoutSetup::NumberL1TauJets;
            break;
        default:
            // do nothing - should not arrive here
            break;
    }

    p_number = numparticles;

    if (static_cast<int> (p_number) >
            maxNumberCands) {
        p_number = maxNumberCands;
    }

    p_wsc = wsc;
    p_particletype = pType;

    memcpy(p_particleparameter, particlep, sizeof(ParticleParameter)*p_number);
    memcpy(&p_conditionparameter, conditionp, sizeof(ConditionParameter));

    //LogTrace("L1GlobalTriggerCaloTemplate")
    //<< "\n  Maximum number of candidates = " << maxNumberCands
    //<< "; particles in condition: p_number = " << p_number
    //<< std::endl;

}

/**
 * blockCondition - Try all permutations of particles and check correlation
 *
 * @return boolean result of the check
 *
 */

const bool L1GlobalTriggerCaloTemplate::blockCondition() const
{

    // maximum number of candidates
    int maxNumberCands = 0;

    switch (p_particletype) {
        case EG:
            maxNumberCands = L1GlobalTriggerReadoutSetup::NumberL1Electrons;
            break;
        case IEG:
            maxNumberCands = L1GlobalTriggerReadoutSetup::NumberL1IsolatedElectrons;
            break;
        case CJET:
            maxNumberCands = L1GlobalTriggerReadoutSetup::NumberL1CentralJets;
            break;
        case FJET:
            maxNumberCands = L1GlobalTriggerReadoutSetup::NumberL1ForwardJets;
            break;
        case TJET:
            maxNumberCands = L1GlobalTriggerReadoutSetup::NumberL1TauJets;
            break;
        default:
            // do nothing - should not arrive here
            break;
    }

    //LogTrace("L1GlobalTriggerCaloTemplate")
    //<< "\n  Maximum number of candidates = " << maxNumberCands
    //<< "; particles in condition: p_number = " << p_number
    //<< std::endl;

    //    // the candidates
    //    L1GctCand* v[maxNumberCands];
    //    int index[maxNumberCands];

    // the candidates
    std::vector<L1GctCand*> v(maxNumberCands);
    std::vector<int> index(maxNumberCands);

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

    // store the indices of the calorimeter objects
    // from the combination evaluated in the condition
    SingleCombInCond objectsInComb;

    // clear the p_combinationsInCond vector
    (*p_combinationsInCond).clear();

    // clear the p_objectsInCond vector
    (*p_objectsInCond).clear();

    ObjectTypeInCond typeInCond;

    do {

        if (--jumpIndex)
            continue;

        LogTrace("L1GlobalTriggerCaloTemplate")
        << "\n  Evaluating new trigger object permutation ... " << std::endl;

        jumpIndex = jump;
        totalLoops++;

        // clean up the indices in the combination
        objectsInComb.clear();
        typeInCond.clear();

        bool tmpResult = true;

        // check if there is a permutation that matches particle conditions
        for (int i = 0; i < static_cast<int> (p_number); i++) {
            LogTrace("L1GlobalTriggerCaloTemplate")
            << "  Current condition index = " << i
            << " < last index = " << p_number
            << ";  checking object with index = " << index[i]
            << std::endl;

            tmpResult &= checkParticle(i, *(v)[index[i]] );
            objectsInComb.push_back(index[i]);

            // TODO dirty - change to L1GtObject...
            switch (p_particletype) {
                case EG:
                    typeInCond.push_back(NoIsoEG);
                    break;
                case IEG:
                    typeInCond.push_back(IsoEG);
                    break;
                case CJET:
                    typeInCond.push_back(CenJet);
                    break;
                case FJET:
                    typeInCond.push_back(ForJet);
                    break;
                case TJET:
                    typeInCond.push_back(TauJet);
                    break;
                default:
                    // do nothing - should not arrive here
                    break;
            }

        }

        std::ostringstream myCout;
        std::copy(objectsInComb.begin(), objectsInComb.end(),
                  std::ostream_iterator<int> (myCout, " "));

        // if permutation does not match particle conditions
        // skip spatial correlations
        if ( !tmpResult ) {

            LogTrace("L1GlobalTriggerCaloTemplate")
            << "  Trigger object permutation ( " <<  myCout.str()
            << ") fails single-particle requirements."
            << std::endl;
            continue;
        } else {
            LogTrace("L1GlobalTriggerCaloTemplate")
            << "  Trigger object permutation ( " <<  myCout.str()
            << ") passes single-particle requirements."
            << std::endl;
        }


        if (p_wsc) {

            LogTrace("L1GlobalTriggerCaloTemplate")
            << "\n  Checking spatial correlations: \n"
            << std::endl;

            // wsc requirements have always p_number = 2
            // one can use directly index[0] and index[1] to compute
            // eta and phi differences
            const int ObjInWscComb = 2;
            if (static_cast<int> (p_number) != ObjInWscComb) {
                edm::LogError("L1GlobalTriggerCaloTemplate")
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

            int etaBits = L1GlobalTriggerReadoutSetup::CaloEtaBits;

            int scaleEta = 1 << (etaBits - 1);
            LogTrace("L1GlobalTriggerCaloTemplate")
            << "  scale factor for eta = " << scaleEta << " ( "
            << etaBits << " bits for eta of the object, MSB - sign bit), "
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

            LogTrace("L1GlobalTriggerCaloTemplate")
            << "  delta_eta = " << delta_eta << " for "
            << "v[" << index[0] <<"]->etaIndex() = " << v[index[0]]->etaIndex()
            << " (scaled value " << signedEta[0] << "), "
            << "v[" << index[1] <<"]->etaIndex() = " << v[index[1]]->etaIndex()
            << " (scaled value " << signedEta[1] << "), "
            << std::endl;

            if ( !checkBit(p_conditionparameter.delta_eta, delta_eta) ) {
                LogTrace("L1GlobalTriggerCaloTemplate") << "  object delta_eta: failed"
                << std::endl;
                continue;
            } else {
                LogTrace("L1GlobalTriggerCaloTemplate")
                << "  ==> object delta_eta: passed"
                << std::endl;
            }

            // check delta_phi

            // calculate absolute value of delta_phi
            if (v[index[0]]->phiIndex() > v[index[1]]->phiIndex()) {
                delta_phi = v[index[0]]->phiIndex() - v[index[1]]->phiIndex();
            } else {
                delta_phi = v[index[1]]->phiIndex() - v[index[0]]->phiIndex();
            }
            LogTrace("L1GlobalTriggerCaloTemplate")
            << std::dec << "  delta_phi = " << delta_phi << " for "
            << "v[" << index[0] <<"]->phiIndex() = " << v[index[0]]->phiIndex() << ", "
            << "v[" << index[1] <<"]->phiIndex() = " << v[index[1]]->phiIndex()
            << std::endl;

            // check if delta_phi > 180 (via delta_phi_maxbits)
            // delta_phi contains bits for 0..180 (0 and 180 included)
            while (delta_phi > p_conditionparameter.delta_phi_maxbits) {
                LogTrace("L1GlobalTriggerCaloTemplate")
                << "  delta_phi = " << delta_phi
                << " > p_conditionparameter.delta_phi_maxbits ==> needs re-scaling"
                << std::endl;

                // delta_phi > 180 ==> take 360 - delta_phi
                delta_phi = (p_conditionparameter.delta_phi_maxbits - 1)*2 - delta_phi;
                LogTrace("L1GlobalTriggerCaloTemplate")
                << "  delta_phi changed to: " <<  delta_phi
                << std::endl;
            }


            if (!checkBit(p_conditionparameter.delta_phi, delta_phi)) {
                continue;
            }


        } // end wsc check

        // if we get here all checks were successfull for this combination
        // set the general result for blockCondition remains "true"

        condResult = true;
        passLoops++;
        (*p_combinationsInCond).push_back(objectsInComb);

        LogTrace("L1GlobalTriggerCaloTemplate")
        << "\n  ... Trigger object permutation ( " <<  myCout.str()
        << ") passes all requirements."
        << std::endl;


        //    } while ( std::next_permutation(index, index + maxNumberCands) );
    } while ( std::next_permutation(index.begin(), index.end()) );

    (*p_objectsInCond) = typeInCond;


    LogTrace("L1GlobalTriggerCaloTemplate")
    << "\n  L1GlobalTriggerCaloTemplate: total number of permutations found:          "
    << totalLoops
    << "\n  L1GlobalTriggerCaloTemplate: number of permutations passing requirements: "
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


    LogTrace("L1GlobalTriggerCaloTemplate")
    << "\n  List of combinations passing all requirements for this condition: \n  "
    <<  myCout1.str()
    << " \n"
    << std::endl;

    return condResult;

}

void L1GlobalTriggerCaloTemplate::printThresholds(std::ostream& myCout) const
{

    myCout << "L1GlobalTriggerCaloTemplate: threshold values " << std::endl;
    myCout << "Condition Name: " << getName() << std::endl;

    switch (p_particletype) {
        case EG:
            myCout << "Particle: " << "EG";
            break;
        case IEG:
            myCout << "Particle: " << "IsoEG";
            break;
        case CJET:
            myCout << "Particle: " << "CJet";
            break;
        case FJET:
            myCout << "Particle: " << "FJet";
            break;
        case TJET:
            myCout << "Particle: " << "TJet";
            break;
        default:
            // write nothing - should not arrive here
            break;
    }

    myCout << "\ngreater or equal bit: " << p_ge_eq << std::endl;

    for(unsigned int i = 0; i < p_number; i++) {

        myCout << "\n  TEMPLATE " << i << std::endl;
        myCout << "    et_threshold          "
        << std::hex << p_particleparameter[i].et_threshold
        << std::endl;
        myCout << "    eta                   "
        <<  std::hex << p_particleparameter[i].eta
        << std::endl;
        myCout << "    phi                   "
        <<  std::hex << p_particleparameter[i].phi << std::endl;
    }

    if (p_wsc) {
        myCout << "    Correlation parameters:" << std::endl;
        myCout << "    delta_eta             "
        << std::hex << p_conditionparameter.delta_eta << std::endl;
        myCout << "    delta_eta_maxbits     "
        << std::dec << p_conditionparameter.delta_eta_maxbits << std::endl;
        myCout << "    delta_phi             "
        << std::hex << p_conditionparameter.delta_phi << std::endl;
        myCout << "    delta_phi_maxbits     "
        << std::dec << p_conditionparameter.delta_phi_maxbits << std::endl;
    }

    //reset to decimal output
    myCout << std::dec << std::endl;

}

/**
 * getCandidate - decides what candidate to get using the particletype
 *
 * 
 * @param indexCand The number of the candidate
 * @return A reference to the candidate
 *
 */

L1GctCand* L1GlobalTriggerCaloTemplate::getCandidate (int indexCand) const
{

    switch (p_particletype) {
        case EG:
            return (*m_GT.gtPSB()->getCandL1NoIsoEG())[indexCand];
            break;
        case IEG:
            return (*m_GT.gtPSB()->getCandL1IsoEG())[indexCand];
            break;
        case CJET:
            return (*m_GT.gtPSB()->getCandL1CenJet())[indexCand];
            break;
        case FJET:
            return (*m_GT.gtPSB()->getCandL1ForJet())[indexCand];
            break;
        case TJET:
            return (*m_GT.gtPSB()->getCandL1TauJet())[indexCand];
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
    int ncondition, L1GctCand &cand) const
{

    std::string checkFalse = "\n  ==> checkParticle = false ";

    if (ncondition >= static_cast<int> (p_number) || ncondition < 0) {
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
    LogTrace("L1GlobalTriggerCaloTemplate") << "  calo object eta check:"
    << " etaIndex() = "
    << std::dec << cand.etaIndex() << " (dec) "
    << std::hex << cand.etaIndex() << " (hex) "
    << std::dec
    << std::endl;

    if (!checkBit(p_particleparameter[ncondition].eta, cand.etaIndex())) {
        LogTrace("L1GlobalTriggerCaloTemplate")
        << "  calo object eta: failed"
        << checkFalse
        << std::endl;
        return false;
    } else {
        LogTrace("L1GlobalTriggerCaloTemplate")
        << "  ==> calo object eta: passed"
        << std::endl;
    }


    // check phi
    LogTrace("L1GlobalTriggerCaloTemplate") << "  calo object phi check:"
    << " phiIndex() = "
    << std::dec << cand.phiIndex() << " (dec) "
    << std::hex << cand.phiIndex() << " (hex) "
    << std::dec
    << std::endl;

    if (!checkBit(p_particleparameter[ncondition].phi, cand.phiIndex())) {
        LogTrace("L1GlobalTriggerCaloTemplate")
        << "  calo object phi: failed"
        << checkFalse
        << std::endl;
        return false;
    } else {
        LogTrace("L1GlobalTriggerCaloTemplate")
        << "  ==> calo object phi: passed"
        << std::endl;
    }

    // particle matches if we get here
    LogTrace("L1GlobalTriggerCaloTemplate")
    << "  checkParticle: calorimeter object OK, passes all requirements\n"
    << std::endl;

    return true;
}

