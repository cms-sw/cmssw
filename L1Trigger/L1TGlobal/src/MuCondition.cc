/**
 * \class MuCondition
 *
 *
 * Description: evaluation of a CondMuon condition.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 *
 */

// this class header
#include "L1Trigger/L1TGlobal/interface/MuCondition.h"

// system include files
#include <iostream>
#include <iomanip>

#include <string>
#include <vector>
#include <algorithm>

// user include files
//   base classes
#include "L1Trigger/L1TGlobal/interface/MuonTemplate.h"
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "DataFormats/L1Trigger/interface/Muon.h"

#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFunctions.h"
#include "L1Trigger/L1TGlobal/interface/GtBoard.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors
//     default
l1t::MuCondition::MuCondition() :
    ConditionEvaluation() {

    // empty

}

//     from base template condition (from event setup usually)
l1t::MuCondition::MuCondition(const GtCondition* muonTemplate,
        const GtBoard* ptrGTL, const int nrL1Mu,
        const int ifMuEtaNumberBits) :
    ConditionEvaluation(),
    m_gtMuonTemplate(static_cast<const MuonTemplate*>(muonTemplate)),
    m_gtGTL(ptrGTL),
    m_ifMuEtaNumberBits(ifMuEtaNumberBits)
{
    m_corrParDeltaPhiNrBins = 0;
    m_condMaxNumberObjects = nrL1Mu;
}

// copy constructor
void l1t::MuCondition::copy(const l1t::MuCondition &cp) {

    m_gtMuonTemplate = cp.gtMuonTemplate();
    m_gtGTL = cp.gtGTL();

    m_ifMuEtaNumberBits = cp.gtIfMuEtaNumberBits();
    m_corrParDeltaPhiNrBins = cp.m_corrParDeltaPhiNrBins;

    m_condMaxNumberObjects = cp.condMaxNumberObjects();
    m_condLastResult = cp.condLastResult();
    m_combinationsInCond = cp.getCombinationsInCond();

    m_verbosity = cp.m_verbosity;

}

l1t::MuCondition::MuCondition(const l1t::MuCondition& cp) :
    ConditionEvaluation() {
    copy(cp);
}

// destructor
l1t::MuCondition::~MuCondition() {

    // empty

}

// equal operator
l1t::MuCondition& l1t::MuCondition::operator= (const l1t::MuCondition& cp)
{
    copy(cp);
    return *this;
}

// methods
void l1t::MuCondition::setGtMuonTemplate(const MuonTemplate* muonTempl) {

    m_gtMuonTemplate = muonTempl;

}

///   set the pointer to GTL
void l1t::MuCondition::setGtGTL(const GtBoard* ptrGTL) {

    m_gtGTL = ptrGTL;

}


//   set the number of bits for eta of muon objects
void l1t::MuCondition::setGtIfMuEtaNumberBits(
        const int& ifMuEtaNumberBitsValue) {

    m_ifMuEtaNumberBits = ifMuEtaNumberBitsValue;

}

//   set the maximum number of bins for the delta phi scales
void l1t::MuCondition::setGtCorrParDeltaPhiNrBins(
        const int& corrParDeltaPhiNrBins) {

    m_corrParDeltaPhiNrBins = corrParDeltaPhiNrBins;

}


// try all object permutations and check spatial correlations, if required
const bool l1t::MuCondition::evaluateCondition(const int bxEval) const {  
 
    // BLW Need to pass this as an argument
    //const int bxEval=0;   //BLW Change for BXVector

    // number of trigger objects in the condition
    int nObjInCond = m_gtMuonTemplate->nrObjects();

    // the candidates
    const BXVector<const l1t::Muon*>* candVec = m_gtGTL->getCandL1Mu();  //BLW Change for BXVector

    // Look at objects in bx = bx + relativeBx
    int useBx = bxEval + m_gtMuonTemplate->condRelativeBx();

    // Fail condition if attempting to get Bx outside of range
    if( ( useBx < candVec->getFirstBX() ) ||
	( useBx > candVec->getLastBX() ) ) {
      return false;
    }

    int numberObjects = candVec->size(useBx);  //BLW Change for BXVector
    //LogTrace("l1t|Global") << "  numberObjects: " << numberObjects
    //    << std::endl;
    if (numberObjects < nObjInCond) {
        return false;
    }

    std::vector<int> index(numberObjects);

    for (int i = 0; i < numberObjects; ++i) {
        index[i] = i;
    }

    int numberForFactorial = numberObjects - nObjInCond;

    // TEMPORARY FIX UNTIL IMPLEMENT NEW MUON CONDITIONS
    int myfactorial = 1;
    for( int i=numberForFactorial; i>0; i-- ) myfactorial *= i;

    int jumpIndex = 1;
    int jump = myfactorial;//factorial(numberObjects - nObjInCond);

    int totalLoops = 0;
    int passLoops = 0;

    // condition result condResult set to true if at least one permutation
    //     passes all requirements
    // all possible permutations are checked
    bool condResult = false;

    // store the indices of the muon objects
    // from the combination evaluated in the condition
    SingleCombInCond objectsInComb;
    objectsInComb.reserve(nObjInCond);

    // clear the m_combinationsInCond vector
    combinationsInCond().clear();

    do {

        if (--jumpIndex)
            continue;

        jumpIndex = jump;
        totalLoops++;

        // clear the indices in the combination
        objectsInComb.clear();

        bool tmpResult = true;

	bool passCondition = false;
        // check if there is a permutation that matches object-parameter requirements
        for (int i = 0; i < nObjInCond; i++) {

	    passCondition = checkObjectParameter(i,  *(candVec->at(useBx,index[i]) )); //BLW Change for BXVector
	    tmpResult &= passCondition;
	    if( passCondition ) 
	      LogDebug("l1t|Global") << "===> MuCondition::evaluateCondition, CONGRATS!! This muon passed the condition." << std::endl;
	    else 
	      LogDebug("l1t|Global") << "===> MuCondition::evaluateCondition, FAIL!! This muon failed the condition." << std::endl;
            objectsInComb.push_back(index[i]);

        }


        // if permutation does not match particle conditions
        // skip charge correlation and spatial correlations
        if ( !tmpResult) {

            continue;

        }

        // get the correlation parameters (chargeCorrelation included here also)
        MuonTemplate::CorrelationParameter corrPar =
            *(m_gtMuonTemplate->correlationParameter());

        // charge_correlation consists of 3 relevant bits (D2, D1, D0)
        unsigned int chargeCorr = corrPar.chargeCorrelation;

        // charge ignore bit (D0) not set?
        if ((chargeCorr & 1) == 0) {

	  for (int i = 0; i < nObjInCond; i++) {
	    // check valid charge - skip if invalid charge
	    int chargeValid = (candVec->at(useBx,index[i]))->hwChargeValid(); //BLW Change for BXVector
	    tmpResult &= chargeValid;

	    if ( chargeValid==0) { //BLW type change for New Muon Class
	      continue;
	    }
	  }

	  if ( !tmpResult) {
	    continue;
	  }


	  if( nObjInCond > 1 ){ // more objects condition

	    // find out if signs are equal
	    bool equalSigns = true;
	    for (int i = 0; i < nObjInCond-1; i++) {
	      if ((candVec->at(useBx,index[i]))->charge() != (candVec->at(useBx,index[i+1]))->charge()) { //BLW Change for BXVector
		equalSigns = false;
		break;
	      }
	    }

	    // two or three particle condition
	    if (nObjInCond == 2 || nObjInCond == 3) {
	      if( !( ((chargeCorr & 2)!=0 && equalSigns) || ((chargeCorr & 4)!=0 && !equalSigns) ) ){
		continue;
	      }
	    }
	    else if (nObjInCond == 4) {
	      //counter to count positive charges to determine if there are pairs
	      unsigned int posCount = 0;

	      for (int i = 0; i < nObjInCond; i++) {
		if ((candVec->at(useBx,index[i]))->charge()> 0) {  //BLW Change for BXVector
		  posCount++;
		}
	      }

	      if( !( ((chargeCorr & 2)!=0 && equalSigns) || ((chargeCorr & 4)!=0 && posCount==2) ) ){
		continue;
	      }
	    }
	  } // end require nObjInCond > 1
        } // end signchecks


        if (m_gtMuonTemplate->wsc()) {

            // wsc requirements have always nObjInCond = 2
            // one can use directly index[0] and index[1] to compute
            // eta and phi differences
            const int ObjInWscComb = 2;
            if (nObjInCond != ObjInWscComb) {

                edm::LogError("l1t|Global") << "\n  Error: "
                    << "number of particles in condition with spatial correlation = " << nObjInCond
                    << "\n  it must be = " << ObjInWscComb << std::endl;
                // TODO Perhaps I should throw here an exception,
                // since something is really wrong if nObjInCond != ObjInWscComb (=2)
                continue;
            }

	    // check delta eta
	    if( !checkRangeDeltaEta( (candVec->at(useBx,0))->hwEta(), (candVec->at(useBx,1))->hwEta(), corrPar.deltaEtaRangeLower, corrPar.deltaEtaRangeUpper, 8) ){
	      LogDebug("l1t|Global") << "\t\t l1t::Candidate failed checkRangeDeltaEta" << std::endl;
	      continue;
	    }

	    // check delta phi
	    if( !checkRangeDeltaPhi( (candVec->at(useBx,0))->hwPhi(), (candVec->at(useBx,1))->hwPhi(), 
				     corrPar.deltaPhiRangeLower, corrPar.deltaPhiRangeUpper) ){
	      LogDebug("l1t|Global") << "\t\t l1t::Candidate failed checkRangeDeltaPhi" << std::endl;
	      continue;
	    }

        } // end wsc check

        // if we get here all checks were successfull for this combination
        // set the general result for evaluateCondition to "true"

        condResult = true;
        passLoops++;
        (combinationsInCond()).push_back(objectsInComb);

    } while (std::next_permutation(index.begin(), index.end()) );

    //LogTrace("l1t|Global")
    //    << "\n  MuCondition: total number of permutations found:          " << totalLoops
    //    << "\n  MuCondition: number of permutations passing requirements: " << passLoops
    //    << "\n" << std::endl;

    return condResult;

}

// load muon candidates
const l1t::Muon* l1t::MuCondition::getCandidate(const int bx, const int indexCand) const {

    return (m_gtGTL->getCandL1Mu())->at(bx,indexCand);  //BLW Change for BXVector
}

/**
 * checkObjectParameter - Compare a single particle with a numbered condition.
 *
 * @param iCondition The number of the condition.
 * @param cand The candidate to compare.
 *
 * @return The result of the comparison (false if a condition does not exist).
 */

const bool l1t::MuCondition::checkObjectParameter(const int iCondition, const l1t::Muon& cand) const {

    // number of objects in condition
    int nObjInCond = m_gtMuonTemplate->nrObjects();

    if (iCondition >= nObjInCond || iCondition < 0) {
        return false;
    }

//     // empty candidates can not be compared
//     if (cand.empty()) {
//         return false;
//     }

    const MuonTemplate::ObjectParameter objPar =
        ( *(m_gtMuonTemplate->objectParameter()) )[iCondition];

    // using the logic table from GTL-9U-module.pdf
    // "Truth table for Isolation bit"

    // check thresholds:

    //   value < low pt threshold
    //       fail trigger

    //   low pt threshold <= value < high pt threshold & non-isolated muon:
    //       requestIso true:                    fail trigger
    //       requestIso false, enableIso true:   fail trigger
    //       requestIso false, enableIso false:  OK,  trigger

    //   low pt threshold <= value < high pt threshold & isolated muon:
    //       requestIso true:                    OK,  trigger
    //       requestIso false, enableIso true:   OK,  trigger
    //       requestIso false, enableIso false:  OK,  trigger

    //   value >= high pt threshold & non-isolated muon:
    //       requestIso true:  fail trigger
    //       requestIso false: OK,  trigger

    //   value >= high pt threshold & isolated muon:
    //       OK, trigger

    LogDebug("l1t|Global")
      << "\n MuonTemplate::ObjectParameter : "
      << "\n\t ptHighThreshold = " << objPar.ptHighThreshold 
      << "\n\t ptLowThreshold  = " << objPar.ptLowThreshold
      << "\n\t requestIso      = " << objPar.requestIso
      << "\n\t enableIso       = " << objPar.enableIso
      << "\n\t etaRange        = " << objPar.etaRange
      << "\n\t phiLow          = " << objPar.phiLow
      << "\n\t phiHigh         = " << objPar.phiHigh
      << "\n\t phiWindowLower  = " << objPar.phiWindowLower
      << "\n\t phiWindowUpper  = " << objPar.phiWindowUpper
      << "\n\t phiWindowVetoLower = " << objPar.phiWindowVetoLower
      << "\n\t phiWindowVetoLower = " << objPar.phiWindowVetoLower
      << "\n\t charge          = " << objPar.charge
      << "\n\t qualityLUT      = " << objPar.qualityLUT
      << "\n\t isolationLUT    = " << objPar.isolationLUT
      << "\n\t enableMip       = " << objPar.enableMip
      << std::endl;

    LogDebug("l1t|Global")
      << "\n l1t::Muon : "
      << "\n\t hwPt   = " <<  cand.hwPt()
      << "\n\t hwEta  = " << cand.hwEta()
      << "\n\t hwPhi  = " << cand.hwPhi()
      << "\n\t hwCharge = " << cand.hwCharge()
      << "\n\t hwQual = " << cand.hwQual()
      << "\n\t hwIso  = " << cand.hwIso()
      << std::endl;


    if ( !checkThreshold(objPar.ptHighThreshold, cand.hwPt(), m_gtMuonTemplate->condGEq()) ) {
      if ( !checkThreshold(objPar.ptLowThreshold, cand.hwPt(), m_gtMuonTemplate->condGEq()) ) {
	LogDebug("l1t|Global") << "\t\t Muon Failed checkThreshold " << std::endl;
	return false;
      }
    }


    // check eta
    if( !checkRangeEta(cand.hwEta(), objPar.etaWindowLower, objPar.etaWindowUpper, objPar.etaWindowVetoLower, objPar.etaWindowVetoLower, 8) ){
      LogDebug("l1t|Global") << "\t\t l1t::Candidate failed checkRange(eta)" << std::endl;
      return false;
    }

    // check phi
    if( !checkRangePhi(cand.hwPhi(), objPar.phiWindowLower, objPar.phiWindowUpper, objPar.phiWindowVetoLower, objPar.phiWindowVetoLower) ){
      LogDebug("l1t|Global") << "\t\t l1t::Candidate failed checkRange(phi)" << std::endl;
      return false;
    }

    // check charge
    if( objPar.charge>=0 ){
      if( cand.hwCharge() != objPar.charge ){
	LogDebug("l1t|Global") << "\t\t l1t::Candidate failed charge requirement" << std::endl;
	return false;
      }
    }



    // check quality ( bit check ) with quality LUT
    // sanity check on candidate quality
    if( cand.hwQual()>16 ){
      LogDebug("l1t|Global") << "\t\t l1t::Candidate has out of range hwQual = " << cand.hwQual() << std::endl;
      return false;
    }
    bool passQualLUT = ( (objPar.qualityLUT >> cand.hwQual()) & 1 );
    if( !passQualLUT ){
      LogDebug("l1t|Global") << "\t\t l1t::Candidate failed quality requirement" << std::endl;
      return false;
    }


    // check isolation ( bit check ) with isolation LUT
    // sanity check on candidate isolation
    if( cand.hwIso()>4 ){
      LogDebug("l1t|Global") << "\t\t l1t::Candidate has out of range hwIso = " << cand.hwIso() << std::endl;
      return false;
    }
    bool passIsoLUT = ( (objPar.isolationLUT >> cand.hwIso()) & 1 );
    if( !passIsoLUT ){
      LogDebug("l1t|Global") << "\t\t l1t::Candidate failed isolation requirement" << std::endl;
      return false;
    }

    // A number of values is required to trigger (at least one).
    // "Don’t care" means that all values are allowed.
    // Qual = 000 means then NO MUON (GTL module)

    // if (cand.hwQual() == 0) {
    // 	LogDebug("l1t|Global") << "\t\t Muon Failed hwQual() == 0" << std::endl;
    //     return false;
    // }

    // if (objPar.qualityRange == 0) {
    // 	LogDebug("l1t|Global") << "\t\t Muon Failed qualityRange == 0" << std::endl;
    //     return false;
    // }
    // else {
    //   if (!checkBit(objPar.qualityRange, cand.hwQual())) {
    // 	LogDebug("l1t|Global") << "\t\t Muon Failed checkBit(qualityRange) " << std::endl;
    //         return false;
    //     }
    // }

    // check mip
    if (objPar.enableMip) {
   //      if (!cand.hwMip()) {
	  // LogDebug("l1t|Global") << "\t\t Muon Failed enableMip" << std::endl;
   //          return false;
   //      }
    }

    // particle matches if we get here
    //LogTrace("l1t|Global")
    //    << "  checkObjectParameter: muon object OK, passes all requirements\n" << std::endl;

    return true;
}

void l1t::MuCondition::print(std::ostream& myCout) const {

    m_gtMuonTemplate->print(myCout);

    myCout << "    Number of bits for eta of muon objects = "
            << m_ifMuEtaNumberBits << std::endl;
    myCout << "    Maximum number of bins for the delta phi scales = "
            << m_corrParDeltaPhiNrBins << "\n " << std::endl;

    ConditionEvaluation::print(myCout);

}

