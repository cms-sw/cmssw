/**
 * \class CorrWithOverlapRemovalCondition
 *
 *
 * Description: evaluation of a correlation-with-overlap-removal condition.
 *
 *
 * Implementation:
 *
 *  The central method of the class is evaluateCondition().
 *  Correlation cuts inherited from the original CorrCondition class.
 *
 *  v1 Logic:
 *  - Define 3 GtConditionCategories: (1st & 2nd correlation legs, and overlap-removal leg)
 *
 *  - Loop leg1: over all objects of 1st correlation leg
 *
 *    - Retrive its coordinates, and do conversions depending on the types
 *
 *    - Loop over all objects of leg3 (overlap-removal leg)
 *      - Retrive its coodrinates, and do conversions depending on the types
 *      - Check for matching with overlap-removal object.
 *    - If metched with any overlap object, next leg1 object.
 *
 *    - Loop leg2: over all object of 2nd correlation leg
 *      - Retrive its coordinates, and do conversion depending on the types
 *      - Loop over all objects of leg3 (overlap-removal leg)
 *        - Retrive its coodrinates, and do conversions depending on the types
 *        - Check for matching with any overlap-removal object.
 *      - If metched with any overlap object, next leg2 object.
 *      - Check for dEta, dPhi, dR, and mass (and charge) correlation with 1st leg object.
 *        If any correlation cut pass, save leg1-leg2 object combination.
 *    - End loop leg2
 *
 *  - End loop leg1
 *
 *  - Return true, if saved at least one leg1-leg2 combintation
 *
 *
 * \author: Vladimir Rekovic
 *
 */

// this class header
#include "L1Trigger/L1TGlobal/interface/CorrWithOverlapRemovalCondition.h"

// system include files
#include <iostream>
#include <iomanip>

#include <string>
#include <vector>
#include <algorithm>

// user include files
//   base classes
#include "L1Trigger/L1TGlobal/interface/CorrelationWithOverlapRemovalTemplate.h"
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"

#include "L1Trigger/L1TGlobal/interface/MuCondition.h"
#include "L1Trigger/L1TGlobal/interface/CaloCondition.h"
#include "L1Trigger/L1TGlobal/interface/EnergySumCondition.h"
#include "L1Trigger/L1TGlobal/interface/MuonTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CaloTemplate.h"
#include "L1Trigger/L1TGlobal/interface/EnergySumTemplate.h"
#include "L1Trigger/L1TGlobal/interface/GlobalScales.h"

#include "DataFormats/L1Trigger/interface/L1Candidate.h"

#include "L1Trigger/L1TGlobal/interface/GlobalBoard.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors
//     default
l1t::CorrWithOverlapRemovalCondition::CorrWithOverlapRemovalCondition() : ConditionEvaluation() {}

//     from base template condition (from event setup usually)
l1t::CorrWithOverlapRemovalCondition::CorrWithOverlapRemovalCondition(const GlobalCondition* corrTemplate,
                                                                      const GlobalCondition* cond0Condition,
                                                                      const GlobalCondition* cond1Condition,
                                                                      const GlobalCondition* cond2Condition,
                                                                      const GlobalBoard* ptrGTB)
    : ConditionEvaluation(),
      m_gtCorrelationWithOverlapRemovalTemplate(
          static_cast<const CorrelationWithOverlapRemovalTemplate*>(corrTemplate)),
      m_gtCond0(cond0Condition),
      m_gtCond1(cond1Condition),
      m_gtCond2(cond2Condition),
      m_uGtB(ptrGTB) {}

// copy constructor
void l1t::CorrWithOverlapRemovalCondition::copy(const l1t::CorrWithOverlapRemovalCondition& cp) {
  m_gtCorrelationWithOverlapRemovalTemplate = cp.gtCorrelationWithOverlapRemovalTemplate();
  m_uGtB = cp.getuGtB();

  m_condMaxNumberObjects = cp.condMaxNumberObjects();
  m_condLastResult = cp.condLastResult();
  m_combinationsInCond = cp.getCombinationsInCond();

  m_verbosity = cp.m_verbosity;
}

l1t::CorrWithOverlapRemovalCondition::CorrWithOverlapRemovalCondition(const l1t::CorrWithOverlapRemovalCondition& cp)
    : ConditionEvaluation() {
  copy(cp);
}

// destructor
l1t::CorrWithOverlapRemovalCondition::~CorrWithOverlapRemovalCondition() {
  // empty
}

// equal operator
l1t::CorrWithOverlapRemovalCondition& l1t::CorrWithOverlapRemovalCondition::operator=(
    const l1t::CorrWithOverlapRemovalCondition& cp) {
  copy(cp);
  return *this;
}

// methods
void l1t::CorrWithOverlapRemovalCondition::setGtCorrelationWithOverlapRemovalTemplate(
    const CorrelationWithOverlapRemovalTemplate* caloTempl) {
  m_gtCorrelationWithOverlapRemovalTemplate = caloTempl;
}

///   set the pointer to uGT GlobalBoard
void l1t::CorrWithOverlapRemovalCondition::setuGtB(const GlobalBoard* ptrGTB) { m_uGtB = ptrGTB; }

void l1t::CorrWithOverlapRemovalCondition::setScales(const GlobalScales* sc) { m_gtScales = sc; }

// try all object permutations and check spatial correlations, if required
const bool l1t::CorrWithOverlapRemovalCondition::evaluateCondition(const int bxEval) const {
  // std::cout << "m_isDebugEnabled = " << m_isDebugEnabled << std::endl;
  // std::cout << "m_verbosity = " << m_verbosity << std::endl;

  //std::ostringstream myCout;
  //m_gtCorrelationWithOverlapRemovalTemplate->print(myCout);
  //LogDebug("L1TGlobal")
  //   << "CorrelationWithOverlapRemoval Condition Evaluation \n" << myCout.str() << std::endl;

  bool condResult = false;
  bool reqObjResult = false;

  // number of objects in condition (it is 3, no need to retrieve from
  // condition template) and their type
  int nObjInCond = 3;
  std::vector<GlobalObject> cndObjTypeVec(nObjInCond);

  // evaluate first the two sub-conditions (Type1s)

  const GtConditionCategory cond0Categ = m_gtCorrelationWithOverlapRemovalTemplate->cond0Category();
  const GtConditionCategory cond1Categ = m_gtCorrelationWithOverlapRemovalTemplate->cond1Category();
  const GtConditionCategory cond2Categ = m_gtCorrelationWithOverlapRemovalTemplate->cond2Category();

  //Decide if we have a mixed (muon + cal) condition
  bool convertCaloScales = false;
  if ((cond0Categ == CondMuon && (cond1Categ == CondCalo || cond1Categ == CondEnergySum)) ||
      (cond1Categ == CondMuon && (cond0Categ == CondCalo || cond0Categ == CondEnergySum)))
    convertCaloScales = true;

  bool convertCaloScalesForOverlapRemovalFromLeg0 = false;
  if ((cond0Categ == CondMuon && (cond2Categ == CondCalo || cond2Categ == CondEnergySum)) ||
      (cond2Categ == CondMuon && (cond0Categ == CondCalo || cond0Categ == CondEnergySum)))
    convertCaloScalesForOverlapRemovalFromLeg0 = true;

  bool convertCaloScalesForOverlapRemovalFromLeg1 = false;
  if ((cond1Categ == CondMuon && (cond2Categ == CondCalo || cond2Categ == CondEnergySum)) ||
      (cond2Categ == CondMuon && (cond1Categ == CondCalo || cond1Categ == CondEnergySum)))
    convertCaloScalesForOverlapRemovalFromLeg1 = true;

  const MuonTemplate* corrMuon = nullptr;
  const CaloTemplate* corrCalo = nullptr;
  const EnergySumTemplate* corrEnergySum = nullptr;

  // FIXME copying is slow...
  CombinationsInCond cond0Comb;
  CombinationsInCond cond1Comb;
  CombinationsInCond cond2Comb;

  int cond0bx(0);
  int cond1bx(0);
  int cond2bx(0);

  switch (cond0Categ) {
    case CondMuon: {
      corrMuon = static_cast<const MuonTemplate*>(m_gtCond0);
      MuCondition muCondition(
          corrMuon, m_uGtB, 0, 0);  //BLW these are counts that don't seem to be used...perhaps remove

      muCondition.evaluateConditionStoreResult(bxEval);
      reqObjResult = muCondition.condLastResult();

      cond0Comb = (muCondition.getCombinationsInCond());
      cond0bx = bxEval + (corrMuon->condRelativeBx());

      cndObjTypeVec[0] = (corrMuon->objectType())[0];

      if (m_verbosity) {
        std::ostringstream myCout;
        muCondition.print(myCout);

        LogDebug("L1TGlobal") << myCout.str() << std::endl;
      }
    } break;
    case CondCalo: {
      corrCalo = static_cast<const CaloTemplate*>(m_gtCond0);

      CaloCondition caloCondition(
          corrCalo, m_uGtB, 0, 0, 0, 0);  //BLW these are counters that don't seem to be used...perhaps remove.

      caloCondition.evaluateConditionStoreResult(bxEval);
      reqObjResult = caloCondition.condLastResult();

      cond0Comb = (caloCondition.getCombinationsInCond());
      cond0bx = bxEval + (corrCalo->condRelativeBx());

      cndObjTypeVec[0] = (corrCalo->objectType())[0];

      if (m_verbosity) {
        std::ostringstream myCout;
        caloCondition.print(myCout);

        LogDebug("L1TGlobal") << myCout.str() << std::endl;
      }
    } break;
    case CondEnergySum: {
      corrEnergySum = static_cast<const EnergySumTemplate*>(m_gtCond0);
      EnergySumCondition eSumCondition(corrEnergySum, m_uGtB);

      eSumCondition.evaluateConditionStoreResult(bxEval);
      reqObjResult = eSumCondition.condLastResult();

      cond0Comb = (eSumCondition.getCombinationsInCond());
      cond0bx = bxEval + (corrEnergySum->condRelativeBx());

      cndObjTypeVec[0] = (corrEnergySum->objectType())[0];

      if (m_verbosity) {
        std::ostringstream myCout;
        eSumCondition.print(myCout);

        LogDebug("L1TGlobal") << myCout.str() << std::endl;
      }
    } break;
    default: {
      // should not arrive here, there are no correlation conditions defined for this object
      return false;
    } break;
  }

  // return if first subcondition is false
  if (!reqObjResult) {
    LogDebug("L1TGlobal") << "\n  First sub-condition false, second sub-condition not evaluated and not printed."
                          << std::endl;
    return false;
  }

  // second object
  reqObjResult = false;

  switch (cond1Categ) {
    case CondMuon: {
      corrMuon = static_cast<const MuonTemplate*>(m_gtCond1);
      MuCondition muCondition(
          corrMuon, m_uGtB, 0, 0);  //BLW these are counts that don't seem to be used...perhaps remove

      muCondition.evaluateConditionStoreResult(bxEval);
      reqObjResult = muCondition.condLastResult();

      cond1Comb = (muCondition.getCombinationsInCond());
      cond1bx = bxEval + (corrMuon->condRelativeBx());
      cndObjTypeVec[1] = (corrMuon->objectType())[0];

      if (m_verbosity) {
        std::ostringstream myCout;
        muCondition.print(myCout);

        LogDebug("L1TGlobal") << myCout.str() << std::endl;
      }
    } break;
    case CondCalo: {
      corrCalo = static_cast<const CaloTemplate*>(m_gtCond1);
      CaloCondition caloCondition(
          corrCalo, m_uGtB, 0, 0, 0, 0);  //BLW these are counters that don't seem to be used...perhaps remove.

      caloCondition.evaluateConditionStoreResult(bxEval);
      reqObjResult = caloCondition.condLastResult();

      cond1Comb = (caloCondition.getCombinationsInCond());
      cond1bx = bxEval + (corrCalo->condRelativeBx());

      cndObjTypeVec[1] = (corrCalo->objectType())[0];

      if (m_verbosity) {
        std::ostringstream myCout;
        caloCondition.print(myCout);

        LogDebug("L1TGlobal") << myCout.str() << std::endl;
      }

    } break;
    case CondEnergySum: {
      corrEnergySum = static_cast<const EnergySumTemplate*>(m_gtCond1);

      EnergySumCondition eSumCondition(corrEnergySum, m_uGtB);

      eSumCondition.evaluateConditionStoreResult(bxEval);
      reqObjResult = eSumCondition.condLastResult();

      cond1Comb = (eSumCondition.getCombinationsInCond());
      cond1bx = bxEval + (corrEnergySum->condRelativeBx());
      cndObjTypeVec[1] = (corrEnergySum->objectType())[0];

      if (m_verbosity) {
        std::ostringstream myCout;
        eSumCondition.print(myCout);

        LogDebug("L1TGlobal") << myCout.str() << std::endl;
      }
    } break;
    default: {
      // should not arrive here, there are no correlation conditions defined for this object
      return false;
    } break;
  }

  // return if second sub-condition is false
  if (!reqObjResult) {
    return false;
  } else {
    LogDebug("L1TGlobal") << "\n"
                          << "    Both sub-conditions true for object requirements."
                          << "    Evaluate correlation requirements.\n"
                          << std::endl;
  }

  // third object (used for overlap removal)
  reqObjResult = false;

  switch (cond2Categ) {
    case CondMuon: {
      corrMuon = static_cast<const MuonTemplate*>(m_gtCond2);
      MuCondition muCondition(
          corrMuon, m_uGtB, 0, 0);  //BLW these are counts that don't seem to be used...perhaps remove

      muCondition.evaluateConditionStoreResult(bxEval);
      reqObjResult = muCondition.condLastResult();

      cond2Comb = (muCondition.getCombinationsInCond());
      cond2bx = bxEval + (corrMuon->condRelativeBx());
      cndObjTypeVec[2] = (corrMuon->objectType())[0];

      if (m_verbosity) {
        std::ostringstream myCout;
        muCondition.print(myCout);

        LogDebug("L1TGlobal") << myCout.str() << std::endl;
      }
    } break;
    case CondCalo: {
      corrCalo = static_cast<const CaloTemplate*>(m_gtCond2);
      CaloCondition caloCondition(
          corrCalo, m_uGtB, 0, 0, 0, 0);  //BLW these are counters that don't seem to be used...perhaps remove.

      caloCondition.evaluateConditionStoreResult(bxEval);
      reqObjResult = caloCondition.condLastResult();

      cond2Comb = (caloCondition.getCombinationsInCond());
      cond2bx = bxEval + (corrCalo->condRelativeBx());
      cndObjTypeVec[2] = (corrCalo->objectType())[0];

      if (m_verbosity) {
        std::ostringstream myCout;
        caloCondition.print(myCout);

        LogDebug("L1TGlobal") << myCout.str() << std::endl;
      }

    } break;
    case CondEnergySum: {
      corrEnergySum = static_cast<const EnergySumTemplate*>(m_gtCond2);

      EnergySumCondition eSumCondition(corrEnergySum, m_uGtB);

      eSumCondition.evaluateConditionStoreResult(bxEval);
      reqObjResult = eSumCondition.condLastResult();

      cond2Comb = (eSumCondition.getCombinationsInCond());
      cond2bx = bxEval + (corrEnergySum->condRelativeBx());
      cndObjTypeVec[2] = (corrEnergySum->objectType())[0];

      if (m_verbosity) {
        std::ostringstream myCout;
        eSumCondition.print(myCout);

        LogDebug("L1TGlobal") << myCout.str() << std::endl;
      }
    } break;
    default: {
      // should not arrive here, there are no correlation conditions defined for this object
      return false;
    } break;
  }

  // if third sub-condition is false, effectively there will no overlap removal
  if (!reqObjResult) {
    LogDebug("L1TGlobal") << "\n"
                          << "    Third sub-condtion false for object requirements."
                          << "    Algorithm returning false.\n"
                          << std::endl;
    return false;
  } else {
    LogDebug("L1TGlobal") << "\n"
                          << "    All three sub-conditions true for object requirements."
                          << "    Evaluate correlation requirements and overlap removal.\n"
                          << std::endl;
  }

  // since we have two good legs and overlap-removal let, get the correlation parameters
  CorrelationWithOverlapRemovalTemplate::CorrelationWithOverlapRemovalParameter corrPar =
      *(m_gtCorrelationWithOverlapRemovalTemplate->correlationParameter());

  // vector to store the indices of the calorimeter objects
  // from the combination evaluated in the condition
  SingleCombInCond objectsInComb;
  objectsInComb.reserve(nObjInCond);

  // clear the m_combinationsInCond vector
  (combinationsInCond()).clear();

  // pointers to objects
  const BXVector<const l1t::Muon*>* candMuVec = nullptr;
  const BXVector<const l1t::L1Candidate*>* candCaloVec = nullptr;
  const BXVector<const l1t::EtSum*>* candEtSumVec = nullptr;

  bool etSumCond = false;

  // make the conversions of the indices, depending on the combination of objects involved
  // (via pair index)

  int phiIndex0 = 0;
  int phiIndex1 = 0;
  int phiORIndex0 = 0;  // hold phi index transformed in case of need with overlap-removal
  int phiORIndex1 = 0;  // hold phi index transformed in case of need with overlap-removal
  double phi0Phy = 0.;
  double phi1Phy = 0.;

  int etaIndex0 = 0;
  int etaIndex1 = 0;
  int etaORIndex0 = 0;
  int etaORIndex1 = 0;
  double eta0Phy = 0.;
  double eta1Phy = 0.;
  int etaBin0 = 0;
  int etaBin1 = 0;

  int etIndex0 = 0;
  int etIndex1 = 0;
  int etBin0 = 0;
  int etBin1 = 0;
  double et0Phy = 0.;
  double et1Phy = 0.;

  int chrg0 = -1;
  int chrg1 = -1;

  // make the conversions of the indices, depending on the combination of objects involved in overlap-removal
  int phiIndex2 = 0;
  int etaIndex2 = 0;
  double phi2Phy = 0.;
  double eta2Phy = 0.;
  int etaBin2 = 0;
  //int etIndex2  = 0;
  //int etBin2    = 0;

  // Determine the number of phi bins to get cutoff at pi
  int phiBound = 0;
  if (cond0Categ == CondMuon || cond1Categ == CondMuon || cond2Categ == CondMuon) {
    GlobalScales::ScaleParameters par = m_gtScales->getMUScales();
    //phiBound = par.phiBins.size()/2;
    phiBound = (int)((par.phiMax - par.phiMin) / par.phiStep) / 2;
  } else {
    //Assumes all calorimeter objects are on same phi scale
    GlobalScales::ScaleParameters par = m_gtScales->getEGScales();
    //phiBound = par.phiBins.size()/2;
    phiBound = (int)((par.phiMax - par.phiMin) / par.phiStep) / 2;
  }
  LogDebug("L1TGlobal") << "Phi Bound = " << phiBound << std::endl;

  // Keep track of objects for LUTS
  std::string lutObj0 = "NULL";
  std::string lutObj1 = "NULL";
  std::string lutObj2 = "NULL";

  LogTrace("L1TGlobal") << "  Sub-condition 0: std::vector<SingleCombInCond> size: " << (cond0Comb.size()) << std::endl;
  LogTrace("L1TGlobal") << "  Sub-condition 1: std::vector<SingleCombInCond> size: " << (cond1Comb.size()) << std::endl;
  LogTrace("L1TGlobal") << "  Sub-condition 2: std::vector<SingleCombInCond> size: " << (cond2Comb.size()) << std::endl;

  // ///////////////////////////////////////////////////////////////////////////////////////////
  // loop over all combinations which produced individually "true" as Type1s
  // ///////////////////////////////////////////////////////////////////////////////////////////
  // BLW: Optimization issue: potentially making the same comparison twice
  //                          if both legs are the same object type.
  // ///////////////////////////////////////////////////////////////////////////////////////////
  for (std::vector<SingleCombInCond>::const_iterator it0Comb = cond0Comb.begin(); it0Comb != cond0Comb.end();
       it0Comb++) {
    // Type1s: there is 1 object only, no need for a loop, index 0 should be OK in (*it0Comb)[0]
    // ... but add protection to not crash
    int obj0Index = -1;

    if (!(*it0Comb).empty()) {
      obj0Index = (*it0Comb)[0];
    } else {
      LogTrace("L1TGlobal") << "\n  SingleCombInCond (*it0Comb).size() " << ((*it0Comb).size()) << std::endl;
      return false;
    }

    // Collect the information on the first leg of the correlation
    switch (cond0Categ) {
      case CondMuon: {
        lutObj0 = "MU";
        candMuVec = m_uGtB->getCandL1Mu();
        phiIndex0 = (candMuVec->at(cond0bx, obj0Index))->hwPhi();  //(*candMuVec)[obj0Index]->phiIndex();
        etaIndex0 = (candMuVec->at(cond0bx, obj0Index))->hwEta();
        etIndex0 = (candMuVec->at(cond0bx, obj0Index))->hwPt();
        chrg0 = (candMuVec->at(cond0bx, obj0Index))->hwCharge();
        int etaBin0 = etaIndex0;
        if (etaBin0 < 0)
          etaBin0 = m_gtScales->getMUScales().etaBins.size() + etaBin0;  //twos complement
        //		LogDebug("L1TGlobal") << "Muon phi" << phiIndex0 << " eta " << etaIndex0 << " etaBin0 = " << etaBin0  << " et " << etIndex0 << std::endl;
        //

        etBin0 = etIndex0;
        int ssize = m_gtScales->getMUScales().etBins.size();
        if (etBin0 >= ssize) {
          etBin0 = ssize - 1;
          LogTrace("L1TGlobal") << "muon0 hw et" << etBin0 << " out of scale range.  Setting to maximum.";
        }

        // Determine Floating Pt numbers for floating point caluclation
        std::pair<double, double> binEdges = m_gtScales->getMUScales().phiBins.at(phiIndex0);
        phi0Phy = 0.5 * (binEdges.second + binEdges.first);
        binEdges = m_gtScales->getMUScales().etaBins.at(etaBin0);
        eta0Phy = 0.5 * (binEdges.second + binEdges.first);
        binEdges = m_gtScales->getMUScales().etBins.at(etBin0);
        et0Phy = 0.5 * (binEdges.second + binEdges.first);

        LogDebug("L1TGlobal") << "Found all quantities for the muon 0" << std::endl;
      } break;

        // Calorimeter Objects (EG, Jet, Tau)
      case CondCalo: {
        switch (cndObjTypeVec[0]) {
          case gtEG: {
            lutObj0 = "EG";
            candCaloVec = m_uGtB->getCandL1EG();
            phiIndex0 = (candCaloVec->at(cond0bx, obj0Index))->hwPhi();
            etaIndex0 = (candCaloVec->at(cond0bx, obj0Index))->hwEta();
            etIndex0 = (candCaloVec->at(cond0bx, obj0Index))->hwPt();
            etaBin0 = etaIndex0;
            if (etaBin0 < 0)
              etaBin0 = m_gtScales->getEGScales().etaBins.size() + etaBin0;
            //		    LogDebug("L1TGlobal") << "EG0 phi" << phiIndex0 << " eta " << etaIndex0 << " etaBin0 = " << etaBin0 << " et " << etIndex0 << std::endl;

            etBin0 = etIndex0;
            int ssize = m_gtScales->getEGScales().etBins.size();
            if (etBin0 >= ssize) {
              etBin0 = ssize - 1;
              LogTrace("L1TGlobal") << "EG0 hw et" << etBin0 << " out of scale range.  Setting to maximum.";
            }

            // Determine Floating Pt numbers for floating point caluclation
            std::pair<double, double> binEdges = m_gtScales->getEGScales().phiBins.at(phiIndex0);
            phi0Phy = 0.5 * (binEdges.second + binEdges.first);
            binEdges = m_gtScales->getEGScales().etaBins.at(etaBin0);
            eta0Phy = 0.5 * (binEdges.second + binEdges.first);
            binEdges = m_gtScales->getEGScales().etBins[etBin0];
            et0Phy = 0.5 * (binEdges.second + binEdges.first);

          } break;
          case gtJet: {
            lutObj0 = "JET";
            candCaloVec = m_uGtB->getCandL1Jet();
            phiIndex0 = (candCaloVec->at(cond0bx, obj0Index))->hwPhi();
            etaIndex0 = (candCaloVec->at(cond0bx, obj0Index))->hwEta();
            etIndex0 = (candCaloVec->at(cond0bx, obj0Index))->hwPt();
            etaBin0 = etaIndex0;
            if (etaBin0 < 0)
              etaBin0 = m_gtScales->getJETScales().etaBins.size() + etaBin0;

            etBin0 = etIndex0;
            int ssize = m_gtScales->getJETScales().etBins.size();
            if (etBin0 >= ssize) {
              //edm::LogWarning("L1TGlobal")
              //<< "jet0 hw et" << etBin0 << " out of scale range.  Setting to maximum.";
              etBin0 = ssize - 1;
            }

            // Determine Floating Pt numbers for floating point caluclation
            std::pair<double, double> binEdges = m_gtScales->getJETScales().phiBins.at(phiIndex0);
            phi0Phy = 0.5 * (binEdges.second + binEdges.first);
            binEdges = m_gtScales->getJETScales().etaBins.at(etaBin0);
            eta0Phy = 0.5 * (binEdges.second + binEdges.first);
            binEdges = m_gtScales->getJETScales().etBins.at(etBin0);
            et0Phy = 0.5 * (binEdges.second + binEdges.first);

          } break;
          case gtTau: {
            candCaloVec = m_uGtB->getCandL1Tau();
            phiIndex0 = (candCaloVec->at(cond0bx, obj0Index))->hwPhi();
            etaIndex0 = (candCaloVec->at(cond0bx, obj0Index))->hwEta();
            etIndex0 = (candCaloVec->at(cond0bx, obj0Index))->hwPt();
            etaBin0 = etaIndex0;
            if (etaBin0 < 0)
              etaBin0 = m_gtScales->getTAUScales().etaBins.size() + etaBin0;

            etBin0 = etIndex0;
            int ssize = m_gtScales->getTAUScales().etBins.size();
            if (etBin0 >= ssize) {
              etBin0 = ssize - 1;
              LogTrace("L1TGlobal") << "tau0 hw et" << etBin0 << " out of scale range.  Setting to maximum.";
            }

            // Determine Floating Pt numbers for floating point caluclation
            std::pair<double, double> binEdges = m_gtScales->getTAUScales().phiBins.at(phiIndex0);
            phi0Phy = 0.5 * (binEdges.second + binEdges.first);
            binEdges = m_gtScales->getTAUScales().etaBins.at(etaBin0);
            eta0Phy = 0.5 * (binEdges.second + binEdges.first);
            binEdges = m_gtScales->getTAUScales().etBins.at(etBin0);
            et0Phy = 0.5 * (binEdges.second + binEdges.first);
            lutObj0 = "TAU";
          } break;
          default: {
          } break;
        }  //end switch on calo type.

        phiORIndex0 = phiIndex0;
        etaORIndex0 = etaIndex0;

        //If needed convert calo scales to muon scales for comparison
        if (convertCaloScales) {
          std::string lutName = lutObj0;
          lutName += "-MU";
          long long tst = m_gtScales->getLUT_CalMuEta(lutName, etaBin0);
          LogDebug("L1TGlobal") << lutName << "  EtaCal = " << etaIndex0 << " etaBin0 = " << etaBin0
                                << " EtaMu = " << tst << std::endl;
          etaIndex0 = tst;
          tst = m_gtScales->getLUT_CalMuPhi(lutName, phiIndex0);
          LogDebug("L1TGlobal") << lutName << "  PhiCal = " << phiIndex0 << " PhiMu = " << tst << std::endl;
          phiIndex0 = tst;
        }

        //If needed convert calo scales to muon scales for comparison
        if (convertCaloScalesForOverlapRemovalFromLeg0) {
          phiORIndex0 = phiIndex0;
          etaORIndex0 = etaIndex0;
        }

      } break;

      // Energy Sums
      case CondEnergySum: {
        etSumCond = true;
        //Stupid mapping between enum types for energy sums.
        l1t::EtSum::EtSumType type;

        switch (cndObjTypeVec[0]) {
          case gtETM:
            type = l1t::EtSum::EtSumType::kMissingEt;
            lutObj0 = "ETM";
            break;
          case gtETT:
            type = l1t::EtSum::EtSumType::kTotalEt;
            lutObj0 = "ETT";
            break;
          case gtETTem:
            type = l1t::EtSum::EtSumType::kTotalEtEm;
            lutObj0 =
                "ETTem";  //should this be just ETT (share LUTs?) Can't be used for CorrCond anyway since now directional information
            break;
          case gtHTM:
            type = l1t::EtSum::EtSumType::kMissingHt;
            lutObj0 = "HTM";
            break;
          case gtHTT:
            type = l1t::EtSum::EtSumType::kTotalHt;
            lutObj0 = "HTT";
            break;
          case gtETMHF:
            type = l1t::EtSum::EtSumType::kMissingEtHF;
            lutObj0 = "ETMHF";
            break;
          case gtMinBiasHFP0:
          case gtMinBiasHFM0:
          case gtMinBiasHFP1:
          case gtMinBiasHFM1:
            type = l1t::EtSum::EtSumType::kMinBiasHFP0;
            lutObj0 =
                "MinBias";  //??Fix?? Not a valid LUT type Can't be used for CorrCond anyway since now directional information
            break;
          default:
            edm::LogError("L1TGlobal") << "\n  Error: "
                                       << "Unmatched object type from template to EtSumType, cndObjTypeVec[0] = "
                                       << cndObjTypeVec[0] << std::endl;
            type = l1t::EtSum::EtSumType::kTotalEt;
            break;
        }

        candEtSumVec = m_uGtB->getCandL1EtSum();

        for (int iEtSum = 0; iEtSum < (int)candEtSumVec->size(cond0bx); iEtSum++) {
          if ((candEtSumVec->at(cond0bx, iEtSum))->getType() == type) {
            phiIndex0 = (candEtSumVec->at(cond0bx, iEtSum))->hwPhi();
            etaIndex0 = (candEtSumVec->at(cond0bx, iEtSum))->hwEta();
            etIndex0 = (candEtSumVec->at(cond0bx, iEtSum))->hwPt();

            //  Get the floating point numbers
            if (cndObjTypeVec[0] == gtETM) {
              std::pair<double, double> binEdges = m_gtScales->getETMScales().phiBins.at(phiIndex0);
              phi0Phy = 0.5 * (binEdges.second + binEdges.first);
              eta0Phy = 0.;  //No Eta for Energy Sums

              etBin0 = etIndex0;
              int ssize = m_gtScales->getETMScales().etBins.size();
              assert(ssize > 0);
              if (etBin0 >= ssize) {
                etBin0 = ssize - 1;
              }

              binEdges = m_gtScales->getETMScales().etBins.at(etBin0);
              et0Phy = 0.5 * (binEdges.second + binEdges.first);
            } else if (cndObjTypeVec[0] == gtHTM) {
              std::pair<double, double> binEdges = m_gtScales->getHTMScales().phiBins.at(phiIndex0);
              phi0Phy = 0.5 * (binEdges.second + binEdges.first);
              eta0Phy = 0.;  //No Eta for Energy Sums

              etBin0 = etIndex0;
              int ssize = m_gtScales->getHTMScales().etBins.size();
              assert(ssize > 0);
              if (etBin0 >= ssize) {
                etBin0 = ssize - 1;
              }

              binEdges = m_gtScales->getHTMScales().etBins.at(etBin0);
              et0Phy = 0.5 * (binEdges.second + binEdges.first);
            } else if (cndObjTypeVec[0] == gtETMHF) {
              std::pair<double, double> binEdges = m_gtScales->getETMHFScales().phiBins.at(phiIndex0);
              phi0Phy = 0.5 * (binEdges.second + binEdges.first);
              eta0Phy = 0.;  //No Eta for Energy Sums

              etBin0 = etIndex0;
              int ssize = m_gtScales->getETMHFScales().etBins.size();
              assert(ssize > 0);
              if (etBin0 >= ssize) {
                etBin0 = ssize - 1;
              }

              binEdges = m_gtScales->getETMHFScales().etBins.at(etBin0);
              et0Phy = 0.5 * (binEdges.second + binEdges.first);
            }

            phiORIndex0 = phiIndex0;
            etaORIndex0 = etaIndex0;

            //If needed convert calo scales to muon scales for comparison (only phi for energy sums)
            if (convertCaloScales) {
              std::string lutName = lutObj0;
              lutName += "-MU";
              long long tst = m_gtScales->getLUT_CalMuPhi(lutName, phiIndex0);
              LogDebug("L1TGlobal") << lutName << "  PhiCal = " << phiIndex0 << " PhiMu = " << tst << std::endl;
              phiIndex0 = tst;
            }

            //If needed convert calo scales to muon scales for comparison (only phi for energy sums)
            if (convertCaloScalesForOverlapRemovalFromLeg0) {
              phiORIndex0 = phiIndex0;
            }

          }  //check it is the EtSum we want
        }    // loop over Etsums

      }  // end case CondEnergySum
      break;

      default: {
        // should not arrive here, there are no correlation conditions defined for this object
        LogDebug("L1TGlobal") << "Error could not find the Cond Category for Leg 0" << std::endl;
        return false;
      } break;
    }  //end switch on first leg type

    LogDebug("L1TGlobal") << "lutObj0 = " << lutObj0 << std::endl;

    unsigned int overlapRemovalMatchLeg1 = 0x0;

    // ///////////////////////////////////////////////////////////////////////////////////////////
    // loop over overlap-removal leg combination which produced individually "true" as Type1s
    // ///////////////////////////////////////////////////////////////////////////////////////////
    for (std::vector<SingleCombInCond>::const_iterator it2Comb = cond2Comb.begin();
         it2Comb != cond2Comb.end() && overlapRemovalMatchLeg1 != 0x1;
         it2Comb++) {
      // Type1s: there is 1 object only, no need for a loop, index 0 should be OK in (*it2Comb)[0]
      // ... but add protection to not crash
      int obj2Index = -1;

      if (!(*it2Comb).empty()) {
        obj2Index = (*it2Comb)[0];
      } else {
        LogTrace("L1TGlobal") << "\n  SingleCombInCond (*it2Comb).size() " << ((*it2Comb).size()) << std::endl;
        return false;
      }

      // Collect the information on the overlap-removal leg
      switch (cond2Categ) {
        case CondMuon: {
          lutObj2 = "MU";
          candMuVec = m_uGtB->getCandL1Mu();
          phiIndex2 = (candMuVec->at(cond2bx, obj2Index))->hwPhi();  //(*candMuVec)[obj2Index]->phiIndex();
          etaIndex2 = (candMuVec->at(cond2bx, obj2Index))->hwEta();
          int etaBin2 = etaIndex2;
          if (etaBin2 < 0)
            etaBin2 = m_gtScales->getMUScales().etaBins.size() + etaBin2;  //twos complement
          //LogDebug("L1TGlobal") << "Muon phi" << phiIndex2 << " eta " << etaIndex2 << " etaBin2 = " << etaBin2  << " et " << etIndex2 << std::endl;

          // Determine Floating Pt numbers for floating point caluclation
          std::pair<double, double> binEdges = m_gtScales->getMUScales().phiBins.at(phiIndex2);
          phi2Phy = 0.5 * (binEdges.second + binEdges.first);
          binEdges = m_gtScales->getMUScales().etaBins.at(etaBin2);
          eta2Phy = 0.5 * (binEdges.second + binEdges.first);

          LogDebug("L1TGlobal") << "Found all quantities for the muon 0" << std::endl;
        } break;

        // Calorimeter Objects (EG, Jet, Tau)
        case CondCalo: {
          switch (cndObjTypeVec[2]) {
            case gtEG: {
              lutObj2 = "EG";
              candCaloVec = m_uGtB->getCandL1EG();
              phiIndex2 = (candCaloVec->at(cond2bx, obj2Index))->hwPhi();
              etaIndex2 = (candCaloVec->at(cond2bx, obj2Index))->hwEta();
              if (etaBin2 < 0)
                etaBin2 = m_gtScales->getEGScales().etaBins.size() + etaBin2;
              //LogDebug("L1TGlobal") << "EG0 phi" << phiIndex2 << " eta " << etaIndex2 << " etaBin2 = " << etaBin2 << " et " << etIndex2 << std::endl;

              // Determine Floating Pt numbers for floating point caluclation
              std::pair<double, double> binEdges = m_gtScales->getEGScales().phiBins.at(phiIndex2);
              phi2Phy = 0.5 * (binEdges.second + binEdges.first);
              binEdges = m_gtScales->getEGScales().etaBins.at(etaBin2);
              eta2Phy = 0.5 * (binEdges.second + binEdges.first);

            } break;
            case gtJet: {
              lutObj2 = "JET";
              candCaloVec = m_uGtB->getCandL1Jet();
              phiIndex2 = (candCaloVec->at(cond2bx, obj2Index))->hwPhi();
              etaIndex2 = (candCaloVec->at(cond2bx, obj2Index))->hwEta();
              etaBin2 = etaIndex2;
              if (etaBin2 < 0)
                etaBin2 = m_gtScales->getJETScales().etaBins.size() + etaBin2;

              // Determine Floating Pt numbers for floating point caluclation
              std::pair<double, double> binEdges = m_gtScales->getJETScales().phiBins.at(phiIndex2);
              phi2Phy = 0.5 * (binEdges.second + binEdges.first);
              binEdges = m_gtScales->getJETScales().etaBins.at(etaBin2);
              eta2Phy = 0.5 * (binEdges.second + binEdges.first);

            } break;
            case gtTau: {
              candCaloVec = m_uGtB->getCandL1Tau();
              phiIndex2 = (candCaloVec->at(cond2bx, obj2Index))->hwPhi();
              etaIndex2 = (candCaloVec->at(cond2bx, obj2Index))->hwEta();
              if (etaBin2 < 0)
                etaBin2 = m_gtScales->getTAUScales().etaBins.size() + etaBin2;

              // Determine Floating Pt numbers for floating point caluclation
              std::pair<double, double> binEdges = m_gtScales->getTAUScales().phiBins.at(phiIndex2);
              phi2Phy = 0.5 * (binEdges.second + binEdges.first);
              binEdges = m_gtScales->getTAUScales().etaBins.at(etaBin2);
              eta2Phy = 0.5 * (binEdges.second + binEdges.first);
              lutObj2 = "TAU";
            } break;
            default: {
            } break;
          }  //end switch on calo type.

          //If needed convert calo scales to muon scales for comparison
          if (convertCaloScales) {
            std::string lutName = lutObj2;
            lutName += "-MU";
            long long tst = m_gtScales->getLUT_CalMuEta(lutName, etaBin2);
            LogDebug("L1TGlobal") << lutName << "  EtaCal = " << etaIndex2 << " etaBin2 = " << etaBin2
                                  << " EtaMu = " << tst << std::endl;
            etaIndex2 = tst;

            tst = m_gtScales->getLUT_CalMuPhi(lutName, phiIndex2);
            LogDebug("L1TGlobal") << lutName << "  PhiCal = " << phiIndex2 << " PhiMu = " << tst << std::endl;
            phiIndex2 = tst;
          }

        }  // end case CondCalo
        break;

        // Energy Sums
        case CondEnergySum: {
          etSumCond = true;
          //Stupid mapping between enum types for energy sums.
          l1t::EtSum::EtSumType type;
          switch (cndObjTypeVec[2]) {
            case gtETM:
              type = l1t::EtSum::EtSumType::kMissingEt;
              lutObj2 = "ETM";
              break;
            case gtETT:
              type = l1t::EtSum::EtSumType::kTotalEt;
              lutObj2 = "ETT";
              break;
            case gtETTem:
              type = l1t::EtSum::EtSumType::kTotalEtEm;
              lutObj2 =
                  "ETTem";  //should this be just ETT (share LUTs?) Can't be used for CorrCond anyway since now directional information
              break;
            case gtHTM:
              type = l1t::EtSum::EtSumType::kMissingHt;
              lutObj2 = "HTM";
              break;
            case gtHTT:
              type = l1t::EtSum::EtSumType::kTotalHt;
              lutObj2 = "HTT";
              break;
            case gtETMHF:
              type = l1t::EtSum::EtSumType::kMissingEtHF;
              lutObj2 = "ETMHF";
              break;
            case gtMinBiasHFP0:
            case gtMinBiasHFM0:
            case gtMinBiasHFP1:
            case gtMinBiasHFM1:
              type = l1t::EtSum::EtSumType::kMinBiasHFP0;
              lutObj2 =
                  "MinBias";  //??Fix?? Not a valid LUT type Can't be used for CorrCond anyway since now directional information
              break;
            default:
              edm::LogError("L1TGlobal") << "\n  Error: "
                                         << "Unmatched object type from template to EtSumType, cndObjTypeVec[2] = "
                                         << cndObjTypeVec[2] << std::endl;
              type = l1t::EtSum::EtSumType::kTotalEt;
              break;
          }

          candEtSumVec = m_uGtB->getCandL1EtSum();

          for (int iEtSum = 0; iEtSum < (int)candEtSumVec->size(cond2bx); iEtSum++) {
            if ((candEtSumVec->at(cond2bx, iEtSum))->getType() == type) {
              phiIndex2 = (candEtSumVec->at(cond2bx, iEtSum))->hwPhi();
              etaIndex2 = (candEtSumVec->at(cond2bx, iEtSum))->hwEta();

              //  Get the floating point numbers
              if (cndObjTypeVec[2] == gtETM) {
                std::pair<double, double> binEdges = m_gtScales->getETMScales().phiBins.at(phiIndex2);
                phi2Phy = 0.5 * (binEdges.second + binEdges.first);
                eta2Phy = 0.;  //No Eta for Energy Sums

              } else if (cndObjTypeVec[2] == gtHTM) {
                std::pair<double, double> binEdges = m_gtScales->getHTMScales().phiBins.at(phiIndex2);
                phi2Phy = 0.5 * (binEdges.second + binEdges.first);
                eta2Phy = 0.;  //No Eta for Energy Sums

              } else if (cndObjTypeVec[2] == gtETMHF) {
                std::pair<double, double> binEdges = m_gtScales->getETMHFScales().phiBins.at(phiIndex2);
                phi2Phy = 0.5 * (binEdges.second + binEdges.first);
                eta2Phy = 0.;  //No Eta for Energy Sums
              }

              //If needed convert calo scales to muon scales for comparison (only phi for energy sums)
              if (convertCaloScales) {
                std::string lutName = lutObj2;
                lutName += "-MU";
                long long tst = m_gtScales->getLUT_CalMuPhi(lutName, phiIndex2);
                LogDebug("L1TGlobal") << lutName << "  PhiCal = " << phiIndex2 << " PhiMu = " << tst << std::endl;
                phiIndex2 = tst;
              }

            }  //check it is the EtSum we want
          }    // loop over Etsums

        } break;

        default: {
          // should not arrive here, there are no correlation conditions defined for this object
          LogDebug("L1TGlobal") << "Error could not find the Cond Category for Leg 3" << std::endl;
          return false;
        } break;
      }  //end switch on overlap-removal leg type

      LogDebug("L1TGlobal") << "lutObj2 = " << lutObj2 << std::endl;

      // /////////////////////////////////////////////////////////////////////////////////////////
      //
      // here check if there is a match of 1st leg with overlap removal object, and store result
      //
      // /////////////////////////////////////////////////////////////////////////////////////////
      // These all require some delta eta and phi calculations.  Do them first...for now real calculation but need to
      // revise this to line up with firmware calculations.
      double deltaPhiPhy = fabs(phi2Phy - phi0Phy);
      if (deltaPhiPhy > M_PI)
        deltaPhiPhy = 2. * M_PI - deltaPhiPhy;
      double deltaEtaPhy = fabs(eta2Phy - eta0Phy);

      // Deter the integer based delta eta and delta phi
      int deltaPhiFW = abs(phiORIndex0 - phiIndex2);
      if (deltaPhiFW >= phiBound)
        deltaPhiFW = 2 * phiBound - deltaPhiFW;
      std::string lutName = lutObj0;
      lutName += "-";
      lutName += lutObj2;
      long long deltaPhiLUT = m_gtScales->getLUT_DeltaPhi(lutName, deltaPhiFW);
      unsigned int precDeltaPhiLUT = m_gtScales->getPrec_DeltaPhi(lutName);

      int deltaEtaFW = abs(etaORIndex0 - etaIndex2);
      long long deltaEtaLUT = 0;
      unsigned int precDeltaEtaLUT = 0;
      if (!etSumCond) {
        deltaEtaLUT = m_gtScales->getLUT_DeltaEta(lutName, deltaEtaFW);
        precDeltaEtaLUT = m_gtScales->getPrec_DeltaEta(lutName);
      }

      LogDebug("L1TGlobal") << "Obj0 phiFW = " << phiORIndex0 << " Obj2 phiFW = " << phiIndex2 << "\n"
                            << "    DeltaPhiFW = " << deltaPhiFW << "\n"
                            << "    LUT Name = " << lutName << " Prec = " << precDeltaPhiLUT
                            << "  DeltaPhiLUT = " << deltaPhiLUT << "\n"
                            << "Obj0 etaFW = " << etaIndex0 << " Obj2 etaFW = " << etaIndex2 << "\n"
                            << "    DeltaEtaFW = " << deltaEtaFW << "\n"
                            << "    LUT Name = " << lutName << " Prec = " << precDeltaEtaLUT
                            << "  DeltaEtaLUT = " << deltaEtaLUT << std::endl;

      // If there is a OverlapRemovalDeltaEta cut, check it.
      if (corrPar.corrCutType & 0x10) {
        unsigned int preShift = precDeltaEtaLUT - corrPar.precOverlapRemovalEtaCut;
        LogDebug("L1TGlobal") << "    Testing Leg1 Overlap Removal Delta Eta Cut (" << lutObj0 << "," << lutObj2
                              << ") [" << (long long)(corrPar.minOverlapRemovalEtaCutValue * pow(10, preShift)) << ","
                              << (long long)(corrPar.maxOverlapRemovalEtaCutValue * pow(10, preShift))
                              << "] with precision = " << corrPar.precOverlapRemovalEtaCut << "\n"
                              << "    deltaEtaLUT = " << deltaEtaLUT << "\n"
                              << "    Precision Shift = " << preShift << "\n"
                              << "    deltaEta (shift)= "
                              << (deltaEtaLUT / pow(10, preShift + corrPar.precOverlapRemovalEtaCut)) << "\n"
                              << "    deltaEtaPhy = " << deltaEtaPhy << std::endl;

        //if(preShift>0) deltaEtaLUT /= pow(10,preShift);
        if (deltaEtaLUT >= (long long)(corrPar.minOverlapRemovalEtaCutValue * pow(10, preShift)) &&
            deltaEtaLUT <= (long long)(corrPar.maxOverlapRemovalEtaCutValue * pow(10, preShift))) {
          overlapRemovalMatchLeg1 |= 0x1;
          LogDebug("L1TGlobal") << "    Satified Leg1 Overlap Removal Delta Eta Cut ["
                                << (long long)(corrPar.minOverlapRemovalEtaCutValue * pow(10, preShift)) << ","
                                << (long long)(corrPar.maxOverlapRemovalEtaCutValue * pow(10, preShift)) << "]"
                                << std::endl;
          // next leg3 object
          continue;

        } else {
          LogDebug("L1TGlobal") << "    Failed Leg1 Overlap Removal Delta Eta Cut ["
                                << (long long)(corrPar.minOverlapRemovalEtaCutValue * pow(10, preShift)) << ","
                                << (long long)(corrPar.maxOverlapRemovalEtaCutValue * pow(10, preShift)) << "]"
                                << std::endl;
        }
      }

      //if there is a OverlapRemovalDeltaPhi cut, check it.
      if (corrPar.corrCutType & 0x20) {
        unsigned int preShift = precDeltaPhiLUT - corrPar.precOverlapRemovalPhiCut;
        LogDebug("L1TGlobal") << "    Testing Leg1 Overlap Removal Delta Phi Cut (" << lutObj0 << "," << lutObj2
                              << ") [" << (long long)(corrPar.minOverlapRemovalPhiCutValue * pow(10, preShift)) << ","
                              << (long long)(corrPar.maxOverlapRemovalPhiCutValue * pow(10, preShift))
                              << "] with precision = " << corrPar.precOverlapRemovalPhiCut << "\n"
                              << "    deltaPhiLUT = " << deltaPhiLUT << "\n"
                              << "    Precision Shift = " << preShift << "\n"
                              << "    deltaPhi (shift)= "
                              << (deltaPhiLUT / pow(10, preShift + corrPar.precOverlapRemovalPhiCut)) << "\n"
                              << "    deltaPhiPhy = " << deltaPhiPhy << std::endl;

        //if(preShift>0) deltaPhiLUT /= pow(10,preShift);
        if (deltaPhiLUT >= (long long)(corrPar.minOverlapRemovalPhiCutValue * pow(10, preShift)) &&
            deltaPhiLUT <= (long long)(corrPar.maxOverlapRemovalPhiCutValue * pow(10, preShift))) {
          overlapRemovalMatchLeg1 |= 0x1;
          LogDebug("L1TGlobal") << "    Satisfied Leg1 Overlap Removal Delta Phi Cut ["
                                << (long long)(corrPar.minOverlapRemovalPhiCutValue * pow(10, preShift)) << ","
                                << (long long)(corrPar.maxOverlapRemovalPhiCutValue * pow(10, preShift)) << "]"
                                << std::endl;
          // next leg3 object
          continue;

        } else {
          LogDebug("L1TGlobal") << "    Failed Leg1 Overlap Removal Delta Phi Cut ["
                                << (long long)(corrPar.minOverlapRemovalPhiCutValue * pow(10, preShift)) << ","
                                << (long long)(corrPar.maxOverlapRemovalPhiCutValue * pow(10, preShift)) << "]"
                                << std::endl;
        }
      }

      //if there is a OverlapRemovalDeltaR cut, check it.
      if (corrPar.corrCutType & 0x40) {
        //Assumes Delta Eta and Delta Phi LUTs have the same precision
        unsigned int preShift = 2 * precDeltaPhiLUT - corrPar.precOverlapRemovalDRCut;
        double deltaRSqPhy = deltaPhiPhy * deltaPhiPhy + deltaEtaPhy * deltaEtaPhy;
        long long deltaRSq = deltaEtaLUT * deltaEtaLUT + deltaPhiLUT * deltaPhiLUT;

        LogDebug("L1TGlobal") << "    Testing Leg1 Overlap Removal Delta R Cut (" << lutObj0 << "," << lutObj2 << ") ["
                              << (long long)(corrPar.minOverlapRemovalDRCutValue * pow(10, preShift)) << ","
                              << (long long)(corrPar.maxOverlapRemovalDRCutValue * pow(10, preShift))
                              << "] with precision = " << corrPar.precOverlapRemovalDRCut << "\n"
                              << "    deltaPhiLUT = " << deltaPhiLUT << "\n"
                              << "    deltaEtaLUT = " << deltaEtaLUT << "\n"
                              << "    deltaRSqLUT = " << deltaRSq << "\n"
                              << "    Precision Shift = " << preShift << "\n"
                              << "    deltaRSqLUT (shift)= " << (deltaRSq / pow(10, preShift + corrPar.precDRCut))
                              << "\n"
                              << "    deltaRSqPhy = " << deltaRSqPhy << std::endl;

        //if(preShift>0) deltaRSq /= pow(10,preShift);
        if (deltaRSq >= (long long)(corrPar.minOverlapRemovalDRCutValue * pow(10, preShift)) &&
            deltaRSq <= (long long)(corrPar.maxOverlapRemovalDRCutValue * pow(10, preShift))) {
          overlapRemovalMatchLeg1 |= 0x1;
          LogDebug("L1TGlobal") << "    Satified Leg1 Overlap Removal Delta R Cut ["
                                << (long long)(corrPar.minOverlapRemovalDRCutValue * pow(10, preShift)) << ","
                                << (long long)(corrPar.maxOverlapRemovalDRCutValue * pow(10, preShift)) << "]"
                                << std::endl;
          // next leg3 object
          continue;

        } else {
          LogDebug("L1TGlobal") << "    Failed Leg1 Overlap Removal Delta R Cut ["
                                << (int)(corrPar.minOverlapRemovalDRCutValue * pow(10, preShift)) << ","
                                << (long long)(corrPar.maxOverlapRemovalDRCutValue * pow(10, preShift)) << "]"
                                << std::endl;
        }
      }

    }  // end loop over combinations in overlap-removal leg.

    // skip object leg1 if matched with overlap removal object
    // ///////////////////////////////////////////////////////
    if (overlapRemovalMatchLeg1 == 0x1) {
      LogDebug("L1TGlobal") << "   Remove Object of Leg1: Satisfied Overlap Removal Cuts " << std::endl;
      continue;
    } else {
      LogDebug("L1TGlobal") << "   Keep Object of Leg1: Failed Overlap Removal Cuts " << std::endl;
    }

    // ///////////////////////////////////////////////////////////////////////////////////////////
    // Now loop over the second leg to get its information
    // ///////////////////////////////////////////////////////////////////////////////////////////
    for (std::vector<SingleCombInCond>::const_iterator it1Comb = cond1Comb.begin(); it1Comb != cond1Comb.end();
         it1Comb++) {
      LogDebug("L1TGlobal") << "Looking at second Condition" << std::endl;
      // Type1s: there is 1 object only, no need for a loop (*it1Comb)[0]
      // ... but add protection to not crash
      int obj1Index = -1;

      if (!(*it1Comb).empty()) {
        obj1Index = (*it1Comb)[0];
      } else {
        LogTrace("L1TGlobal") << "\n  SingleCombInCond (*it1Comb).size() " << ((*it1Comb).size()) << std::endl;
        return false;
      }

      //If we are dealing with the same object type avoid the two legs
      // either being the same object
      if (cndObjTypeVec[0] == cndObjTypeVec[1] && obj0Index == obj1Index && cond0bx == cond1bx) {
        LogDebug("L1TGlobal") << "Corr Condition looking at same leg...skip" << std::endl;
        continue;
      }

      switch (cond1Categ) {
        case CondMuon: {
          lutObj1 = "MU";
          candMuVec = m_uGtB->getCandL1Mu();
          phiIndex1 = (candMuVec->at(cond1bx, obj1Index))->hwPhi();  //(*candMuVec)[obj0Index]->phiIndex();
          etaIndex1 = (candMuVec->at(cond1bx, obj1Index))->hwEta();
          etIndex1 = (candMuVec->at(cond1bx, obj1Index))->hwPt();
          chrg1 = (candMuVec->at(cond1bx, obj1Index))->hwCharge();
          etaBin1 = etaIndex1;
          if (etaBin1 < 0)
            etaBin1 = m_gtScales->getMUScales().etaBins.size() + etaBin1;
          //		   LogDebug("L1TGlobal") << "Muon phi" << phiIndex1 << " eta " << etaIndex1 << " etaBin1 = " << etaBin1  << " et " << etIndex1 << std::endl;
          etBin1 = etIndex1;
          int ssize = m_gtScales->getMUScales().etBins.size();

          if (etBin1 >= ssize) {
            LogTrace("L1TGlobal") << "muon2 hw et" << etBin1 << " out of scale range.  Setting to maximum.";
            etBin1 = ssize - 1;
          }

          // Determine Floating Pt numbers for floating point caluclation
          std::pair<double, double> binEdges = m_gtScales->getMUScales().phiBins.at(phiIndex1);
          phi1Phy = 0.5 * (binEdges.second + binEdges.first);
          binEdges = m_gtScales->getMUScales().etaBins.at(etaBin1);
          eta1Phy = 0.5 * (binEdges.second + binEdges.first);
          binEdges = m_gtScales->getMUScales().etBins.at(etBin1);
          et1Phy = 0.5 * (binEdges.second + binEdges.first);

        } break;

        case CondCalo: {
          switch (cndObjTypeVec[1]) {
            case gtEG: {
              candCaloVec = m_uGtB->getCandL1EG();
              phiIndex1 = (candCaloVec->at(cond1bx, obj1Index))->hwPhi();
              etaIndex1 = (candCaloVec->at(cond1bx, obj1Index))->hwEta();
              etIndex1 = (candCaloVec->at(cond1bx, obj1Index))->hwPt();
              etaBin1 = etaIndex1;
              if (etaBin1 < 0)
                etaBin1 = m_gtScales->getEGScales().etaBins.size() + etaBin1;

              etBin1 = etIndex1;
              int ssize = m_gtScales->getEGScales().etBins.size();
              if (etBin1 >= ssize) {
                LogTrace("L1TGlobal") << "EG1 hw et" << etBin1 << " out of scale range.  Setting to maximum.";
                etBin1 = ssize - 1;
              }

              // Determine Floating Pt numbers for floating point caluclation
              std::pair<double, double> binEdges = m_gtScales->getEGScales().phiBins.at(phiIndex1);
              phi1Phy = 0.5 * (binEdges.second + binEdges.first);
              binEdges = m_gtScales->getEGScales().etaBins.at(etaBin1);
              eta1Phy = 0.5 * (binEdges.second + binEdges.first);
              binEdges = m_gtScales->getEGScales().etBins.at(etBin1);
              et1Phy = 0.5 * (binEdges.second + binEdges.first);
              lutObj1 = "EG";
            } break;

            case gtJet: {
              candCaloVec = m_uGtB->getCandL1Jet();
              phiIndex1 = (candCaloVec->at(cond1bx, obj1Index))->hwPhi();
              etaIndex1 = (candCaloVec->at(cond1bx, obj1Index))->hwEta();
              etIndex1 = (candCaloVec->at(cond1bx, obj1Index))->hwPt();
              etaBin1 = etaIndex1;
              if (etaBin1 < 0)
                etaBin1 = m_gtScales->getJETScales().etaBins.size() + etaBin1;
              etBin1 = etIndex1;
              int ssize = m_gtScales->getJETScales().etBins.size();
              assert(ssize);
              if (etBin1 >= ssize) {
                //edm::LogWarning("L1TGlobal")
                //<< "jet2 hw et" << etBin1 << " out of scale range.  Setting to maximum.";
                etBin1 = ssize - 1;
              }

              // Determine Floating Pt numbers for floating point caluclation
              std::pair<double, double> binEdges = m_gtScales->getJETScales().phiBins.at(phiIndex1);
              phi1Phy = 0.5 * (binEdges.second + binEdges.first);
              binEdges = m_gtScales->getJETScales().etaBins.at(etaBin1);
              eta1Phy = 0.5 * (binEdges.second + binEdges.first);

              binEdges = m_gtScales->getJETScales().etBins.at(etBin1);
              et1Phy = 0.5 * (binEdges.second + binEdges.first);
              lutObj1 = "JET";
            } break;

            case gtTau: {
              candCaloVec = m_uGtB->getCandL1Tau();
              phiIndex1 = (candCaloVec->at(cond1bx, obj1Index))->hwPhi();
              etaIndex1 = (candCaloVec->at(cond1bx, obj1Index))->hwEta();
              etIndex1 = (candCaloVec->at(cond1bx, obj1Index))->hwPt();
              etaBin1 = etaIndex1;
              if (etaBin1 < 0)
                etaBin1 = m_gtScales->getTAUScales().etaBins.size() + etaBin1;
              etBin1 = etIndex1;
              int ssize = m_gtScales->getTAUScales().etBins.size();
              if (etBin1 >= ssize) {
                LogTrace("L1TGlobal") << "tau2 hw et" << etBin1 << " out of scale range.  Setting to maximum.";
                etBin1 = ssize - 1;
              }

              // Determine Floating Pt numbers for floating point caluclation
              std::pair<double, double> binEdges = m_gtScales->getTAUScales().phiBins.at(phiIndex1);
              phi1Phy = 0.5 * (binEdges.second + binEdges.first);
              binEdges = m_gtScales->getTAUScales().etaBins.at(etaBin1);
              eta1Phy = 0.5 * (binEdges.second + binEdges.first);
              binEdges = m_gtScales->getTAUScales().etBins.at(etBin1);
              et1Phy = 0.5 * (binEdges.second + binEdges.first);
              lutObj1 = "TAU";
            } break;
            default: {
            } break;
          }  //end switch on calo type.

          phiORIndex1 = phiIndex1;
          etaORIndex1 = etaIndex1;

          //If needed convert calo scales to muon scales for comparison
          if (convertCaloScales) {
            std::string lutName = lutObj1;
            lutName += "-MU";
            long long tst = m_gtScales->getLUT_CalMuEta(lutName, etaBin1);
            LogDebug("L1TGlobal") << lutName << "  EtaCal = " << etaIndex1 << " EtaMu = " << tst << std::endl;
            etaIndex1 = tst;
            tst = m_gtScales->getLUT_CalMuPhi(lutName, phiIndex1);
            LogDebug("L1TGlobal") << lutName << "  PhiCal = " << phiIndex1 << " PhiMu = " << tst << std::endl;
            phiIndex1 = tst;
          }

          //If needed convert calo scales to muon scales for comparison
          if (convertCaloScalesForOverlapRemovalFromLeg1) {
            phiORIndex1 = phiIndex1;
            etaORIndex1 = etaIndex1;
          }

        }  // end case CondCalo
        break;
        case CondEnergySum: {
          LogDebug("L1TGlobal") << "Looking at second Condition as Energy Sum: " << cndObjTypeVec[1] << std::endl;
          etSumCond = true;

          //Stupid mapping between enum types for energy sums.
          l1t::EtSum::EtSumType type;

          switch (cndObjTypeVec[1]) {
            case gtETM:
              type = l1t::EtSum::EtSumType::kMissingEt;
              lutObj1 = "ETM";
              break;
            case gtETT:
              type = l1t::EtSum::EtSumType::kTotalEt;
              lutObj1 = "ETT";
              break;
            case gtETTem:
              type = l1t::EtSum::EtSumType::kTotalEtEm;
              lutObj1 = "ETTem";
              break;
            case gtHTM:
              type = l1t::EtSum::EtSumType::kMissingHt;
              lutObj1 = "HTM";
              break;
            case gtHTT:
              type = l1t::EtSum::EtSumType::kTotalHt;
              lutObj1 = "HTT";
              break;
            case gtETMHF:
              type = l1t::EtSum::EtSumType::kMissingEtHF;
              lutObj1 = "ETMHF";
              break;
            case gtMinBiasHFP0:
            case gtMinBiasHFM0:
            case gtMinBiasHFP1:
            case gtMinBiasHFM1:
              type = l1t::EtSum::EtSumType::kMinBiasHFP0;
              lutObj1 = "MinBias";
              break;
            default:
              edm::LogError("L1TGlobal") << "\n  Error: "
                                         << "Unmatched object type from template to EtSumType, cndObjTypeVec[1] = "
                                         << cndObjTypeVec[1] << std::endl;
              type = l1t::EtSum::EtSumType::kTotalEt;
              break;
          }

          candEtSumVec = m_uGtB->getCandL1EtSum();

          LogDebug("L1TGlobal") << "obj " << lutObj1 << " Vector Size " << candEtSumVec->size(cond1bx) << std::endl;
          for (int iEtSum = 0; iEtSum < (int)candEtSumVec->size(cond1bx); iEtSum++) {
            if ((candEtSumVec->at(cond1bx, iEtSum))->getType() == type) {
              phiIndex1 = (candEtSumVec->at(cond1bx, iEtSum))->hwPhi();
              etaIndex1 = (candEtSumVec->at(cond1bx, iEtSum))->hwEta();
              etIndex1 = (candEtSumVec->at(cond1bx, iEtSum))->hwPt();

              // Determine Floating Pt numbers for floating point caluclation

              if (cndObjTypeVec[1] == gtETM) {
                std::pair<double, double> binEdges = m_gtScales->getETMScales().phiBins.at(phiIndex1);
                phi1Phy = 0.5 * (binEdges.second + binEdges.first);
                eta1Phy = 0.;  //No Eta for Energy Sums

                etBin1 = etIndex1;
                int ssize = m_gtScales->getETMScales().etBins.size();
                assert(ssize > 0);
                if (etBin1 >= ssize) {
                  etBin1 = ssize - 1;
                }

                binEdges = m_gtScales->getETMScales().etBins.at(etBin1);
                et1Phy = 0.5 * (binEdges.second + binEdges.first);
              } else if (cndObjTypeVec[1] == gtHTM) {
                std::pair<double, double> binEdges = m_gtScales->getHTMScales().phiBins.at(phiIndex1);
                phi1Phy = 0.5 * (binEdges.second + binEdges.first);
                eta1Phy = 0.;  //No Eta for Energy Sums

                etBin1 = etIndex1;
                int ssize = m_gtScales->getHTMScales().etBins.size();
                assert(ssize > 0);
                if (etBin1 >= ssize) {
                  etBin1 = ssize - 1;
                }

                binEdges = m_gtScales->getHTMScales().etBins.at(etBin1);
                et1Phy = 0.5 * (binEdges.second + binEdges.first);
              } else if (cndObjTypeVec[1] == gtETMHF) {
                std::pair<double, double> binEdges = m_gtScales->getETMHFScales().phiBins.at(phiIndex1);
                phi1Phy = 0.5 * (binEdges.second + binEdges.first);
                eta1Phy = 0.;  //No Eta for Energy Sums
                etBin1 = etIndex1;
                int ssize = m_gtScales->getETMHFScales().etBins.size();
                assert(ssize > 0);
                if (etBin1 >= ssize) {
                  etBin1 = ssize - 1;
                }
                binEdges = m_gtScales->getETMHFScales().etBins.at(etBin1);
                et1Phy = 0.5 * (binEdges.second + binEdges.first);
              }

              phiORIndex1 = phiIndex1;
              etaORIndex1 = etaIndex1;

              //If needed convert calo scales to muon scales for comparison (only phi for energy sums)
              if (convertCaloScales) {
                std::string lutName = lutObj1;
                lutName += "-MU";
                long long tst = m_gtScales->getLUT_CalMuPhi(lutName, phiIndex1);
                LogDebug("L1TGlobal") << lutName << "  PhiCal = " << phiIndex1 << " PhiMu = " << tst << std::endl;
                phiIndex1 = tst;
              }

              //If needed convert calo scales to muon scales for comparison (only phi for energy sums)
              if (convertCaloScalesForOverlapRemovalFromLeg1) {
                phiORIndex1 = phiIndex1;
              }

            }  //check it is the EtSum we want
          }    // loop over Etsums

        }  // end case EnergySum
        break;
        default: {
          // should not arrive here, there are no correlation conditions defined for this object
          LogDebug("L1TGlobal") << "Error could not find the Cond Category for Leg 0" << std::endl;
          return false;
        } break;
      }  //end switch on second leg

      unsigned int overlapRemovalMatchLeg2 = 0x0;

      // ///////////////////////////////////////////////////////////////////////////////////////////
      // loop over overlap-removal leg combination which produced individually "true" as Type1s
      // ///////////////////////////////////////////////////////////////////////////////////////////
      for (std::vector<SingleCombInCond>::const_iterator it2Comb = cond2Comb.begin();
           it2Comb != cond2Comb.end() && overlapRemovalMatchLeg2 != 0x1;
           it2Comb++) {
        // Type1s: there is 1 object only, no need for a loop, index 0 should be OK in (*it2Comb)[0]
        // ... but add protection to not crash
        int obj2Index = -1;

        if (!(*it2Comb).empty()) {
          obj2Index = (*it2Comb)[0];
        } else {
          LogTrace("L1TGlobal") << "\n  SingleCombInCond (*it2Comb).size() " << ((*it2Comb).size()) << std::endl;
          return false;
        }

        // Collect the information on the overlap-removal leg
        switch (cond2Categ) {
          case CondMuon: {
            lutObj2 = "MU";
            candMuVec = m_uGtB->getCandL1Mu();
            phiIndex2 = (candMuVec->at(cond2bx, obj2Index))->hwPhi();  //(*candMuVec)[obj2Index]->phiIndex();
            etaIndex2 = (candMuVec->at(cond2bx, obj2Index))->hwEta();
            int etaBin2 = etaIndex2;
            if (etaBin2 < 0)
              etaBin2 = m_gtScales->getMUScales().etaBins.size() + etaBin2;  //twos complement
            //LogDebug("L1TGlobal") << "Muon phi" << phiIndex2 << " eta " << etaIndex2 << " etaBin2 = " << etaBin2  << " et " << etIndex2 << std::endl;

            // Determine Floating Pt numbers for floating point caluclation
            std::pair<double, double> binEdges = m_gtScales->getMUScales().phiBins.at(phiIndex2);
            phi2Phy = 0.5 * (binEdges.second + binEdges.first);
            binEdges = m_gtScales->getMUScales().etaBins.at(etaBin2);
            eta2Phy = 0.5 * (binEdges.second + binEdges.first);

            LogDebug("L1TGlobal") << "Found all quantities for the muon 0" << std::endl;
          } break;

          // Calorimeter Objects (EG, Jet, Tau)
          case CondCalo: {
            switch (cndObjTypeVec[2]) {
              case gtEG: {
                lutObj2 = "EG";
                candCaloVec = m_uGtB->getCandL1EG();
                phiIndex2 = (candCaloVec->at(cond2bx, obj2Index))->hwPhi();
                etaIndex2 = (candCaloVec->at(cond2bx, obj2Index))->hwEta();
                if (etaBin2 < 0)
                  etaBin2 = m_gtScales->getEGScales().etaBins.size() + etaBin2;
                //LogDebug("L1TGlobal") << "EG0 phi" << phiIndex2 << " eta " << etaIndex2 << " etaBin2 = " << etaBin2 << " et " << etIndex2 << std::endl;

                // Determine Floating Pt numbers for floating point caluclation
                std::pair<double, double> binEdges = m_gtScales->getEGScales().phiBins.at(phiIndex2);
                phi2Phy = 0.5 * (binEdges.second + binEdges.first);
                binEdges = m_gtScales->getEGScales().etaBins.at(etaBin2);
                eta2Phy = 0.5 * (binEdges.second + binEdges.first);
              } break;

              case gtJet: {
                lutObj2 = "JET";
                candCaloVec = m_uGtB->getCandL1Jet();
                phiIndex2 = (candCaloVec->at(cond2bx, obj2Index))->hwPhi();
                etaIndex2 = (candCaloVec->at(cond2bx, obj2Index))->hwEta();
                etaBin2 = etaIndex2;
                if (etaBin2 < 0)
                  etaBin2 = m_gtScales->getJETScales().etaBins.size() + etaBin2;
                // Determine Floating Pt numbers for floating point caluclation
                std::pair<double, double> binEdges = m_gtScales->getJETScales().phiBins.at(phiIndex2);
                phi2Phy = 0.5 * (binEdges.second + binEdges.first);
                binEdges = m_gtScales->getJETScales().etaBins.at(etaBin2);
                eta2Phy = 0.5 * (binEdges.second + binEdges.first);
              } break;
              case gtTau: {
                candCaloVec = m_uGtB->getCandL1Tau();
                phiIndex2 = (candCaloVec->at(cond2bx, obj2Index))->hwPhi();
                etaIndex2 = (candCaloVec->at(cond2bx, obj2Index))->hwEta();
                if (etaBin2 < 0)
                  etaBin2 = m_gtScales->getTAUScales().etaBins.size() + etaBin2;

                // Determine Floating Pt numbers for floating point caluclation
                std::pair<double, double> binEdges = m_gtScales->getTAUScales().phiBins.at(phiIndex2);
                phi2Phy = 0.5 * (binEdges.second + binEdges.first);
                binEdges = m_gtScales->getTAUScales().etaBins.at(etaBin2);
                eta2Phy = 0.5 * (binEdges.second + binEdges.first);
                lutObj2 = "TAU";
              } break;
              default: {
              } break;
            }  //end switch on calo type.

            //If needed convert calo scales to muon scales for comparison
            if (convertCaloScales) {
              std::string lutName = lutObj2;
              lutName += "-MU";
              long long tst = m_gtScales->getLUT_CalMuEta(lutName, etaBin2);
              LogDebug("L1TGlobal") << lutName << "  EtaCal = " << etaIndex2 << " etaBin2 = " << etaBin2
                                    << " EtaMu = " << tst << std::endl;
              etaIndex2 = tst;

              tst = m_gtScales->getLUT_CalMuPhi(lutName, phiIndex2);
              LogDebug("L1TGlobal") << lutName << "  PhiCal = " << phiIndex2 << " PhiMu = " << tst << std::endl;
              phiIndex2 = tst;
            }

          } break;

          // Energy Sums
          case CondEnergySum: {
            etSumCond = true;
            //Stupid mapping between enum types for energy sums.
            l1t::EtSum::EtSumType type;
            switch (cndObjTypeVec[2]) {
              case gtETM:
                type = l1t::EtSum::EtSumType::kMissingEt;
                lutObj2 = "ETM";
                break;
              case gtETT:
                type = l1t::EtSum::EtSumType::kTotalEt;
                lutObj2 = "ETT";
                break;
              case gtETTem:
                type = l1t::EtSum::EtSumType::kTotalEtEm;
                lutObj2 =
                    "ETTem";  //should this be just ETT (share LUTs?) Can't be used for CorrCond anyway since now directional information
                break;
              case gtHTM:
                type = l1t::EtSum::EtSumType::kMissingHt;
                lutObj2 = "HTM";
                break;
              case gtHTT:
                type = l1t::EtSum::EtSumType::kTotalHt;
                lutObj2 = "HTT";
                break;
              case gtETMHF:
                type = l1t::EtSum::EtSumType::kMissingEtHF;
                lutObj2 = "ETMHF";
                break;
              case gtMinBiasHFP0:
              case gtMinBiasHFM0:
              case gtMinBiasHFP1:
              case gtMinBiasHFM1:
                type = l1t::EtSum::EtSumType::kMinBiasHFP0;
                lutObj2 =
                    "MinBias";  //??Fix?? Not a valid LUT type Can't be used for CorrCond anyway since now directional information
                break;
              default:
                edm::LogError("L1TGlobal")
                    << "\n  Error: "
                    << "Unmatched object type from template to EtSumType, cndObjTypeVec[2] = " << cndObjTypeVec[2]
                    << std::endl;
                type = l1t::EtSum::EtSumType::kTotalEt;
                break;
            }

            candEtSumVec = m_uGtB->getCandL1EtSum();

            for (int iEtSum = 0; iEtSum < (int)candEtSumVec->size(cond2bx); iEtSum++) {
              if ((candEtSumVec->at(cond2bx, iEtSum))->getType() == type) {
                phiIndex2 = (candEtSumVec->at(cond2bx, iEtSum))->hwPhi();
                etaIndex2 = (candEtSumVec->at(cond2bx, iEtSum))->hwEta();

                //  Get the floating point numbers
                if (cndObjTypeVec[2] == gtETM) {
                  std::pair<double, double> binEdges = m_gtScales->getETMScales().phiBins.at(phiIndex2);
                  phi2Phy = 0.5 * (binEdges.second + binEdges.first);
                  eta2Phy = 0.;  //No Eta for Energy Sums

                } else if (cndObjTypeVec[2] == gtHTM) {
                  std::pair<double, double> binEdges = m_gtScales->getHTMScales().phiBins.at(phiIndex2);
                  phi2Phy = 0.5 * (binEdges.second + binEdges.first);
                  eta2Phy = 0.;  //No Eta for Energy Sums

                } else if (cndObjTypeVec[2] == gtETMHF) {
                  std::pair<double, double> binEdges = m_gtScales->getETMHFScales().phiBins.at(phiIndex2);
                  phi2Phy = 0.5 * (binEdges.second + binEdges.first);
                  eta2Phy = 0.;  //No Eta for Energy Sums
                }

                //If needed convert calo scales to muon scales for comparison (only phi for energy sums)
                if (convertCaloScales) {
                  std::string lutName = lutObj2;
                  lutName += "-MU";
                  long long tst = m_gtScales->getLUT_CalMuPhi(lutName, phiIndex2);
                  LogDebug("L1TGlobal") << lutName << "  PhiCal = " << phiIndex2 << " PhiMu = " << tst << std::endl;
                  phiIndex2 = tst;
                }

              }  //check it is the EtSum we want
            }    // loop over Etsums

          }  // end case CondEnerySum
          break;
          default: {
            // should not arrive here, there are no correlation conditions defined for this object
            LogDebug("L1TGlobal") << "Error could not find the Cond Category for Leg 3" << std::endl;
            return false;
          } break;
        }  //end switch on overlap-removal leg type

        // /////////////////////////////////////////////////////////////////////////////////////////
        //
        // here check if there is a match of 2st leg with overlap removal object ...if yes, continue
        //
        // /////////////////////////////////////////////////////////////////////////////////////////
        // These all require some delta eta and phi calculations.  Do them first...for now real calculation but need to
        // revise this to line up with firmware calculations.
        double deltaPhiPhy = fabs(phi2Phy - phi1Phy);
        if (deltaPhiPhy > M_PI)
          deltaPhiPhy = 2. * M_PI - deltaPhiPhy;
        double deltaEtaPhy = fabs(eta2Phy - eta1Phy);

        // Deter the integer based delta eta and delta phi
        int deltaPhiFW = abs(phiORIndex1 - phiIndex2);
        if (deltaPhiFW >= phiBound)
          deltaPhiFW = 2 * phiBound - deltaPhiFW;
        std::string lutName = lutObj1;
        lutName += "-";
        lutName += lutObj2;
        long long deltaPhiLUT = m_gtScales->getLUT_DeltaPhi(lutName, deltaPhiFW);
        unsigned int precDeltaPhiLUT = m_gtScales->getPrec_DeltaPhi(lutName);

        int deltaEtaFW = abs(etaORIndex1 - etaIndex2);
        long long deltaEtaLUT = 0;
        unsigned int precDeltaEtaLUT = 0;
        if (!etSumCond) {
          deltaEtaLUT = m_gtScales->getLUT_DeltaEta(lutName, deltaEtaFW);
          precDeltaEtaLUT = m_gtScales->getPrec_DeltaEta(lutName);
        }

        LogDebug("L1TGlobal") << "Obj1 phiFW = " << phiORIndex1 << " Obj2 phiFW = " << phiIndex2 << "\n"
                              << "    DeltaPhiFW = " << deltaPhiFW << "\n"
                              << "    LUT Name = " << lutName << " Prec = " << precDeltaPhiLUT
                              << "  DeltaPhiLUT = " << deltaPhiLUT << "\n"
                              << "Obj1 etaFW = " << etaIndex1 << " Obj1 etaFW = " << etaIndex1 << "\n"
                              << "    DeltaEtaFW = " << deltaEtaFW << "\n"
                              << "    LUT Name = " << lutName << " Prec = " << precDeltaEtaLUT
                              << "  DeltaEtaLUT = " << deltaEtaLUT << std::endl;

        // If there is a OverlapRemovalDeltaEta cut, check it.
        // /////////////////////////////////////////////////
        if (corrPar.corrCutType & 0x10) {
          unsigned int preShift = precDeltaEtaLUT - corrPar.precOverlapRemovalEtaCut;
          LogDebug("L1TGlobal") << "    Testing Leg2 Overlap Removal Delta Eta Cut (" << lutObj1 << "," << lutObj2
                                << ") [" << (long long)(corrPar.minOverlapRemovalEtaCutValue * pow(10, preShift)) << ","
                                << (long long)(corrPar.maxOverlapRemovalEtaCutValue * pow(10, preShift))
                                << "] with precision = " << corrPar.precOverlapRemovalEtaCut << "\n"
                                << "    deltaEtaLUT = " << deltaEtaLUT << "\n"
                                << "    Precision Shift = " << preShift << "\n"
                                << "    deltaEta (shift)= "
                                << (deltaEtaLUT / pow(10, preShift + corrPar.precOverlapRemovalEtaCut)) << "\n"
                                << "    deltaEtaPhy = " << deltaEtaPhy << std::endl;

          //if(preShift>0) deltaEtaLUT /= pow(10,preShift);
          if (deltaEtaLUT >= (long long)(corrPar.minOverlapRemovalEtaCutValue * pow(10, preShift)) &&
              deltaEtaLUT <= (long long)(corrPar.maxOverlapRemovalEtaCutValue * pow(10, preShift))) {
            overlapRemovalMatchLeg2 |= 0x1;
            LogDebug("L1TGlobal") << "    Satisfied Leg2 Overlap Removal Delta Eta Cut ["
                                  << (long long)(corrPar.minOverlapRemovalEtaCutValue * pow(10, preShift)) << ","
                                  << (long long)(corrPar.maxOverlapRemovalEtaCutValue * pow(10, preShift)) << "]"
                                  << std::endl;
            // next leg3 object
            continue;

          } else {
            LogDebug("L1TGlobal") << "    Failed Leg2 Overlap Removal Delta Eta Cut ["
                                  << (long long)(corrPar.minOverlapRemovalEtaCutValue * pow(10, preShift)) << ","
                                  << (long long)(corrPar.maxOverlapRemovalEtaCutValue * pow(10, preShift)) << "]"
                                  << std::endl;
          }
        }
        // If there is a OverlapRemovalDeltaPhi cut, check it.
        // /////////////////////////////////////////////////
        if (corrPar.corrCutType & 0x20) {
          unsigned int preShift = precDeltaPhiLUT - corrPar.precOverlapRemovalPhiCut;
          LogDebug("L1TGlobal") << "    Testing Delta Phi Cut (" << lutObj1 << "," << lutObj2 << ") ["
                                << (long long)(corrPar.minOverlapRemovalPhiCutValue * pow(10, preShift)) << ","
                                << (long long)(corrPar.maxOverlapRemovalPhiCutValue * pow(10, preShift))
                                << "] with precision = " << corrPar.precOverlapRemovalPhiCut << "\n"
                                << "    deltaPhiLUT = " << deltaPhiLUT << "\n"
                                << "    Precision Shift = " << preShift << "\n"
                                << "    deltaPhi (shift)= "
                                << (deltaPhiLUT / pow(10, preShift + corrPar.precOverlapRemovalPhiCut)) << "\n"
                                << "    deltaPhiPhy = " << deltaPhiPhy << std::endl;

          //if(preShift>0) deltaPhiLUT /= pow(10,preShift);
          if (deltaPhiLUT >= (long long)(corrPar.minOverlapRemovalPhiCutValue * pow(10, preShift)) &&
              deltaPhiLUT <= (long long)(corrPar.maxOverlapRemovalPhiCutValue * pow(10, preShift))) {
            overlapRemovalMatchLeg2 |= 0x1;
            LogDebug("L1TGlobal") << "    Satisfied Leg2 Overlap Removal Delta Phi Cut ["
                                  << (long long)(corrPar.minOverlapRemovalPhiCutValue * pow(10, preShift)) << ","
                                  << (long long)(corrPar.maxOverlapRemovalPhiCutValue * pow(10, preShift)) << "]"
                                  << std::endl;
            // next leg3 object
            continue;

          } else {
            LogDebug("L1TGlobal") << "    Failed Leg2 Overlap Removal Delta Phi Cut ["
                                  << (long long)(corrPar.minOverlapRemovalPhiCutValue * pow(10, preShift)) << ","
                                  << (long long)(corrPar.maxOverlapRemovalPhiCutValue * pow(10, preShift)) << "]"
                                  << std::endl;
          }
        }

        //if there is a OverlapRemovalDeltaR cut, check it.
        // /////////////////////////////////////////////////
        if (corrPar.corrCutType & 0x40) {
          //Assumes Delta Eta and Delta Phi LUTs have the same precision
          unsigned int preShift = 2 * precDeltaPhiLUT - corrPar.precOverlapRemovalDRCut;
          double deltaRSqPhy = deltaPhiPhy * deltaPhiPhy + deltaEtaPhy * deltaEtaPhy;
          long long deltaRSq = deltaEtaLUT * deltaEtaLUT + deltaPhiLUT * deltaPhiLUT;

          LogDebug("L1TGlobal") << "    Testing Leg2 Overlap Removal Delta R Cut (" << lutObj1 << "," << lutObj2
                                << ") [" << (long long)(corrPar.minOverlapRemovalDRCutValue * pow(10, preShift)) << ","
                                << (long long)(corrPar.maxOverlapRemovalDRCutValue * pow(10, preShift))
                                << "] with precision = " << corrPar.precOverlapRemovalDRCut << "\n"
                                << "    deltaPhiLUT = " << deltaPhiLUT << "\n"
                                << "    deltaEtaLUT = " << deltaEtaLUT << "\n"
                                << "    deltaRSqLUT = " << deltaRSq << "\n"
                                << "    Precision Shift = " << preShift << "\n"
                                << "    deltaRSqLUT (shift)= " << (deltaRSq / pow(10, preShift + corrPar.precDRCut))
                                << "\n"
                                << "    deltaRSqPhy = " << deltaRSqPhy << std::endl;

          //if(preShift>0) deltaRSq /= pow(10,preShift);
          if (deltaRSq >= (long long)(corrPar.minOverlapRemovalDRCutValue * pow(10, preShift)) &&
              deltaRSq <= (long long)(corrPar.maxOverlapRemovalDRCutValue * pow(10, preShift))) {
            overlapRemovalMatchLeg2 |= 0x1;
            LogDebug("L1TGlobal") << "    Satisfied Leg2 Overlap Removal Delta R Cut ["
                                  << (long long)(corrPar.minOverlapRemovalDRCutValue * pow(10, preShift)) << ","
                                  << (long long)(corrPar.maxOverlapRemovalDRCutValue * pow(10, preShift)) << "]"
                                  << std::endl;
            // next leg3 object
            continue;

          } else {
            LogDebug("L1TGlobal") << "    Failed Leg2 Overlap Removal Delta R Cut ["
                                  << (int)(corrPar.minOverlapRemovalDRCutValue * pow(10, preShift)) << ","
                                  << (long long)(corrPar.maxOverlapRemovalDRCutValue * pow(10, preShift)) << "]"
                                  << std::endl;
          }
        }

      }  // end loop over combinations in overlap-removal leg.

      // skip object leg2 if matched with overlap removal object
      // ///////////////////////////////////////////////////////
      if (overlapRemovalMatchLeg2 == 0x1) {
        LogDebug("L1TGlobal") << "   Remove Object of Leg2: Satisfied Overlap Removal Cuts" << std::endl;
        continue;
      } else {
        LogDebug("L1TGlobal") << "   Keep Object of Leg2: Failed Overlap Removal Cuts " << std::endl;
      }

      // ///////////////////////////////////////////////////////
      // At this point, both correlation legs are cleared of overlaps
      // ///////////////////////////////////////////////////////

      if (m_verbosity) {
        LogDebug("L1TGlobal") << "    Correlation pair [" << l1TGtObjectEnumToString(cndObjTypeVec[0]) << ", "
                              << l1TGtObjectEnumToString(cndObjTypeVec[1]) << "] with collection indices [" << obj0Index
                              << ", " << obj1Index << "] "
                              << " has: \n"
                              << "     Et  value   = [" << etIndex0 << ", " << etIndex1 << "]\n"
                              << "     phi indices = [" << phiIndex0 << ", " << phiIndex1 << "]\n"
                              << "     eta indices = [" << etaIndex0 << ", " << etaIndex1 << "]\n"
                              << "     chrg        = [" << chrg0 << ", " << chrg1 << "]\n"
                              << std::endl;
      }

      // Now perform the desired correlation on these two objects. Assume true until we find a contradition
      bool reqResult = true;

      // clear the indices in the combination
      objectsInComb.clear();

      objectsInComb.push_back(obj0Index);
      objectsInComb.push_back(obj1Index);

      // if we get here all checks were successful for this combination
      // set the general result for evaluateCondition to "true"

      // These all require some delta eta and phi calculations.  Do them first...for now real calculation but need to
      // revise this to line up with firmware calculations.
      double deltaPhiPhy = fabs(phi1Phy - phi0Phy);
      if (deltaPhiPhy > M_PI)
        deltaPhiPhy = 2. * M_PI - deltaPhiPhy;
      double deltaEtaPhy = fabs(eta1Phy - eta0Phy);

      // Determine the integer based delta eta and delta phi
      int deltaPhiFW = abs(phiIndex0 - phiIndex1);
      if (deltaPhiFW >= phiBound)
        deltaPhiFW = 2 * phiBound - deltaPhiFW;
      std::string lutName = lutObj0;
      lutName += "-";
      lutName += lutObj1;
      long long deltaPhiLUT = m_gtScales->getLUT_DeltaPhi(lutName, deltaPhiFW);
      unsigned int precDeltaPhiLUT = m_gtScales->getPrec_DeltaPhi(lutName);

      int deltaEtaFW = abs(etaIndex0 - etaIndex1);
      long long deltaEtaLUT = 0;
      unsigned int precDeltaEtaLUT = 0;
      if (!etSumCond) {
        deltaEtaLUT = m_gtScales->getLUT_DeltaEta(lutName, deltaEtaFW);
        precDeltaEtaLUT = m_gtScales->getPrec_DeltaEta(lutName);
      }

      //
      LogDebug("L1TGlobal") << "Obj0 phiFW = " << phiIndex0 << " Obj1 phiFW = " << phiIndex1 << "\n"
                            << "    DeltaPhiFW = " << deltaPhiFW << "\n"
                            << "    LUT Name = " << lutName << " Prec = " << precDeltaPhiLUT
                            << "  DeltaPhiLUT = " << deltaPhiLUT << "\n"
                            << "Obj0 etaFW = " << etaIndex0 << " Obj1 etaFW = " << etaIndex1 << "\n"
                            << "    DeltaEtaFW = " << deltaEtaFW << "\n"
                            << "    LUT Name = " << lutName << " Prec = " << precDeltaEtaLUT
                            << "  DeltaEtaLUT = " << deltaEtaLUT << std::endl;

      // If there is a delta eta, check it.
      if (corrPar.corrCutType & 0x1) {
        unsigned int preShift = precDeltaEtaLUT - corrPar.precEtaCut;
        LogDebug("L1TGlobal") << "    Testing Delta Eta Cut (" << lutObj0 << "," << lutObj1 << ") ["
                              << (long long)(corrPar.minEtaCutValue * pow(10, preShift)) << ","
                              << (long long)(corrPar.maxEtaCutValue * pow(10, preShift))
                              << "] with precision = " << corrPar.precEtaCut << "\n"
                              << "    deltaEtaLUT = " << deltaEtaLUT << "\n"
                              << "    Precision Shift = " << preShift << "\n"
                              << "    deltaEta (shift)= " << (deltaEtaLUT / pow(10, preShift + corrPar.precEtaCut))
                              << "\n"
                              << "    deltaEtaPhy = " << deltaEtaPhy << std::endl;

        //if(preShift>0) deltaEtaLUT /= pow(10,preShift);
        if (deltaEtaLUT >= (long long)(corrPar.minEtaCutValue * pow(10, preShift)) &&
            deltaEtaLUT <= (long long)(corrPar.maxEtaCutValue * pow(10, preShift))) {
          LogDebug("L1TGlobal") << "    Passed Delta Eta Cut ["
                                << (long long)(corrPar.minEtaCutValue * pow(10, preShift)) << ","
                                << (long long)(corrPar.maxEtaCutValue * pow(10, preShift)) << "]" << std::endl;

        } else {
          LogDebug("L1TGlobal") << "    Failed Delta Eta Cut ["
                                << (long long)(corrPar.minEtaCutValue * pow(10, preShift)) << ","
                                << (long long)(corrPar.maxEtaCutValue * pow(10, preShift)) << "]" << std::endl;
          reqResult = false;
        }
      }

      //if there is a delta phi check it.
      if (corrPar.corrCutType & 0x2) {
        unsigned int preShift = precDeltaPhiLUT - corrPar.precPhiCut;
        LogDebug("L1TGlobal") << "    Testing Delta Phi Cut (" << lutObj0 << "," << lutObj1 << ") ["
                              << (long long)(corrPar.minPhiCutValue * pow(10, preShift)) << ","
                              << (long long)(corrPar.maxPhiCutValue * pow(10, preShift))
                              << "] with precision = " << corrPar.precPhiCut << "\n"
                              << "    deltaPhiLUT = " << deltaPhiLUT << "\n"
                              << "    Precision Shift = " << preShift << "\n"
                              << "    deltaPhi (shift)= " << (deltaPhiLUT / pow(10, preShift + corrPar.precPhiCut))
                              << "\n"
                              << "    deltaPhiPhy = " << deltaPhiPhy << std::endl;

        //if(preShift>0) deltaPhiLUT /= pow(10,preShift);
        if (deltaPhiLUT >= (long long)(corrPar.minPhiCutValue * pow(10, preShift)) &&
            deltaPhiLUT <= (long long)(corrPar.maxPhiCutValue * pow(10, preShift))) {
          LogDebug("L1TGlobal") << "    Passed Delta Phi Cut ["
                                << (long long)(corrPar.minPhiCutValue * pow(10, preShift)) << ","
                                << (long long)(corrPar.maxPhiCutValue * pow(10, preShift)) << "]" << std::endl;

        } else {
          LogDebug("L1TGlobal") << "    Failed Delta Phi Cut ["
                                << (long long)(corrPar.minPhiCutValue * pow(10, preShift)) << ","
                                << (long long)(corrPar.maxPhiCutValue * pow(10, preShift)) << "]" << std::endl;
          reqResult = false;
        }
      }

      if (corrPar.corrCutType & 0x4) {
        //Assumes Delta Eta and Delta Phi LUTs have the same precision
        unsigned int preShift = 2 * precDeltaPhiLUT - corrPar.precDRCut;
        double deltaRSqPhy = deltaPhiPhy * deltaPhiPhy + deltaEtaPhy * deltaEtaPhy;
        long long deltaRSq = deltaEtaLUT * deltaEtaLUT + deltaPhiLUT * deltaPhiLUT;

        LogDebug("L1TGlobal") << "    Testing Delta R Cut (" << lutObj0 << "," << lutObj1 << ") ["
                              << (long long)(corrPar.minDRCutValue * pow(10, preShift)) << ","
                              << (long long)(corrPar.maxDRCutValue * pow(10, preShift))
                              << "] with precision = " << corrPar.precDRCut << "\n"
                              << "    deltaPhiLUT = " << deltaPhiLUT << "\n"
                              << "    deltaEtaLUT = " << deltaEtaLUT << "\n"
                              << "    deltaRSqLUT = " << deltaRSq << "\n"
                              << "    Precision Shift = " << preShift << "\n"
                              << "    deltaRSqLUT (shift)= " << (deltaRSq / pow(10, preShift + corrPar.precDRCut))
                              << "\n"
                              << "    deltaRSqPhy = " << deltaRSqPhy << std::endl;

        //if(preShift>0) deltaRSq /= pow(10,preShift);
        if (deltaRSq >= (long long)(corrPar.minDRCutValue * pow(10, preShift)) &&
            deltaRSq <= (long long)(corrPar.maxDRCutValue * pow(10, preShift))) {
          LogDebug("L1TGlobal") << "    Passed Delta R Cut [" << (long long)(corrPar.minDRCutValue * pow(10, preShift))
                                << "," << (long long)(corrPar.maxDRCutValue * pow(10, preShift)) << "]" << std::endl;

        } else {
          LogDebug("L1TGlobal") << "    Failed Delta R Cut [" << (int)(corrPar.minDRCutValue * pow(10, preShift)) << ","
                                << (long long)(corrPar.maxDRCutValue * pow(10, preShift)) << "]" << std::endl;
          reqResult = false;
        }
      }

      if (corrPar.corrCutType & 0x20) {
        // Two body pt: pt^2 = pt1^2+pt2^2+2*pt1*pt2*(cos(phi1)*cos(phi2)+sin(phi1)*sin(phi2)).

        LogDebug("L1TGlobal") << " corrPar.corrCutType: " << corrPar.corrCutType << "\n";

        //calculate math sins and cosines for debugging
        double cosPhi1Phy = cos(phi0Phy);
        double sinPhi1Phy = sin(phi0Phy);
        double cosPhi2Phy = cos(phi1Phy);
        double sinPhi2Phy = sin(phi1Phy);

        double tbptSqPhy = et0Phy * et0Phy + et1Phy * et1Phy +
                           2 * et0Phy * et1Phy * (cosPhi1Phy * cosPhi2Phy + sinPhi1Phy * sinPhi2Phy);
        // get values from LUT's

        const std::string& lutName0 = lutObj0;
        unsigned int precCosLUT0 = m_gtScales->getPrec_Cos(lutName0);
        unsigned int precSinLUT0 = m_gtScales->getPrec_Sin(lutName0);

        const std::string& lutName1 = lutObj1;
        unsigned int precCosLUT1 = m_gtScales->getPrec_Cos(lutName1);
        unsigned int precSinLUT1 = m_gtScales->getPrec_Sin(lutName1);

        if (precCosLUT0 - precCosLUT1 != 0)
          LogDebug("L1TGlobal") << "Warning: Cos LUTs for TwoBodyPt on different Precision" << std::endl;
        if (precSinLUT0 - precSinLUT1 != 0)
          LogDebug("L1TGlobal") << "Warning: Sin LUTs for TwoBodyPt on different Precision" << std::endl;
        if (precSinLUT0 - precCosLUT1 != 0)
          LogDebug("L1TGlobal") << "Warning: Sin and Cos LUTs for TwoBodyPt on different Precision" << std::endl;
        if (precSinLUT1 - precCosLUT0 != 0)
          LogDebug("L1TGlobal") << "Warning: Sin and Cos LUTs for TwoBodyPt on different Precision" << std::endl;

        long long cosPhi1LUT = m_gtScales->getLUT_Cos(lutName0, phiIndex0);
        long long sinPhi1LUT = m_gtScales->getLUT_Sin(lutName0, phiIndex0);

        long long cosPhi2LUT = m_gtScales->getLUT_Cos(lutName1, phiIndex1);
        long long sinPhi2LUT = m_gtScales->getLUT_Sin(lutName1, phiIndex1);

        // now get pt LUTs
        std::string lutName = lutObj0;
        lutName += "-ET";
        long long ptObj0 = m_gtScales->getLUT_Pt("TwoBody_" + lutName, etIndex0);
        unsigned int precPtLUTObj0 = m_gtScales->getPrec_Pt("TwoBody_" + lutName);

        lutName = lutObj1;
        lutName += "-ET";
        long long ptObj1 = m_gtScales->getLUT_Pt("TwoBody_" + lutName, etIndex1);
        unsigned int precPtLUTObj1 = m_gtScales->getPrec_Pt("TwoBody_" + lutName);

        LogTrace("L1TGlobal") << " TBPT Trig precisions:\t " << precCosLUT0 << "\t" << precCosLUT1 << "\t"
                              << precSinLUT0 << "\t" << precSinLUT1;
        LogTrace("L1TGlobal") << " TBPT Pt precisions:\t " << precPtLUTObj0 << "\t" << precPtLUTObj1;
        LogTrace("L1TGlobal") << " TBPT Pt cut:\t " << corrPar.minTBPTCutValue << "\tPrecTBPTCut\t"
                              << corrPar.precTBPTCut;
        LogTrace("L1TGlobal") << " TBPT Pt1*Pt1 -- Phys:\t " << et0Phy * et0Phy << "\tHW:\t"
                              << ptObj0 * ptObj0 * (pow(10, 6));
        LogTrace("L1TGlobal") << " TBPT Pt2*Pt2 -- Phys:\t " << et1Phy * et1Phy << "\tHW:\t"
                              << ptObj1 * ptObj1 * (pow(10, 6));
        LogTrace("L1TGlobal") << " TBPT 2Pt1*Pt2 -- Phys:\t " << 2 * et0Phy * et1Phy << "\tHW:\t"
                              << 2 * (ptObj0 * pow(10, 0)) * (ptObj1 * pow(10, 0));
        LogTrace("L1TGlobal") << " TBPT Trig -- Phys:\t " << cosPhi1Phy * cosPhi2Phy + sinPhi1Phy * sinPhi2Phy
                              << "\tHW:\t" << cosPhi1LUT * cosPhi2LUT + sinPhi1LUT * sinPhi2LUT;

        //double tbptSqPhy =   et0Phy*et0Phy             + et1Phy*et1Phy + 2*et0Phy*et1Phy*(cosPhi1Phy*cosPhi2Phy + sinPhi1Phy*sinPhi2Phy);
        long long tbptSqHW = ptObj0 * ptObj0 * (pow(10, 2 * precCosLUT0)) +
                             ptObj1 * ptObj1 * (pow(10, 2 * precCosLUT0)) +
                             2 * ptObj0 * ptObj1 * (cosPhi1LUT * cosPhi2LUT + sinPhi1LUT * sinPhi2LUT);

        unsigned int preShift = precPtLUTObj0 + precPtLUTObj1 + 2 * precCosLUT0;

        LogTrace("L1TGlobal") << "TBPT Result -- Phys: " << tbptSqPhy << "\tHW: " << tbptSqHW << "\tShifted\t"
                              << tbptSqHW / pow(10, preShift) << std::endl;

        preShift = preShift - corrPar.precTBPTCut;

        LogDebug("L1TGlobal")
            << "    Testing Two Body Pt Cut (" << lutObj0 << "," << lutObj1 << ") ["
            << (long long)(corrPar.minTBPTCutValue * pow(10, preShift)) << ","
            << (long long)(corrPar.maxTBPTCutValue * pow(10, preShift)) << "] with precision = " << corrPar.precTBPTCut
            << "\n"
            << "    etIndex0     = " << etIndex0 << "    pt0LUT      = " << ptObj0 << " PhyEt0 = " << et0Phy << "\n"
            << "    etIndex1     = " << etIndex1 << "    pt1LUT      = " << ptObj1 << " PhyEt1 = " << et1Phy << "\n"
            << "    Precision Shift = " << preShift << "\n"
            << "    Sin(phi1): LUT/Phys\t " << sinPhi1LUT << " / " << sinPhi1Phy << "\n"
            << "    Sin(phi2): LUT/Phys\t " << sinPhi2LUT << " / " << sinPhi2Phy << "\n"
            << "    Cos(phi1): LUT/Phys\t " << cosPhi1LUT << " / " << cosPhi1Phy << "\n"
            << "    Cos(phi2): LUT/Phys\t " << cosPhi2LUT << " / " << cosPhi2Phy
            << "\n"

            //    << "    deltaPhiLUT = " << deltaPhiLUT << "\n"
            //    << "    deltaEtaLUT = " << deltaEtaLUT << "\n"
            //    << "    deltaRSqLUT = " << deltaRSq <<  "\n"
            //    << "    Precision Shift = " << preShift << "\n"
            //    << "    deltaRSqLUT (shift)= " << (deltaRSq/pow(10,preShift+corrPar.precDRCut))	<< "\n"
            //    << "    deltaRSqPhy = " << deltaRSqPhy
            << std::endl;

        if (tbptSqHW > 0. && tbptSqHW >= (long long)(corrPar.minTBPTCutValue * pow(10, preShift))) {
          LogDebug("L1TGlobal") << "    Passed Two Body pT Cut ["
                                << (long long)(corrPar.minTBPTCutValue * pow(10, preShift)) << "]"
                                << "\twith value: " << tbptSqHW << "\n"
                                << "\tPhysics Cut[" << corrPar.minTBPTCutValue / pow(10, corrPar.precTBPTCut)
                                << "]\tPhysics Value: " << tbptSqPhy << std::endl;

        } else {
          LogDebug("L1TGlobal") << "    Failed Two Body pT Cut ["
                                << (long long)(corrPar.minTBPTCutValue * pow(10, preShift)) << "]"
                                << "\t with value: " << tbptSqHW << "\n"
                                << "\tPhysics Cut[" << corrPar.minTBPTCutValue / pow(10, corrPar.precTBPTCut)
                                << "]\tPhysics Value: " << tbptSqPhy << std::endl;
          reqResult = false;
        }
      }

      if (corrPar.corrCutType & 0x8 || corrPar.corrCutType & 0x10) {
        //invariant mass calculation based on
        // M = sqrt(2*p1*p2(cosh(eta1-eta2) - cos(phi1 - phi2)))
        // but we calculate (1/2)M^2
        //
        double cosDeltaPhiPhy = cos(deltaPhiPhy);
        double coshDeltaEtaPhy = cosh(deltaEtaPhy);
        if (corrPar.corrCutType & 0x10)
          coshDeltaEtaPhy = 1.;
        double massSqPhy = et0Phy * et1Phy * (coshDeltaEtaPhy - cosDeltaPhiPhy);

        long long cosDeltaPhiLUT = m_gtScales->getLUT_DeltaPhi_Cos(lutName, deltaPhiFW);
        unsigned int precCosLUT = m_gtScales->getPrec_DeltaPhi_Cos(lutName);

        long long coshDeltaEtaLUT;
        if (corrPar.corrCutType & 0x10) {
          coshDeltaEtaLUT = 1 * pow(10, precCosLUT);
        } else {
          coshDeltaEtaLUT = m_gtScales->getLUT_DeltaEta_Cosh(lutName, deltaEtaFW);
          unsigned int precCoshLUT = m_gtScales->getPrec_DeltaEta_Cosh(lutName);
          if (precCoshLUT - precCosLUT != 0)
            LogDebug("L1TGlobal") << "Warning: Cos and Cosh LUTs on different Precision" << std::endl;
        }

        std::string lutName = lutObj0;
        lutName += "-ET";
        long long ptObj0 = m_gtScales->getLUT_Pt("Mass_" + lutName, etIndex0);
        unsigned int precPtLUTObj0 = m_gtScales->getPrec_Pt("Mass_" + lutName);

        lutName = lutObj1;
        lutName += "-ET";
        long long ptObj1 = m_gtScales->getLUT_Pt("Mass_" + lutName, etIndex1);
        unsigned int precPtLUTObj1 = m_gtScales->getPrec_Pt("Mass_" + lutName);

        // Pt and Angles are at different precission.
        long long massSq = ptObj0 * ptObj1 * (coshDeltaEtaLUT - cosDeltaPhiLUT);

        //Note: There is an assumption here that Cos and Cosh have the same precission
        unsigned int preShift = precPtLUTObj0 + precPtLUTObj1 + precCosLUT - corrPar.precMassCut;

        LogDebug("L1TGlobal") << "    Testing Invariant Mass (" << lutObj0 << "," << lutObj1 << ") ["
                              << (long long)(corrPar.minMassCutValue * pow(10, preShift)) << ","
                              << (long long)(corrPar.maxMassCutValue * pow(10, preShift))
                              << "] with precision = " << corrPar.precMassCut << "\n"
                              << "    deltaPhiLUT  = " << deltaPhiLUT << "  cosLUT  = " << cosDeltaPhiLUT << "\n"
                              << "    deltaEtaLUT  = " << deltaEtaLUT << "  coshLUT = " << coshDeltaEtaLUT << "\n"
                              << "    etIndex0     = " << etIndex0 << "    pt0LUT      = " << ptObj0
                              << " PhyEt0 = " << et0Phy << "\n"
                              << "    etIndex1     = " << etIndex1 << "    pt1LUT      = " << ptObj1
                              << " PhyEt1 = " << et1Phy << "\n"
                              << "    massSq/2     = " << massSq << "\n"
                              << "    Precision Shift = " << preShift << "\n"
                              << "    massSq   (shift)= " << (massSq / pow(10, preShift + corrPar.precMassCut)) << "\n"
                              << "    deltaPhiPhy  = " << deltaPhiPhy << "  cos() = " << cosDeltaPhiPhy << "\n"
                              << "    deltaEtaPhy  = " << deltaEtaPhy << "  cosh()= " << coshDeltaEtaPhy << "\n"
                              << "    massSqPhy/2  = " << massSqPhy
                              << "  sqrt(|massSq|) = " << sqrt(fabs(2. * massSqPhy)) << std::endl;

        //if(preShift>0) massSq /= pow(10,preShift);
        if (massSq >= 0 && massSq >= (long long)(corrPar.minMassCutValue * pow(10, preShift)) &&
            massSq <= (long long)(corrPar.maxMassCutValue * pow(10, preShift))) {
          LogDebug("L1TGlobal") << "    Passed Invariant Mass Cut ["
                                << (long long)(corrPar.minMassCutValue * pow(10, preShift)) << ","
                                << (long long)(corrPar.maxMassCutValue * pow(10, preShift)) << "]" << std::endl;

        } else {
          LogDebug("L1TGlobal") << "    Failed Invariant Mass Cut ["
                                << (long long)(corrPar.minMassCutValue * pow(10, preShift)) << ","
                                << (long long)(corrPar.maxMassCutValue * pow(10, preShift)) << "]" << std::endl;
          reqResult = false;
        }
      }

      // For Muon-Muon Correlation Check the Charge Correlation if requested
      bool chrgCorrel = true;
      if (cond0Categ == CondMuon && cond1Categ == CondMuon) {
        // Check for like-sign
        if (corrPar.chargeCorrelation == 2 && chrg0 != chrg1)
          chrgCorrel = false;
        // Check for opp-sign
        if (corrPar.chargeCorrelation == 4 && chrg0 == chrg1)
          chrgCorrel = false;
      }

      if (reqResult & chrgCorrel) {
        condResult = true;
        (combinationsInCond()).push_back(objectsInComb);
      }

    }  //end loop over second leg

  }  //end loop over first leg

  if (m_verbosity && condResult) {
    LogDebug("L1TGlobal") << " pass(es) the correlation condition.\n" << std::endl;
  }

  return condResult;
}

// load calo candidates
const l1t::L1Candidate* l1t::CorrWithOverlapRemovalCondition::getCandidate(const int bx, const int indexCand) const {
  // objectType() gives the type for nrObjects() only,
  // but in a CondCalo all objects have the same type
  // take type from the type of the first object
  switch ((m_gtCorrelationWithOverlapRemovalTemplate->objectType())[0]) {
    case gtEG:
      return (m_uGtB->getCandL1EG())->at(bx, indexCand);
      break;

    case gtJet:
      return (m_uGtB->getCandL1Jet())->at(bx, indexCand);
      break;

    case gtTau:
      return (m_uGtB->getCandL1Tau())->at(bx, indexCand);
      break;
    default:
      return nullptr;
      break;
  }

  return nullptr;
}

/**
 * checkObjectParameter - Compare a single particle with a numbered condition.
 *
 * @param iCondition The number of the condition.
 * @param cand The candidate to compare.
 *
 * @return The result of the comparison (false if a condition does not exist).
 */

const bool l1t::CorrWithOverlapRemovalCondition::checkObjectParameter(const int iCondition,
                                                                      const l1t::L1Candidate& cand) const {
  return true;
}

void l1t::CorrWithOverlapRemovalCondition::print(std::ostream& myCout) const {
  myCout << "Dummy Print for CorrWithOverlapRemovalCondition" << std::endl;
  m_gtCorrelationWithOverlapRemovalTemplate->print(myCout);

  ConditionEvaluation::print(myCout);
}
