/**
 * \class CorrThreeBodyCondition
 *
 * \orig author: Elisa Fontanesi - Boston University
 *               CorrCondition and CorrWithOverlapRemovalCondition classes used as a starting point
 *
 * Description: L1 Global Trigger three-body correlation conditions:                                                                                     
 *              evaluation of a three-body correlation condition (= three-muon invariant mass)
 *
 */

// this class header
#include "L1Trigger/L1TGlobal/interface/CorrCondition.h"
#include "L1Trigger/L1TGlobal/interface/CorrThreeBodyCondition.h"

// system include files
#include <iostream>
#include <iomanip>

#include <string>
#include <vector>
#include <algorithm>

// user include files
//   base classes
#include "L1Trigger/L1TGlobal/interface/CorrelationTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CorrelationThreeBodyTemplate.h"
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"

#include "L1Trigger/L1TGlobal/interface/MuCondition.h"
#include "L1Trigger/L1TGlobal/interface/MuonTemplate.h"
#include "L1Trigger/L1TGlobal/interface/GlobalScales.h"
#include "L1Trigger/L1TGlobal/interface/GlobalBoard.h"

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors
//     default
l1t::CorrThreeBodyCondition::CorrThreeBodyCondition() : ConditionEvaluation() {}

//     from base template condition (from event setup usually)
l1t::CorrThreeBodyCondition::CorrThreeBodyCondition(const GlobalCondition* corrTemplate,
                                                    const GlobalCondition* cond0Condition,
                                                    const GlobalCondition* cond1Condition,
                                                    const GlobalCondition* cond2Condition,
                                                    const GlobalBoard* ptrGTB)
    : ConditionEvaluation(),
      m_gtCorrelationThreeBodyTemplate(static_cast<const CorrelationThreeBodyTemplate*>(corrTemplate)),
      m_gtCond0(cond0Condition),
      m_gtCond1(cond1Condition),
      m_gtCond2(cond2Condition),
      m_uGtB(ptrGTB) {}

// copy constructor
void l1t::CorrThreeBodyCondition::copy(const l1t::CorrThreeBodyCondition& cp) {
  m_gtCorrelationThreeBodyTemplate = cp.gtCorrelationThreeBodyTemplate();
  m_uGtB = cp.getuGtB();

  m_condMaxNumberObjects = cp.condMaxNumberObjects();
  m_condLastResult = cp.condLastResult();
  m_combinationsInCond = cp.getCombinationsInCond();

  m_verbosity = cp.m_verbosity;
}

l1t::CorrThreeBodyCondition::CorrThreeBodyCondition(const l1t::CorrThreeBodyCondition& cp) : ConditionEvaluation() {
  copy(cp);
}

// destructor
l1t::CorrThreeBodyCondition::~CorrThreeBodyCondition() {
  // empty
}

// equal operator
l1t::CorrThreeBodyCondition& l1t::CorrThreeBodyCondition::operator=(const l1t::CorrThreeBodyCondition& cp) {
  copy(cp);
  return *this;
}

///   set the pointer to uGT GlobalBoard
void l1t::CorrThreeBodyCondition::setuGtB(const GlobalBoard* ptrGTB) { m_uGtB = ptrGTB; }

void l1t::CorrThreeBodyCondition::setScales(const GlobalScales* sc) { m_gtScales = sc; }

// try all object permutations and check spatial correlations, if required
const bool l1t::CorrThreeBodyCondition::evaluateCondition(const int bxEval) const {
  if (m_verbosity) {
    std::ostringstream myCout;
    m_gtCorrelationThreeBodyTemplate->print(myCout);
    LogDebug("L1TGlobal") << "Three-body Correlation Condition Evaluation..." << std::endl;
  }

  bool condResult = false;

  // number of objects in the condition (three) and their type
  int nObjInCond = 3;
  std::vector<GlobalObject> cndObjTypeVec(nObjInCond);

  // evaluate first the three subconditions (Type1s)
  const GtConditionCategory cond0Categ = m_gtCorrelationThreeBodyTemplate->cond0Category();
  const GtConditionCategory cond1Categ = m_gtCorrelationThreeBodyTemplate->cond1Category();
  const GtConditionCategory cond2Categ = m_gtCorrelationThreeBodyTemplate->cond2Category();

  const MuonTemplate* corrMuon = nullptr;

  CombinationsInCond cond0Comb;
  CombinationsInCond cond1Comb;
  CombinationsInCond cond2Comb;

  int cond0bx(0);
  int cond1bx(0);
  int cond2bx(0);

  // FIRST OBJECT
  bool reqObjResult = false;

  if (cond0Categ == CondMuon) {
    LogDebug("L1TGlobal") << "\n --------------------- First muon checks ---------------------" << std::endl;
    corrMuon = static_cast<const MuonTemplate*>(m_gtCond0);
    MuCondition muCondition(corrMuon, m_uGtB, 0, 0);

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
  } else {
    // Interested only in three-muon correlations
    LogDebug("L1TGlobal") << "CondMuon not satisfied for Leg 0" << std::endl;
    return false;
  }

  // return if first subcondition is false
  if (!reqObjResult) {
    LogDebug("L1TGlobal") << "\n  First subcondition false, second subcondition not evaluated and not printed."
                          << std::endl;
    return false;
  }

  // SECOND OBJECT
  reqObjResult = false;

  if (cond1Categ == CondMuon) {
    LogDebug("L1TGlobal") << "\n --------------------- Second muon checks ---------------------" << std::endl;
    corrMuon = static_cast<const MuonTemplate*>(m_gtCond1);
    MuCondition muCondition(corrMuon, m_uGtB, 0, 0);

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
  }

  else {
    // Interested only in three-muon correlations
    LogDebug("L1TGlobal") << "CondMuon not satisfied for Leg 1" << std::endl;
    return false;
  }

  // return if second subcondition is false
  if (!reqObjResult) {
    LogDebug("L1TGlobal") << "\n  Second subcondition false, third subcondition not evaluated and not printed."
                          << std::endl;
    return false;
  }

  // THIRD OBJECT
  reqObjResult = false;

  if (cond2Categ == CondMuon) {
    LogDebug("L1TGlobal") << "\n --------------------- Third muon checks ---------------------" << std::endl;
    corrMuon = static_cast<const MuonTemplate*>(m_gtCond2);
    MuCondition muCondition(corrMuon, m_uGtB, 0, 0);

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
  }

  else {
    // Interested only in three-muon correlations
    LogDebug("L1TGlobal") << "CondMuon not satisfied for Leg 2" << std::endl;
    return false;
  }

  // Return if third subcondition is false
  if (!reqObjResult) {
    return false;
  } else {
    LogDebug("L1TGlobal")
        << "\n"
        << "Found three objects satisfying subconditions: evaluate three-body correlation requirements.\n"
        << std::endl;
  }

  // Since we have three good legs get the correlation parameters
  CorrelationThreeBodyTemplate::CorrelationThreeBodyParameter corrPar =
      *(m_gtCorrelationThreeBodyTemplate->correlationThreeBodyParameter());

  // Vector to store the indices of the objects involved in the condition evaluation
  SingleCombInCond objectsInComb;
  objectsInComb.reserve(nObjInCond);

  // Clear the m_combinationsInCond vector:
  // it will store the set of objects satisfying the condition evaluated as true
  (combinationsInCond()).clear();

  // Pointers to objects
  const BXVector<const l1t::Muon*>* candMuVec = nullptr;

  // Make the conversions of the indices, depending on the combination of objects involved
  int phiIndex0 = 0;
  double phi0Phy = 0.;
  int phiIndex1 = 0;
  double phi1Phy = 0.;
  int phiIndex2 = 0;
  double phi2Phy = 0.;

  int etaIndex0 = 0;
  double eta0Phy = 0.;
  int etaBin0 = 0;
  int etaIndex1 = 0;
  double eta1Phy = 0.;
  int etaBin1 = 0;
  int etaIndex2 = 0;
  double eta2Phy = 0.;
  int etaBin2 = 0;

  int etIndex0 = 0;
  int etBin0 = 0;
  double et0Phy = 0.;
  int etIndex1 = 0;
  int etBin1 = 0;
  double et1Phy = 0.;
  int etIndex2 = 0;
  int etBin2 = 0;
  double et2Phy = 0.;

  // Charges to take into account the charge correlation
  int chrg0 = -1;
  int chrg1 = -1;
  int chrg2 = -1;

  // Determine the number of phi bins to get cutoff at pi
  int phiBound = 0;
  if (cond0Categ == CondMuon || cond1Categ == CondMuon || cond2Categ == CondMuon) {
    const GlobalScales::ScaleParameters& par = m_gtScales->getMUScales();
    phiBound = (int)((par.phiMax - par.phiMin) / par.phiStep) / 2;
  } else {
    //Assumes all objects are on same phi scale
    const GlobalScales::ScaleParameters& par = m_gtScales->getEGScales();
    phiBound = (int)((par.phiMax - par.phiMin) / par.phiStep) / 2;
  }
  LogDebug("L1TGlobal") << "Phi Bound = " << phiBound << std::endl;

  // Keep track of objects for LUTS
  std::string lutObj0 = "NULL";
  std::string lutObj1 = "NULL";
  std::string lutObj2 = "NULL";

  LogTrace("L1TGlobal") << "  Number of objects satisfying the subcondition 0: " << (cond0Comb.size()) << std::endl;
  LogTrace("L1TGlobal") << "  Number of objects satisfying the subcondition 1: " << (cond1Comb.size()) << std::endl;
  LogTrace("L1TGlobal") << "  Number of objects satisfying the subcondition 2: " << (cond2Comb.size()) << std::endl;

  ////////////////////////////////
  // LOOP OVER ALL COMBINATIONS //
  ////////////////////////////////
  unsigned int preShift = 0;

  // *** Looking for a set of three objects
  for (std::vector<SingleCombInCond>::const_iterator it0Comb = cond0Comb.begin(); it0Comb != cond0Comb.end();
       it0Comb++) {
    // Type1s: there is 1 object only, no need for a loop, index 0 should be OK in (*it0Comb)[0]
    // ... but add protection to not crash
    LogDebug("L1TGlobal") << "Looking at first subcondition" << std::endl;
    int obj0Index = -1;

    if (!(*it0Comb).empty()) {
      obj0Index = (*it0Comb)[0];
    } else {
      LogTrace("L1TGlobal") << "\n  SingleCombInCond (*it0Comb).size() " << ((*it0Comb).size()) << std::endl;
      return false;
    }

    // FIRST OBJECT: Collect the information on the first leg of the correlation
    if (cond0Categ == CondMuon) {
      lutObj0 = "MU";
      candMuVec = m_uGtB->getCandL1Mu();
      phiIndex0 = (candMuVec->at(cond0bx, obj0Index))->hwPhiAtVtx();
      etaIndex0 = (candMuVec->at(cond0bx, obj0Index))->hwEtaAtVtx();
      etIndex0 = (candMuVec->at(cond0bx, obj0Index))->hwPt();
      chrg0 = (candMuVec->at(cond0bx, obj0Index))->hwCharge();

      etaBin0 = etaIndex0;
      if (etaBin0 < 0)
        etaBin0 = m_gtScales->getMUScales().etaBins.size() + etaBin0;

      etBin0 = etIndex0;
      int ssize = m_gtScales->getMUScales().etBins.size();
      if (etBin0 >= ssize) {
        etBin0 = ssize - 1;
        LogTrace("L1TGlobal") << "MU0 hw et" << etBin0 << " out of scale range.  Setting to maximum.";
      }

      // Determine Floating Pt numbers for floating point calculation
      std::pair<double, double> binEdges = m_gtScales->getMUScales().phiBins.at(phiIndex0);
      phi0Phy = 0.5 * (binEdges.second + binEdges.first);
      binEdges = m_gtScales->getMUScales().etaBins.at(etaBin0);
      eta0Phy = 0.5 * (binEdges.second + binEdges.first);
      binEdges = m_gtScales->getMUScales().etBins.at(etBin0);
      et0Phy = 0.5 * (binEdges.second + binEdges.first);
      LogDebug("L1TGlobal") << "Found all quantities for MU0" << std::endl;
    } else {
      // Interested only in three-muon correlations
      LogDebug("L1TGlobal") << "CondMuon not satisfied for Leg 0" << std::endl;
      return false;
    }

    // SECOND OBJECT: Now loop over the second leg to get its information
    for (std::vector<SingleCombInCond>::const_iterator it1Comb = cond1Comb.begin(); it1Comb != cond1Comb.end();
         it1Comb++) {
      LogDebug("L1TGlobal") << "Looking at second subcondition" << std::endl;
      int obj1Index = -1;

      if (!(*it1Comb).empty()) {
        obj1Index = (*it1Comb)[0];
      } else {
        LogTrace("L1TGlobal") << "\n  SingleCombInCond (*it1Comb).size() " << ((*it1Comb).size()) << std::endl;
        return false;
      }

      // If we are dealing with the same object type avoid the two legs either being the same object
      if (cndObjTypeVec[0] == cndObjTypeVec[1] && obj0Index == obj1Index && cond0bx == cond1bx) {
        LogDebug("L1TGlobal") << "Corr Condition looking at same leg...skip" << std::endl;
        continue;
      }

      if (cond1Categ == CondMuon) {
        lutObj1 = "MU";
        candMuVec = m_uGtB->getCandL1Mu();
        phiIndex1 = (candMuVec->at(cond1bx, obj1Index))->hwPhiAtVtx();
        etaIndex1 = (candMuVec->at(cond1bx, obj1Index))->hwEtaAtVtx();
        etIndex1 = (candMuVec->at(cond1bx, obj1Index))->hwPt();
        chrg1 = (candMuVec->at(cond1bx, obj1Index))->hwCharge();

        etaBin1 = etaIndex1;
        if (etaBin1 < 0)
          etaBin1 = m_gtScales->getMUScales().etaBins.size() + etaBin1;

        etBin1 = etIndex1;
        int ssize = m_gtScales->getMUScales().etBins.size();
        if (etBin1 >= ssize) {
          LogTrace("L1TGlobal") << "MU1 hw et" << etBin1 << " out of scale range.  Setting to maximum.";
          etBin1 = ssize - 1;
        }

        // Determine Floating Pt numbers for floating point calculation
        std::pair<double, double> binEdges = m_gtScales->getMUScales().phiBins.at(phiIndex1);
        phi1Phy = 0.5 * (binEdges.second + binEdges.first);
        binEdges = m_gtScales->getMUScales().etaBins.at(etaBin1);
        eta1Phy = 0.5 * (binEdges.second + binEdges.first);
        binEdges = m_gtScales->getMUScales().etBins.at(etBin1);
        et1Phy = 0.5 * (binEdges.second + binEdges.first);
        LogDebug("L1TGlobal") << "Found all quantities for MU1" << std::endl;
      } else {
        // Interested only in three-muon correlations
        LogDebug("L1TGlobal") << "CondMuon not satisfied for Leg 1" << std::endl;
        return false;
      }

      // THIRD OBJECT: Finally loop over the third leg to get its information
      for (std::vector<SingleCombInCond>::const_iterator it2Comb = cond2Comb.begin(); it2Comb != cond2Comb.end();
           it2Comb++) {
        LogDebug("L1TGlobal") << "Looking at the third object for the three-body condition" << std::endl;
        int obj2Index = -1;

        if (!(*it2Comb).empty()) {
          obj2Index = (*it2Comb)[0];
        } else {
          LogTrace("L1TGlobal") << "\n  SingleCombInCond (*it2Comb).size() " << ((*it2Comb).size()) << std::endl;
          return false;
        }

        // If we are dealing with the same object type avoid the two legs
        // either being the same object
        if ((cndObjTypeVec[0] == cndObjTypeVec[2] && obj0Index == obj2Index && cond0bx == cond2bx) ||
            (cndObjTypeVec[1] == cndObjTypeVec[2] && obj1Index == obj2Index && cond1bx == cond2bx)) {
          LogDebug("L1TGlobal") << "Corr Condition looking at same leg...skip" << std::endl;
          continue;
        }

        if (cond2Categ == CondMuon) {
          lutObj2 = "MU";
          candMuVec = m_uGtB->getCandL1Mu();
          phiIndex2 = (candMuVec->at(cond2bx, obj2Index))->hwPhiAtVtx();
          etaIndex2 = (candMuVec->at(cond2bx, obj2Index))->hwEtaAtVtx();
          etIndex2 = (candMuVec->at(cond2bx, obj2Index))->hwPt();
          chrg2 = (candMuVec->at(cond2bx, obj2Index))->hwCharge();

          etaBin2 = etaIndex2;
          if (etaBin2 < 0)
            etaBin2 = m_gtScales->getMUScales().etaBins.size() + etaBin2;

          etBin2 = etIndex2;
          int ssize = m_gtScales->getMUScales().etBins.size();
          if (etBin2 >= ssize) {
            LogTrace("L1TGlobal") << "MU2 hw et" << etBin2 << " out of scale range.  Setting to maximum.";
            etBin2 = ssize - 1;
          }

          // Determine Floating Pt numbers for floating point calculation
          std::pair<double, double> binEdges = m_gtScales->getMUScales().phiBins.at(phiIndex2);
          phi2Phy = 0.5 * (binEdges.second + binEdges.first);
          binEdges = m_gtScales->getMUScales().etaBins.at(etaBin2);
          eta2Phy = 0.5 * (binEdges.second + binEdges.first);
          binEdges = m_gtScales->getMUScales().etBins.at(etBin2);
          et2Phy = 0.5 * (binEdges.second + binEdges.first);
          LogDebug("L1TGlobal") << "Found all quantities for MU2" << std::endl;
        }

        else {
          // Interested only in three-muon correlations
          LogDebug("L1TGlobal") << "CondMuon not satisfied for Leg 2" << std::endl;
          return false;
        };

        if (m_verbosity) {
          LogDebug("L1TGlobal") << "\n >>>>>> THREE-MUON EVENT!" << std::endl;
          LogDebug("L1TGlobal") << ">>>>>> Object involved in the three-body correlation condition are ["
                                << l1t::GlobalObjectEnumToString(cndObjTypeVec[0]) << ", "
                                << l1t::GlobalObjectEnumToString(cndObjTypeVec[1]) << ", "
                                << l1t::GlobalObjectEnumToString(cndObjTypeVec[2]) << "] with collection indices ["
                                << obj0Index << ", " << obj1Index << obj2Index << "] "
                                << " having: \n"
                                << "     Et  values  = [" << etIndex0 << ", " << etIndex1 << ", " << etIndex2 << "]\n"
                                << "     phi indices = [" << phiIndex0 << ", " << phiIndex1 << ", " << phiIndex2
                                << "]\n"
                                << "     eta indices = [" << etaIndex0 << ", " << etaIndex1 << ", " << etaIndex2
                                << "]\n"
                                << "     charge values = [" << chrg0 << ", " << chrg1 << ", " << chrg2 << "]\n";
        }

        // Now perform the desired correlation on these three objects:
        // reqResult will be set true in case all checks were successful for a given combination of three muons
        bool reqResult = false;
        bool chrgCorrel = true;

        // Check the three-muon charge correlation, if requested.
        // NOTE that the charge can be 1 (positive) or 0 (negative), so [SS] SUM(chrg) == 3 OR 0, [OS] SUM(chrg) == 1 OR 2
        if (cond0Categ == CondMuon && cond1Categ == CondMuon && cond2Categ == CondMuon) {
          // Check for opp-sign requirement:
          if (corrPar.chargeCorrelation == 4 && ((chrg0 + chrg1 + chrg2) == 3 || (chrg0 + chrg1 + chrg2) == 0)) {
            chrgCorrel = false;
          }
          // Check for same-sign
          if (corrPar.chargeCorrelation == 2 && ((chrg0 + chrg1 + chrg2) == 1 || (chrg0 + chrg1 + chrg2) == 2)) {
            chrgCorrel = false;
          }
          // Ignore the charge correlation requirement
          if (corrPar.chargeCorrelation == 1) {
            chrgCorrel = true;
          }
        }

        // Clear the vector containing indices of the objects of the combination involved in the condition evaluation
        objectsInComb.clear();
        objectsInComb.push_back(obj0Index);
        objectsInComb.push_back(obj1Index);
        objectsInComb.push_back(obj2Index);

        // Delta eta and phi calculations needed to evaluate the three-body invariant mass
        double deltaPhiPhy_01 = fabs(phi1Phy - phi0Phy);
        if (deltaPhiPhy_01 > M_PI)
          deltaPhiPhy_01 = 2. * M_PI - deltaPhiPhy_01;
        double deltaEtaPhy_01 = fabs(eta1Phy - eta0Phy);

        double deltaPhiPhy_02 = fabs(phi2Phy - phi0Phy);
        if (deltaPhiPhy_02 > M_PI)
          deltaPhiPhy_02 = 2. * M_PI - deltaPhiPhy_02;
        double deltaEtaPhy_02 = fabs(eta2Phy - eta0Phy);

        double deltaPhiPhy_12 = fabs(phi2Phy - phi1Phy);
        if (deltaPhiPhy_12 > M_PI)
          deltaPhiPhy_12 = 2. * M_PI - deltaPhiPhy_12;
        double deltaEtaPhy_12 = fabs(eta2Phy - eta1Phy);

        // Determine the integer based delta eta and delta phi
        int deltaPhiFW_01 = abs(phiIndex0 - phiIndex1);
        if (deltaPhiFW_01 >= phiBound)
          deltaPhiFW_01 = 2 * phiBound - deltaPhiFW_01;
        std::string lutName_01 = lutObj0;
        lutName_01 += "-";
        lutName_01 += lutObj1;
        long long deltaPhiLUT_01 = m_gtScales->getLUT_DeltaPhi(lutName_01, deltaPhiFW_01);
        unsigned int precDeltaPhiLUT_01 = m_gtScales->getPrec_DeltaPhi(lutName_01);

        int deltaEtaFW_01 = abs(etaIndex0 - etaIndex1);
        long long deltaEtaLUT_01 = 0;
        unsigned int precDeltaEtaLUT_01 = 0;
        deltaEtaLUT_01 = m_gtScales->getLUT_DeltaEta(lutName_01, deltaEtaFW_01);
        precDeltaEtaLUT_01 = m_gtScales->getPrec_DeltaEta(lutName_01);
        ///
        int deltaPhiFW_02 = abs(phiIndex0 - phiIndex2);
        if (deltaPhiFW_02 >= phiBound)
          deltaPhiFW_02 = 2 * phiBound - deltaPhiFW_02;
        std::string lutName_02 = lutObj0;
        lutName_02 += "-";
        lutName_02 += lutObj2;
        long long deltaPhiLUT_02 = m_gtScales->getLUT_DeltaPhi(lutName_02, deltaPhiFW_02);
        unsigned int precDeltaPhiLUT_02 = m_gtScales->getPrec_DeltaPhi(lutName_02);

        int deltaEtaFW_02 = abs(etaIndex0 - etaIndex2);
        long long deltaEtaLUT_02 = 0;
        unsigned int precDeltaEtaLUT_02 = 0;
        deltaEtaLUT_02 = m_gtScales->getLUT_DeltaEta(lutName_02, deltaEtaFW_02);
        precDeltaEtaLUT_02 = m_gtScales->getPrec_DeltaEta(lutName_02);
        ///
        int deltaPhiFW_12 = abs(phiIndex1 - phiIndex2);
        if (deltaPhiFW_12 >= phiBound)
          deltaPhiFW_12 = 2 * phiBound - deltaPhiFW_12;
        std::string lutName_12 = lutObj1;
        lutName_12 += "-";
        lutName_12 += lutObj2;
        long long deltaPhiLUT_12 = m_gtScales->getLUT_DeltaPhi(lutName_12, deltaPhiFW_12);
        unsigned int precDeltaPhiLUT_12 = m_gtScales->getPrec_DeltaPhi(lutName_12);

        int deltaEtaFW_12 = abs(etaIndex1 - etaIndex2);
        long long deltaEtaLUT_12 = 0;
        unsigned int precDeltaEtaLUT_12 = 0;
        deltaEtaLUT_12 = m_gtScales->getLUT_DeltaEta(lutName_12, deltaEtaFW_12);
        precDeltaEtaLUT_12 = m_gtScales->getPrec_DeltaEta(lutName_12);
        ///

        LogDebug("L1TGlobal") << "### Obj0 phiFW = " << phiIndex0 << " Obj1 phiFW = " << phiIndex1 << "\n"
                              << "    DeltaPhiFW = " << deltaPhiFW_01 << "    LUT Name 01= " << lutName_01
                              << " Prec = " << precDeltaPhiLUT_01 << "\n"
                              << "    LUT Name 02= " << lutName_02 << " Prec = " << precDeltaPhiLUT_02 << "\n"
                              << "    LUT Name 12= " << lutName_12 << " Prec = " << precDeltaPhiLUT_12 << "\n"
                              << "    DeltaPhiLUT_01 = " << deltaPhiLUT_01 << "\n"
                              << "    DeltaPhiLUT_02 = " << deltaPhiLUT_02 << "\n"
                              << "    DeltaPhiLUT_12 = " << deltaPhiLUT_12 << "\n"
                              << "### Obj0 etaFW = " << etaIndex0 << " Obj1 etaFW = " << etaIndex1 << "\n"
                              << "    DeltaEtaFW = " << deltaEtaFW_01 << "    LUT Name 01 = " << lutName_01
                              << " Prec 01 = " << precDeltaEtaLUT_01 << "\n"
                              << "    LUT Name 02 = " << lutName_02 << " Prec 02 = " << precDeltaEtaLUT_02 << "\n"
                              << "    LUT Name 12 = " << lutName_12 << " Prec 12 = " << precDeltaEtaLUT_12 << "\n"
                              << "    DeltaEtaLUT_01 = " << deltaEtaLUT_01 << "    DeltaEtaLUT_02 = " << deltaEtaLUT_02
                              << "    DeltaEtaLUT_12 = " << deltaEtaLUT_12 << std::endl;

        if (corrPar.corrCutType & 0x9) {
          // Invariant mass calculation based for each pair on
          // M = sqrt(2*p1*p2(cosh(eta1-eta2) - cos(phi1 - phi2)))
          // NOTE: we calculate (1/2)M^2
          ///
          double cosDeltaPhiPhy_01 = cos(deltaPhiPhy_01);
          double coshDeltaEtaPhy_01 = cosh(deltaEtaPhy_01);
          double massSqPhy_01 = et0Phy * et1Phy * (coshDeltaEtaPhy_01 - cosDeltaPhiPhy_01);

          long long cosDeltaPhiLUT_01 = m_gtScales->getLUT_DeltaPhi_Cos(lutName_01, deltaPhiFW_01);
          unsigned int precCosLUT_01 = m_gtScales->getPrec_DeltaPhi_Cos(lutName_01);

          long long coshDeltaEtaLUT_01;
          coshDeltaEtaLUT_01 = m_gtScales->getLUT_DeltaEta_Cosh(lutName_01, deltaEtaFW_01);
          unsigned int precCoshLUT_01 = m_gtScales->getPrec_DeltaEta_Cosh(lutName_01);
          if (precCoshLUT_01 - precCosLUT_01 != 0)
            LogDebug("L1TGlobal") << "Warning: Cos and Cosh LUTs on different precision" << std::endl;

          double cosDeltaPhiPhy_02 = cos(deltaPhiPhy_02);
          double coshDeltaEtaPhy_02 = cosh(deltaEtaPhy_02);
          if (corrPar.corrCutType & 0x10)
            coshDeltaEtaPhy_02 = 1.;
          double massSqPhy_02 = et0Phy * et2Phy * (coshDeltaEtaPhy_02 - cosDeltaPhiPhy_02);
          long long cosDeltaPhiLUT_02 = m_gtScales->getLUT_DeltaPhi_Cos(lutName_02, deltaPhiFW_02);
          unsigned int precCosLUT_02 = m_gtScales->getPrec_DeltaPhi_Cos(lutName_02);
          long long coshDeltaEtaLUT_02;
          if (corrPar.corrCutType & 0x10) {
            coshDeltaEtaLUT_02 = 1 * pow(10, precCosLUT_02);
          } else {
            coshDeltaEtaLUT_02 = m_gtScales->getLUT_DeltaEta_Cosh(lutName_02, deltaEtaFW_02);
            unsigned int precCoshLUT_02 = m_gtScales->getPrec_DeltaEta_Cosh(lutName_02);
            if (precCoshLUT_02 - precCosLUT_02 != 0)
              LogDebug("L1TGlobal") << "Warning: Cos and Cosh LUTs on different precision" << std::endl;
          }

          double cosDeltaPhiPhy_12 = cos(deltaPhiPhy_12);
          double coshDeltaEtaPhy_12 = cosh(deltaEtaPhy_12);
          if (corrPar.corrCutType & 0x10)
            coshDeltaEtaPhy_12 = 1.;
          double massSqPhy_12 = et1Phy * et2Phy * (coshDeltaEtaPhy_12 - cosDeltaPhiPhy_12);
          long long cosDeltaPhiLUT_12 = m_gtScales->getLUT_DeltaPhi_Cos(lutName_12, deltaPhiFW_12);
          unsigned int precCosLUT_12 = m_gtScales->getPrec_DeltaPhi_Cos(lutName_12);
          long long coshDeltaEtaLUT_12;
          if (corrPar.corrCutType & 0x10) {
            coshDeltaEtaLUT_12 = 1 * pow(10, precCosLUT_12);
          } else {
            coshDeltaEtaLUT_12 = m_gtScales->getLUT_DeltaEta_Cosh(lutName_12, deltaEtaFW_12);
            unsigned int precCoshLUT_12 = m_gtScales->getPrec_DeltaEta_Cosh(lutName_12);
            if (precCoshLUT_12 - precCosLUT_12 != 0)
              LogDebug("L1TGlobal") << "Warning: Cos and Cosh LUTs on different precision" << std::endl;
          }

          std::string lutName = lutObj0;
          lutName += "-ET";
          long long ptObj0 = m_gtScales->getLUT_Pt("Mass_" + lutName, etIndex0);
          unsigned int precPtLUTObj0 = m_gtScales->getPrec_Pt("Mass_" + lutName);

          lutName = lutObj1;
          lutName += "-ET";
          long long ptObj1 = m_gtScales->getLUT_Pt("Mass_" + lutName, etIndex1);
          unsigned int precPtLUTObj1 = m_gtScales->getPrec_Pt("Mass_" + lutName);

          lutName = lutObj2;
          lutName += "-ET";
          long long ptObj2 = m_gtScales->getLUT_Pt("Mass_" + lutName, etIndex2);
          unsigned int precPtLUTObj2 = m_gtScales->getPrec_Pt("Mass_" + lutName);

          // Pt and Angles are at different precision
          long long massSq_01 = ptObj0 * ptObj1 * (coshDeltaEtaLUT_01 - cosDeltaPhiLUT_01);
          long long massSq_02 = ptObj0 * ptObj2 * (coshDeltaEtaLUT_02 - cosDeltaPhiLUT_02);
          long long massSq_12 = ptObj1 * ptObj2 * (coshDeltaEtaLUT_12 - cosDeltaPhiLUT_12);

          // Note: There is an assumption here that Cos and Cosh have the same precision
          //       unsigned int preShift_01 = precPtLUTObj0 + precPtLUTObj1 + precCosLUT - corrPar.precMassCut;
          unsigned int preShift_01 = precPtLUTObj0 + precPtLUTObj1 + precCosLUT_01 - corrPar.precMassCut;
          unsigned int preShift_02 = precPtLUTObj0 + precPtLUTObj2 + precCosLUT_02 - corrPar.precMassCut;
          unsigned int preShift_12 = precPtLUTObj1 + precPtLUTObj2 + precCosLUT_12 - corrPar.precMassCut;

          LogDebug("L1TGlobal") << "####################################\n";
          LogDebug("L1TGlobal") << "    Testing the dimuon invariant mass between the FIRST PAIR 0-1 (" << lutObj0
                                << "," << lutObj1 << ") \n"
                                << "    massSq/2     = " << massSq_01 << "\n"
                                << "    Precision Shift = " << preShift_01 << "\n"
                                << "    massSq   (shift)= " << (massSq_01 / pow(10, preShift_01 + corrPar.precMassCut))
                                << "\n"
                                << "    massSqPhy/2  = " << massSqPhy_01
                                << "  sqrt(|massSq|) = " << sqrt(fabs(2. * massSqPhy_01)) << std::endl;

          LogDebug("L1TGlobal") << "####################################\n";
          LogDebug("L1TGlobal") << "    Testing the dimuon invariant mass between the SECOND PAIR 0-2 (" << lutObj0
                                << "," << lutObj2 << ") \n"
                                << "    massSq/2     = " << massSq_02 << "\n"
                                << "    Precision Shift = " << preShift_02 << "\n"
                                << "    massSq   (shift)= " << (massSq_02 / pow(10, preShift_02 + corrPar.precMassCut))
                                << "\n"
                                << "    massSqPhy/2  = " << massSqPhy_02
                                << "  sqrt(|massSq|) = " << sqrt(fabs(2. * massSqPhy_02)) << std::endl;

          LogDebug("L1TGlobal") << "####################################\n";
          LogDebug("L1TGlobal") << "    Testing the dimuon invariant mass between the THIRD PAIR 1-2 (" << lutObj1
                                << "," << lutObj2 << ") \n"
                                << "    massSq/2     = " << massSq_12 << "\n"
                                << "    Precision Shift = " << preShift_12 << "\n"
                                << "    massSq   (shift)= " << (massSq_12 / pow(10, preShift_12 + corrPar.precMassCut))
                                << "\n"
                                << "    massSqPhy/2  = " << massSqPhy_12
                                << "  sqrt(|massSq|) = " << sqrt(fabs(2. * massSqPhy_12)) << std::endl;

          LogDebug("L1TGlobal") << "\n ########### THREE-BODY INVARIANT MASS #########################\n";
          long long massSq = 0;

          if (preShift_01 == preShift_02 && preShift_01 == preShift_12 && preShift_02 == preShift_12) {
            LogDebug("L1TGlobal") << "Check the preshift value: " << preShift_01 << " = " << preShift_02 << " = "
                                  << preShift_12 << std::endl;
            preShift = preShift_01;
          } else {
            LogDebug("L1TGlobal")
                << "Preshift values considered for the sum of the dimuon invariant masses are different!" << std::endl;
          }

          if ((massSq_01 != massSq_02) && (massSq_01 != massSq_12) && (massSq_02 != massSq_12)) {
            massSq = massSq_01 + massSq_02 + massSq_12;
            LogDebug("L1TGlobal") << "massSq = " << massSq << std::endl;
          } else {
            LogDebug("L1TGlobal") << "Same pair of muons considered, three-body invariant mass do not computed"
                                  << std::endl;
          }

          if (massSq >= 0 && massSq >= (long long)(corrPar.minMassCutValue * pow(10, preShift)) &&
              massSq <= (long long)(corrPar.maxMassCutValue * pow(10, preShift))) {
            LogDebug("L1TGlobal") << "    Passed Invariant Mass Cut ["
                                  << (long long)(corrPar.minMassCutValue * pow(10, preShift)) << ","
                                  << (long long)(corrPar.maxMassCutValue * pow(10, preShift)) << "]" << std::endl;
            reqResult = true;
          } else {
            LogDebug("L1TGlobal") << "    Failed Invariant Mass Cut ["
                                  << (long long)(corrPar.minMassCutValue * pow(10, preShift)) << ","
                                  << (long long)(corrPar.maxMassCutValue * pow(10, preShift)) << "]" << std::endl;
            reqResult = false;
          }
        }

        if (reqResult && chrgCorrel) {
          condResult = true;
          (combinationsInCond()).push_back(objectsInComb);
        }

      }  //end loop over third leg
    }    //end loop over second leg
  }      //end loop over first leg

  if (m_verbosity && condResult) {
    LogDebug("L1TGlobal") << " pass(es) the correlation condition.\n" << std::endl;
  }
  return condResult;
}

/**
 * checkObjectParameter - Compare a single particle with a numbered condition
 *
 * @param iCondition: The number of the condition.
 * @param cand: The candidate to compare.
 *
 * @return: The result of the comparison (false if a condition does not exist)
 */

const bool l1t::CorrThreeBodyCondition::checkObjectParameter(const int iCondition, const l1t::L1Candidate& cand) const {
  return true;
}

void l1t::CorrThreeBodyCondition::print(std::ostream& myCout) const {
  myCout << "Dummy Print for CorrThreeBodyCondition" << std::endl;
  m_gtCorrelationThreeBodyTemplate->print(myCout);

  ConditionEvaluation::print(myCout);
}
