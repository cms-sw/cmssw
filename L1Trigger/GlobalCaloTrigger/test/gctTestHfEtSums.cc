#include "L1Trigger/GlobalCaloTrigger/test/gctTestHtAndJetCounts.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCounter.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCounterLut.h"

#include <math.h>
#include <iostream>
#include <cassert>

using namespace std;

//=================================================================================================================
//
/// Constructor and destructor

gctTestHtAndJetCounts::gctTestHtAndJetCounts() {}
gctTestHtAndJetCounts::~gctTestHtAndJetCounts() {}

//=================================================================================================================
//
/// Read the input jet data from the jetfinders (after GCT processing).
void gctTestHtAndJetCounts::fillRawJetData(const L1GlobalCaloTrigger* gct) {

  minusWheelJetDta.clear();
  plusWheelJetData.clear();

  int leaf=0;

  // Minus Wheel
  for ( ; leaf<3; leaf++) {
    minusWheelJetDta.push_back(rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderA()));
    minusWheelJetDta.push_back(rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderB()));
    minusWheelJetDta.push_back(rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderC()));
  }

  // Plus Wheel
  for ( ; leaf<6; leaf++) {
    plusWheelJetData.push_back(rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderA()));
    plusWheelJetData.push_back(rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderB()));
    plusWheelJetData.push_back(rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderC()));
  }

}

//=================================================================================================================
//
/// Check the Ht summing algorithms
bool gctTestHtAndJetCounts::checkHtSums(const L1GlobalCaloTrigger* gct) const
{
  bool testPass = true;
  L1GctGlobalEnergyAlgos* myGlobalEnergy = gct->getEnergyFinalStage();

  unsigned htMinusVl = 0;
  unsigned htPlusVal = 0;

  vector<rawJetData>::const_iterator jfList;
  //
  // Check the Ht calculation (starting from the found jets)
  //--------------------------------------------------------------------------------------
  //
  // Minus Wheel
  jfList = minusWheelJetDta.begin();
  int leaf=0;
  for ( ; leaf<3; leaf++) {
    unsigned leafHtSum = 0;
    for (int jf=0; jf<3; jf++) {
      assert (jfList < minusWheelJetDta.end());
      leafHtSum += jfList->htSum;
      jfList++;
    }
    if (leafHtSum == gct->getJetLeafCards().at(leaf)->getOutputHt().value()) {
      htMinusVl += leafHtSum;
    } else { cout << "Ht sum check leaf " << leaf << endl; testPass = false; }
  }

  // Plus Wheel
  jfList = plusWheelJetData.begin();
  for ( ; leaf<6; leaf++) {
    unsigned leafHtSum = 0;
    for (int jf=0; jf<3; jf++) {
      assert (jfList < plusWheelJetData.end());
      leafHtSum += jfList->htSum;
      jfList++;
    }
    if (leafHtSum == gct->getJetLeafCards().at(leaf)->getOutputHt().value()) {
      htPlusVal += leafHtSum;
    } else { cout << "Ht sum check leaf " << leaf << endl; testPass = false; }
  }

  unsigned htTotal = htMinusVl + htPlusVal;

  bool htMinusOvrFlow = (htMinusVl>=4096);
  bool htPlusOverFlow = (htPlusVal>=4096);
  bool htTotalOvrFlow = (htTotal>=4096) || htMinusOvrFlow  || htPlusOverFlow;
  //
  // Check the input to the final GlobalEnergyAlgos is as expected
  //--------------------------------------------------------------------------------------
  //
  if (!myGlobalEnergy->getInputHtVlMinusWheel().overFlow() && !htMinusOvrFlow &&
      (myGlobalEnergy->getInputHtVlMinusWheel().value()!=htMinusVl)) { cout << "ht Minus " << htMinusVl <<endl; testPass = false; }
  if (!myGlobalEnergy->getInputHtValPlusWheel().overFlow() && !htPlusOverFlow &&
      (myGlobalEnergy->getInputHtValPlusWheel().value()!=htPlusVal)) { cout << "ht Plus " << htPlusVal <<endl; testPass = false; }

  // Check the output value
  if (!myGlobalEnergy->getEtHad().overFlow() && !htTotalOvrFlow &&
      (myGlobalEnergy->getEtHad().value() != htTotal)) {cout << "Algo etHad" << endl; testPass = false;}
  return testPass;
}

//=================================================================================================================
//
/// Check the jet counting algorithms
bool gctTestHtAndJetCounts::checkJetCounts(const L1GlobalCaloTrigger* gct) const
{
  bool testPass = true;
  L1GctGlobalEnergyAlgos* myGlobalEnergy = gct->getEnergyFinalStage();
  const L1GctJetEtCalibrationLut* myLut = gct->getJetEtCalibLut();
  //
  // Emulate the jet counting
  //--------------------------------------------------------------------------------------
  //
  vector<unsigned> JcMinusWheel(L1GctWheelJetFpga::N_JET_COUNTERS);
  vector<unsigned> JcPlusWheel (L1GctWheelJetFpga::N_JET_COUNTERS);
  vector<unsigned> JcResult(L1GctWheelJetFpga::N_JET_COUNTERS);

  for (unsigned jcnum = 0 ; jcnum<L1GctWheelJetFpga::N_JET_COUNTERS ; jcnum++) {

    L1GctJetCounterSetup::cutsListForJetCounter
      cutListPos=gct->getWheelJetFpgas().at(0)->getJetCounter(jcnum)->getJetCounterLut()->cutList();
    L1GctJetCounterSetup::cutsListForJetCounter
      cutListNeg=gct->getWheelJetFpgas().at(1)->getJetCounter(jcnum)->getJetCounterLut()->cutList();

    unsigned count0 = countJetsInCut(minusWheelJetDta, cutListNeg, myLut) ;
    JcMinusWheel.at(jcnum) = count0;
    unsigned count1 = countJetsInCut(plusWheelJetData, cutListPos, myLut) ;
    JcPlusWheel.at(jcnum) = count1;
    JcResult.at(jcnum) = ( (count0<7) && (count1<7) ? (count0 + count1) : 31 ) ;
  }

  // Check the inputs from the two wheels
  for (unsigned int i=0; i<L1GctWheelJetFpga::N_JET_COUNTERS; i++) {
    if ((myGlobalEnergy->getInputJcVlMinusWheel(i).value()!=JcMinusWheel.at(i)) &&
	(myGlobalEnergy->getInputJcVlMinusWheel(i).overFlow() ^ (JcMinusWheel.at(i)==7))) {
      cout << "jc Minus " << i << " value " << JcMinusWheel.at(i) <<endl;
      testPass = false;
    }
    if ((myGlobalEnergy->getInputJcValPlusWheel(i).value()!=JcPlusWheel.at(i)) ||
	(myGlobalEnergy->getInputJcValPlusWheel(i).overFlow() ^ (JcPlusWheel.at(i)==7))) {
      cout << "jc Plus " << i << " value " << JcPlusWheel.at(i) <<endl;
      testPass = false;
    }
  }

  // Check the outputs
  for (unsigned int j=0 ; j<L1GctWheelJetFpga::N_JET_COUNTERS ; j++) {
    if ((myGlobalEnergy->getJetCount(j).value() != JcResult.at(j)) || 
	(myGlobalEnergy->getJetCount(j).overFlow() ^ (JcResult.at(j)==31))) { 
//      cout << "Algo jCount " << j << endl;
//      cout << "Expected " << JcResult.at(j) << " found " << myGlobalEnergy->getJetCount(j) << endl;
//      cout << "PlusWheel " << myGlobalEnergy->getInputJcValPlusWheel(j) << endl;
//      cout << *myGlobalEnergy->getPlusWheelJetFpga() << endl;
//      cout << "MinusWheel " << myGlobalEnergy->getInputJcVlMinusWheel(j) << endl;
//      cout << *myGlobalEnergy->getMinusWheelJetFpga() << endl;
      testPass = false;
    }
  }
  return testPass;
}

//=================================================================================================================
//
// PRIVATE MEMBER FUNCTIONS
//
gctTestHtAndJetCounts::rawJetData gctTestHtAndJetCounts::rawJetFinderOutput(const L1GctJetFinderBase* jf) const
{
  RawJetsVector jetList;
  unsigned sumHt = 0;
  for (unsigned i=0; i<L1GctJetFinderBase::MAX_JETS_OUT; i++) {
    if (!jf->getRawJets().at(i).isNullJet()) {
//       cout << "found a jet " << jf->getRawJets().at(i).rawsum()
// 	   << " eta " << jf->getRawJets().at(i).globalEta()
// 	   << " phi " << jf->getRawJets().at(i).globalPhi() << endl;
      jetList.push_back(jf->getRawJets().at(i));
      sumHt += jetHtSum(jf, i);
    }
  }
  rawJetData result;
  result.jets  = jetList;
  result.htSum = sumHt;
  return result;
}

/// Return the Ht for a given jet in a jetFinder
unsigned gctTestHtAndJetCounts::jetHtSum(const L1GctJetFinderBase* jf, const int jn) const
{
  //
  return jf->getRawJets().at(jn).calibratedEt(jf->getJetEtCalLut());
  //
}

//
// Function definition for jet count checking
//=========================================================================
// Does what it says ...
unsigned gctTestHtAndJetCounts::countJetsInCut(const vector<rawJetData>& jetList,
                                               const L1GctJetCounterSetup::cutsListForJetCounter& cutList,
                                               const L1GctJetEtCalibrationLut* lut) const
{
  unsigned count = 0;
  L1GctJetCount<3> dummy;
  const unsigned MAX_VALUE = (1<<(dummy.size()))-1;

  // Loop over all jets in all jetFinders in a wheel
  for (vector<rawJetData>::const_iterator jList=jetList.begin(); jList<jetList.end(); jList++) {
    for (RawJetsVector::const_iterator jet=jList->jets.begin(); jet<jList->jets.end(); jet++) {
      if (!jet->jetCand(lut).empty()) {
        bool jetPassesCut = true;
        for (L1GctJetCounterSetup::cutsListForJetCounter::const_iterator cut=cutList.begin(); cut<cutList.end(); cut++) {
          switch (cut->cutType) {
            case L1GctJetCounterSetup::minRank :
              jetPassesCut &= (jet->jetCand(lut).rank() >= cut->cutValue1);
              break;

            case L1GctJetCounterSetup::maxRank:
              jetPassesCut &= (jet->jetCand(lut).rank() <= cut->cutValue1);
              break;

            case L1GctJetCounterSetup::centralEta:
              jetPassesCut &= (jet->rctEta() <= cut->cutValue1);
              break;

            case L1GctJetCounterSetup::forwardEta:
              jetPassesCut &= (jet->rctEta() >= cut->cutValue1);
              break;

            case L1GctJetCounterSetup::phiWindow:
              jetPassesCut &= (cut->cutValue2 > cut->cutValue1 ?
                ((jet->globalPhi() >= cut->cutValue1) && (jet->globalPhi() <= cut->cutValue2)) :
                ((jet->globalPhi() >= cut->cutValue1) || (jet->globalPhi() <= cut->cutValue2)));
              break;

            case L1GctJetCounterSetup::nullCutType:
              jetPassesCut &= false;
              break;

          }
        }
        if (jetPassesCut && (count<MAX_VALUE)) { count++; }
      }
    }
  }
  return count;
}



