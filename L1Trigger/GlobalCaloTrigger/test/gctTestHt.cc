#include "L1Trigger/GlobalCaloTrigger/test/gctTestHt.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"

#include <math.h>
#include <iostream>
#include <cassert>

using namespace std;

//=================================================================================================================
//
/// Constructor and destructor

gctTestHt::gctTestHt() {}
gctTestHt::~gctTestHt() {}

//=================================================================================================================
//
/// Read the input jet data from the jetfinders (after GCT processing).
void gctTestHt::fillRawJetData(const L1GlobalCaloTrigger* gct) {

  minusWheelJetDta.clear();
  plusWheelJetData.clear();
  minusWheelJetDta.resize(9*m_numOfBx);
  plusWheelJetData.resize(9*m_numOfBx);

  int bx=m_bxStart;
  int mPos=0;
  int pPos=0;
  for (int ibx=0; ibx<m_numOfBx; ibx++) { 
    int leaf=0;

    // Minus Wheel
    for ( ; leaf<3; leaf++) {
      minusWheelJetDta.at(mPos++) = rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderA(), bx);
      minusWheelJetDta.at(mPos++) = rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderB(), bx);
      minusWheelJetDta.at(mPos++) = rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderC(), bx);
    }

    // Plus Wheel
    for ( ; leaf<6; leaf++) {
      plusWheelJetData.at(pPos++) = rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderA(), bx);
      plusWheelJetData.at(pPos++) = rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderB(), bx);
      plusWheelJetData.at(pPos++) = rawJetFinderOutput(gct->getJetLeafCards().at(leaf)->getJetFinderC(), bx);
    }

    bx++;
  }

}

//=================================================================================================================
//
/// Set array sizes for the number of bunch crossings
void gctTestHt::setBxRange(const int bxStart, const int numOfBx){
  // Allow the start of the bunch crossing range to be altered
  // without losing previously stored jet data
  rawJetData temp;
  for (int bx=bxStart; bx<m_bxStart; bx++) {
    minusWheelJetDta.insert(minusWheelJetDta.begin(), 9, temp);
    plusWheelJetData.insert(plusWheelJetData.begin(), 9, temp);
  }

  m_bxStart = bxStart;

  // Resize the vectors without clearing previously stored values
  minusWheelJetDta.resize(9*numOfBx);
  plusWheelJetData.resize(9*numOfBx);
  m_numOfBx = numOfBx;
}

//=================================================================================================================
//
/// Check the Ht summing algorithms
bool gctTestHt::checkHtSums(const L1GlobalCaloTrigger* gct) const
{
  bool testPass = true;
  L1GctGlobalEnergyAlgos* myGlobalEnergy = gct->getEnergyFinalStage();

  L1GctInternJetDataCollection internalJets = gct->getInternalJets();

  std::vector<rawJetData>::const_iterator mJet=minusWheelJetDta.begin();
  std::vector<rawJetData>::const_iterator pJet=plusWheelJetData.begin();

  for (int bx=0; bx<m_numOfBx; bx++) {
    unsigned htMinusVl = 0;
    unsigned htPlusVal = 0;
    int hxMinusVl = 0;
    int hxPlusVal = 0;
    int hyMinusVl = 0;
    int hyPlusVal = 0;
    bool htMinusInputOf = false;
    bool htPlusInputOvf = false;

    unsigned fRotX0 = 17;
    //
    // Check the Ht calculation (starting from the found jets)
    //--------------------------------------------------------------------------------------
    //
    // Minus Wheel
    int leaf=0;
    for ( ; leaf<3; leaf++) {
      unsigned leafHtSum = 0;
      int leafHxSum = 0;
      int leafHySum = 0;
      bool leafHtOvf = false;

      for (int jf=0; jf<3; jf++) {
	assert (mJet != minusWheelJetDta.end());
	leafHtSum += (mJet->htStripSum0 + mJet->htStripSum1); 
	leafHtOvf |= (mJet->htOverFlow);

        unsigned fRotX1 = (fRotX0+ 6) % 36;
        unsigned fRotY0 = (fRotX0+27) % 36;
        unsigned fRotY1 = (fRotX0+33) % 36;
	leafHxSum += etComponent(mJet->htStripSum0, fRotX0, mJet->htStripSum1, fRotX1); 
	leafHySum += etComponent(mJet->htStripSum0, fRotY0, mJet->htStripSum1, fRotY1); 
	fRotX0 = (fRotX0+32) % 36;
	mJet++;
      }
      if (leafHtSum >= 4096) { leafHtSum -= 4096; }
      if (leafHtSum == gct->getJetLeafCards().at(leaf)->getAllOutputHt().at(bx).value()) {
	htMinusVl += leafHtSum;
      } else { cout << "Ht sum check leaf " << leaf << endl; testPass = false; }
      if (leafHxSum == gct->getJetLeafCards().at(leaf)->getAllOutputHx().at(bx).value()) {
	hxMinusVl += leafHxSum;
      } else { cout << "Hx sum check leaf " << leaf << endl; testPass = false; }
      if (leafHySum == gct->getJetLeafCards().at(leaf)->getAllOutputHy().at(bx).value()) {
	hyMinusVl += leafHySum;
      } else { cout << "Hy sum check leaf " << leaf << endl; testPass = false; }
      if ((gct->getJetLeafCards().at(leaf)->getAllOutputHt().at(bx).overFlow() == leafHtOvf) &&
	  (gct->getJetLeafCards().at(leaf)->getAllOutputHx().at(bx).overFlow() == leafHtOvf) &&
	  (gct->getJetLeafCards().at(leaf)->getAllOutputHy().at(bx).overFlow() == leafHtOvf)) {
	htMinusInputOf |= leafHtOvf;
      } else { cout << "Ht minus overflow check leaf " << leaf << endl; testPass = false; }
    }

    // Plus Wheel
    for ( ; leaf<6; leaf++) {
      unsigned leafHtSum = 0;
      int leafHxSum = 0;
      int leafHySum = 0;
      bool leafHtOvf = false;
      for (int jf=0; jf<3; jf++) {
	assert (pJet != plusWheelJetData.end());
	leafHtSum += (pJet->htStripSum0 + pJet->htStripSum1);
	leafHtOvf |= (pJet->htOverFlow);

        unsigned fRotX1 = (fRotX0+ 6) % 36;
        unsigned fRotY0 = (fRotX0+27) % 36;
        unsigned fRotY1 = (fRotX0+33) % 36;
	leafHxSum += etComponent(pJet->htStripSum0, fRotX0, pJet->htStripSum1, fRotX1); 
	leafHySum += etComponent(pJet->htStripSum0, fRotY0, pJet->htStripSum1, fRotY1); 
	fRotX0 = (fRotX0+32) % 36;
	pJet++;
      }
      if (leafHtSum >= 4096) { leafHtSum -= 4096; }
      if (leafHtSum == gct->getJetLeafCards().at(leaf)->getAllOutputHt().at(bx).value()) {
	htPlusVal += leafHtSum;
      } else { cout << "Ht sum check leaf " << leaf << endl; testPass = false; }
      if (leafHxSum == gct->getJetLeafCards().at(leaf)->getAllOutputHx().at(bx).value()) {
	hxPlusVal += leafHxSum;
      } else { cout << "Hx sum check leaf " << leaf << endl; testPass = false; }
      if (leafHySum == gct->getJetLeafCards().at(leaf)->getAllOutputHy().at(bx).value()) {
	hyPlusVal += leafHySum;
      } else { cout << "Hy sum check leaf " << leaf << endl; testPass = false; }
      if ((gct->getJetLeafCards().at(leaf)->getAllOutputHt().at(bx).overFlow() == leafHtOvf) &&
	  (gct->getJetLeafCards().at(leaf)->getAllOutputHx().at(bx).overFlow() == leafHtOvf) &&
	  (gct->getJetLeafCards().at(leaf)->getAllOutputHy().at(bx).overFlow() == leafHtOvf)) {
	htPlusInputOvf |= leafHtOvf;
      } else { cout << "Ht plus overflow check leaf " << leaf << endl; testPass = false; }
    }

    unsigned htTotal = htMinusVl + htPlusVal;

    bool htMinusOvrFlow = (htMinusVl>=4096) || htMinusInputOf;
    bool htPlusOverFlow = (htPlusVal>=4096) || htPlusInputOvf;

    htMinusVl = htMinusVl%4096;
    htPlusVal = htPlusVal%4096;

    bool htTotalOvrFlow = (htTotal>=4096) || htMinusOvrFlow  || htPlusOverFlow;

    htTotal = htTotal%4096;

    int hxTotal = hxMinusVl + hxPlusVal;
    int hyTotal = hyMinusVl + hyPlusVal;
    //
    // Check the input to the final GlobalEnergyAlgos is as expected
    //--------------------------------------------------------------------------------------
    //
    if (!myGlobalEnergy->getInputHtVlMinusWheel().at(bx).overFlow() && !htMinusOvrFlow &&
	(myGlobalEnergy->getInputHtVlMinusWheel().at(bx).value()!=htMinusVl)) { 
      cout << "ht Minus " << htMinusVl <<endl; 
      testPass = false; 
    }

    if (myGlobalEnergy->getInputHtVlMinusWheel().at(bx).value()!=htMinusVl) { 
      cout << "ht Minus ovF " << htMinusVl << " found " << myGlobalEnergy->getInputHtVlMinusWheel().at(bx) <<endl; 
      testPass = false; 
    }

    if (!myGlobalEnergy->getInputHtValPlusWheel().at(bx).overFlow() && !htPlusOverFlow &&
	(myGlobalEnergy->getInputHtValPlusWheel().at(bx).value()!=htPlusVal)) { 
      cout << "ht Plus " << htPlusVal <<endl; 
      testPass = false; 
    }

    if (myGlobalEnergy->getInputHtValPlusWheel().at(bx).value()!=htPlusVal) { 
      cout << "ht Plus ovF " << htPlusVal << " found " << myGlobalEnergy->getInputHtValPlusWheel().at(bx) <<endl; 
      testPass = false; 
    }

    if (myGlobalEnergy->getInputHxVlMinusWheel().at(bx).value()!=hxMinusVl) { 
      cout << "hx Minus " << hxMinusVl << endl;
      testPass = false;
    }
    if (myGlobalEnergy->getInputHxValPlusWheel().at(bx).value()!=hxPlusVal) {
      cout << "hx Plus " << hxPlusVal <<endl;
      testPass = false; }

    if (myGlobalEnergy->getInputHyVlMinusWheel().at(bx).value()!=hyMinusVl) { 
      cout << "hy Minus " << hyMinusVl <<endl;
      testPass = false;
    }
    if (myGlobalEnergy->getInputHyValPlusWheel().at(bx).value()!=hyPlusVal) {
      cout << "hy Plus " << hyPlusVal <<endl;
      testPass = false; 
    }

    if ((myGlobalEnergy->getInputHtVlMinusWheel().at(bx).overFlow() == htMinusOvrFlow) &&
	(myGlobalEnergy->getInputHxVlMinusWheel().at(bx).overFlow() == htMinusOvrFlow) &&
	(myGlobalEnergy->getInputHyVlMinusWheel().at(bx).overFlow() == htMinusOvrFlow)) {
    } else { cout << "Ht minus overflow check wheel" << endl; testPass = false; }

    if ((myGlobalEnergy->getInputHtValPlusWheel().at(bx).overFlow() == htPlusOverFlow) &&
	(myGlobalEnergy->getInputHxValPlusWheel().at(bx).overFlow() == htPlusOverFlow) &&
	(myGlobalEnergy->getInputHyValPlusWheel().at(bx).overFlow() == htPlusOverFlow)) {
    } else { cout << "Ht plus overflow check wheel" << endl; testPass = false; }

    // Check the output value
    if (!myGlobalEnergy->getEtHadColl().at(bx).overFlow() && !htTotalOvrFlow &&
	(myGlobalEnergy->getEtHadColl().at(bx).value() != htTotal)) {
      cout << "Algo etHad" << endl; 
      testPass = false;
    }

    if (myGlobalEnergy->getEtHadColl().at(bx).value() != htTotal) {
      cout << "Algo etHad ovf" << " found " << myGlobalEnergy->getEtHadColl().at(bx) << endl; 
      testPass = false;
    }

    // Check the missing Ht calculation
    double dhx = static_cast<double>(-hxTotal);
    double dhy = static_cast<double>(-hyTotal);
    double dhm = sqrt(dhx*dhx + dhy*dhy);
    double phi = atan2(dhy, dhx);
    if (phi < 0) phi += 2.*M_PI;

    unsigned htMiss = static_cast<unsigned>(dhm/16.);
    unsigned htMPhi = static_cast<unsigned>(phi/M_PI*18.)*2;

    if (htMiss>63 || htMinusOvrFlow || htPlusOverFlow) htMiss = 63;

    if ((htMiss != myGlobalEnergy->getHtMissColl().at(bx).value()) ||
	(htMPhi != myGlobalEnergy->getHtMissPhiColl().at(bx).value())) {
      cout << "Missing Ht: expected " << htMiss << " phi " << htMPhi 
	   << " from x input " << hxTotal << " y input " << hyTotal
	   << ", found " << myGlobalEnergy->getHtMissColl().at(bx).value()
	   << " phi " << myGlobalEnergy->getHtMissPhiColl().at(bx).value() << endl;
      testPass = false;
    }

    // Check the storage of internal jet candidates
    unsigned htFromInternalJets = 0;
    for (L1GctInternJetDataCollection::const_iterator jet=internalJets.begin();
	 jet != internalJets.end(); jet++) {
      if (jet->bx() == bx+m_bxStart) {
	htFromInternalJets += jet->et();
      }
    }
    if (htFromInternalJets%4096 != htTotal) {
      cout << "Found ht from jets " << htFromInternalJets << " expected " << htTotal << endl;
      testPass = false;
    }

    // end of loop over bunch crossings
  }
  return testPass;
}


//=================================================================================================================
//
// PRIVATE MEMBER FUNCTIONS
//
gctTestHt::rawJetData gctTestHt::rawJetFinderOutput(const L1GctJetFinderBase* jf, const int bx) const
{
  lutPtrVector  lutsFromJf = jf->getJetEtCalLuts();
  RawJetsVector jetsFromJf = jf->getRawJets();
  RawJetsVector jetList;
  unsigned sumHtStrip0 = 0;
  unsigned sumHtStrip1 = 0;
  bool     sumHtOvrFlo = false;
  for (RawJetsVector::const_iterator jet=jetsFromJf.begin(); jet!=jetsFromJf.end(); jet++) {
    if (jet->bx()==bx && !jet->isNullJet()) {
//       cout << "found a jet " << jet->rawsum()
//  	   << " eta " << jet->globalEta()
//  	   << " phi " << jet->globalPhi()
// 	   << (jet->overFlow() ? " overflow set " : " ") 
//  	   << " bx " << jet->bx() << endl;
      jetList.push_back(*jet);
      unsigned etaBin = jet->rctEta();
      if (jet->rctPhi() == 0) {
	sumHtStrip0 += jet->calibratedEt(lutsFromJf.at(etaBin));
      }
      if (jet->rctPhi() == 1) {
	sumHtStrip1 += jet->calibratedEt(lutsFromJf.at(etaBin));
      }
      sumHtOvrFlo |= (jet->overFlow());
    }
  }
  rawJetData result(jetList, sumHtStrip0, sumHtStrip1, sumHtOvrFlo);
  return result;
}

int gctTestHt::etComponent(const unsigned Emag0, const unsigned fact0,
			   const unsigned Emag1, const unsigned fact1) const {
  // Copy the Ex, Ey conversion from the hardware emulation
  const unsigned sinFact[10] = {0, 2845, 5603, 8192, 10531, 12550, 14188, 15395, 16134, 16383};
  unsigned myFact;
  bool neg0=false, neg1=false, negativeResult;
  int res0=0, res1=0, result;
  unsigned Emag, fact;

  for (int i=0; i<2; i++) {
    if (i==0) { Emag = Emag0; fact = fact0; }
    else { Emag = Emag1; fact = fact1; }

    switch (fact/9) {
    case 0:
      myFact = sinFact[fact];
      negativeResult = false;
      break;
    case 1:
      myFact = sinFact[(18-fact)];
      negativeResult = false;
      break;
    case 2:
      myFact = sinFact[(fact-18)];
      negativeResult = true;
      break;
    case 3:
      myFact = sinFact[(36-fact)];
      negativeResult = true;
      break;
    default:
      cout << "Invalid factor " << fact << endl;
      return 0;
    }
    result = static_cast<int>(Emag*myFact);
    if (i==0) { res0 = result; neg0 = negativeResult; }
    else { res1 = result; neg1 = negativeResult; }
  }
  if ( neg0==neg1 ) {
    result = res0 + res1;
    negativeResult = neg0;
  } else {
    if ( res0>=res1 ) {
      result = res0 - res1;
      negativeResult = neg0;
    } else {
      result = res1 - res0;
      negativeResult = neg1;
    }
  }
  // Divide by 8192 using bit-shift; but emulate
  // twos-complement arithmetic for negative numbers
  if ( negativeResult ) {
    result = (1<<28)-result;
    result = (result+0x1000)>>13;
    result = result-(1<<15);
  } else { result = (result+0x1000)>>13; }
  return result;
}



