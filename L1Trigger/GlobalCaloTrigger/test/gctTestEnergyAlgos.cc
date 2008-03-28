#include "L1Trigger/GlobalCaloTrigger/test/gctTestEnergyAlgos.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"

#include <math.h>
#include <algorithm>
#include <cassert>
#include <iostream>

using namespace std;

//=================================================================================================================
//
/// Constructor and destructor

gctTestEnergyAlgos::gctTestEnergyAlgos() : etStripSums(36) {}
gctTestEnergyAlgos::~gctTestEnergyAlgos() {}

//=================================================================================================================
//
/// Load another event into the gct. Overloaded for the various ways of doing this.
//  Here's a random event generator. Loads isolated input regions to check the energy sums.
std::vector<L1CaloRegion> gctTestEnergyAlgos::loadEvent(L1GlobalCaloTrigger* &gct, const bool simpleEvent, const int16_t bx)
{
  static const unsigned MAX_ET_CENTRAL=0x3ff; 
  static const unsigned MAX_ET_FORWARD=0x0ff; 
  std::vector<L1CaloRegion> inputRegions;

  int bxRel=bx-m_bxStart;
  int base=36*bxRel;
  assert( ( (base >= 0) && (base+36) <= (int) etStripSums.size() ) );

  for (int i=0; i<36; i++) {
    etStripSums.at(i+base)=0;
  }
  bool tempMinusOvrFlow = false;
  bool tempPlusOverFlow = false;

  // For initial tests just try things out with one region input
  // Then test with summing multiple regions. Choose one value
  // of energy and phi for each eta to avoid trying to set the
  // same region several times.
  for (unsigned i=0; i<(simpleEvent ? 1 : L1CaloRegionDetId::N_ETA); i++) {
    etmiss_vec etVector=randomMissingEtVector();
    //cout << "Region et " << etVector.mag << " phi " << etVector.phi << endl;
    // Set a single region input
    unsigned etaRegion = i;
    unsigned phiRegion = etVector.phi/4;

    unsigned maxEtForThisEta = MAX_ET_CENTRAL;
    if (etaRegion<4 || etaRegion>=18) {
      etVector.mag = etVector.mag >> 2;
      maxEtForThisEta = MAX_ET_FORWARD;
    } 
    //cout << "Region et " << etVector.mag << " eta " << etaRegion << " phi " << etVector.phi << endl;

    bool regionOf = etVector.mag > maxEtForThisEta;
    L1CaloRegion temp(etVector.mag, regionOf, true, false, false, etaRegion, phiRegion);
    temp.setBx(bx);
    inputRegions.push_back(temp);
        
    // Here we fill the expected values. Et values restricted to
    // eight bits in HF and ten bits in the rest of the system.
    if (etaRegion<(L1CaloRegionDetId::N_ETA)/2) {
      if (etVector.mag > maxEtForThisEta) {
        etStripSums.at(phiRegion+base) += MAX_ET_CENTRAL;
        tempMinusOvrFlow = true;
      } else {
        etStripSums.at(phiRegion+base) += etVector.mag;
      }
    } else {
      if (etVector.mag > maxEtForThisEta) {
        etStripSums.at(phiRegion+L1CaloRegionDetId::N_PHI+base) += MAX_ET_CENTRAL;
        tempPlusOverFlow = true;
      } else {
        etStripSums.at(phiRegion+L1CaloRegionDetId::N_PHI+base) += etVector.mag;
      }
    }
  }
  inMinusOvrFlow.at(bxRel) = tempMinusOvrFlow;
  inPlusOverFlow.at(bxRel) = tempPlusOverFlow;

  gct->fillRegions(inputRegions);
  return inputRegions;
}

// This method reads the gct input data for jetfinding from a file
// as an array of region energies, one for each eta-phi bin
std::vector<L1CaloRegion> gctTestEnergyAlgos::loadEvent(L1GlobalCaloTrigger* &gct, const std::string &fileName, bool &endOfFile, const int16_t bx)
{
  std::vector<L1CaloRegion> inputRegions;

  //Open the file
  if (!regionEnergyMapInputFile.is_open()) {
    regionEnergyMapInputFile.open(fileName.c_str(), ios::in);
  }

  //Error message and abandon ship if we can't read the file
  if(!regionEnergyMapInputFile.good())
  {
    throw cms::Exception("fileReadError")
    << " in gctTestEnergyAlgos::loadEvent(L1GlobalCaloTrigger *&, const std::string)\n"
    << "Couldn't read data from file " << fileName << "!";
  }

  int bxRel=bx-m_bxStart;
  int base=36*bxRel;
  assert( ( (base >= 0) && (base+36) <= (int) etStripSums.size() ) );

  //Initialise our local sums etc for this event
  for (int i=0; i<36; i++) {
    etStripSums.at(i+base)=0;
  }
  bool tempMinusOvrFlow = false;
  bool tempPlusOverFlow = false;

  // Here we read the energy map from the file.
  // Set each region at the input to the gct, and fill the expected strip sums etc.
  for (unsigned jphi=0; jphi<L1CaloRegionDetId::N_PHI; ++jphi) {
    unsigned iphi = (L1CaloRegionDetId::N_PHI + 4 - jphi)%L1CaloRegionDetId::N_PHI;
    for (unsigned ieta=0; ieta<L1CaloRegionDetId::N_ETA; ++ieta) {
      L1CaloRegion temp = nextRegionFromFile(ieta, iphi, bx);
      inputRegions.push_back(temp);
      if (ieta<(L1CaloRegionDetId::N_ETA/2)) {
	unsigned strip = iphi;
	etStripSums.at(strip+base) += temp.et();
	tempMinusOvrFlow           |= temp.overFlow();
      } else {
	unsigned strip = iphi+L1CaloRegionDetId::N_PHI;
	etStripSums.at(strip+base) += temp.et();
	tempPlusOverFlow           |= temp.overFlow();
      }
    }
  }
  inMinusOvrFlow.at(bxRel) = tempMinusOvrFlow;
  inPlusOverFlow.at(bxRel) = tempPlusOverFlow;
  endOfFile = regionEnergyMapInputFile.eof();

  gct->fillRegions(inputRegions);
  return inputRegions;
}

/// Set array sizes for the number of bunch crossings
void gctTestEnergyAlgos::setBxRange(const int bxStart, const int numOfBx){
  // Allow the start of the bunch crossing range to be altered
  // without losing previously stored etStripSums
  for (int bx=bxStart; bx<m_bxStart; bx++) {
    etStripSums.insert(etStripSums.begin(), 36, (unsigned) 0);
    inMinusOvrFlow.insert(inMinusOvrFlow.begin(), false); 
    inPlusOverFlow.insert(inPlusOverFlow.begin(), false); 
  }

  m_bxStart = bxStart;

  // Resize the vectors without clearing previously stored values
  etStripSums.resize(36*numOfBx);
  inMinusOvrFlow.resize(numOfBx);
  inPlusOverFlow.resize(numOfBx);
  m_numOfBx = numOfBx;
}

//=================================================================================================================
//
/// Check the energy sums algorithms
bool gctTestEnergyAlgos::checkEnergySums(const L1GlobalCaloTrigger* gct) const
{
  bool testPass=true;
  L1GctGlobalEnergyAlgos* myGlobalEnergy = gct->getEnergyFinalStage();

  for (int bx=0; bx<m_numOfBx; bx++) {
    int exMinusVl = 0;
    int eyMinusVl = 0;
    unsigned etMinusVl = 0;
    bool exMinusOvrFlow = inMinusOvrFlow.at(bx);
    bool eyMinusOvrFlow = inMinusOvrFlow.at(bx);
    bool etMinusOvrFlow = inMinusOvrFlow.at(bx);

    int exPlusVal = 0;
    int eyPlusVal = 0;
    unsigned etPlusVal = 0;
    bool exPlusOverFlow = inPlusOverFlow.at(bx);
    bool eyPlusOverFlow = inPlusOverFlow.at(bx);
    bool etPlusOverFlow = inPlusOverFlow.at(bx);

    unsigned rctStrip = 0;
    for (unsigned leaf=0; leaf<3; leaf++) {
      int exMinusSm = 0;
      int eyMinusSm = 0;
      unsigned etMinusSm = 0;
      int exPlusSum = 0;
      int eyPlusSum = 0;
      unsigned etPlusSum = 0;

      for (unsigned col=0; col<6; col++) {

	unsigned strip = (22-rctStrip)%18 + 36*bx;
	unsigned etm = etStripSums.at(strip);
	int exm = etComponent(etm, ((2*strip+9)%36) );
	int eym = etComponent(etm, (( 2*strip )%36) );

	exMinusSm += exm;
	eyMinusSm += eym;
	etMinusSm += etm; 

	unsigned etp = etStripSums.at(strip+18);
	int exp = etComponent(etp, ((2*strip+9)%36) );
	int eyp = etComponent(etp, (( 2*strip )%36) );

	exPlusSum += exp;
	eyPlusSum += eyp;
	etPlusSum += etp; 

	rctStrip++;
      }
      // Check overflow for each leaf card
      if (exMinusSm<-8192) { exMinusSm += 16384; exMinusOvrFlow = true; }
      if (exMinusSm>=8192) { exMinusSm -= 16384; exMinusOvrFlow = true; }
      if (eyMinusSm<-8192) { eyMinusSm += 16384; eyMinusOvrFlow = true; }
      if (eyMinusSm>=8192) { eyMinusSm -= 16384; eyMinusOvrFlow = true; }
      if (etMinusSm>=4096) { etMinusSm -= 4096; etMinusOvrFlow = true; }
      exMinusVl += exMinusSm;
      eyMinusVl += eyMinusSm;
      etMinusVl += etMinusSm;
      if (exPlusSum<-8192) { exPlusSum += 16384; exPlusOverFlow = true; }
      if (exPlusSum>=8192) { exPlusSum -= 16384; exPlusOverFlow = true; }
      if (eyPlusSum<-8192) { eyPlusSum += 16384; eyPlusOverFlow = true; }
      if (eyPlusSum>=8192) { eyPlusSum -= 16384; eyPlusOverFlow = true; }
      if (etPlusSum>=4096) { etPlusSum -= 4096; etPlusOverFlow = true; }
      exPlusVal += exPlusSum;
      eyPlusVal += eyPlusSum;
      etPlusVal += etPlusSum;
    }
    // Check overflow for the overall sums
    if (exMinusVl<-8192) { exMinusVl += 16384; exMinusOvrFlow = true; }
    if (exMinusVl>=8192) { exMinusVl -= 16384; exMinusOvrFlow = true; }
    if (eyMinusVl<-8192) { eyMinusVl += 16384; eyMinusOvrFlow = true; }
    if (eyMinusVl>=8192) { eyMinusVl -= 16384; eyMinusOvrFlow = true; }
    if (etMinusVl>=4096) { etMinusVl -= 4096; etMinusOvrFlow = true; }

    if (exPlusVal<-8192) { exPlusVal += 16384; exPlusOverFlow = true; }
    if (exPlusVal>=8192) { exPlusVal -= 16384; exPlusOverFlow = true; }
    if (eyPlusVal<-8192) { eyPlusVal += 16384; eyPlusOverFlow = true; }
    if (eyPlusVal>=8192) { eyPlusVal -= 16384; eyPlusOverFlow = true; }
    if (etPlusVal>=4096) { etPlusVal -= 4096; etPlusOverFlow = true; }

    int exTotal = exMinusVl + exPlusVal;
    int eyTotal = eyMinusVl + eyPlusVal;
    unsigned etTotal = (etMinusVl + etPlusVal) & 0xfff;

    bool exTotalOvrFlow = exMinusOvrFlow || exPlusOverFlow;
    bool eyTotalOvrFlow = eyMinusOvrFlow || eyPlusOverFlow;
    bool etTotalOvrFlow = etMinusOvrFlow || etPlusOverFlow;

    if (exTotal<-8192) { exTotal += 16384; exTotalOvrFlow = true; }
    if (exTotal>=8192) { exTotal -= 16384; exTotalOvrFlow = true; }
    if (eyTotal<-8192) { eyTotal += 16384; eyTotalOvrFlow = true; }
    if (eyTotal>=8192) { eyTotal -= 16384; eyTotalOvrFlow = true; }
    if (etTotal>=4096) { etTotal -= 4096; etTotalOvrFlow = true; }

    etmiss_vec etResult = trueMissingEt(-exTotal/2, -eyTotal/2);

    bool etMissOverFlow = exTotalOvrFlow || eyTotalOvrFlow;
    if (etResult.mag>=4096) { etResult.mag -= 4096; etMissOverFlow = true; }

    //
    // Check the input to the final GlobalEnergyAlgos is as expected
    //--------------------------------------------------------------------------------------
    //
    if ((myGlobalEnergy->getInputExVlMinusWheel().at(bx).overFlow()!=exMinusOvrFlow) || 
	(myGlobalEnergy->getInputExVlMinusWheel().at(bx).value()!=exMinusVl)) { cout << "ex Minus " << exMinusVl 
									      << (exMinusOvrFlow ? " overflow " : "  " )
									      << " from Gct " << myGlobalEnergy->getInputExVlMinusWheel().at(bx) <<endl; testPass = false; }
    if ((myGlobalEnergy->getInputExValPlusWheel().at(bx).overFlow()!=exPlusOverFlow) || 
	(myGlobalEnergy->getInputExValPlusWheel().at(bx).value()!=exPlusVal)) { cout << "ex Plus " << exPlusVal 
									      << (exPlusOverFlow ? " overflow " : "  " )
									      << " from Gct " << myGlobalEnergy->getInputExValPlusWheel().at(bx) <<endl; testPass = false; }
    if ((myGlobalEnergy->getInputEyVlMinusWheel().at(bx).overFlow()!=eyMinusOvrFlow) || 
	(myGlobalEnergy->getInputEyVlMinusWheel().at(bx).value()!=eyMinusVl)) { cout << "ey Minus " << eyMinusVl 
									      << (eyMinusOvrFlow ? " overflow " : "  " )
									      << " from Gct " << myGlobalEnergy->getInputEyVlMinusWheel().at(bx) <<endl; testPass = false; }
    if ((myGlobalEnergy->getInputEyValPlusWheel().at(bx).overFlow()!=eyPlusOverFlow) || 
	(myGlobalEnergy->getInputEyValPlusWheel().at(bx).value()!=eyPlusVal)) { cout << "ey Plus " << eyPlusVal 
									      << (eyPlusOverFlow ? " overflow " : "  " )
									      << " from Gct " << myGlobalEnergy->getInputEyValPlusWheel().at(bx) <<endl; testPass = false; }
    if ((myGlobalEnergy->getInputEtVlMinusWheel().at(bx).overFlow()!=etMinusOvrFlow) || 
	(myGlobalEnergy->getInputEtVlMinusWheel().at(bx).value()!=etMinusVl)) { cout << "et Minus " << etMinusVl 
									      << (etMinusOvrFlow ? " overflow " : "  " )
									      << " from Gct " << myGlobalEnergy->getInputEtVlMinusWheel().at(bx) <<endl; testPass = false; }
    if ((myGlobalEnergy->getInputEtValPlusWheel().at(bx).overFlow()!=etPlusOverFlow) || 
	(myGlobalEnergy->getInputEtValPlusWheel().at(bx).value()!=etPlusVal)) { cout << "et Plus " << etPlusVal 
									      << (etPlusOverFlow ? " overflow " : "  " )
									      << " from Gct " << myGlobalEnergy->getInputEtValPlusWheel().at(bx) <<endl; testPass = false; }
 
    //
    // Now check the processing in the final stage GlobalEnergyAlgos
    //--------------------------------------------------------------------------------------
    //
    // Check the missing Et calculation. Allow some margin for the
    // integer calculation of missing Et.
    unsigned etDiff, phDiff;
    unsigned etMargin, phMargin;

    etDiff = (unsigned) abs((long int) etResult.mag - (long int) myGlobalEnergy->getEtMissColl().at(bx).value());
    phDiff = (unsigned) abs((long int) etResult.phi - (long int) myGlobalEnergy->getEtMissPhiColl().at(bx).value());
    if (etDiff>2000) {etDiff=4096-etDiff;}
    if (phDiff>60)   {phDiff=72-phDiff;}
    //
    etMargin = (etMissOverFlow ? 40 : max((etResult.mag/100), (unsigned) 1) + 2);
    if (etResult.mag==0) { phMargin = 72; } else { phMargin = (30/etResult.mag) + 1; }
    if ((etDiff > etMargin) || (phDiff > phMargin)) {cout << "Algo etMiss diff "
							  << etDiff << " phi diff " << phDiff << endl; testPass = false; }

    if (etMissOverFlow != myGlobalEnergy->getEtMissColl().at(bx).overFlow()) {
      cout << "etMiss overFlow " << (etMissOverFlow ? "expected but not found in Gct" :
				     "found in Gct but not expected" ) << std::endl;
      testPass = false;
    }
    // Check the total Et calculation
    if (!myGlobalEnergy->getEtSumColl().at(bx).overFlow() && !etTotalOvrFlow &&
	(myGlobalEnergy->getEtSumColl().at(bx).value() != etTotal)) {cout << "Algo etSum" << endl; testPass = false;}
    // end of loop over bunch crossings
  }
  return testPass;
}

//=================================================================================================================
//
// PRIVATE MEMBER FUNCTIONS
//
// Function definitions for event generation
//=========================================================================
// Generates 2-d missing Et vector
gctTestEnergyAlgos::etmiss_vec gctTestEnergyAlgos::randomMissingEtVector() const
{
  // This produces random variables distributed as a 2-d Gaussian
  // with a standard deviation of sigma for each component,
  // and magnitude ranging up to 5*sigma.
  //
  // With sigma set to 400 we will always be in the range
  // of 12-bit signed integers (-2048 to 2047).
  // With sigma set to 200 we are always in the range
  // of 10-bit input region Et values.
  const float sigma=400.;

  // rmax controls the magnitude range
  // Chosen as a power of two conveniently close to
  // exp(5*5/2) to give a 5*sigma range.
  const unsigned rmax=262144;

  // Generate a pair of uniform pseudo-random integers
  vector<unsigned> components = randomTestData((int) 2, rmax);

  // Exclude the value zero for the first random integer
  // (Alternatively, return an overflow bit)
  while (components[0]==0) {components = randomTestData((int) 2, rmax);}

  // Convert to the 2-d Gaussian
  float p,r,s;
  unsigned Emag, Ephi;

  const float nbins = 18.;

  r = float(rmax);
  s = r/float(components[0]);
  p = float(components[1])/r;
  // Force phi value into the centre of a bin
  Emag = int(sigma*sqrt(2.*log(s)));
  Ephi = int(nbins*p);
  // Copy to the output
  etmiss_vec Et;
  Et.mag = Emag;
  Et.phi = (4*Ephi);
  return Et;

}

/// Generates test data consisting of a vector of energies
/// uniformly distributed between zero and max
vector<unsigned> gctTestEnergyAlgos::randomTestData(const int size, const unsigned max) const
{
  vector<unsigned> energies;
  int r,e;
  float p,q,s,t;

  p = float(max);
  q = float(RAND_MAX);
  for (int i=0; i<size; i++) {
    r = rand();
    s = float(r);
    t = s*p/q;
    e = int(t);

    energies.push_back(e);
  }
  return energies;
}

// Loads test input regions from a text file.
L1CaloRegion gctTestEnergyAlgos::nextRegionFromFile(const unsigned ieta, const unsigned iphi, const int16_t bx)
{
  // The file just contains lists of region energies
  unsigned et;
  regionEnergyMapInputFile >> et;
  L1CaloRegion temp(et, false, true, false, false, ieta, iphi);
  temp.setBx(bx);
  return temp;
}

//
// Function definitions for energy sum checking
//=========================================================================
int gctTestEnergyAlgos::etComponent(const unsigned Emag, const unsigned fact) const {
  // Copy the Ex, Ey conversion from the hardware emulation
  const unsigned sinFact[10] = {0, 89, 175, 256, 329, 392, 443, 481, 504, 512};
  unsigned myFact;
  bool negativeResult;
  int result;
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
  // Divide by 256 using bit-shift; but emulate
  // twos-complement arithmetic for negative numbers
  if ( negativeResult ) {
    result = (1<<24)-result;
    result = (result+0x80)>>8;
    result = result-(1<<16);
  } else { result = (result+0x80)>>8; }
  return result;
}

/// Calculate the expected missing Et vector for a given
/// ex and ey sum, for comparison with the hardware
gctTestEnergyAlgos::etmiss_vec gctTestEnergyAlgos::trueMissingEt(const int ex, const int ey) const {

  etmiss_vec result;

  double fx = static_cast<double>(ex);
  double fy = static_cast<double>(ey);
  double fmag = sqrt(fx*fx + fy*fy);
  double fphi = 36.*atan2(fy, fx)/3.1415927;

  result.mag = static_cast<int>(fmag);
  if (fphi>=0) {
    result.phi = static_cast<int>(fphi);
  } else {
    result.phi = static_cast<int>(fphi+72.);
  }

  return result;

}
