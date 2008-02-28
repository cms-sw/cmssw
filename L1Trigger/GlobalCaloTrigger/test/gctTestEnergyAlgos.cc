#include "L1Trigger/GlobalCaloTrigger/test/gctTestEnergyAlgos.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"

#include <math.h>
#include <algorithm>
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
std::vector<L1CaloRegion> gctTestEnergyAlgos::loadEvent(L1GlobalCaloTrigger* &gct, const bool simpleEvent)
{
  static const unsigned MAX_ET_CENTRAL=0x3ff; 
  static const unsigned MAX_ET_FORWARD=0x0ff; 
  std::vector<L1CaloRegion> inputRegions;

  for (int i=0; i<36; i++) {
    etStripSums.at(i)=0;
  }
  inMinusOvrFlow = false;
  inPlusOverFlow = false;

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
    inputRegions.push_back(temp);
        
    // Here we fill the expected values. Et values restricted to
    // eight bits in HF and ten bits in the rest of the system.
    if (etaRegion<(L1CaloRegionDetId::N_ETA)/2) {
      if (etVector.mag > maxEtForThisEta) {
        etStripSums.at(phiRegion) += MAX_ET_CENTRAL;
        inMinusOvrFlow = true;
      } else {
        etStripSums.at(phiRegion) += etVector.mag;
      }
    } else {
      if (etVector.mag > maxEtForThisEta) {
        etStripSums.at(phiRegion+L1CaloRegionDetId::N_PHI) += MAX_ET_CENTRAL;
        inPlusOverFlow = true;
      } else {
        etStripSums.at(phiRegion+L1CaloRegionDetId::N_PHI) += etVector.mag;
      }
    }
  }

  gct->fillRegions(inputRegions);
  return inputRegions;
}

// This method reads the gct input data for jetfinding from a file
// as an array of region energies, one for each eta-phi bin
std::vector<L1CaloRegion> gctTestEnergyAlgos::loadEvent(L1GlobalCaloTrigger* &gct, const std::string &fileName, bool &endOfFile)
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

  //Initialise our local sums etc for this event
  for (int i=0; i<36; i++) {
    etStripSums.at(i)=0;
  }
  inMinusOvrFlow = false;
  inPlusOverFlow = false;

  // Here we read the energy map from the file.
  // Set each region at the input to the gct, and fill the expected strip sums etc.
  for (unsigned jphi=0; jphi<L1CaloRegionDetId::N_PHI; ++jphi) {
    unsigned iphi = (L1CaloRegionDetId::N_PHI + 4 - jphi)%L1CaloRegionDetId::N_PHI;
    for (unsigned ieta=0; ieta<L1CaloRegionDetId::N_ETA; ++ieta) {
      L1CaloRegion temp = nextRegionFromFile(ieta, iphi);
      inputRegions.push_back(temp);
      if (ieta<(L1CaloRegionDetId::N_ETA/2)) {
	unsigned strip = iphi;
	etStripSums.at(strip) += temp.et();
	inMinusOvrFlow        |= temp.overFlow();
      } else {
	unsigned strip = iphi+L1CaloRegionDetId::N_PHI;
	etStripSums.at(strip) += temp.et();
	inPlusOverFlow        |= temp.overFlow();
      }
    }
  }
  endOfFile = regionEnergyMapInputFile.eof();

  gct->fillRegions(inputRegions);
  return inputRegions;
}

//=================================================================================================================
//
/// Check the energy sums algorithms
bool gctTestEnergyAlgos::checkEnergySums(const L1GlobalCaloTrigger* gct) const
{
  bool testPass=true;
  L1GctGlobalEnergyAlgos* myGlobalEnergy = gct->getEnergyFinalStage();

  int exMinusVl = 0;
  int eyMinusVl = 0;
  unsigned etMinusVl = 0;
  bool exMinusOvrFlow = inMinusOvrFlow;
  bool eyMinusOvrFlow = inMinusOvrFlow;
  bool etMinusOvrFlow = inMinusOvrFlow;

  int exPlusVal = 0;
  int eyPlusVal = 0;
  unsigned etPlusVal = 0;
  bool exPlusOverFlow = inPlusOverFlow;
  bool eyPlusOverFlow = inPlusOverFlow;
  bool etPlusOverFlow = inPlusOverFlow;

  unsigned rctStrip = 0;
  for (unsigned leaf=0; leaf<3; leaf++) {
    int exMinusSm = 0;
    int eyMinusSm = 0;
    unsigned etMinusSm = 0;
    int exPlusSum = 0;
    int eyPlusSum = 0;
    unsigned etPlusSum = 0;

    for (unsigned col=0; col<6; col++) {

      unsigned strip = (22-rctStrip)%18;
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
  if ((myGlobalEnergy->getInputExVlMinusWheel().overFlow()!=exMinusOvrFlow) || 
      (myGlobalEnergy->getInputExVlMinusWheel().value()!=exMinusVl)) { cout << "ex Minus " << exMinusVl 
         << (exMinusOvrFlow ? " overflow " : "  " )
         << " expected " << myGlobalEnergy->getInputExVlMinusWheel() <<endl; testPass = false; }
  if ((myGlobalEnergy->getInputExValPlusWheel().overFlow()!=exPlusOverFlow) || 
      (myGlobalEnergy->getInputExValPlusWheel().value()!=exPlusVal)) { cout << "ex Plus " << exPlusVal 
         << (exPlusOverFlow ? " overflow " : "  " )
         << " expected " << myGlobalEnergy->getInputExValPlusWheel() <<endl; testPass = false; }
  if ((myGlobalEnergy->getInputEyVlMinusWheel().overFlow()!=eyMinusOvrFlow) || 
      (myGlobalEnergy->getInputEyVlMinusWheel().value()!=eyMinusVl)) { cout << "ey Minus " << eyMinusVl 
         << (eyMinusOvrFlow ? " overflow " : "  " )
         << " from Gct " << myGlobalEnergy->getInputEyVlMinusWheel() <<endl; testPass = false; }
  if ((myGlobalEnergy->getInputEyValPlusWheel().overFlow()!=eyPlusOverFlow) || 
      (myGlobalEnergy->getInputEyValPlusWheel().value()!=eyPlusVal)) { cout << "ey Plus " << eyPlusVal 
         << (eyPlusOverFlow ? " overflow " : "  " )
         << " from Gct " << myGlobalEnergy->getInputEyValPlusWheel() <<endl; testPass = false; }
  if ((myGlobalEnergy->getInputEtVlMinusWheel().overFlow()!=etMinusOvrFlow) || 
      (myGlobalEnergy->getInputEtVlMinusWheel().value()!=etMinusVl)) { cout << "et Minus " << etMinusVl 
         << (etMinusOvrFlow ? " overflow " : "  " )
         << " from Gct " << myGlobalEnergy->getInputEtVlMinusWheel() <<endl; testPass = false; }
  if ((myGlobalEnergy->getInputEtValPlusWheel().overFlow()!=etPlusOverFlow) || 
      (myGlobalEnergy->getInputEtValPlusWheel().value()!=etPlusVal)) { cout << "et Plus " << etPlusVal 
         << (etPlusOverFlow ? " overflow " : "  " )
         << " from Gct " << myGlobalEnergy->getInputEtValPlusWheel() <<endl; testPass = false; }
 
  //
  // Now check the processing in the final stage GlobalEnergyAlgos
  //--------------------------------------------------------------------------------------
  //
  // Check the missing Et calculation. Allow some margin for the
  // integer calculation of missing Et.
  unsigned etDiff, phDiff;
  unsigned etMargin, phMargin;

  etDiff = (unsigned) abs((long int) etResult.mag - (long int) myGlobalEnergy->getEtMiss().value());
  phDiff = (unsigned) abs((long int) etResult.phi - (long int) myGlobalEnergy->getEtMissPhi().value());
  if (phDiff>60) {phDiff=72-phDiff;}
  //
  etMargin = (etMissOverFlow ? 40 : max((etResult.mag/100), (unsigned) 1) + 2);
  if (etResult.mag==0) { phMargin = 72; } else { phMargin = (30/etResult.mag) + 1; }
  if ((etDiff > etMargin) || (phDiff > phMargin)) {cout << "Algo etMiss diff "
					            << etDiff << " phi diff " << phDiff << endl; testPass = false;}

  if (etMissOverFlow != myGlobalEnergy->getEtMiss().overFlow()) {
    cout << "etMiss overFlow " << (etMissOverFlow ? "expected but not found in Gct" :
                                                    "found in Gct but not expected" ) << std::endl;
    testPass = false;
  }
  // Check the total Et calculation
  if (!myGlobalEnergy->getEtSum().overFlow() && !etTotalOvrFlow &&
      (myGlobalEnergy->getEtSum().value() != etTotal)) {cout << "Algo etSum" << endl; testPass = false;}
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
L1CaloRegion gctTestEnergyAlgos::nextRegionFromFile(const unsigned ieta, const unsigned iphi)
{
  // The file just contains lists of region energies
  unsigned et;
  regionEnergyMapInputFile >> et;
  L1CaloRegion temp(et, false, true, false, false, ieta, iphi);
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
