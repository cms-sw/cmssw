#include "L1Trigger/GlobalCaloTrigger/test/gctTestEnergyAlgos.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace std;

//=================================================================================================================
//
/// Constructor and destructor

gctTestEnergyAlgos::gctTestEnergyAlgos()
    : m_bxStart(0), m_numOfBx(1), etStripSums(36), inMinusOvrFlow(1), inPlusOverFlow(1) {}
gctTestEnergyAlgos::~gctTestEnergyAlgos() {}

//=================================================================================================================
//
/// Load another event into the gct. Overloaded for the various ways of doing this.
//  Here's a random event generator. Loads isolated input regions to check the energy sums.
std::vector<L1CaloRegion> gctTestEnergyAlgos::loadEvent(L1GlobalCaloTrigger*& gct,
                                                        const bool simpleEvent,
                                                        const int16_t bx) {
  static const unsigned MAX_ET_CENTRAL = 0x3ff;
  static const unsigned MAX_ET_FORWARD = 0x0ff;
  std::vector<L1CaloRegion> inputRegions;

  // For initial tests just try things out with one region input
  // Then test with summing multiple regions. Choose one value
  // of energy and phi for each eta to avoid trying to set the
  // same region several times.
  for (unsigned i = 0; i < (simpleEvent ? 1 : L1CaloRegionDetId::N_ETA); i++) {
    etmiss_vec etVector = randomMissingEtVector();
    //    cout << "Region et " << etVector.mag << " phi " << etVector.phi << endl;
    // Set a single region input
    unsigned etaRegion = i;
    unsigned phiRegion = etVector.phi / 4;

    bool regionOf = false;
    bool regionFg = false;

    // Central or forward region?
    if (etaRegion < 4 || etaRegion >= 18) {
      // forward
      etVector.mag = etVector.mag >> 2;
      if (etVector.mag >= MAX_ET_FORWARD) {
        // deal with et values in overflow
        etVector.mag = MAX_ET_FORWARD;
      }

    } else {
      // central
      regionOf = etVector.mag > MAX_ET_CENTRAL;
      regionFg = true;
    }
    //    cout << "Region et " << etVector.mag << " eta " << etaRegion << " phi " << etVector.phi << " bx " << bx << endl;
    // Arguments to named ctor are (et, overflow, finegrain, mip, quiet, eta, phi)
    L1CaloRegion temp =
        L1CaloRegion::makeRegionFromGctIndices(etVector.mag, regionOf, regionFg, false, false, etaRegion, phiRegion);
    temp.setBx(bx);
    inputRegions.push_back(temp);
  }

  loadInputRegions(gct, inputRegions, bx);
  return inputRegions;
}

// This method reads the gct input data for jetfinding from a file
// as an array of region energies, one for each eta-phi bin
std::vector<L1CaloRegion> gctTestEnergyAlgos::loadEvent(L1GlobalCaloTrigger*& gct,
                                                        const std::string& fileName,
                                                        bool& endOfFile,
                                                        const int16_t bx) {
  std::vector<L1CaloRegion> inputRegions;

  //Open the file
  if (!regionEnergyMapInputFile.is_open()) {
    regionEnergyMapInputFile.open(fileName.c_str(), ios::in);
  }

  //Error message and abandon ship if we can't read the file
  if (!regionEnergyMapInputFile.good()) {
    throw cms::Exception("fileReadError")
        << " in gctTestEnergyAlgos::loadEvent(L1GlobalCaloTrigger *&, const std::string)\n"
        << "Couldn't read data from file " << fileName << "!";
  }

  // Here we read the energy map from the file.
  // Set each region at the input to the gct, and fill the expected strip sums etc.
  for (unsigned jphi = 0; jphi < L1CaloRegionDetId::N_PHI; ++jphi) {
    unsigned iphi = (L1CaloRegionDetId::N_PHI + 4 - jphi) % L1CaloRegionDetId::N_PHI;
    for (unsigned ieta = 0; ieta < L1CaloRegionDetId::N_ETA; ++ieta) {
      L1CaloRegion temp = nextRegionFromFile(ieta, iphi, bx);
      inputRegions.push_back(temp);
    }
  }
  endOfFile = regionEnergyMapInputFile.eof();

  loadInputRegions(gct, inputRegions, bx);
  return inputRegions;
}

// Load a vector of regions found elsewhere
std::vector<L1CaloRegion> gctTestEnergyAlgos::loadEvent(L1GlobalCaloTrigger*& gct,
                                                        const std::vector<L1CaloRegion>& inputRegions,
                                                        const int16_t bx) {
  loadInputRegions(gct, inputRegions, bx);
  return inputRegions;
}

//=================================================================================================================
//
/// Sends input regions to the gct and remembers strip sums for checking
/// This routine is called from all of the above input methods
void gctTestEnergyAlgos::loadInputRegions(L1GlobalCaloTrigger*& gct,
                                          const std::vector<L1CaloRegion>& inputRegions,
                                          const int16_t bx) {
  int bxRel = bx - m_bxStart;
  int base = 36 * bxRel;
  assert(((base >= 0) && (base + 36) <= (int)etStripSums.size()));

  //Initialise our local sums etc for this event
  for (int i = 0; i < 36; i++) {
    etStripSums.at(i + base) = 0;
  }
  inMinusOvrFlow.at(bxRel) = false;
  inPlusOverFlow.at(bxRel) = false;

  gct->fillRegions(inputRegions);

  // Now add up the input regions
  for (std::vector<L1CaloRegion>::const_iterator reg = inputRegions.begin(); reg != inputRegions.end(); ++reg) {
    assert(reg->bx() == bx);

    static const unsigned MAX_ET_CENTRAL = 0x3ff;
    static const unsigned MAX_ET_FORWARD = 0x0ff;
    // Check the channel masking for Et sums
    if (!m_chanMask->totalEtMask(reg->gctEta())) {
      if (reg->id().ieta() < (L1CaloRegionDetId::N_ETA / 2)) {
        unsigned strip = reg->id().iphi();
        if (reg->overFlow() || ((reg->rctEta() >= 7) && (reg->et() == MAX_ET_FORWARD))) {
          etStripSums.at(strip + base) += MAX_ET_CENTRAL;
          inMinusOvrFlow.at(bxRel) = true;
        } else {
          etStripSums.at(strip + base) += reg->et();
        }
      } else {
        unsigned strip = reg->id().iphi() + L1CaloRegionDetId::N_PHI;
        if (reg->overFlow() || ((reg->rctEta() >= 7) && (reg->et() == MAX_ET_FORWARD))) {
          etStripSums.at(strip + base) += MAX_ET_CENTRAL;
          inPlusOverFlow.at(bxRel) = true;
        } else {
          etStripSums.at(strip + base) += reg->et();
        }
      }
    }
  }
}

//=================================================================================================================
//
/// Set array sizes for the number of bunch crossings
void gctTestEnergyAlgos::setBxRange(const int bxStart, const int numOfBx) {
  // Allow the start of the bunch crossing range to be altered
  // without losing previously stored etStripSums
  for (int bx = bxStart; bx < m_bxStart; bx++) {
    etStripSums.insert(etStripSums.begin(), 36, (unsigned)0);
    inMinusOvrFlow.insert(inMinusOvrFlow.begin(), false);
    inPlusOverFlow.insert(inPlusOverFlow.begin(), false);
  }

  m_bxStart = bxStart;

  // Resize the vectors without clearing previously stored values
  etStripSums.resize(36 * numOfBx);
  inMinusOvrFlow.resize(numOfBx);
  inPlusOverFlow.resize(numOfBx);
  m_numOfBx = numOfBx;
}

//=================================================================================================================
//
/// Check the energy sums algorithms
bool gctTestEnergyAlgos::checkEnergySums(const L1GlobalCaloTrigger* gct) const {
  bool testPass = true;
  L1GctGlobalEnergyAlgos* myGlobalEnergy = gct->getEnergyFinalStage();

  for (int bx = 0; bx < m_numOfBx; bx++) {
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
    for (unsigned leaf = 0; leaf < 3; leaf++) {
      int exMinusSm = 0;
      int eyMinusSm = 0;
      unsigned etMinusSm = 0;
      int exPlusSum = 0;
      int eyPlusSum = 0;
      unsigned etPlusSum = 0;

      unsigned etm0 = 0, etm1 = 0, etp0 = 0, etp1 = 0;

      for (unsigned col = 0; col < 6; col++) {
        unsigned strip = (22 - rctStrip) % 18 + 36 * bx;
        unsigned etm = etStripSums.at(strip) % 4096;
        unsigned etp = etStripSums.at(strip + 18) % 4096;
        etMinusSm += etm;
        etPlusSum += etp;

        if (col % 2 == 0) {
          etm0 = etm;
          etp0 = etp;
        } else {
          etm1 = etm;
          etp1 = etp;

          int exm = etComponent(etm0, ((2 * strip + 11) % 36), etm1, ((2 * strip + 9) % 36));
          int eym = etComponent(etm0, ((2 * strip + 2) % 36), etm1, ((2 * strip) % 36));

          int exp = etComponent(etp0, ((2 * strip + 11) % 36), etp1, ((2 * strip + 9) % 36));
          int eyp = etComponent(etp0, ((2 * strip + 2) % 36), etp1, ((2 * strip) % 36));

          exMinusSm += exm;
          eyMinusSm += eym;

          exPlusSum += exp;
          eyPlusSum += eyp;
        }
        rctStrip++;
      }
      // Check overflow for each leaf card
      if (exMinusSm < -65535) {
        exMinusSm += 131072;
        exMinusOvrFlow = true;
      }
      if (exMinusSm >= 65535) {
        exMinusSm -= 131072;
        exMinusOvrFlow = true;
      }
      if (eyMinusSm < -65535) {
        eyMinusSm += 131072;
        eyMinusOvrFlow = true;
      }
      if (eyMinusSm >= 65535) {
        eyMinusSm -= 131072;
        eyMinusOvrFlow = true;
      }
      if (etMinusSm >= 4096) {
        etMinusOvrFlow = true;
      }
      exMinusVl += exMinusSm;
      eyMinusVl += eyMinusSm;
      etMinusVl += etMinusSm;
      if (exPlusSum < -65535) {
        exPlusSum += 131072;
        exPlusOverFlow = true;
      }
      if (exPlusSum >= 65535) {
        exPlusSum -= 131072;
        exPlusOverFlow = true;
      }
      if (eyPlusSum < -65535) {
        eyPlusSum += 131072;
        eyPlusOverFlow = true;
      }
      if (eyPlusSum >= 65535) {
        eyPlusSum -= 131072;
        eyPlusOverFlow = true;
      }
      if (etPlusSum >= 4096) {
        etPlusOverFlow = true;
      }
      exPlusVal += exPlusSum;
      eyPlusVal += eyPlusSum;
      etPlusVal += etPlusSum;
    }
    // Check overflow for the overall sums
    if (exMinusVl < -65535) {
      exMinusVl += 131072;
      exMinusOvrFlow = true;
    }
    if (exMinusVl >= 65535) {
      exMinusVl -= 131072;
      exMinusOvrFlow = true;
    }
    if (eyMinusVl < -65535) {
      eyMinusVl += 131072;
      eyMinusOvrFlow = true;
    }
    if (eyMinusVl >= 65535) {
      eyMinusVl -= 131072;
      eyMinusOvrFlow = true;
    }
    if (etMinusVl >= 4096 || etMinusOvrFlow) {
      etMinusVl = 4095;
      etMinusOvrFlow = true;
    }

    if (exPlusVal < -65535) {
      exPlusVal += 131072;
      exPlusOverFlow = true;
    }
    if (exPlusVal >= 65535) {
      exPlusVal -= 131072;
      exPlusOverFlow = true;
    }
    if (eyPlusVal < -65535) {
      eyPlusVal += 131072;
      eyPlusOverFlow = true;
    }
    if (eyPlusVal >= 65535) {
      eyPlusVal -= 131072;
      eyPlusOverFlow = true;
    }
    if (etPlusVal >= 4096 || etPlusOverFlow) {
      etPlusVal = 4095;
      etPlusOverFlow = true;
    }

    int exTotal = exMinusVl + exPlusVal;
    int eyTotal = eyMinusVl + eyPlusVal;
    unsigned etTotal = etMinusVl + etPlusVal;

    bool exTotalOvrFlow = exMinusOvrFlow || exPlusOverFlow;
    bool eyTotalOvrFlow = eyMinusOvrFlow || eyPlusOverFlow;
    bool etTotalOvrFlow = etMinusOvrFlow || etPlusOverFlow;

    if (exTotal < -65535) {
      exTotal += 131072;
      exTotalOvrFlow = true;
    }
    if (exTotal >= 65535) {
      exTotal -= 131072;
      exTotalOvrFlow = true;
    }
    if (eyTotal < -65535) {
      eyTotal += 131072;
      eyTotalOvrFlow = true;
    }
    if (eyTotal >= 65535) {
      eyTotal -= 131072;
      eyTotalOvrFlow = true;
    }
    if (etTotal >= 4096 || etTotalOvrFlow) {
      etTotal = 4095;
      etTotalOvrFlow = true;
    }

    etmiss_vec etResult = trueMissingEt(-exTotal / 2, -eyTotal / 2);

    bool etMissOverFlow = exTotalOvrFlow || eyTotalOvrFlow;
    if (etMissOverFlow) {
      etResult.mag = 4095;
      etResult.phi = 45;
    }
    if (etResult.mag >= 4095) {
      etResult.mag = 4095;
      etMissOverFlow = true;
    }

    //
    // Check the input to the final GlobalEnergyAlgos is as expected
    //--------------------------------------------------------------------------------------
    //
    if ((myGlobalEnergy->getInputExVlMinusWheel().at(bx).overFlow() != exMinusOvrFlow) ||
        (myGlobalEnergy->getInputExVlMinusWheel().at(bx).value() != exMinusVl)) {
      cout << "ex Minus at GlobalEnergy input " << exMinusVl << (exMinusOvrFlow ? " overflow " : "  ") << " from Gct "
           << myGlobalEnergy->getInputExVlMinusWheel().at(bx) << endl;
      testPass = false;
    }
    if ((myGlobalEnergy->getInputExValPlusWheel().at(bx).overFlow() != exPlusOverFlow) ||
        (myGlobalEnergy->getInputExValPlusWheel().at(bx).value() != exPlusVal)) {
      cout << "ex Plus at GlobalEnergy input " << exPlusVal << (exPlusOverFlow ? " overflow " : "  ") << " from Gct "
           << myGlobalEnergy->getInputExValPlusWheel().at(bx) << endl;
      testPass = false;
    }
    if ((myGlobalEnergy->getInputEyVlMinusWheel().at(bx).overFlow() != eyMinusOvrFlow) ||
        (myGlobalEnergy->getInputEyVlMinusWheel().at(bx).value() != eyMinusVl)) {
      cout << "ey Minus at GlobalEnergy input " << eyMinusVl << (eyMinusOvrFlow ? " overflow " : "  ") << " from Gct "
           << myGlobalEnergy->getInputEyVlMinusWheel().at(bx) << endl;
      testPass = false;
    }
    if ((myGlobalEnergy->getInputEyValPlusWheel().at(bx).overFlow() != eyPlusOverFlow) ||
        (myGlobalEnergy->getInputEyValPlusWheel().at(bx).value() != eyPlusVal)) {
      cout << "ey Plus at GlobalEnergy input " << eyPlusVal << (eyPlusOverFlow ? " overflow " : "  ") << " from Gct "
           << myGlobalEnergy->getInputEyValPlusWheel().at(bx) << endl;
      testPass = false;
    }
    if ((myGlobalEnergy->getInputEtVlMinusWheel().at(bx).overFlow() != etMinusOvrFlow) ||
        (myGlobalEnergy->getInputEtVlMinusWheel().at(bx).value() != etMinusVl)) {
      cout << "et Minus at GlobalEnergy input " << etMinusVl << (etMinusOvrFlow ? " overflow " : "  ") << " from Gct "
           << myGlobalEnergy->getInputEtVlMinusWheel().at(bx) << endl;
      testPass = false;
    }
    if ((myGlobalEnergy->getInputEtValPlusWheel().at(bx).overFlow() != etPlusOverFlow) ||
        (myGlobalEnergy->getInputEtValPlusWheel().at(bx).value() != etPlusVal)) {
      cout << "et Plus at GlobalEnergy input " << etPlusVal << (etPlusOverFlow ? " overflow " : "  ") << " from Gct "
           << myGlobalEnergy->getInputEtValPlusWheel().at(bx) << endl;
      testPass = false;
    }

    //
    // Now check the processing in the final stage GlobalEnergyAlgos
    //--------------------------------------------------------------------------------------
    //
    // Check the missing Et calculation. Allow some margin for the
    // integer calculation of missing Et.
    unsigned etDiff, phDiff;
    unsigned etMargin, phMargin;

    etDiff = (unsigned)abs((long int)etResult.mag - (long int)myGlobalEnergy->getEtMissColl().at(bx).value());
    phDiff = (unsigned)abs((long int)etResult.phi - (long int)myGlobalEnergy->getEtMissPhiColl().at(bx).value());
    if (etDiff > 2000) {
      etDiff = 4096 - etDiff;
      etMissOverFlow = true;
    }
    if (phDiff > 60) {
      phDiff = 72 - phDiff;
    }
    //
    etMargin = (etMissOverFlow ? 40 : max((etResult.mag / 100), (unsigned)1) + 2);
    if (etResult.mag == 0) {
      phMargin = 72;
    } else {
      phMargin = (30 / etResult.mag) + 1;
    }
    if ((etDiff > etMargin) || (phDiff > phMargin)) {
      cout << "Algo etMiss diff " << etDiff << " phi diff " << phDiff << endl;
      testPass = false;
      cout << " exTotal " << exTotal << " eyTotal " << eyTotal << endl;
      cout << "etMiss mag " << etResult.mag << " phi " << etResult.phi << "; from Gct mag "
           << myGlobalEnergy->getEtMissColl().at(bx).value() << " phi "
           << myGlobalEnergy->getEtMissPhiColl().at(bx).value() << endl;
      cout << "ex Minus " << exMinusVl << (exMinusOvrFlow ? " overflow " : "  ") << " from Gct "
           << myGlobalEnergy->getInputExVlMinusWheel().at(bx) << endl;
      cout << "ex Plus " << exPlusVal << (exPlusOverFlow ? " overflow " : "  ") << " from Gct "
           << myGlobalEnergy->getInputExValPlusWheel().at(bx) << endl;
      cout << "ey Minus " << eyMinusVl << (eyMinusOvrFlow ? " overflow " : "  ") << " from Gct "
           << myGlobalEnergy->getInputEyVlMinusWheel().at(bx) << endl;
      cout << "ey Plus " << eyPlusVal << (eyPlusOverFlow ? " overflow " : "  ") << " from Gct "
           << myGlobalEnergy->getInputEyValPlusWheel().at(bx) << endl;
      rctStrip = 0;
      for (unsigned i = 0; i < gct->getJetLeafCards().size(); i++) {
        cout << "Leaf card " << i << " ex " << gct->getJetLeafCards().at(i)->getAllOutputEx().at(bx) << " ey "
             << gct->getJetLeafCards().at(i)->getAllOutputEy().at(bx) << endl;
        cout << "strip sums ";
        for (unsigned col = 0; col < 6; col++) {
          unsigned strip = (40 - rctStrip) % 18 + 36 * bx;
          cout << " s " << strip << " e " << etStripSums.at(strip);
          rctStrip++;
        }
        cout << endl;
      }
    }

    if (etMissOverFlow != myGlobalEnergy->getEtMissColl().at(bx).overFlow()) {
      cout << "etMiss overFlow " << (etMissOverFlow ? "expected but not found in Gct" : "found in Gct but not expected")
           << std::endl;
      unsigned etm0 = myGlobalEnergy->getEtMissColl().at(bx).value();
      unsigned etm1 = etResult.mag;
      cout << "etMiss value from Gct " << etm0 << "; expected " << etm1 << endl;
      if ((etm0 > 4090) && (etm1 > 4090) && (etm0 < 4096) && (etm1 < 4096)) {
        cout << "Known effect - continue testing" << endl;
      } else {
        testPass = false;
      }
    }
    // Check the total Et calculation
    if (!myGlobalEnergy->getEtSumColl().at(bx).overFlow() && !etTotalOvrFlow &&
        (myGlobalEnergy->getEtSumColl().at(bx).value() != etTotal)) {
      cout << "Algo etSum" << endl;
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
// Function definitions for event generation
//=========================================================================
// Generates 2-d missing Et vector
gctTestEnergyAlgos::etmiss_vec gctTestEnergyAlgos::randomMissingEtVector() const {
  // This produces random variables distributed as a 2-d Gaussian
  // with a standard deviation of sigma for each component,
  // and magnitude ranging up to 5*sigma.
  //
  // With sigma set to 400 we will always be in the range
  // of 12-bit signed integers (-2048 to 2047).
  // With sigma set to 200 we are always in the range
  // of 10-bit input region Et values.
  const float sigma = 400.;

  // rmax controls the magnitude range
  // Chosen as a power of two conveniently close to
  // exp(5*5/2) to give a 5*sigma range.
  const unsigned rmax = 262144;

  // Generate a pair of uniform pseudo-random integers
  vector<unsigned> components = randomTestData((int)2, rmax);

  // Exclude the value zero for the first random integer
  // (Alternatively, return an overflow bit)
  while (components[0] == 0) {
    components = randomTestData((int)2, rmax);
  }

  // Convert to the 2-d Gaussian
  float p, r, s;
  unsigned Emag, Ephi;

  const float nbins = 18.;

  r = float(rmax);
  s = r / float(components[0]);
  p = float(components[1]) / r;
  // Force phi value into the centre of a bin
  Emag = int(sigma * sqrt(2. * log(s)));
  Ephi = int(nbins * p);
  // Copy to the output
  etmiss_vec Et;
  Et.mag = Emag;
  Et.phi = (4 * Ephi);
  return Et;
}

/// Generates test data consisting of a vector of energies
/// uniformly distributed between zero and max
vector<unsigned> gctTestEnergyAlgos::randomTestData(const int size, const unsigned max) const {
  vector<unsigned> energies;
  int r, e;
  float p, q, s, t;

  p = float(max);
  q = float(RAND_MAX);
  for (int i = 0; i < size; i++) {
    r = rand();
    s = float(r);
    t = s * p / q;
    e = int(t);

    energies.push_back(e);
  }
  return energies;
}

// Loads test input regions from a text file.
L1CaloRegion gctTestEnergyAlgos::nextRegionFromFile(const unsigned ieta, const unsigned iphi, const int16_t bx) {
  // The file just contains lists of region energies
  unsigned et;
  regionEnergyMapInputFile >> et;
  L1CaloRegion temp = L1CaloRegion::makeRegionFromGctIndices(et, false, true, false, false, ieta, iphi);
  temp.setBx(bx);
  return temp;
}

//
// Function definitions for energy sum checking
//=========================================================================
int gctTestEnergyAlgos::etComponent(const unsigned Emag0,
                                    const unsigned fact0,
                                    const unsigned Emag1,
                                    const unsigned fact1) const {
  // Copy the Ex, Ey conversion from the hardware emulation
  const unsigned sinFact[10] = {0, 2845, 5603, 8192, 10531, 12550, 14188, 15395, 16134, 16383};
  unsigned myFact;
  bool neg0 = false, neg1 = false, negativeResult;
  int res0 = 0, res1 = 0, result;
  unsigned Emag, fact;

  for (int i = 0; i < 2; i++) {
    if (i == 0) {
      Emag = Emag0;
      fact = fact0;
    } else {
      Emag = Emag1;
      fact = fact1;
    }

    switch (fact / 9) {
      case 0:
        myFact = sinFact[fact];
        negativeResult = false;
        break;
      case 1:
        myFact = sinFact[(18 - fact)];
        negativeResult = false;
        break;
      case 2:
        myFact = sinFact[(fact - 18)];
        negativeResult = true;
        break;
      case 3:
        myFact = sinFact[(36 - fact)];
        negativeResult = true;
        break;
      default:
        cout << "Invalid factor " << fact << endl;
        return 0;
    }
    result = static_cast<int>(Emag * myFact);
    if (i == 0) {
      res0 = result;
      neg0 = negativeResult;
    } else {
      res1 = result;
      neg1 = negativeResult;
    }
  }
  if (neg0 == neg1) {
    result = res0 + res1;
    negativeResult = neg0;
  } else {
    if (res0 >= res1) {
      result = res0 - res1;
      negativeResult = neg0;
    } else {
      result = res1 - res0;
      negativeResult = neg1;
    }
  }
  // Divide by 8192 using bit-shift; but emulate
  // twos-complement arithmetic for negative numbers
  if (negativeResult) {
    result = (1 << 28) - result;
    result = (result + 0x1000) >> 13;
    result = result - (1 << 15);
  } else {
    result = (result + 0x1000) >> 13;
  }
  return result;
}

/// Calculate the expected missing Et vector for a given
/// ex and ey sum, for comparison with the hardware
gctTestEnergyAlgos::etmiss_vec gctTestEnergyAlgos::trueMissingEt(const int ex, const int ey) const {
  etmiss_vec result;

  double fx = static_cast<double>(ex);
  double fy = static_cast<double>(ey);
  double fmag = sqrt(fx * fx + fy * fy);
  double fphi = 36. * atan2(fy, fx) / 3.1415927;

  result.mag = static_cast<int>(fmag);
  if (fphi >= 0) {
    result.phi = static_cast<int>(fphi);
  } else {
    result.phi = static_cast<int>(fphi + 72.);
  }

  return result;
}
