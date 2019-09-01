#include "L1Trigger/GlobalCaloTrigger/test/gctTestFirmware.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"

#include <iostream>
#include <cassert>

using std::cout;
using std::endl;
using std::vector;

//=================================================================================================================
//
/// Constructor and destructor

gctTestFirmware::gctTestFirmware() : jetsFromFile(L1CaloRegionDetId::N_PHI) {}
gctTestFirmware::~gctTestFirmware() {}

void gctTestFirmware::fillJetsFromFirmware(const std::string& fileName, const int bxStart, const int numOfBx) {
  //Open the file
  if (!jetsFromFirmwareInputFile.is_open()) {
    jetsFromFirmwareInputFile.open(fileName.c_str(), std::ios::in);
  }

  //Error message and abandon ship if we can't read the file
  if (!jetsFromFirmwareInputFile.good()) {
    throw cms::Exception("fileReadError")
        << " in gctTestFirmware::checkJetFinder(const L1GlobalCaloTrigger*, const std::string &)\n"
        << "Couldn't read data from file " << fileName << "!";
  }

  jetsFromFile = getJetsFromFile(bxStart, numOfBx);
}

//
//=========================================================================
// Here's the procedure for checking the jet finding
//=========================================================================
//
/// Check the jet finder against results from the firmware
bool gctTestFirmware::checkJetFinder(const L1GlobalCaloTrigger* gct) const {
  bool testPass = true;
  unsigned jf = 0;
  for (int jlc = 0; jlc < L1GlobalCaloTrigger::N_JET_LEAF_CARDS; ++jlc) {
    testPass &= (jetsFromFile.at(jf++) == gct->getJetLeafCards().at(jlc)->getJetFinderA()->getRawJets());
    testPass &= (jetsFromFile.at(jf++) == gct->getJetLeafCards().at(jlc)->getJetFinderB()->getRawJets());
    testPass &= (jetsFromFile.at(jf++) == gct->getJetLeafCards().at(jlc)->getJetFinderC()->getRawJets());
  }

  // Diagnostics if we've found an error
  if (!testPass) {
    cout << "checkJetFinder() failed" << endl;
    unsigned jf = 0;
    for (int jlc = 0; jlc < L1GlobalCaloTrigger::N_JET_LEAF_CARDS; ++jlc) {
      for (int i = 0; i < 3; i++) {
        JetsVector jetlist1, jetlist2;
        cout << "Jet Finder " << jf << endl;
        jetlist1 = jetsFromFile.at(jf++);
        switch (i) {
          case 0:
            jetlist2 = gct->getJetLeafCards().at(jlc)->getJetFinderA()->getRawJets();
            break;
          case 1:
            jetlist2 = gct->getJetLeafCards().at(jlc)->getJetFinderB()->getRawJets();
            break;
          case 2:
            jetlist2 = gct->getJetLeafCards().at(jlc)->getJetFinderC()->getRawJets();
            break;
        }
        unsigned numOfBx = jetlist1.size() / L1GctJetFinderBase::MAX_JETS_OUT;
        unsigned jj = 0;
        for (unsigned i = 0; i < numOfBx; i++) {
          cout << "   Bunch crossing " << i;
          bool ok = true;
          for (unsigned j = 0; j < L1GctJetFinderBase::MAX_JETS_OUT; j++) {
            if (jetlist1.at(jj) != jetlist2.at(jj)) {
              cout << "\nJet Number " << j;
              cout << "\nexpected " << jetlist1.at(jj);
              cout << "\nfound    " << jetlist2.at(jj) << endl;
              ok = false;
            }
            ++jj;
          }
          if (ok) {
            cout << " all ok!" << endl;
          }
        }
      }
    }
  }

  return testPass;
}

/// Read one event's worth of jets from the file
vector<gctTestFirmware::JetsVector> gctTestFirmware::getJetsFromFile(const int bxStart, const int numOfBx) {
  vector<JetsVector> result(L1CaloRegionDetId::N_PHI);
  char textFromFile[10];
  std::string strFromFile;
  unsigned jf, ev;

  int bx = bxStart;
  for (int i = 0; i < numOfBx && jetsFromFirmwareInputFile.good(); i++) {
    jetsFromFirmwareInputFile.width(10);
    jetsFromFirmwareInputFile >> textFromFile;
    jetsFromFirmwareInputFile >> ev;
    strFromFile = textFromFile;
    assert(strFromFile == "Event");
    for (unsigned j = 0; j < L1CaloRegionDetId::N_PHI; ++j) {
      jetsFromFirmwareInputFile >> textFromFile;
      jetsFromFirmwareInputFile >> jf;
      strFromFile = textFromFile;
      assert((strFromFile == "JetFinder") && (jf == j));
      JetsVector temp;
      for (unsigned i = 0; i < L1GctJetFinderBase::MAX_JETS_OUT; ++i) {
        temp.push_back(nextJetFromFile(jf, bx));
      }
      // Sort the jets coming from the hardware to match the order from the jetFinderBase
      // *** The sort is currently commented. Note that it won't work unless the ***
      // *** same et->rank lookup table is used in the test and in the emulator  ***
      // sort(temp.begin(), temp.end(), L1GctJet::rankGreaterThan());

      // Shift the jetfinders around in phi
      static const unsigned JF_PHI_OFFSET = 1;
      static const unsigned JF_NPHI = L1CaloRegionDetId::N_PHI / 2;
      unsigned pos = JF_NPHI * (j / JF_NPHI) + (j + JF_PHI_OFFSET) % JF_NPHI;
      JetsVector::iterator itr = result.at(pos).end();
      result.at(pos).insert(itr, temp.begin(), temp.end());
    }
    bx++;
  }
  return result;
}

/// Read a single jet
L1GctJet gctTestFirmware::nextJetFromFile(const unsigned jf, const int bx) {
  unsigned et, eta, phi;
  bool of, tv;
  jetsFromFirmwareInputFile >> et;
  jetsFromFirmwareInputFile >> of;
  jetsFromFirmwareInputFile >> tv;
  jetsFromFirmwareInputFile >> eta;
  jetsFromFirmwareInputFile >> phi;

  // Declare local constants to save typing
  const unsigned NE = L1CaloRegionDetId::N_ETA / 2;
  const unsigned NP = L1CaloRegionDetId::N_PHI / 2;
  // Convert local jetfinder to global coordinates
  // Note about phi - the jetfinders are mapped onto
  // the RCT crates with jf #0 covering 50-90 degrees
  // and jf #8 covering 90-130 degrees
  unsigned globalEta = (eta == NE + 1) ? 0 : ((jf < NP) ? (NE - eta) : (NE + eta - 1));
  unsigned globalPhi = (eta == NE + 1) ? 0 : ((2 * (NP + 1 - (jf % NP)) + 3 * phi) % (2 * NP));

  L1GctJet temp(et, globalEta, globalPhi, of, (eta > 7), tv);
  temp.setBx(bx);
  return temp;
}

/// Analyse calculation of energy sums in firmware
bool gctTestFirmware::checkEnergySumsFromFirmware(const L1GlobalCaloTrigger* gct,
                                                  const std::string& fileName,
                                                  const int numOfBx) {
  bool testPass = true;

  //Open the file
  if (!esumsFromFirmwareInputFile.is_open()) {
    esumsFromFirmwareInputFile.open(fileName.c_str(), std::ios::in);
  }

  //Error message and abandon ship if we can't read the file
  if (!esumsFromFirmwareInputFile.good()) {
    throw cms::Exception("fileReadError")
        << " in gctTestFirmware::checkEnergySumsFromFirmware(const L1GlobalCaloTrigger*, const std::string &)\n"
        << "Couldn't read data from file " << fileName << "!";
  }

  //Loop reading events from the file (one event per line)
  for (int bx = 0; bx < numOfBx; bx++) {
    unsigned evno;
    unsigned etGct, htGct, magGct, phiGct;
    unsigned etEmv, htEmv, magEmv, phiEmv;
    int exGct, eyGct;
    unsigned magTest, phiTest;

    esumsFromFirmwareInputFile >> evno;
    // Values output from the GCT firmware
    esumsFromFirmwareInputFile >> etGct;
    esumsFromFirmwareInputFile >> htGct;
    esumsFromFirmwareInputFile >> magGct;
    esumsFromFirmwareInputFile >> phiGct;
    // Values output from "procedural VHDL" emulator
    esumsFromFirmwareInputFile >> etEmv;
    esumsFromFirmwareInputFile >> htEmv;
    esumsFromFirmwareInputFile >> magEmv;
    esumsFromFirmwareInputFile >> phiEmv;
    // Values of ex, ey components input
    esumsFromFirmwareInputFile >> exGct;
    esumsFromFirmwareInputFile >> eyGct;
    // Values of missing Et from VHDL "algorithm-under-test"
    esumsFromFirmwareInputFile >> magTest;
    esumsFromFirmwareInputFile >> phiTest;

    // Check total Et calculation
    if (etGct != etEmv) {
      cout << "Reading firmware values from file, et from Gct vhdl " << etGct << " from procedural VHDL " << etEmv
           << endl;
      testPass = false;
    }
    if (etGct != gct->getEtSumCollection().at(bx).et()) {
      cout << "Checking firmware values from file, et from Gct " << etGct << " from CMSSW "
           << gct->getEtSumCollection().at(bx).et() << endl;
      testPass = false;
    }

    // Ignore ht check against emulator since it depends on the jet calibration
    // Just check the two firmware values against each other
    if (htGct != htEmv) {
      cout << "Reading firmware values from file, ht from Gct vhdl " << htGct << " from procedural VHDL " << htEmv
           << endl;
      testPass = false;
    }

    int exPlus = gct->getEnergyFinalStage()->getInputExValPlusWheel().at(bx).value();
    int eyPlus = gct->getEnergyFinalStage()->getInputEyValPlusWheel().at(bx).value();
    int exMinus = gct->getEnergyFinalStage()->getInputExVlMinusWheel().at(bx).value();
    int eyMinus = gct->getEnergyFinalStage()->getInputEyVlMinusWheel().at(bx).value();

    int exEmu = exPlus + exMinus;
    int eyEmu = eyPlus + eyMinus;
    if (exGct != exEmu || eyGct != eyEmu) {
      cout << "Checking firmware values from file, met components from Gct vhdl " << exGct << " and " << eyGct
           << "; from CMSSW " << exEmu << " and " << eyEmu << endl;
      testPass = false;
    }

    if (magTest != gct->getEtMissCollection().at(bx).et() || phiTest != gct->getEtMissCollection().at(bx).phi()) {
      cout << "Checking met calculation, components from vhdl " << exGct << " and " << eyGct << ", result mag "
           << magTest << " phi " << phiTest << endl;
      cout << "Components from CMSSW " << exEmu << " and " << eyEmu << ", result mag "
           << gct->getEtMissCollection().at(bx).et() << " phi " << gct->getEtMissCollection().at(bx).phi() << endl;
      testPass = false;
    }
  }

  return testPass;
}
