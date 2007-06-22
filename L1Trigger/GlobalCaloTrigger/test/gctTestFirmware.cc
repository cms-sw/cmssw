#include "L1Trigger/GlobalCaloTrigger/test/gctTestFirmware.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

#include <iostream>

using std::vector;
using std::cout;
using std::endl;

//=================================================================================================================
//
/// Constructor and destructor

gctTestFirmware::gctTestFirmware() : jetsFromFile(L1CaloRegionDetId::N_PHI) {}
gctTestFirmware::~gctTestFirmware() {}

void gctTestFirmware::fillJetsFromFirmware(const std::string &fileName)
{
  //Open the file
  if (!jetsFromFirmwareInputFile.is_open()) {
    jetsFromFirmwareInputFile.open(fileName.c_str(), std::ios::in);
  }

  //Error message and abandon ship if we can't read the file
  if(!jetsFromFirmwareInputFile.good())
  {
    throw cms::Exception("fileReadError")
    << " in gctTestFirmware::checkJetFinder(const L1GlobalCaloTrigger*, const std::string &)\n"
    << "Couldn't read data from file " << fileName << "!";
  }

  jetsFromFile = getJetsFromFile();
}

//
//=========================================================================
// Here's the procedure for checking the jet finding
//=========================================================================
//
/// Check the jet finder against results from the firmware
bool gctTestFirmware::checkJetFinder(const L1GlobalCaloTrigger* gct) const
{
  bool testPass = true;
  unsigned jf = 0;
  for (int jlc=0; jlc<L1GlobalCaloTrigger::N_JET_LEAF_CARDS; ++jlc) {
    testPass &= (jetsFromFile.at(jf++) == gct->getJetLeafCards().at(jlc)->getJetFinderA()->getRawJets());
    testPass &= (jetsFromFile.at(jf++) == gct->getJetLeafCards().at(jlc)->getJetFinderB()->getRawJets());
    testPass &= (jetsFromFile.at(jf++) == gct->getJetLeafCards().at(jlc)->getJetFinderC()->getRawJets());
  }

  // Diagnostics if we've found an error
  if (!testPass) {
    unsigned jf = 0;
    for (int jlc=0; jlc<L1GlobalCaloTrigger::N_JET_LEAF_CARDS; ++jlc) {
      for (int i=0; i<3; i++) {
	JetsVector jetlist1, jetlist2;
	cout << "Jet Finder " << jf;
	jetlist1 = jetsFromFile.at(jf++);
	switch (i) {
	case 0 :
	  jetlist2 = gct->getJetLeafCards().at(jlc)->getJetFinderA()->getRawJets(); break;
	case 1 :
	  jetlist2 = gct->getJetLeafCards().at(jlc)->getJetFinderB()->getRawJets(); break;
	case 2 :
	  jetlist2 = gct->getJetLeafCards().at(jlc)->getJetFinderC()->getRawJets(); break;
	}
	bool ok = true;
	for (unsigned j=0; j<L1GctJetFinderBase::MAX_JETS_OUT; j++) {
	  if (jetlist1.at(j)!=jetlist2.at(j)) {
	    cout << "\nJet Number " << j;
	    cout << "\nexpected " << jetlist1.at(j);
	    cout << "\nfound    " << jetlist2.at(j) << endl;
	    ok = false;
	  }
	}
	if (ok) { cout << " all ok!" << endl; }
      }
    }
  }

  return testPass;		 

}

/// Read one event's worth of jets from the file
vector<gctTestFirmware::JetsVector> gctTestFirmware::getJetsFromFile()
{
  vector<JetsVector> result;
  char textFromFile[10];
  std::string strFromFile;
  unsigned jf, ev;
  jetsFromFirmwareInputFile.width(10);
  jetsFromFirmwareInputFile >> textFromFile;
  jetsFromFirmwareInputFile >> ev;
  strFromFile = textFromFile;
  assert (strFromFile=="Event");
  for (unsigned j=0; j<L1CaloRegionDetId::N_PHI; ++j) {
    jetsFromFirmwareInputFile >> textFromFile;
    jetsFromFirmwareInputFile >> jf;
    strFromFile = textFromFile;
    assert ((strFromFile=="JetFinder") && (jf==j));
    JetsVector temp;
    for (unsigned i=0; i<L1GctJetFinderBase::MAX_JETS_OUT; ++i) {
      temp.push_back(nextJetFromFile(jf));
    }
    // Sort the jets coming from the hardware to match the order from the jetFinderBase
	// *** The sort is currently commented. Note that it won't work unless the ***
	// *** same et->rank lookup table is used in the test and in the emulator  ***
    // sort(temp.begin(), temp.end(), L1GctJet::rankGreaterThan());
    result.push_back(temp);
  }
  return result;
}

/// Read a single jet
L1GctJet gctTestFirmware::nextJetFromFile (const unsigned jf)
{

  unsigned et, eta, phi;
  bool of, tv;
  jetsFromFirmwareInputFile >> et;
  jetsFromFirmwareInputFile >> of;
  jetsFromFirmwareInputFile >> tv;
  jetsFromFirmwareInputFile >> eta;
  jetsFromFirmwareInputFile >> phi;

  // Declare local constants to save typing
  const unsigned NE = L1CaloRegionDetId::N_ETA/2;
  const unsigned NP = L1CaloRegionDetId::N_PHI/2;
  // Convert local jetfinder to global coordinates
  // Note about phi - the jetfinders are mapped onto
  // the RCT crates with jf #0 covering 50-90 degrees
  // and jf #8 covering 90-130 degrees
  unsigned globalEta = (eta==NE+1) ? 0 : ((jf<NP) ? (NE-eta) : (NE+eta-1));
  unsigned globalPhi = (eta==NE+1) ? 0 : ((2*(NP+1-(jf%NP))+3*phi)%(2*NP));

  if (of) { et |= (1<<L1GctJet::RAWSUM_BITWIDTH); }

  L1GctJet temp(et, globalEta, globalPhi, tv);
  return temp;
}

