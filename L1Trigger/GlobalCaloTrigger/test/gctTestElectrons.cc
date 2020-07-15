#include "L1Trigger/GlobalCaloTrigger/test/gctTestElectrons.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronSorter.h"

#include <fstream>  //for file IO
#include <iostream>
#include <sstream>  //for int->char conversion

using std::cout;
using std::endl;
using std::string;
using std::vector;

//=================================================================================================================
//
/// Constructor and destructor

gctTestElectrons::gctTestElectrons() {
  m_theIsoEmCandSorter = new L1GctElectronSorter(18, false);
  m_nonIsoEmCandSorter = new L1GctElectronSorter(18, true);
}

gctTestElectrons::~gctTestElectrons() {
  delete m_theIsoEmCandSorter;
  delete m_nonIsoEmCandSorter;
}

//=================================================================================================================
//
/// Load another event into the gct. Overloaded for the various ways of doing this.
std::vector<L1CaloEmCand> gctTestElectrons::loadEvent(L1GlobalCaloTrigger*& gct,
                                                      const std::string fileName,
                                                      const int16_t bx) {
  m_fileNameUsed = fileName;
  //  gct->openSourceCardFiles(fileName);
  //Open the input files using the function LoadFile and send the EmCands to the gct
  for (int i = 0; i < 18; i++) {
    std::stringstream ss;
    string fileNo;
    ss << i;
    ss >> fileNo;
    LoadFileData(m_fileNameUsed + fileNo, bx);
  }

  // copy both local vectors of input candidates into a single vector
  std::vector<L1CaloEmCand> temp(m_theIsoEmCandsFromFileInput);
  temp.insert(temp.end(), m_nonIsoEmCandsFromFileInput.begin(), m_nonIsoEmCandsFromFileInput.end());

  // send the input candidates to the gct
  gct->fillEmCands(temp);

  return temp;
}

//=================================================================================================================
//
/// Read the input electron data (after GCT processing).
void gctTestElectrons::fillElectronData(const L1GlobalCaloTrigger* gct) {
  m_theIsoEmCandsFromGct = gct->getIsoElectrons();
  m_nonIsoEmCandsFromGct = gct->getNonIsoElectrons();

  cout << "=========== From the GCT chain ===============" << endl;
  cout << "Iso electrons are: " << endl;
  print(m_theIsoEmCandsFromGct);
  cout << "Non-iso electrons are: " << endl;
  print(m_nonIsoEmCandsFromGct);
}

//=================================================================================================================
//
/// Repeat the sort locally and check the result
bool gctTestElectrons::checkElectrons(const L1GlobalCaloTrigger* gct, const int bxStart, const int numOfBx) {
  bool testPass = true;

  // Process multiple bunch crossings
  m_theIsoEmCandsFromFileSorted.clear();
  m_nonIsoEmCandsFromFileSorted.clear();

  vector<L1GctEmCand>::iterator theIsoItr = m_theIsoEmCandsFromFileSorted.end();
  vector<L1GctEmCand>::iterator nonIsoItr = m_nonIsoEmCandsFromFileSorted.end();

  int16_t bx = bxStart;
  for (int i = 0; i < numOfBx; i++) {
    //Open the input files using the function LoadFile and sort them and see if same output is returned
    for (int i = 0; i < 18; i++) {
      std::stringstream ss;
      string fileNo;
      ss << i;
      ss >> fileNo;
      LoadFileData(m_fileNameUsed + fileNo, bx);
    }

    std::cout << "Filling sorters" << std::endl;
    for (unsigned int i = 0; i != 72; i++) {
      m_theIsoEmCandSorter->setInputEmCand(m_theIsoEmCandsFromFileInput[i]);
      m_nonIsoEmCandSorter->setInputEmCand(m_nonIsoEmCandsFromFileInput[i]);
    }
    m_theIsoEmCandSorter->process();
    m_nonIsoEmCandSorter->process();

    m_theIsoEmCandsFromFileSorted.insert(
        theIsoItr, m_theIsoEmCandSorter->getOutputCands().begin(), m_theIsoEmCandSorter->getOutputCands().end());
    m_nonIsoEmCandsFromFileSorted.insert(
        nonIsoItr, m_nonIsoEmCandSorter->getOutputCands().begin(), m_nonIsoEmCandSorter->getOutputCands().end());

    bx++;
  }

  cout << "=================From files externally sorted=============" << endl;
  cout << "Iso electrons are:" << endl;
  print(m_theIsoEmCandsFromFileSorted);
  cout << "Non-iso electrons are:" << endl;
  print(m_nonIsoEmCandsFromFileSorted);

  testPass &= m_theIsoEmCandsFromFileSorted == m_theIsoEmCandsFromGct;
  testPass &= m_nonIsoEmCandsFromFileSorted == m_nonIsoEmCandsFromGct;

  return testPass;
}

/// PRIVATE MEMBER FUNCTIONS
// Function definition of function that reads in dummy data and load it into inputCands vector
void gctTestElectrons::LoadFileData(const string& inputFile, const int16_t bx) {
  using namespace std;

  ifstream file;

  //Opens the file
  file.open(inputFile.c_str(), ios::in);

  if (!file) {
    throw cms::Exception("ErrorOpenFile") << "Cannot open input data file" << endl;
  }

  unsigned candRank = 0, candRegion = 0, candCard = 0, candCrate = 0;
  short dummy;
  string bxNo = "poels";
  bool candIso = false;

  //Reads in first crossing, then crossing no
  //then 8 electrons, 4 first is iso, next non iso.
  //The 3x14 jet stuff and 8 mip and quiet bits are skipped over.

  file >> bxNo;
  file >> std::dec >> dummy;
  for (int i = 0; i < 58; i++) {  //Loops over one bunch-crossing
    if (i < 8) {
      file >> std::hex >> dummy;
      candRank = dummy & 0x3f;
      candRegion = (dummy >> 6) & 0x1;
      candCard = (dummy >> 7) & 0x7;
      candIso = (i >= 4);
      L1CaloEmCand electron(candRank, candRegion, candCard, candCrate, candIso);
      electron.setBx(bx);
      if (candIso) {
        m_theIsoEmCandsFromFileInput.push_back(electron);
      } else {
        m_nonIsoEmCandsFromFileInput.push_back(electron);
      }
    } else {
      file >> std::hex >> dummy;
    }
  }
  file.close();

  return;
}

void gctTestElectrons::print(const vector<L1GctEmCand> cands) const {
  for (unsigned int i = 0; i != cands.size(); i++) {
    cout << "          Rank: " << cands[i].rank() << "  Eta: " << cands[i].etaIndex()
         << "  Phi: " << cands[i].phiIndex() << endl;
  }
  return;
}
