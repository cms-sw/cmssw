/*! \file testSortAlgo.cpp
 * \test file for testing the electron sorter
 *
 *  This test program reads in dummy data, followed by testing
 *  the electron sorter methods are working correctly. 
 *  
 *
 * \author Maria Hansen
 * \date March 2006
 */

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronSorter.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"
#include "FWCore/Utilities/interface/Exception.h"

//Standard library headers
#include <fstream>  //for file IO
#include <string>
#include <vector>
#include <iostream>
#include <exception>  //for exception handling
#include <stdexcept>  //for std::runtime_error()

using namespace std;

//Typedefs and other definitions
typedef vector<L1CaloEmCand> EmInputCandVec;
typedef vector<L1GctEmCand> EmOutputCandVec;
ifstream file;
ofstream ofile;
EmInputCandVec indata;
EmOutputCandVec gctData;

//  Function for reading in the dummy data
void LoadFileData(const string& inputFile, int electrons, bool isolation);
//Function that outputs a txt file with the output
void WriteFileData(EmOutputCandVec outputs);
//Function to easily output a gctEmCand vector to screen
void print(EmOutputCandVec cands);

//Function to convert from CaloEmCand to GctEmCand, copied from private method in ElectronSorter
void convertToGct(EmInputCandVec cands);

int main() {
  indata.clear();
  gctData.clear();

  cout << "**************************************" << endl;
  cout << "  L1GctElectronSorter class unit tester." << endl;
  cout << "****************************************" << endl;

  EmInputCandVec inputs;
  EmOutputCandVec outputs;
  bool checkIn = false;
  bool checkOut = false;
  bool nInputs = false;
  bool nOutputs = false;

  // Number of electrons in sorter
  unsigned int noElectrons = 16;
  // Name of file with test data
  // testElectrons_0 and 9 contain 1 electron iso/non-iso per bunch crossing, split over all cards and region combinations
  // testElectrons_1 and 10 contain 0 electrons, all entries are 0
  // testElectrons_2 and 11 have energy of electron 0, but still got 4 bits position information
  // testElectrons_3 and 12 contain 2 iso/non-iso electrons per bunch crossing
  // testElectrons_4 and 13 contain 3 iso/non-iso electrons per bunch crossing
  // testElectrons_5 and 14 contain 4 iso/non-iso electrons per bunch crossing
  // testElectrons_6 and 15 contain electrons with rank but no position information
  // testElectrons_7 and 16 contain electrons with equal energies
  // testElectrons_8 and 17 contain electrons with equal energies and phi
  // testElectrons_9 and 10 contain electrons with equal energy, phi and rank

  const string testFile = "data/testElectrons_0";

  try {
    //Constructor with noElectrons non iso (iso = 0) electron candidates
    L1GctElectronSorter* testSort = new L1GctElectronSorter(noElectrons / 4, 1);

    // Check the number of inputs/size of input vector corresponds to the expected value
    inputs = testSort->getInputCands();
    if (inputs.size() != noElectrons) {
      throw cms::Exception("ErrorSizeOfInput")
          << "The electronSorter constructor holds the wrong number of inputs" << endl;
      nInputs = true;
    }

    // Check the number of ouputs/size of output vector corresponds to the expected value
    outputs = testSort->getOutputCands();
    if (outputs.size() != 4) {
      throw cms::Exception("ErrorSizeOfOutput")
          << "The size of the output vector in electron sorter is incorrect" << endl;
      nOutputs = true;
    }

    //Looking at data passed through the sort algorithm
    //Load data from file " " and no of electrons and isolation given as in electron sorter constructor
    LoadFileData(testFile, noElectrons, 1);
    cout << " Data loaded in from input file" << endl;
    print(gctData);
    for (unsigned int i = 0; i < indata.size(); i++) {
      testSort->setInputEmCand(indata[i]);
    }
    inputs = testSort->getInputCands();

    //This part checks that the data read in is what is stored in the private vector of the sort algorithm
    //is the same as what was read in from the data file
    for (unsigned int i = 0; i != indata.size(); i++) {
      if (indata[i].rank() != inputs[i].rank()) {
        throw cms::Exception("ErrorInPrivateVectorRank")
            << "Error in data: Discrepancy between Rank in file and input buffer!" << endl;
        checkIn = true;
      }
      if (indata[i].rctRegion() != inputs[i].rctRegion()) {
        throw cms::Exception("ErrorInRegion")
            << "Error in data:Discrepancy between region in file and input buffer!" << endl;
        checkIn = true;
      }
      if (indata[i].rctCard() != inputs[i].rctCard()) {
        throw cms::Exception("ErrorInCard")
            << "Error in data:Discrepancy between card in file and input buffer!" << endl;
        checkIn = true;
      }
    }

    //sort the electron candidates by rank
    testSort->process();

    //This part checks that the values returned by the getOutput() method are indeed the largest 4 electron candidates sorted by rank
    outputs = testSort->getOutputCands();
    cout << "Output from sort algorithm: " << endl;
    print(outputs);
    for (unsigned int n = 0; n != outputs.size(); n++) {
      int count = 0;
      for (unsigned int i = 0; i != indata.size(); i++) {
        if (indata[i].rank() > outputs[n].rank()) {
          count = count + 1;
          if (n == 0 && count > 1) {
            cout << "Error in getOutput method, highest ranking electron candidate isn't returned" << endl;
            checkOut = true;
          }
          if (n == 1 && count > 2) {
            cout << "Error in getOutput method, 2nd highest ranking electron candidate isn't returned" << endl;
            checkOut = true;
          }
          if (n == 2 && count > 3) {
            cout << "Error in getOutput method, 3rd highest ranking electron candidate isn't returned" << endl;
            checkOut = true;
          }
          if (n == 3 && count > 4) {
            cout << "Error in getOutput method, 4th highest ranking electron candidate isn't returned" << endl;
            checkOut = true;
          }
        }
      }
    }

    if (checkIn) {
      cout << "Error: Discrepancy between data read in and data stored as inputs!" << endl;
    }
    if (checkIn) {
      cout << "Error: Discrepancy between the sorted data and data outputted!" << endl;
    }
    if (nInputs) {
      cout << "The number of inputs/size of input vector in electron sorter is incorrect" << endl;
    }
    if (nOutputs) {
      cout << "The number of outputs/size of output vector in electron sorter is incorrect" << endl;
    }
    if (!nInputs && !nOutputs && !checkIn && !checkOut) {
      cout << "============================================" << endl;
      cout << " Class ElectronSorter has passed unit tester" << endl;
      cout << "============================================" << endl;
    }

    WriteFileData(outputs);
    delete testSort;

  } catch (cms::Exception& e) {
    if (e.category() == "ErrorOpenFile") {
      std::cout << "No input file - exiting" << std::endl;
    } else {
      cerr << e.what() << endl;
    }
  }
  return 0;
}

// Function definition of function that reads in dummy data and load it into inputCands vector
void LoadFileData(const string& inputFile, int elecs, bool iso) {
  //Number of bunch crossings to read in
  int bx = elecs / 4;

  //Opens the file
  file.open(inputFile.c_str(), ios::in);

  if (!file) {
    throw cms::Exception("ErrorOpenFile") << "Cannot open input data file" << endl;
  }

  unsigned candRank = 0, candRegion = 0, candCard = 0, candCrate = 0;
  short dummy;
  string bxNo = "poels";
  bool candIso = 0;

  //Reads in first crossing, then crossing no
  //then 8 electrons, 4 first is iso, next non iso.
  //The 3x14 jet stuff and 8 mip and quiet bits are skipped over.

  for (int n = 0; n < bx; n++) {  //Loop over the bx bunch crossings
    file >> bxNo;
    file >> std::dec >> dummy;
    for (int i = 0; i < 58; i++) {  //Loops over one bunch-crossing
      if (i < 8) {
        if (i > 3 && iso) {
          file >> std::hex >> dummy;
          candRank = dummy & 0x3f;
          candRegion = (dummy >> 6) & 0x1;
          candCard = (dummy >> 7) & 0x7;
          candIso = 1;
          L1CaloEmCand electrons(candRank, candRegion, candCard, candCrate, candIso);
          indata.push_back(electrons);
        } else {
          if (i < 4 && !iso) {
            file >> std::hex >> dummy;
            candRank = dummy & 0x3f;
            candRegion = (dummy >> 6) & 0x1;
            candCard = (dummy >> 7) & 0x7;
            candIso = 0;
            L1CaloEmCand electrons(candRank, candRegion, candCard, candCrate, candIso);
            indata.push_back(electrons);
          } else {
            file >> dummy;
          }
        }
      } else {
        file >> std::hex >> dummy;
      }
    }
  }
  file.close();
  convertToGct(indata);

  return;
}

//Function definition, that writes the output to a file
void WriteFileData(EmOutputCandVec outputs) {
  EmOutputCandVec writeThis = outputs;
  ofile.open("sortOutput.txt", ios::out);
  for (unsigned int i = 0; i != writeThis.size(); i++) {
    ofile << std::hex << writeThis[i].rank();
    ofile << " ";
    ofile << std::hex << writeThis[i].etaIndex();
    ofile << " ";
    ofile << std::hex << writeThis[i].phiIndex();
    ofile << "\n";
  }
  return;
}

void print(EmOutputCandVec candidates) {
  EmOutputCandVec cands = candidates;
  for (unsigned int i = 0; i != cands.size(); i++) {
    cout << "          Rank: " << cands[i].rank() << "  Eta: " << cands[i].etaIndex()
         << "  Phi: " << cands[i].phiIndex() << endl;
  }
  return;
}

// Copy of the private function in ElectronSorter to convert a CaloEmCand to a gctEmCand
void convertToGct(EmInputCandVec candidates) {
  EmInputCandVec cand = candidates;
  for (unsigned int i = 0; i != cand.size(); i++) {
    unsigned rank = cand[i].rank();
    unsigned card = cand[i].rctCard();
    unsigned region = cand[i].rctRegion();
    //unsigned crate = cand[i].rctCrate();
    //bool sign = (crate<9?-1:1); to be used later when giving sign to phi regions
    bool isolation = cand[i].isolated();
    unsigned eta = 10;  //initialisation value, outside eta range
    unsigned phi = 50;

    switch (card) {
      case 0:
        phi = 1;
        if (region == 0) {
          eta = 0;
        } else {
          eta = 1;
        }
        break;
      case 1:
        phi = 1;
        if (region == 0) {
          eta = 2;
        } else {
          eta = 3;
        }
        break;
      case 2:
        phi = 1;
        if (region == 0) {
          eta = 4;
        } else {
          eta = 5;
        }
        [[fallthrough]];
      case 3:
        phi = 0;
        if (region == 0) {
          eta = 0;
        } else {
          eta = 1;
        }
        break;
      case 4:
        phi = 0;
        if (region == 0) {
          eta = 2;
        } else {
          eta = 3;
        }
        [[fallthrough]];
      case 5:
        phi = 0;
        if (region == 0) {
          eta = 4;
        } else {
          eta = 5;
        }
        break;
      case 6:
        if (region == 0) {
          eta = 6;
          phi = 1;
        } else {
          eta = 6;
          phi = 0;
        }
        break;
    }
    L1GctEmCand gctTemp(rank, phi, eta, isolation);
    gctData.push_back(gctTemp);
  }
  return;
}
