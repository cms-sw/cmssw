/*! \file testJetFinder.cpp
 * \brief Procedural unit-test code for the L1GctHardwareJetFinder class.
 *
 *  This is code that tests each public method of the L1GctTdrJetFinder
 *  class.  It takes data from a file to test the methods against known
 *  results.  Results are also output to file to allow debugging.
 *
 * \author Robert Frazier
 * \date March 2006
 */

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHardwareJetFinder.h"  //The class to be tested

//Custom headers needed for this test
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

#include "L1Trigger/GlobalCaloTrigger/test/produceTrivialCalibrationLut.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

//Standard library headers
#include <fstream>  //for file IO
#include <string>
#include <vector>
#include <iostream>
#include <exception>  //for exception handling
#include <stdexcept>  //for std::runtime_error()
using namespace std;

//Typedefs for the vector templates and other types used
typedef vector<L1GctRegion> RegionsVector;
typedef vector<L1GctJet> RawJetsVector;
typedef vector<L1GctJetCand> JetsVector;
typedef unsigned long int ULong;

typedef L1GctJetFinderBase::lutPtr lutPtr;
typedef L1GctJetFinderBase::lutPtrVector lutPtrVector;

// Name of the files for test input data and results output.
const string testDataFile = "testJetFinderInputMismatch.txt";
const string resultsFile = "testJetFinderOutputMismatch.txt";

// Global constants to tell the program how many things to read in from file
// THESE ARE TOTAL GUESSES!
const int numInputRegions = 52;  //Num. calorimeter regions given as input
const int numOutputJets = 18;    //Num. jets expected out
//There will be Jet Counts to be added here at some point

// Test any jetFinder between 0 and 17
const unsigned jfNumber = 5;

//  FUNCTION PROTOTYPES
/// Runs the test on the L1GctJetFinder instance passed into it.
void classTest(L1GctHardwareJetFinder *myJetFinder, std::vector<L1GctJetFinderBase *> myNeighbours);
/// Loads test input regions and also the known results from a text file.
void loadTestData(
    RegionsVector &inputRegions, RawJetsVector &trueJets, ULong &trueHt, ULong &stripSum0, ULong &stripSum1);
/// Function to safely open input files of any name, using a referenced return ifstream
void safeOpenInputFile(ifstream &fin, const string name);
/// Function to safely open output files of any name, using a referenced return ofstream
void safeOpenOutputFile(ofstream &fout, const string name);
/// Reads regions from file and pushes the specified number into a vector of regions
void putRegionsInVector(ifstream &fin, RegionsVector &regions, const int numRegions);
/// Gets the data of a single region from the testDataFile (reasonably safely).
L1GctRegion readSingleRegion(ifstream &fin);
/// Reads jets from file and pushes the specified number into a vector of jets
void putJetsInVector(ifstream &fin, RawJetsVector &jets, const int numJets);
/// Gets the data of a single jet from the testDataFile (reasonably safely).
L1GctJet readSingleJet(ifstream &fin);
/// Compares RegionsVectors, prints a message about the comparison, returns true if identical, else false.
bool compareRegionsVectors(RegionsVector &vector1, RegionsVector &vector2, const string description);
/// Compares JetsVectors, prints a message about the comparison, returns true if identical, else false.
bool compareJetsVectors(RawJetsVector &vector1, RawJetsVector &vector2, const string description);
/// Writes out the entire contents of a RegionsVector to the given file output stream
void outputRegionsVector(ofstream &fout, RegionsVector &regions, string description = "Regions");
/// Writes out the entire contents of a JetsVector to the given file output stream
void outputJetsVector(ofstream &fout, RawJetsVector &jets, string description = "Jets");

/// Entrypoint of unit test code + error handling
int main(int argc, char **argv) {
  cout << "\n*************************************" << endl;
  cout << "  L1GctJetFinder class unit tester." << endl;
  cout << "*************************************" << endl;

  try {
    produceTrivialCalibrationLut *lutProducer = new produceTrivialCalibrationLut();

    // Instance of the class
    const lutPtrVector &myJetEtCalLut = lutProducer->produce();
    const L1GctJetFinderParams *myJfPars = lutProducer->jfPars();
    delete lutProducer;

    // The jetfinder number to be tested is given by jfNumber
    if (jfNumber < 18) {
      // Work out the numbers of the two neighbours
      unsigned jfNeigh0 = (jfNumber + 8) % 9;
      unsigned jfNeigh1 = (jfNumber + 1) % 9;
      if (jfNumber >= 9) {
        jfNeigh0 += 9;
        jfNeigh1 += 9;
      }

      L1GctHardwareJetFinder *myJetFinder = new L1GctHardwareJetFinder(jfNumber);  //TEST OBJECT on heap;
      std::vector<L1GctJetFinderBase *> myNeighbours;
      myNeighbours.push_back(new L1GctHardwareJetFinder(jfNeigh0));
      myNeighbours.push_back(new L1GctHardwareJetFinder(jfNeigh1));

      myJetFinder->setJetEtCalibrationLuts(myJetEtCalLut);
      myJetFinder->setJetFinderParams(myJfPars);
      myNeighbours.at(0)->setJetEtCalibrationLuts(myJetEtCalLut);
      myNeighbours.at(0)->setJetFinderParams(myJfPars);
      myNeighbours.at(1)->setJetEtCalibrationLuts(myJetEtCalLut);
      myNeighbours.at(1)->setJetFinderParams(myJfPars);

      myJetFinder->setNeighbourJetFinders(myNeighbours);

      // Connect the jetFinders around in a little circle
      std::vector<L1GctJetFinderBase *> dummyNeighbours(2);
      dummyNeighbours.at(0) = myNeighbours.at(1);
      dummyNeighbours.at(1) = myJetFinder;
      myNeighbours.at(0)->setNeighbourJetFinders(dummyNeighbours);
      dummyNeighbours.at(0) = myJetFinder;
      dummyNeighbours.at(1) = myNeighbours.at(0);
      myNeighbours.at(1)->setNeighbourJetFinders(dummyNeighbours);

      classTest(myJetFinder, myNeighbours);

      //clean up
      delete myNeighbours.at(0);
      delete myNeighbours.at(1);
      delete myJetFinder;
    } else {
      cout << "Invalid jet finder number " << jfNumber << "; should be less than 18" << endl;
    }
  } catch (cms::Exception &e) {
    if (e.category() == "FileReadError") {
      std::cout << "No input file - exiting" << std::endl;
    } else {
      cerr << e.what() << endl;
    }
  } catch (...) {
    cerr << "\nError! An unknown exception has occurred!" << endl;
  }

  return 0;
}

// Runs the test, and returns a string with the test result message in.
void classTest(L1GctHardwareJetFinder *myJetFinder, std::vector<L1GctJetFinderBase *> myNeighbours) {
  bool testPass = true;  //flag to mark test failure

  // Vectors for reading in test data from the text file.
  RegionsVector inputRegions;  //Size?
  RawJetsVector trueJets;      //Size?
  ULong trueHt;
  ULong outputEt;
  ULong stripSum0, stripSum1;

  // Vectors for receiving the output from the object under test.
  RegionsVector outputRegions;  //Size?
  RawJetsVector outputJets;     //Size?
  ULong outputHt;
  ULong sumOfJetHt;
  //Jet Counts to be added at some point

  // Load our test input data and known results
  loadTestData(inputRegions, trueJets, trueHt, stripSum0, stripSum1);

  //Fill the L1GctJetFinder with regions.
  for (int i = 0; i < numInputRegions; ++i) {
    myJetFinder->setInputRegion(inputRegions.at(i));
    myNeighbours.at(0)->setInputRegion(inputRegions.at(i));
    myNeighbours.at(1)->setInputRegion(inputRegions.at(i));
  }

  // Test the getInputRegion method
  RegionsVector centralRegions;
  const unsigned COL_OFFSET = L1GctJetFinderBase::COL_OFFSET;
  unsigned pos = 0;
  for (unsigned col = 0; col < 3; ++col) {
    for (unsigned row = 0; row < COL_OFFSET; ++row) {
      if (col == 1 || col == 2) {
        centralRegions.push_back(inputRegions.at(pos));
      }
      ++pos;
    }
  }
  outputRegions = myJetFinder->getInputRegions();
  if (!compareRegionsVectors(outputRegions, centralRegions, "initial data input/output")) {
    testPass = false;
  }

  myJetFinder->fetchInput();         //Run algorithm
  myNeighbours.at(0)->fetchInput();  //Run algorithm
  myNeighbours.at(1)->fetchInput();  //Run algorithm
  myJetFinder->process();            //Run algorithm
  myNeighbours.at(0)->process();     //Run algorithm
  myNeighbours.at(1)->process();     //Run algorithm

  //Get the outputted data and store locally
  outputJets = myJetFinder->getRawJets();
  RawJetsVector nJets0 = myNeighbours.at(0)->getRawJets();
  RawJetsVector nJets1 = myNeighbours.at(1)->getRawJets();
  outputJets.insert(outputJets.end(), nJets0.begin(), nJets0.end());
  outputJets.insert(outputJets.end(), nJets1.begin(), nJets1.end());
  outputHt = myJetFinder->getHtSum().value();

  sumOfJetHt = 0;
  for (RawJetsVector::const_iterator it = outputJets.begin(); it != outputJets.end(); ++it) {
    sumOfJetHt += it->calibratedEt(myJetFinder->getJetEtCalLuts().at(it->rctEta()));
  }

  //Test the outputted jets against the known results
  if (!compareJetsVectors(outputJets, trueJets, "outputted jets")) {
    testPass = false;
  }

  //Test the outputted Ht against known result.
  //NOTE: this is lookup table dependent, so the reference file
  //needs to change when the lookup table changes.
  if (outputHt != sumOfJetHt) {
    cout << "output Ht " << outputHt << " true Ht " << sumOfJetHt << endl;
    cout << "\nTest class has FAILED Ht comparison!" << endl;
    testPass = false;
  } else {
    cout << "\nTest class has passed Ht comparison." << endl;
    if (outputHt != trueHt) {
      cout << "The value recorded in the file " << trueHt << " is wrong; should be " << sumOfJetHt << endl;
      cout << "Have you changed the calibration function??" << endl;
    }
  }

  //Test the Et strip sums against known results
  if ((stripSum0 + stripSum1) != myJetFinder->getEtSum().value()) {
    cout << "Et sum 0 comparison: expected strips " << stripSum0 << " and " << stripSum1 << " found "
         << myJetFinder->getEtSum() << endl;
    cout << "\nTest class has FAILED Et sum comparison!" << endl;
    testPass = false;
  } else {
    cout << "\nTest class has passed Et strip sum comparison." << endl;
  }

  //Write out all outputtable information to file
  cout << "\nWriting results of processing to file " << resultsFile << "..." << endl;
  ;
  ofstream fout;
  safeOpenOutputFile(fout, resultsFile);
  outputRegionsVector(fout, outputRegions, "Inputted Regions");
  outputJetsVector(fout, outputJets, "Outputted Jets");
  fout << "Outputted Ht" << endl;
  fout << outputHt << endl << endl;
  fout << "Outputted Et strip sums" << endl;
  fout << stripSum0 << "  " << stripSum1 << endl << endl;
  fout.close();
  cout << "Done!" << endl;

  //Run the reset method.
  myJetFinder->reset();

  //get all the data again - should all be empty
  outputRegions = myJetFinder->getInputRegions();
  outputJets = myJetFinder->getRawJets();
  outputHt = myJetFinder->getHtSum().value();
  outputEt = myJetFinder->getEtSum().value();

  //an empty regions vector for reset comparison
  vector<L1GctRegion> blankRegionsVec(numInputRegions);
  vector<L1GctJet> blankJetsVec(numOutputJets);

  //Test that all the vectors/values are empty/zero
  if (compareRegionsVectors(outputRegions, blankRegionsVec, "input regions reset") &&
      compareJetsVectors(outputJets, blankJetsVec, "output jets reset") && outputHt == 0 && outputEt == 0) {
    cout << "\nTest class has passed reset method testing." << endl;
  } else {
    cout << "\nTest class has FAILED reset method testing!" << endl;
    testPass = false;
  }

  //Print overall results summary to screen
  if (testPass) {
    cout << "\n*************************************" << endl;
    cout << "      Class has passed testing." << endl;
    cout << "*************************************" << endl;
  } else {
    cout << "\n*************************************" << endl;
    cout << "      Class has FAILED testing!" << endl;
    cout << "*************************************" << endl;
  }

  return;
}

// Loads test input regions from a text file.
void loadTestData(
    RegionsVector &inputRegions, RawJetsVector &trueJets, ULong &trueHt, ULong &stripSum0, ULong &stripSum1) {
  // File input stream
  ifstream fin;

  safeOpenInputFile(fin, testDataFile);  //open the file

  putRegionsInVector(fin, inputRegions, numInputRegions);  //How many input regions? See global constants.
  putJetsInVector(fin, trueJets, numOutputJets);           //How many?? See global constants.

  if (fin.eof() || fin.bad()) {
    throw std::runtime_error("Error reading Ht data from " + testDataFile + "!");
  } else {
    fin >> trueHt;
    fin >> stripSum0;
    fin >> stripSum1;
  }

  fin.close();

  return;
}

// Function to safely open input files of any name, using a referenced return ifstream
void safeOpenInputFile(ifstream &fin, const string name) {
  //Opens the file
  fin.open(name.c_str(), ios::in);

  //Throw an exception if something is wrong
  if (!fin.good()) {
    throw cms::Exception("FileReadError") << "Couldn't open the file " << name << " for reading!\n";
  }
  return;
}

// Function to safely open output files of any name, using a referenced return ofstream
void safeOpenOutputFile(ofstream &fout, const string name) {
  //Opens the file
  fout.open(name.c_str(), ios::trunc);

  //Throw an exception if something is wrong
  if (!fout.good()) {
    throw cms::Exception("FileWriteError") << "Couldn't open the file " << name << " for writing!\n";
  }
  return;
}

//Reads regions from file and pushes the specified number into a vector of regions
void putRegionsInVector(ifstream &fin, RegionsVector &regions, const int numRegions) {
  for (int i = 0; i < numRegions; ++i) {
    regions.push_back(readSingleRegion(fin));
  }
}

//Gets the data of a single region from the testDataFile (reasonably safely).
L1GctRegion readSingleRegion(ifstream &fin) {
  //Represents how many numbers there are per line for a region in the input file
  const int numRegionComponents = 6;  //the id, et, overFlow, tauVeto, mip, quiet, tauVeto.

  ULong regionComponents[numRegionComponents];

  for (int i = 0; i < numRegionComponents; ++i) {
    //check to see if the input stream is still ok first
    if (fin.eof() || fin.bad()) {
      throw cms::Exception("FileReadError") << "Error reading region data from " << testDataFile << "!\n";
    } else {
      fin >> regionComponents[i];  //read in the components.
    }
  }

  // First input value is position in JetFinder array.
  // Convert to eta and phi in global coordinates.
  // Give the two central columns (out of four) the
  // (phi, eta) values corresponding to the required RCT crate number.
  // These depend on the jetfinder number to be tested
  static const unsigned COL_OFFSET = L1GctJetFinderBase::COL_OFFSET;
  static const unsigned NEXTRA = L1GctJetFinderBase::N_EXTRA_REGIONS_ETA00;
  unsigned localEta = regionComponents[0] % COL_OFFSET;
  unsigned localPhi = regionComponents[0] / COL_OFFSET;
  unsigned ieta = ((jfNumber < 9) ? (10 + NEXTRA - localEta) : (11 - NEXTRA + localEta));
  ;
  unsigned phiOffset = 43 - 2 * jfNumber;
  unsigned iphi = (phiOffset - localPhi) % 18;
  //return object
  L1CaloRegion tempRegion(regionComponents[1],
                          static_cast<bool>(regionComponents[2]),
                          static_cast<bool>(regionComponents[3]),
                          static_cast<bool>(regionComponents[4]),
                          static_cast<bool>(regionComponents[5]),
                          ieta,
                          iphi);

  return L1GctRegion::makeJfInputRegion(tempRegion);
}

//Reads jets from file and pushes the specified number into a vector of jets
void putJetsInVector(ifstream &fin, RawJetsVector &jets, const int numJets) {
  for (int i = 0; i < numJets; ++i) {
    jets.push_back(readSingleJet(fin));
  }
}

//Gets the data of a single jet from the testDataFile (reasonably safely).
L1GctJet readSingleJet(ifstream &fin) {
  //This reperesents how many numbers there are per line for a jet in the input file
  const int numJetComponents = 4;  //4 since we have rank, eta, phi & tauVeto.

  ULong jetComponents[numJetComponents];

  //read in the data from the file
  for (int i = 0; i < numJetComponents; ++i) {
    //check to see if the input stream is still ok first
    if (fin.eof() || fin.bad()) {
      throw cms::Exception("FileReadError") << "Error reading jet data from " << testDataFile << "!\n";
    } else {
      fin >> jetComponents[i];  //read in the components.
    }
  }

  //return object
  // Arguments to ctor are: rank, eta, phi, overFlow, forwardJet, tauVeto, bx
  L1GctJet tempJet(jetComponents[0],
                   jetComponents[1],
                   jetComponents[2],
                   false,
                   ((jetComponents[1] < 4) || (jetComponents[1] >= 18)),
                   static_cast<bool>(jetComponents[3]),
                   0);
  return tempJet;
}

// Compares RegionsVectors, prints a message about the comparison, returns true if identical, else false.
bool compareRegionsVectors(RegionsVector &vector1, RegionsVector &vector2, const string description) {
  bool testPass = true;

  if (vector1.size() != vector2.size())  //First check overall size is the same
  {
    testPass = false;
  } else {
    if (!vector1.empty())  //make sure it isn't empty
    {
      //compare the vectors
      for (ULong i = 0; i < vector1.size(); ++i) {
        if (vector1[i].id() != vector2[i].id()) {
          testPass = false;
          break;
        }
        if (vector1[i].et() != vector2[i].et()) {
          testPass = false;
          break;
        }
        if (vector1[i].overFlow() != vector2[i].overFlow()) {
          testPass = false;
          break;
        }
        if (vector1[i].tauVeto() != vector2[i].tauVeto()) {
          testPass = false;
          break;
        }
        if (vector1[i].mip() != vector2[i].mip()) {
          testPass = false;
          break;
        }
        if (vector1[i].quiet() != vector2[i].quiet()) {
          testPass = false;
          break;
        }
      }
    }
  }

  //Print results to screen
  if (testPass == false) {
    cout << "\nTest class has FAILED " << description << " comparison!" << endl;
    return false;
  } else {
    cout << "\nTest class has passed " << description << " comparison." << endl;
  }
  return true;
}

// Compares RawJetsVectors, prints a message about the comparison, returns true if identical, else false.
bool compareJetsVectors(RawJetsVector &vector1, RawJetsVector &vector2, const string description) {
  bool testPass = true;

  if (vector1.size() != vector2.size())  //First check overall size is the same
  {
    cout << "Failed size comparison\n";
    testPass = false;
  } else {
    if (!vector1.empty())  //Make sure it isn't empty
    {
      //compare the vectors
      for (unsigned int i = 0; i < vector1.size(); ++i) {
        if (vector1[i].rawsum() != vector2[i].rawsum()) {
          cout << "Jet " << i << " Failed rawsum comparison -- ";
          cout << "first " << vector1[i].rawsum() << " second " << vector2[i].rawsum() << "\n";
          testPass = false;
        }
        if (vector1[i].rctEta() != vector2[i].rctEta()) {
          cout << "Jet " << i << " Failed rctEta comparison\n";
          testPass = false;
        }
        if (vector1[i].rctPhi() != vector2[i].rctPhi()) {
          cout << "Jet " << i << " Failed rctPhi comparison\n";
          testPass = false;
        }
        if (vector1[i].tauVeto() != vector2[i].tauVeto()) {
          cout << "Jet " << i << " Failed tauV comparison\n";
          testPass = false;
        }
      }
    }
  }

  //Print results to screen
  if (testPass == false) {
    cout << "\nTest class has FAILED " << description << " comparison!" << endl;
    return false;
  } else {
    cout << "\nTest class has passed " << description << " comparison." << endl;
  }
  return true;
}

// Writes out the entire contents of a RegionsVector to the given file output stream
void outputRegionsVector(ofstream &fout, RegionsVector &regions, string description) {
  fout << description << endl;  //brief description of the RegionsVector content

  if (!regions.empty())  //check it isn't an empty vector
  {
    for (unsigned int i = 0; i < regions.size(); ++i) {
      fout << regions[i].et() << "\t" << regions[i].gctEta() << "\t" << regions[i].gctPhi() << "\t"
           << regions[i].overFlow() << "\t" << regions[i].tauVeto() << "\t" << regions[i].mip() << "\t"
           << regions[i].quiet() << endl;
    }
  }
  fout << endl;  //write a blank line to separate data
}

// Writes out the entire contents of a JetsVector to the given file output stream
void outputJetsVector(ofstream &fout, RawJetsVector &jets, string description) {
  fout << description << endl;  //brief description for each JetsVector content

  if (!jets.empty())  //check it isn't an empty vector
  {
    for (unsigned int i = 0; i < jets.size(); ++i) {
      fout << jets[i].rawsum() << "\t" << jets[i].globalEta() << "\t" << jets[i].globalPhi() << "\t"
           << jets[i].tauVeto() << endl;
    }
  }
  fout << endl;  //write a blank line to separate data
}
