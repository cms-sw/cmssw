/*! \file testJetFinalStage.cpp
 * \brief Procedural unit-test code for the L1GctJetFinalStage class.
 *
 *  This is code that tests each public method of the L1GctJetFinalStage
 *  class.  It takes data from a file to test the methods against known
 *  results.  Results are also output to file to allow debugging.
 *
 * \author Robert Frazier
 * \date March 2006
 */

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinalStage.h"  //The class to be tested

//Custom headers needed for this test
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

#include "L1Trigger/GlobalCaloTrigger/test/produceTrivialCalibrationLut.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

//Standard library headers
#include <fstream>   //for file IO
#include <string>
#include <vector>
#include <iostream>
#include <exception> //for exception handling
#include <stdexcept> //for std::runtime_error()

using namespace std;

//Typedefs for the vector templates and other types used
typedef vector<L1GctJetCand> JetsVector;
typedef vector<L1GctJet>  RawJetsVector;
typedef unsigned long int ULong;

typedef L1GctJetFinderBase::lutPtr       lutPtr;
typedef L1GctJetFinderBase::lutPtrVector lutPtrVector;

// Name of the files for test input data and results output.
const string testDataFile = "testJetFinalStageInput.txt";  
const string resultsFile = "testJetFinalStageOutput.txt";


// Global constants to tell the program how many jets to read in from file
// THESE ARE TOTAL GUESSES!
const int numInputJets = 8;  //Num. jets of each type given as input
const int numOutputJets = 4; //Num. Jets of each type outputted.


//  FUNCTION PROTOTYPES
/// Runs the test on the L1GctJetFinalStage instance passed into it.
void classTest(L1GctJetFinalStage *myJetFinalStage, const lutPtrVector myLut);
/// Loads test input and also the known results from a file.
void loadTestData(JetsVector &inputCentralJets, JetsVector &inputForwardJets,
                  JetsVector &inputTauJets, JetsVector &trueCentralJets,
                  JetsVector &trueForwardJets, JetsVector &trueTauJets,
                  const lutPtrVector lut);
/// Sanity checks on the data read from file.
bool checkTestData(JetsVector &inputCentralJets, JetsVector &inputForwardJets,
                   JetsVector &inputTauJets, JetsVector &trueCentralJets,
                   JetsVector &trueForwardJets, JetsVector &trueTauJets);
/// Function to safely open input files of any name, using a referenced return ifstream
void safeOpenInputFile(ifstream &fin, const string name);
/// Function to safely open output files of any name, using a referenced return ofstream
void safeOpenOutputFile(ofstream &fout, const string name);
/// Reads jets from file and pushes the specified number into a vector of jets
void putJetsInVector(ifstream &fin, JetsVector &jets, const int numJets, const lutPtrVector lut);
/// Gets the data of a single jet from the testDataFile (reasonably safely).  
L1GctJet readSingleJet(ifstream &fin);
/// Compares JetsVectors, prints a message about the comparison, returns true if identical, else false.
bool compareJetsVectors(JetsVector &vector1, JetsVector &vector2, const string description);
/// Writes out the entire contents of a JetsVector to the given file output stream
void outputJetsVector(ofstream &fout, JetsVector &jets, string description = "Jets");


/// Entrypoint of unit test code + error handling
int main(int argc, char **argv)
{
  cout << "\n*************************************" << endl;
  cout << "L1GctJetFinalStage class unit tester." << endl;
  cout << "*************************************" << endl;

  try
  {
    //Create a whole bunch of objects to allow various other objects
    //to be constructed...
    //create jet calibration lookup table
    produceTrivialCalibrationLut* lutProducer=new produceTrivialCalibrationLut();

    // Instance of the class
    lutPtrVector myJetEtCalLut = lutProducer->produce();
    delete lutProducer;
  
    vector<L1GctJetLeafCard*> jetLeafCrds(L1GctWheelJetFpga::MAX_LEAF_CARDS);
    for(unsigned i=0; i < L1GctWheelJetFpga::MAX_LEAF_CARDS; ++i)
    {
      jetLeafCrds[i] = new L1GctJetLeafCard(0, 0);
      jetLeafCrds[i]->getJetFinderA()->setJetEtCalibrationLuts(myJetEtCalLut);
      jetLeafCrds[i]->getJetFinderB()->setJetEtCalibrationLuts(myJetEtCalLut);
      jetLeafCrds[i]->getJetFinderC()->setJetEtCalibrationLuts(myJetEtCalLut);
    }
    
    vector<L1GctWheelJetFpga*> wheelJetFpgas(L1GctJetFinalStage::MAX_WHEEL_FPGAS);
    for(unsigned i=0; i < L1GctJetFinalStage::MAX_WHEEL_FPGAS; ++i)
    {
      wheelJetFpgas[i] = new L1GctWheelJetFpga((int) i, jetLeafCrds);
    }
    
    L1GctJetFinalStage * myJetFinalStage = new L1GctJetFinalStage(wheelJetFpgas); //TEST OBJECT on heap;    
    classTest(myJetFinalStage, myJetEtCalLut); //run the test

    //clean up
    delete myJetFinalStage;
    for(vector<L1GctWheelJetFpga*>::iterator it = wheelJetFpgas.begin(); it != wheelJetFpgas.end(); ++it)
    {
      delete *it;
    }
    for(vector<L1GctJetLeafCard*>::iterator it = jetLeafCrds.begin(); it != jetLeafCrds.end(); ++it)
    {
      delete *it;
    }
  }
  catch (cms::Exception& e)
  {
    if (e.category() == "FileReadError") {
      std::cout << "No input file - exiting" << std::endl;
    } else {
      cerr << e.what() << endl;
    }
  }
  catch(...)
  {
    cerr << "\nError! An unknown exception has occurred!" << endl;
  }
  
  return 0;   
}

// Runs the test on the L1GctJetFinalStage passed into it.
void classTest(L1GctJetFinalStage *myJetFinalStage, const lutPtrVector lut)
{
  bool testPass = true; //flag to mark test failure
    
  // Vectors for reading in test data from the text file.
  JetsVector inputCentralJets; //Size?
  JetsVector inputForwardJets; //Size?
  JetsVector inputTauJets;     //Size?
  JetsVector trueCentralJets;  //Size?
  JetsVector trueForwardJets;  //Size?
  JetsVector trueTauJets;      //Size?
  
  // Vectors for receiving the output from the object under test.
  JetsVector outputInCentralJets;
  JetsVector outputInForwardJets;
  JetsVector outputInTauJets; 
  JetsVector outputCentralJets; 
  JetsVector outputForwardJets; 
  JetsVector outputTauJets;     
  
  // Load our test input data and known results
  loadTestData(inputCentralJets, inputForwardJets, inputTauJets,
                trueCentralJets,  trueForwardJets,  trueTauJets, lut);

  if (checkTestData(inputCentralJets, inputForwardJets, inputTauJets,
                     trueCentralJets,  trueForwardJets,  trueTauJets)) { testPass=false; }
  
  //Fill the L1GctJetFinalStage with input data. See me care that I'm doing this three times...
  for(int i = 0; i < numInputJets; ++i)  
  {
    myJetFinalStage->setInputCentralJet(i, inputCentralJets[i]);
  }
  
  for(int i = 0; i < numInputJets; ++i) 
  {
    myJetFinalStage->setInputForwardJet(i, inputForwardJets[i]);
  }
  
  for(int i = 0; i < numInputJets; ++i) 
  {
    myJetFinalStage->setInputTauJet(i, inputTauJets[i]);
  }

  // Test the getInputJets() method
  outputInCentralJets = myJetFinalStage->getInputCentralJets();
  if(!compareJetsVectors(outputInCentralJets, inputCentralJets, "central jets initial data input/output")) { testPass = false; }
  outputInForwardJets = myJetFinalStage->getInputForwardJets();
  if(!compareJetsVectors(outputInForwardJets, inputForwardJets, "forward jets initial data input/output")) { testPass = false; }
  outputInTauJets = myJetFinalStage->getInputTauJets();
  if(!compareJetsVectors(outputInTauJets, inputTauJets, "tau jets initial data input/output")) { testPass = false; }
      
  myJetFinalStage->process();  //Run algorithm
  
  //Get the outputted jets and store locally
  outputCentralJets = myJetFinalStage->getCentralJets();
  outputForwardJets = myJetFinalStage->getForwardJets();
  outputTauJets = myJetFinalStage->getTauJets();

  //Test the outputted jets against the known true results
  if(!compareJetsVectors(outputCentralJets, trueCentralJets, "central jets")) { testPass = false; }
  if(!compareJetsVectors(outputForwardJets, trueForwardJets, "forward jets")) { testPass = false; }
  if(!compareJetsVectors(outputTauJets, trueTauJets, "tau jets")) { testPass = false; }

  //Write out all outputtable information to file
  cout << "\nWriting results of processing to file " << resultsFile << "..." << endl;;
  ofstream fout;
  safeOpenOutputFile(fout, resultsFile);
  outputJetsVector(fout, outputInCentralJets, "Inputted Central Jets");
  outputJetsVector(fout, outputInForwardJets, "Inputted Forward Jets");
  outputJetsVector(fout, outputInTauJets, "Inputted Tau Jets");
  outputJetsVector(fout, outputCentralJets, "Central Jets");
  outputJetsVector(fout, outputForwardJets, "Forward Jets");
  outputJetsVector(fout, outputTauJets, "Tau Jets");
  fout.close();
  cout << "Done!" << endl;
  
  //Run the reset method.
  myJetFinalStage->reset();
  
  //get all the data again - should all be empty
  outputInCentralJets = myJetFinalStage->getInputCentralJets();
  outputInForwardJets = myJetFinalStage->getInputForwardJets();
  outputInTauJets = myJetFinalStage->getInputTauJets();
  outputCentralJets = myJetFinalStage->getCentralJets();
  outputForwardJets = myJetFinalStage->getForwardJets();
  outputTauJets = myJetFinalStage->getTauJets();
  
  //vectors just for reset comparison
  JetsVector empty8JetVec(8);
  JetsVector emtpy4JetVec(4);
  
  //Test that all the outputted vectors are indeed empty
  if(!compareJetsVectors(outputInCentralJets, empty8JetVec, "input central jets reset")) { testPass = false; }
  if(!compareJetsVectors(outputInForwardJets, empty8JetVec, "input forward jets reset")) { testPass = false; }
  if(!compareJetsVectors(outputInTauJets, empty8JetVec, "input tau jets reset")) { testPass = false; }
  if(!compareJetsVectors(outputCentralJets, emtpy4JetVec, "output central jets reset")) { testPass = false; }
  if(!compareJetsVectors(outputForwardJets, emtpy4JetVec, "output forward jets reset")) { testPass = false; }
  if(!compareJetsVectors(outputTauJets, emtpy4JetVec, "output tau jets reset")) { testPass = false; }
  
  //Print overall results summary to screen
  if(testPass)
  {
    cout << "\n*************************************" << endl;
    cout << "      Class has passed testing." << endl;
    cout << "*************************************" << endl;
  }
  else
  {
    cout << "\n*************************************" << endl;
    cout << "      Class has FAILED testing!" << endl;
    cout << "*************************************" << endl;
  }
  
  return;        
}


/// Loads test input and also the known results from a file.
void loadTestData(JetsVector &inputCentralJets, JetsVector &inputForwardJets,
                  JetsVector &inputTauJets, JetsVector &trueCentralJets,
                  JetsVector &trueForwardJets, JetsVector &trueTauJets,
                  const lutPtrVector lut)
{
  // File input stream
  ifstream fin;
  
  safeOpenInputFile(fin, testDataFile);  //open the file
  
  // Loads the input data, and the correct results of processing from the file
  putJetsInVector(fin, inputCentralJets, numInputJets, lut);
  putJetsInVector(fin, inputForwardJets, numInputJets, lut);
  putJetsInVector(fin, inputTauJets, numInputJets, lut);
  putJetsInVector(fin, trueCentralJets, numOutputJets, lut);  
  putJetsInVector(fin, trueForwardJets, numOutputJets, lut);  
  putJetsInVector(fin, trueTauJets, numOutputJets, lut);          

  fin.close();    
      
  return;
}
    
/// Loads test input and also the known results from a file.
bool checkTestData(JetsVector &inputCentralJets, JetsVector &inputForwardJets,
                   JetsVector &inputTauJets, JetsVector &trueCentralJets,
                   JetsVector &trueForwardJets, JetsVector &trueTauJets)
{
  bool checkOk=true;

  // Check the data read from file makes sense before we try to use it for testing
  JetsVector::const_iterator jet;
  for (jet=inputCentralJets.begin(); jet!=inputCentralJets.end(); ++jet) { checkOk &= jet->isCentral(); 
    if (!jet->isCentral() && !jet->empty()) {cout << "data check fail input Central rank=" << jet->rank() << endl;} } 
  for (jet=inputForwardJets.begin(); jet!=inputForwardJets.end(); ++jet) { checkOk &= jet->isForward();
    if (!jet->isForward() && !jet->empty()) {cout << "data check fail input Forward rank=" << jet->rank() << endl;} }
  for (jet=inputTauJets.begin();     jet!=inputTauJets.end();     ++jet) { checkOk &= jet->isTau();
    if (!jet->isTau() && !jet->empty()) {cout << "data check fail input Tau rank=" << jet->rank() << endl;} }
  for (jet=trueCentralJets.begin();  jet!=trueCentralJets.end();  ++jet) { checkOk &= jet->isCentral();
    if (!jet->isCentral() && !jet->empty()) {cout << "data check fail true Central rank=" << jet->rank() << endl;} }
  for (jet=trueForwardJets.begin();  jet!=trueForwardJets.end();  ++jet) { checkOk &= jet->isForward();
    if (!jet->isForward() && !jet->empty()) {cout << "data check fail true Forward rank=" << jet->rank() << endl;} }
  for (jet=trueTauJets.begin();      jet!=trueTauJets.end();      ++jet) { checkOk &= jet->isTau();
    if (!jet->isTau() && !jet->empty()) {cout << "data check fail true Tau rank=" << jet->rank() << endl;} }
  return checkOk;
}
    
    
// Function to safely open input files of any name, using a referenced return ifstream
void safeOpenInputFile(ifstream &fin, const string name)
{
  //Opens the file
  fin.open(name.c_str(), ios::in);

  //Throw an exception if something is wrong
  if(!fin.good())
  {
    throw cms::Exception("FileReadError")
    << "Couldn't open the file " << name << " for reading!\n";
  }
  return;
}

// Function to safely open output files of any name, using a referenced return ofstream
void safeOpenOutputFile(ofstream &fout, const string name)
{
  //Opens the file
  fout.open(name.c_str(), ios::trunc);
  
  //Throw an exception if something is wrong
  if(!fout.good())
  {
    throw cms::Exception("FileWriteError")
    << "Couldn't open the file " << name << " for writing!\n";
  }
  return;
}

//Reads jets from file and pushes the specified number into a vector of jets
void putJetsInVector(ifstream &fin, JetsVector &jets, const int numJets, const lutPtrVector lut)
{
  for(int i=0; i < numJets; ++i)
  {
    L1GctJet tempJet=readSingleJet(fin);
    jets.push_back(tempJet.jetCand(lut.at(tempJet.rctEta())));
  }
}

//Gets the data of a single jet from the testDataFile (reasonably safely). 
L1GctJet readSingleJet(ifstream &fin)
{
  //This reperesents how many numbers there are per line for a jet in the input file
  const int numJetComponents = 4; //4 since we have rank, eta, phi & tauVeto.
  
  ULong jetComponents[numJetComponents];

  //read in the data from the file
  for(int i=0; i < numJetComponents; ++i)
  {
    //check to see if the input stream is still ok first
    if(fin.eof() || fin.bad())
    {
      throw cms::Exception("FileReadError")
      << "Error reading jet data from " << testDataFile << "!\n";     
    }
    else
    {
      fin >> jetComponents[i];  //read in the components.
    }
  }
 
  //return object
  // Arguments to ctor are: rank, eta, phi, overFlow, forwardJet, tauVeto, bx
  L1GctJet tempJet(jetComponents[0], jetComponents[1],
		   jetComponents[2], false,
		   ((jetComponents[1]<4) || (jetComponents[1]>=18)), 
		   static_cast<bool>(jetComponents[3]), 0);

  std::cout << "Read jet " << tempJet << std::endl;
  return tempJet;
}

// Compares JetsVectors, prints a message about the comparison, returns true if identical, else false.
bool compareJetsVectors(JetsVector &vector1, JetsVector &vector2, const string description)
{
  // vector1 is output from the JetFinalStage
  // vector2 is expected based on file input
  bool testPass = true;
  
  if(vector1.size() != vector2.size())  //First check overall size is the same
  {
    testPass = false;
  }
  else
  {
    if (!vector1.empty())  //Make sure it isn't empty
    {
      cout << "Number of jets to compare is " << vector1.size() << endl;
      //compare the vectors
      for(ULong i = 0; i < vector1.size(); ++i)
      {
	cout << "jet1: " << vector1[i] << endl;
	cout << "jet2: " << vector2[i] << endl;
        if(vector1[i].rank()      != vector2[i].rank())      { cout << "rank fail " << endl;
                                                               cout << "found "     << vector1[i].rank()
                                                                    << " expected " << vector2[i].rank()
                                                                                    <<endl; testPass = false; break; }
        if((vector1[i].etaIndex() != vector2[i].etaIndex()) ||
           (vector1[i].etaSign()  != vector2[i].etaSign()))  { cout << "eta fail " << endl;
                                                               cout << "found "     << vector1[i].etaIndex() << " " << vector1[i].etaSign()
                                                                    << " expected " << vector2[i].etaIndex() << " " << vector2[i].etaSign()
                                                                                    <<endl; testPass = false; break; }
        if(vector1[i].phiIndex()  != vector2[i].phiIndex())  { cout << "phi fail " << endl; testPass = false; break; }
        if(vector1[i].isTau()     != vector2[i].isTau())     { cout << "tau fail " << endl; testPass = false; break; }
        if(vector1[i].isForward() != vector2[i].isForward()) { cout << "fwd fail " << endl; testPass = false; break; }
      }
    }
  }
  
  //Print results to screen
  if(testPass == false)
  {
    cout << "\nTest class has FAILED " << description << " comparison!" << endl;
    return false;
  }
  else
  {
    cout << "\nTest class has passed " << description << " comparison." << endl;
  }
  return true;
}

// Writes out the entire contents of a JetsVector to the given file output stream
void outputJetsVector(ofstream &fout, JetsVector &jets, string description)
{
  fout << description << endl; //brief description of the JetsVector content
  
  if(!jets.empty())  //check it isn't an empty vector
  {
    for(ULong i=0; i < jets.size(); ++i)
    {
      fout << jets[i].rank() << "\t" 
           << jets[i].etaIndex()  << "\t"
           << jets[i].etaSign()  << "\t"
           << jets[i].phiIndex()  << "\t"
           << jets[i].isTau()  << "\t"
           << jets[i].isForward() << endl;
    }
  }
  fout << endl;  //write a blank line to separate data
}
