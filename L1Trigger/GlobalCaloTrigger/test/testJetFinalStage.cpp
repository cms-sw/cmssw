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
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"

//Standard library headers
#include <fstream>   //for file IO
#include <string>
#include <vector>
#include <iostream>
#include <exception> //for exception handling
#include <stdexcept> //for std::runtime_error()
using namespace std;


//Typedefs for the vector templates used
typedef vector<L1GctJet> JetsVector;
typedef unsigned long int ULong;


// Name of the files for test input data and results output.
const string testDataFile = "testJetFinalStageInput.txt";  
const string resultsFile = "testJetFinalStageOutput.txt";


// Constants to tell the program how many jets to read in and out
// THESE ARE TOTAL GUESSES!
const int numInputJets = 12;
const int numCentralJets = 4;
const int numForwardJets = 4;
const int numTauJets = 4;


//  FUNCTION PROTOTYPES
/// Runs the test, and returns a string with the test result message in.
void classTest(L1GctJetFinalStage *myJetFinalStage);
/// Loads test input and also the known results from a file.
void loadTestData(JetsVector &inputJets, JetsVector &trueCentralJets,
                  JetsVector &trueForwardJets, JetsVector &trueTauJets);
/// Function to safely open files of any name, using a referenced return ifstream
void safeOpenFile(ifstream &fin, const string &name);
/// Reads jets from file and pushes the specified number into a vector of jets
void putJetsInVector(ifstream &fin, JetsVector &jets, int numJets);
/// Gets jet data from the testDataFile (reasonably safely). 
L1GctJet readSingleJet(ifstream &fin);
/// Compares JetsVectors, prints a message about the comparison, returns true if identical, else false.
bool compareJetsVectors(JetsVector &vector1, JetsVector &vector2, const string description);


/// Entrypoint of unit test code + error handling
int main(int argc, char **argv)
{
    cout << "\n*************************************" << endl;
    cout << "L1GctJetFinalStage class unit tester." << endl;
    cout << "*************************************" << endl;

    try
    {
        L1GctJetFinalStage * myJetFinalStage = new L1GctJetFinalStage(); //TEST OBJECT on heap;    
        classTest(myJetFinalStage);
        delete myJetFinalStage;
    }
    catch(const exception &e)
    {
        cerr << "\nError! " << e.what() << endl;
    }
    catch(...)
    {
        cerr << "\nError! An unknown exception has occurred!" << endl;
    }
    
    return 0;   
}

// Runs the test, and returns a string with the test result message in.
void classTest(L1GctJetFinalStage *myJetFinalStage)
{
    // Vectors for reading in test data from the text file.
    JetsVector inputJets;        //Size?
    JetsVector trueCentralJets;  //Size?
    JetsVector trueForwardJets;  //Size?
    JetsVector trueTauJets;      //Size?
    
    // Vectors for receiving the output from the object under test.
    JetsVector outputJets;         //Size?
    JetsVector outputCentralJets;  //Size?
    JetsVector outputForwardJets;  //Size?
    JetsVector outputTauJets;      //Size?
    
    // Load our test input data and known results
    loadTestData(inputJets, trueCentralJets, trueForwardJets, trueTauJets);
    
    //Fill the L1GctJetFinalStage with input data.
    for(int i = 0; i < numInputJets; ++i)  //How many? See global constants at top of file
    {
        myJetFinalStage->setInputJet(i, inputJets[i]);
    }

    // Test the getInputJets() method
    outputJets = myJetFinalStage->getInputJets();
    if(!compareJetsVectors(outputJets, inputJets, "initial data input/output")) { return; }
    
    myJetFinalStage->process();  //Run algorithm
    
    //Get and then test the output jets against known results
    outputCentralJets = myJetFinalStage->getCentralJets();
    outputForwardJets = myJetFinalStage->getForwardJets();
    outputTauJets = myJetFinalStage->getTauJets();

    if(!compareJetsVectors(outputCentralJets, trueCentralJets, "central jets")) { return; }
    if(!compareJetsVectors(outputForwardJets, trueForwardJets, "forward jets")) { return; }
    if(!compareJetsVectors(outputTauJets, trueTauJets, "tau jets")) { return; }

    cout << "\nWriting results of processing to file...";
    /*
     * Add file output of results
    */
    cout << "done!" << endl;
    
    //test the reset method.
    myJetFinalStage->reset();
    //get all the data again
    outputJets = myJetFinalStage->getInputJets();
    outputCentralJets = myJetFinalStage->getCentralJets();
    outputForwardJets = myJetFinalStage->getForwardJets();
    outputTauJets = myJetFinalStage->getTauJets();
    
    if(outputJets.empty() && outputCentralJets.empty() &&
       outputForwardJets.empty() && outputTauJets.empty())
    {   
        cout << "\nTest class has passed reset method testing." << endl;
    }
    else
    {
        cout << "\nTest class has FAILED reset method testing!" << endl;
        return;
    }

    cout << "\nTest class has passed all tests." << endl;

    return;        
}


/// Loads test input and also the known results from a file.
void loadTestData(JetsVector &inputJets, JetsVector &trueCentralJets,
                  JetsVector &trueForwardJets, JetsVector &trueTauJets)
{
    // File input stream
    ifstream fin;
    
    safeOpenFile(fin, testDataFile);  //open the file
    
    // Loads the input data, and the correct results of processing from the file
    putJetsInVector(fin, inputJets, numInputJets);          //How many input jets? See global constants.
    putJetsInVector(fin, trueCentralJets, numCentralJets);  //How many?? See global constants.
    putJetsInVector(fin, trueForwardJets, numForwardJets);  //How many?? See global constants.
    putJetsInVector(fin, trueTauJets, numTauJets);          //How many?? See global constants.   

    fin.close();    
        
    return;
}
    
    
// Function to safely open files of any name, using a referenced return ifstream
void safeOpenFile(ifstream &fin, const string &name)
{
    //Opens the file
    fin.open(name.c_str(), ios::in);

    //Error message, and return false if it goes pair shaped
    if(!fin.good())
    {
        throw std::runtime_error("Couldn't open the file " + name + "!");
    }
    return;
}

//Reads jets from file and pushes the specified number into a vector of jets
void putJetsInVector(ifstream &fin, JetsVector &jets, int numJets)
{
    for(int i=0; i < numJets; ++i)
    {
        jets.push_back(readSingleJet(fin));
    }
}

//Gets jet data from the testDataFile (reasonably safely). 
L1GctJet readSingleJet(ifstream &fin)
{
    //This reperesents how many numbers there are per line in the input file
    const int numJetComponents = 3; //3 since we have rank, eta & phi.
    
    ULong jetComponents[numJetComponents];

    //read in the data from the file
    for(int i=0; i < numJetComponents; ++i)
    {
        if(fin.eof() || fin.bad())
        {
           throw std::runtime_error("Error reading data from " + testDataFile + "!");
        }
        else
        {
            fin >> jetComponents[i];
        }
    }
   
    L1GctJet tempJet(jetComponents[0], jetComponents[1], jetComponents[2]);    

    return tempJet;
}

// Compares JetsVectors, prints a message about the comparison, returns true if identical, else false.
bool compareJetsVectors(JetsVector &vector1, JetsVector &vector2, const string description)
{
    bool testPass = true;
    
    if(vector1.size() != vector2.size())  
    {
        testPass = false;
    }
    else
    {
        if (!vector1.empty())
        {
            for(unsigned int i = 0; i < vector1.size(); ++i)
            {
                if(vector1[i].getRank() != vector2[i].getRank()) { testPass = false; break; }
                if(vector1[i].getEta() != vector2[i].getEta()) { testPass = false; break; }
                if(vector1[i].getPhi() != vector2[i].getPhi()) { testPass = false; break; }
            }
        }
    }
    
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
