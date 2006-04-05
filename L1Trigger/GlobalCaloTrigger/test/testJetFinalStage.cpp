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


//Typedefs for the vector templates and other types used
typedef vector<L1GctJet> JetsVector;
typedef unsigned long int ULong;


// Name of the files for test input data and results output.
const string testDataFile = "testJetFinalStageInput.txt";  
const string resultsFile = "testJetFinalStageOutput.txt";


// Global constants to tell the program how many jets to read in from file
// THESE ARE TOTAL GUESSES!
const int numInputJets = 12;  //Num. jets given as input
const int numCentralJets = 4; //Num. central jets expected out
const int numForwardJets = 4; //Num. forward jets expected out
const int numTauJets = 4;     //Num. tau jets expected out


//  FUNCTION PROTOTYPES
/// Runs the test on the L1GctJetFinalStage instance passed into it.
void classTest(L1GctJetFinalStage *myJetFinalStage);
/// Loads test input and also the known results from a file.
void loadTestData(JetsVector &inputJets, JetsVector &trueCentralJets,
                  JetsVector &trueForwardJets, JetsVector &trueTauJets);
/// Function to safely open input files of any name, using a referenced return ifstream
void safeOpenInputFile(ifstream &fin, const string name);
/// Function to safely open output files of any name, using a referenced return ofstream
void safeOpenOutputFile(ofstream &fout, const string name);
/// Reads jets from file and pushes the specified number into a vector of jets
void putJetsInVector(ifstream &fin, JetsVector &jets, const int numJets);
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

// Runs the test on the L1GctJetFinalStage passed into it.
void classTest(L1GctJetFinalStage *myJetFinalStage)
{
    bool testPass = true; //flag to mark test failure
    
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
    if(!compareJetsVectors(outputJets, inputJets, "initial data input/output")) { testPass = false; }
    
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
    outputJetsVector(fout, outputJets, "Inputted Jets");
    outputJetsVector(fout, outputCentralJets, "Central Jets");
    outputJetsVector(fout, outputForwardJets, "Forward Jets");
    outputJetsVector(fout, outputTauJets, "Tau Jets");
    fout.close();
    cout << "Done!" << endl;
    
    //Run the reset method.
    myJetFinalStage->reset();
    
    //get all the data again - should all be empty
    outputJets = myJetFinalStage->getInputJets();
    outputCentralJets = myJetFinalStage->getCentralJets();
    outputForwardJets = myJetFinalStage->getForwardJets();
    outputTauJets = myJetFinalStage->getTauJets();
    
    //Test that all the outputted vectors are indeed empty
    if(outputJets.empty() && outputCentralJets.empty() &&
       outputForwardJets.empty() && outputTauJets.empty())
    {   
        cout << "\nTest class has passed reset method testing." << endl;
    }
    else
    {
        cout << "\nTest class has FAILED reset method testing!" << endl;
        testPass = false;
    }
    
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
void loadTestData(JetsVector &inputJets, JetsVector &trueCentralJets,
                  JetsVector &trueForwardJets, JetsVector &trueTauJets)
{
    // File input stream
    ifstream fin;
    
    safeOpenInputFile(fin, testDataFile);  //open the file
    
    // Loads the input data, and the correct results of processing from the file
    putJetsInVector(fin, inputJets, numInputJets);          //How many input jets? See global constants.
    putJetsInVector(fin, trueCentralJets, numCentralJets);  //How many?? See global constants.
    putJetsInVector(fin, trueForwardJets, numForwardJets);  //How many?? See global constants.
    putJetsInVector(fin, trueTauJets, numTauJets);          //How many?? See global constants.   

    fin.close();    
        
    return;
}
    
    
// Function to safely open input files of any name, using a referenced return ifstream
void safeOpenInputFile(ifstream &fin, const string name)
{
    //Opens the file
    fin.open(name.c_str(), ios::in);

    //Throw an exception if something is wrong
    if(!fin.good())
    {
        throw std::runtime_error("Couldn't open the file " + name + " for reading!");
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
        throw std::runtime_error("Couldn't open the file " + name + " for writing!");
    }
    return;
}

//Reads jets from file and pushes the specified number into a vector of jets
void putJetsInVector(ifstream &fin, JetsVector &jets, const int numJets)
{
    for(int i=0; i < numJets; ++i)
    {
        jets.push_back(readSingleJet(fin));
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
           throw std::runtime_error("Error reading jet data from " + testDataFile + "!");
        }
        else
        {
            fin >> jetComponents[i];  //read in the components.
        }
    }
   
    //return object
    L1GctJet tempJet(jetComponents[0], jetComponents[1],
                     jetComponents[2], static_cast<bool>(jetComponents[3]));

    return tempJet;
}

// Compares JetsVectors, prints a message about the comparison, returns true if identical, else false.
bool compareJetsVectors(JetsVector &vector1, JetsVector &vector2, const string description)
{
    bool testPass = true;
    
    if(vector1.size() != vector2.size())  //First check overall size is the same
    {
        testPass = false;
    }
    else
    {
        if (!vector1.empty())  //Make sure it isn't empty
        {
            //compare the vectors
            for(ULong i = 0; i < vector1.size(); ++i)
            {
                if(vector1[i].getRank() != vector2[i].getRank()) { testPass = false; break; }
                if(vector1[i].getEta() != vector2[i].getEta()) { testPass = false; break; }
                if(vector1[i].getPhi() != vector2[i].getPhi()) { testPass = false; break; }
                if(vector1[i].getTauVeto() != vector2[i].getTauVeto()) { testPass = false; break; }
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
            fout << jets[i].getRank() << "\t" 
                 << jets[i].getEta()  << "\t"
                 << jets[i].getPhi()  << "\t"
                 << jets[i].getTauVeto() << endl;
        }
    }
    fout << endl;  //write a blank line to separate data
}
