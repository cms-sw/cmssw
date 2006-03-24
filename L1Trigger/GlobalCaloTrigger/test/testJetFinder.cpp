/*! \file testJetFinder.cpp
 * \brief Procedural unit-test code for the L1GctJetFinder class.
 *
 *  This is code that reads in data from a text file to feed into
 *  the setInputRegions() method, runs the process() method, and then
 *  checks the data from the outputting methods getInputRegions() and
 *  getJets() against known results also stored in the text file.
 *
 * \author Robert Frazier
 * \date March 2006
 */

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinder.h"  //The class to be tested

//Custom headers needed for this test
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"
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
typedef vector<L1GctRegion> RegionsVector;
typedef vector<L1GctJet> JetsVector;

//  FUNCTION PROTOTYPES
/// Runs the test, and returns a string with the test result message in.
string classTest();
/// Loads test input regions and also the known results from a text file.
void loadTestData(RegionsVector &regions, JetsVector &jets, const string &fileName);
/// Function to safely open files of any name, using a referenced return ifstream
void safeOpenFile(ifstream &fin, const string &name);

/// Entrypoint of unit test code + error handling
int main(int argc, char **argv)
{
    cout << "\n*********************************" << endl;
    cout << "L1GctJetFinder class unit tester." << endl;
    cout << "*********************************" << endl;

    try
    {
        cout << "\n" << classTest() << endl;
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
string classTest()
{
    L1GctJetFinder myJetFinder; //TEST OBJECT;    
    bool testPass = true;       //Test passing flag.
    
    // Number of calorimter regions to be fed to the jet finder.
    const int maxRegions = 64;  //64 based on old GCT design
    
    // Name of the file containing the test input data.
    const string testDataFile = "JetFinderTesterData.txt";  
    
    // Vectors for reading in test data from the text file.
    RegionsVector inputRegions(maxRegions);
    JetsVector correctJets;
    // Vectors for receiving the output from the object under test.
    RegionsVector outputRegions(maxRegions);
    JetsVector outputJets;
    
    // Load our test input data and known results
    loadTestData(inputRegions, correctJets, testDataFile);
    
    //Fill the L1GctJetFinder with regions.
    for(int i = 0; i < maxRegions; ++i)
    {
        myJetFinder.setInputRegion(i, inputRegions[i]);
    }

    // Test the getInputRegion method
    outputRegions = myJetFinder.getInputRegions();
    if(outputRegions.size() != inputRegions.size())  
    {
        testPass = false;
    }
    else
    {
        for(int i = 0; i < maxRegions; ++i)
        {
            if(outputRegions[i].getEt() != inputRegions[i].getEt()) { testPass = false; break; }
            if(outputRegions[i].getMip() != inputRegions[i].getMip()) { testPass = false; break; }
            if(outputRegions[i].getQuiet() != inputRegions[i].getQuiet()) {testPass = false; break; }
        }
    }
        
    if(testPass == false)
    {
        return "Test class has failed initial data input/output comparison!";
    }
    
    myJetFinder.process();  //Run algorithm
    
    //Get and then test the output jets against known results
    //NEEDS FILLING IN SIMILARLY TO ABOVE
    outputJets = myJetFinder.getJets();
    // **Fill in***
    
    
    if(testPass == false)
    {
        return "Test class has failed algorithm processing!";
    }

    return "Test class has passed!";        
}


// Loads test input regions from a text file.
void loadTestData(RegionsVector &regions, JetsVector &jets, const string &fileName)
{
    // File input stream
    ifstream fin;
    
    safeOpenFile(fin, fileName);  //open the file
    
    unsigned long int tempEt = 0;
    unsigned short int tempMip = 0;
    unsigned short int tempQuiet = 0;
    
    // Loads the input data
    for(unsigned int i = 0; i < regions.size(); ++i)
    {
        //read in the data from the file
        fin >> tempEt;
        fin >> tempMip;
        fin >> tempQuiet;
        
        regions[i].setEt(tempEt);
        if(tempMip == 0) { regions[i].setMip(false); } else { regions[i].setMip(true); }
        if(tempQuiet == 0) { regions[i].setQuiet(false); } else { regions[i].setQuiet(true); }
    }
    
    // Do similar to load the 'known' output jets (that we currently don't know...)
    
    // Close the file
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
