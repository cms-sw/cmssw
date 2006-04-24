/*! \file testSortAlgo.cpp
 * \test file for testing the electron sorter
 *
 *  This test program reads in dummy data, followed by testing
 *  its methods are working correctly. Any discrepancies found
 *  between read in data and output will be written to the screen
 *  
 *
 * \author Maria Hansen
 * \date March 2006
 */

//To be sorted out!
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronSorter.h" 
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"

//Standard library headers
#include <fstream>   //for file IO
#include <string>
#include <vector>
#include <iostream>

using std::ifstream;
using std::ofstream;

using namespace std;

//Typedefs and other definitions
typedef vector<L1GctEmCand> EmCandidate;
ifstream file;
ofstream ofile;
vector<L1GctEmCand> data;

//  Function for reading in the dummy data
void LoadFileData(const string &inputFile);

//Function that outputs a txt file with the output
void WriteFileData(EmCandidate outputs);

int main()
{
  EmCandidate inputs;
  EmCandidate outputs;
  bool checkIn = false;
  bool checkOut = false;
  

  L1GctElectronSorter* testSort = new L1GctElectronSorter();
  LoadFileData("dummyData.txt");

  for(unsigned int i=0;i<data.size();i++){
    testSort->setInputEmCand(data[i]);
  }

  inputs = testSort->getInput();

  //This part checks that the data read in is what is stored in the private vector of the sort algorithm

  for(unsigned int i=0;i!=data.size();i++){
    if(data[i].getRank() != inputs[i].getRank()){
      cout << "Error in data: Discrepancy between Rank in file and input buffer!"<<endl;
      checkIn = true;
    }
    if(data[i].getEta() != inputs[i].getEta()){
      cout << "Error in data:Discrepancy between Eta in file and input buffer!"<<endl;
      checkIn = true;
    }
    if(data[i].getPhi() != inputs[i].getPhi()){
      cout << "Error in data:Discrepancy between Phi in file and input buffer!"<<endl;
      checkIn = true;
    }
  }

  //sort the electron candidates by rank
  testSort->process();

  //This part checks that the values returned by the getOutput() method are indeed the largest 4 electron candidates sorted by rank

  outputs = testSort->getOutput();
  for(unsigned int n=0;n!=outputs.size();n++){
    int count = 0;
    for(unsigned int i=0;i!=data.size();i++){
      if(data[i].getRank() > outputs[n].getRank()){
       count = count + 1;
       if(n==0 && count > 1){
	 cout <<"Error in getOutput method, highest ranking electron candidate isn't returned"<<endl;
	 checkOut = true;
       }
       if(n==1 && count > 2){
	 cout <<"Error in getOutput method, 2nd highest ranking electron candidate isn't returned"<<endl;
      	 checkOut = true;
       }
       if(n==2 && count > 3){
	 cout <<"Error in getOutput method, 3rd highest ranking electron candidate isn't returned"<<endl;
  	 checkOut = true;
       }
       if(n==3 && count > 4){
	 cout <<"Error in getOutput method, 4th highest ranking electron candidate isn't returned"<<endl;
      	 checkOut = true;
       }
     }
    }
  }
  if(checkIn){
    cout <<"Error: Discrepancy between data read in and data stored as inputs!"<<endl;
  }
  if(checkIn){
      cout <<"Error: Discrepancy between the sorted data and data outputted!"<<endl;
  }
  if(!checkIn&&!checkOut){
     cout <<"No errors found in the electron sorter"<<endl;
  }
  WriteFileData(outputs);
  delete testSort;
  return 0;   
}


// Function definition of function that reads in dummy data and load it into inputCands vector
void LoadFileData(const string &inputFile)
{
    //Opens the file
  file.open(inputFile.c_str(), ios::in);
  
  if(!file){
    cout << "Error: Cannot open input data file" << endl;
    exit(1);
  }

  unsigned long candidate;
  L1GctEmCand electron;
 
  //Read in 20 electrons for now
   for(int i=0;i<20;i++){     
     file >>std::hex>>candidate;
     electron.setRank(candidate);
     file >>std::hex>>candidate;
     electron.setEta(candidate);
     file >>std::hex>>candidate;
     electron.setPhi(candidate);
     data.push_back(electron);
    }
   
  return;
}

//Function definition, that writes the output to a file
void WriteFileData(EmCandidate outputs)
{
  EmCandidate writeThis;
  writeThis = outputs;
  ofile.open("sortOutput.txt",ios::out);
  for(unsigned int i=0;i!=writeThis.size();i++){ 
      ofile<<std::hex<<writeThis[i].getRank();
      ofile<<" ";
      ofile<<std::hex<<writeThis[i].getEta();
      ofile<<" ";
      ofile<<std::hex<<writeThis[i].getPhi();
      ofile<<"\n";
    }
    return;
}
