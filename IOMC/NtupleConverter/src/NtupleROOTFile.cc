/** \class NtupleROOTFile  ***********
* Reads a "h2root" converted cmkin ntpl;
* Joanna Weng 1/2006 
***************************************/    

#include "FWCore/Utilities/interface/Exception.h"
#include "IOMC/NtupleConverter/interface/NtupleROOTFile.h"
#include <iostream>
#include <map>
#include <iomanip>

#include <fstream>
using namespace std;

//-------------------------------------------------------
NtupleROOTFile::NtupleROOTFile(string filename,int id) {
	
        // check if File exists 
	std::ifstream * input= new ifstream(filename.c_str(), ios::in | ios::binary);
	if(! (*input))
	{
	    	throw cms::Exception("NtplNotFound", "NtupleROOTFile: Ntpl not found")
		<< "File " << filename << " could not be opened.\n"; 
	}
	if (input){
		input->close();
		delete input;
        }
	file = new TFile(filename.c_str(),"READ");

        tree = (TTree*)file->Get("h101");
        if (tree != NULL) id = 101;
        if (tree == NULL) {
           tree = (TTree*)file->Get("h100");
           if (tree != NULL) id = 100;
        }

	// Check: Is is a cmkin root file ?
	switch(id){
		
	        case 101:
		tree = (TTree*)file->Get("h101");
                if (tree==NULL){
                   throw cms::Exception("NtplNotValid", 
                       "NtupleROOTFile: Ntpl seems not to be a valid ")
                   << "File " << filename << " could not be opened.\n";
                 }
		tree->SetBranchAddress("Jsmhep",Jsmhep);
		tree->SetBranchAddress("Jsdhep",Jsdhep);
		break;
		case 100:
		tree = (TTree*)file->Get("h100");
		 if (tree==NULL){
                   throw cms::Exception("NtplNotValid",
                       "NtupleROOTFile: Ntpl seems not to be a valid ")
                   << "File; " << filename << " could not be opened.\n";
                 }
                tree->SetBranchAddress("Jmohep",Jmohep);
		tree->SetBranchAddress("Jdahep",Jdahep);
		tree->SetBranchAddress("Isthep",Isthep);
		break;
		// Not supported:
		default:
	
		  // Check: Is it a valid id ?
		throw cms::Exception("NtplNotValid", "NtupleROOTFile: Ntpl not valid")
		<< "File " << filename << " could not be opened.\n"; 
		
	}
	id_ =id;
	tree->SetBranchAddress("Nhep",&Nhep);  
	tree->SetBranchAddress("Nevhep",&Nevhep);
	tree->SetBranchAddress("Phep",Phep);
	tree->SetBranchAddress("Vhep",Vhep);
	tree->SetBranchAddress("Idhep",Idhep);
}

//-------------------------------------------------------
NtupleROOTFile::~NtupleROOTFile() {
	delete tree;
	delete file;
}

//-------------------------------------------------------
void NtupleROOTFile::setEvent(int event) const {
	tree->GetEntry(event);
}

//-------------------------------------------------------
int NtupleROOTFile::getNhep()const{
	return Nhep;
}
//-------------------------------------------------------
int NtupleROOTFile::getNevhep() const{
  
  return  tree->GetEntries();
  	
	// jw 8.3.2006:changed, since there seem to be ntpls 
	// where this entry is not correctly filled
	//return Nevhep;
}

//-------------------------------------------------------
int NtupleROOTFile::getIdhep(int j) const{
	return Idhep[j-1];
}


//-------------------------------------------------------
int NtupleROOTFile::getJsmhep(int j) const{
	if(getId() != 101){
		cout<<"NtupleROOTFile::getJsmhep: ERROR: "
		<<"only available for ID 101 ntuples"<<endl;
		return 0;
	}
	return Jsmhep[j-1];
}


//-------------------------------------------------------
int NtupleROOTFile::getJsdhep(int j) const{
	if(getId() != 101){
		cout<<"NtupleROOTFile::getJsdhep: ERROR: "
		<<"only available for ID 101 ntuples"<<endl;
		return 0;
	}
	return Jsdhep[j-1];
}

//-------------------------------------------------------
int NtupleROOTFile::getIsthep(int j) const{
	if(getId() != 100){  
		
		int jsm= this->getJsmhep(j);
		int jsd= this->getJsdhep(j);
		// this is the CMS compression for ntuple id 101
		int idj=jsm/16000000;
		int idk=jsd/16000000;
		int status = idk*100+idj;
		return status;
	}
	return Isthep[j-1];
}


//-------------------------------------------------------
int NtupleROOTFile::getJmohep(int j, int idx) const{
	if(getId() != 100){
		
		int jsm= this->getJsmhep(j);
		//we have to compute it	
		// this is the CMS compression for ntuple id 101
		int mo1 = (jsm%16000000)/4000;
		int mo2 = jsm%4000;
		// 1. mother
		if (idx==0) return mo1;
		// 2. mother
		if (idx==1) return mo2;		
	}
	return Jmohep[j-1][idx];
}

//-------------------------------------------------------
int NtupleROOTFile::getJdahep(int j, int idx) const{
	if(getId() != 100){
		int jsd= this->getJsdhep(j);
		// we have to compute it	
		// this is the CMS compression for ntuple id 101
		int da1 = (jsd%16000000)/4000;
		int da2= jsd%4000;
		// 1. daughter
		if (idx==0) return da1;
		// 2. daughter
		if (idx==1) return da2;	
	}
	return Jdahep[j-1][idx];
}

//-------------------------------------------------------
double NtupleROOTFile::getPhep(int j, int idx) const{
	return Phep[j-1][idx];
}

//-------------------------------------------------------
double NtupleROOTFile::getVhep(int j, int idx) const{
	return Vhep[j-1][idx];
}

//-------------------------------------------------------
int NtupleROOTFile::getEntries() const{
	return (int)tree->GetEntries();
}
