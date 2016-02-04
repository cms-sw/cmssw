/** \class NtupleROOTFile  ***********
* Reads a "h2root" converted cmkin ntpl;
* Joanna Weng 1/2006 
***************************************/    
#ifndef NTUPLEROOTFILE_
#define NTUPLEROOTFILE_


#include "TFile.h"
#include "TTree.h"
#include <string>
#include <map>

class NtupleROOTFile {
	public:
	NtupleROOTFile(std::string filename, int id);
	virtual ~NtupleROOTFile();
	
	virtual void   setEvent(int event) const;
	virtual int getNhep()const;
	virtual int getNevhep() const;
	virtual int getIdhep(int j) const;
	virtual int getJsmhep(int j) const;
	virtual int getJsdhep(int j) const;
	virtual int getIsthep(int j) const;
	virtual int getJmohep(int j, int idx) const;
	virtual int getJdahep(int j, int idx) const;
	virtual double getPhep(int j, int idx) const;
	virtual double getVhep(int j, int idx) const;
	virtual int    getEntries() const;
	virtual int getId() const {return id_;} 
	
	protected:
  	int id_;
	private:
	TFile *file;
	TTree* tree;
	
	//  Int_t           Nevhep;
	Int_t           Nhep;
	Int_t           Nevhep;
	Int_t           Idhep[4000];   //[Nhep]
	Int_t           Jsmhep[4000];   //[Nhep]
	Int_t           Jsdhep[4000];   //[Nhep]
	Float_t         Phep[4000][5];   //[Nhep]
	Float_t         Vhep[4000][4];   //[Nhep]
	Int_t           Isthep[4000];
	Int_t           Jmohep[4000][2];
	Int_t           Jdahep[4000][2];
	
	NtupleROOTFile() {}
};

#endif
