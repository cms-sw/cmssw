//////////////////////////////////////
// Example root macro for l1 ntuples
//////////////////////////////////////

#ifndef L1GtNtuple_h
#define L1GtNtuple_h

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TFriendElement.h>
#include <TList.h>
#include <TMatrix.h>
#include <TH1D.h>
#include <TH1F.h>
#include <TH2D.h>
#include <TH2F.h>
#include <TCanvas.h>
#include <TFileSet.h>

#include "L1AnalysisEventDataFormat.h"
#include "L1AnalysisGTDataFormat.h"


class L1GtNtuple {
public:
  
TChain          *fChain;   //!pointer to the analyzed TTree or TChain
  
  
  L1Analysis::L1AnalysisEventDataFormat        *event_;
  L1Analysis::L1AnalysisGTDataFormat	       *gt_;
  Int_t fCurrent; 
  std::string 					directory_;

  L1GtNtuple();
  L1GtNtuple(const std::string & fname);

  virtual ~L1GtNtuple();
 
  bool Open(const std::string & fname);
  bool OpenWithList(const std::string & fname);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init();
  //virtual void     Loop();
  void     Test();
  Long64_t GetEntries();

 private :
  bool CheckFirstFile();
  bool OpenWithoutInit();
  bool OpenNtupleList(const std::string & fname);

  std::vector<std::string> listNtuples;
  Long64_t nentries_;
  TFile* rf;
};

#endif

#ifdef L1GtNtuple_cxx

Long64_t L1GtNtuple::GetEntries()
{
  return nentries_;
}

L1GtNtuple::L1GtNtuple()
{
  //doreco=true; domuonreco=true; dol1extra=true;
}

L1GtNtuple::L1GtNtuple(const std::string & fname)
{
  //doreco=true; domuonreco=true; dol1extra=true;
  Open(fname);
}

bool L1GtNtuple::OpenWithList(const std::string & fname)
{
  if (!OpenNtupleList(fname)) exit(0);
  if (!CheckFirstFile())      exit(0);  
  if (!OpenWithoutInit())     exit(0);
    
  std::cout<<"Going to init the available trees..."<<std::endl;
  Init();

  return true;
}

bool L1GtNtuple::Open(const std::string & fname)
{
  
  directory_ = fname;

  listNtuples.push_back(fname);
 
  //if (!CheckFirstFile())  exit(0);
  if (!OpenWithoutInit()) exit(0);

  std::cout<<"Going to init the available trees..."<<std::endl;
  Init();

  return true;
}
 
bool L1GtNtuple::OpenNtupleList(const std::string & fname)
{
  std::ifstream flist(fname.c_str());
  if (!flist)
    {
      std::cout << "File "<<fname<<" is not found !"<<std::endl;
      return false;
    }

  while(!flist.eof())
    {
      std::string str;
      getline(flist,str);
      if (!flist.fail())
	{
           if (str!="") listNtuples.push_back(str);
	}
    }

  return true;
}

bool L1GtNtuple::CheckFirstFile()
{
  if (listNtuples.size()==0) return false;

  rf = TFile::Open(listNtuples[0].c_str());

  if (rf==0) return false;
  if (rf->IsOpen()==0) return false;

  TTree * myChain     = (TTree*) rf->Get("l1NtupleProducer/L1Tree");


  if (!myChain) {
    std::cout<<"L1Tree not found .... "<<std::endl;
    return false;
  } else {
    std::cout<<"Main tree is found .."<<std::endl;
  }
    

  return true;
}


bool L1GtNtuple::OpenWithoutInit()
{
	fChain = new TChain("l1NtupleProducer/L1Tree");
 	
	std::cout << "Dir: " << directory_ << std::endl;
	TFileSet set2(directory_);
	TList* slist2 = set2.GetList();
	TIter next2(slist2);
	for(TObject* obj2=next2(); obj2; obj2=next2())
	{			
		std::cout << obj2->GetName() << std::endl;

		std::ostringstream oss;
		oss<<directory_<<"/"<<obj2->GetName();

		const std::string fname(oss.str());
		std::cout<<fname<<std::endl;

		if (fname.find(".root")!=std::string::npos)
		{
			std::cout << "Add to chain: " << fname << std::endl;
			fChain->Add(fname.c_str());
		}
	}

	return true;
}

L1GtNtuple::~L1GtNtuple()
{
	delete fChain;
}



Int_t L1GtNtuple::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}

Long64_t L1GtNtuple::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);

   if (centry < 0) return centry;
   if (!fChain->InheritsFrom(TChain::Class()))  return centry;
   TChain *chain = (TChain*)fChain;
   if (chain->GetTreeNumber() != fCurrent) {
      fCurrent = chain->GetTreeNumber();
   }
   return centry;
}

void L1GtNtuple::Init()
{
   if (!fChain) return;
   fCurrent = -1;
   /*
   fChain->SetMakeClass(1);
   ftreemuon->SetMakeClass(1);
   ftreereco->SetMakeClass(1);
   ftreeExtra->SetMakeClass(1); */  
   
   std::cout << "Estimate the number of entries ..."<<std::endl;
   nentries_=fChain->GetEntries();

   std::cout << nentries_ <<std::endl;
   
   event_ = new L1Analysis::L1AnalysisEventDataFormat();
   gt_    = new L1Analysis::L1AnalysisGTDataFormat();

   std::cout<<"Setting branch addresses for L1Tree...  "<<std::flush;
   
   fChain->SetBranchAddress("Event", &event_ );
   fChain->SetBranchAddress("GT",    &gt_    );
   
}





#endif
