//////////////////////////////////////
// Example root macro for l1 ntuples
//////////////////////////////////////

#ifndef L1Ntuple_h
#define L1Ntuple_h

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

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

#include "L1AnalysisEventDataFormat.h"
#include "L1AnalysisSimulationDataFormat.h"
#include "L1AnalysisGCTDataFormat.h"
#include "L1AnalysisGTDataFormat.h"
#include "L1AnalysisGMTDataFormat.h"
#include "L1AnalysisGTDataFormat.h"
#include "L1AnalysisRCTDataFormat.h"
#include "L1AnalysisDTTFDataFormat.h"
#include "L1AnalysisCSCTFDataFormat.h"
#include "L1AnalysisRecoMetDataFormat.h"
#include "L1AnalysisRecoJetDataFormat.h"
#include "L1AnalysisRecoClusterDataFormat.h"
#include "L1AnalysisRecoMuonDataFormat.h"
#include "L1AnalysisRecoRpcHitDataFormat.h"
#include "L1AnalysisL1ExtraDataFormat.h"
#include "L1AnalysisRecoVertexDataFormat.h"
#include "L1AnalysisRecoTrackDataFormat.h"
#include "L1AnalysisL1MenuDataFormat.h"

class L1Ntuple {
public:
  TChain          *fChain;   //!pointer to the analyzed TTree or TChain
  TChain          *ftreemuon;
  TChain          *ftreereco;
  TChain          *ftreeExtra;
  TChain          *ftreeMenu;
  TChain          *ftreeEmuExtra;
  Int_t            fCurrent; //!current Tree number in a TChain

  bool doreco;
  bool domuonreco;
  bool dol1extra;
  bool dol1emuextra;
  bool dol1menu;

  L1Analysis::L1AnalysisEventDataFormat        *event_;
  L1Analysis::L1AnalysisSimulationDataFormat   *simulation_;
  L1Analysis::L1AnalysisGCTDataFormat          *gct_;
  L1Analysis::L1AnalysisGMTDataFormat          *gmt_;
  L1Analysis::L1AnalysisGTDataFormat           *gt_;
  L1Analysis::L1AnalysisRCTDataFormat          *rct_;
  L1Analysis::L1AnalysisDTTFDataFormat         *dttf_;
  L1Analysis::L1AnalysisCSCTFDataFormat        *csctf_;
  L1Analysis::L1AnalysisRecoMetDataFormat      *recoMet_;
  L1Analysis::L1AnalysisRecoMuonDataFormat     *recoMuon_;
  L1Analysis::L1AnalysisRecoRpcHitDataFormat   *recoRpcHit_;
  L1Analysis::L1AnalysisRecoJetDataFormat      *recoJet_;
  L1Analysis::L1AnalysisRecoClusterDataFormat  *recoBasicCluster_;
  L1Analysis::L1AnalysisRecoClusterDataFormat  *recoSuperCluster_;
  L1Analysis::L1AnalysisL1ExtraDataFormat      *l1extra_;
  L1Analysis::L1AnalysisL1ExtraDataFormat      *l1emuextra_;
  L1Analysis::L1AnalysisRecoVertexDataFormat   *recoVertex_;
  L1Analysis::L1AnalysisRecoTrackDataFormat    *recoTrack_;
  L1Analysis::L1AnalysisL1MenuDataFormat       *l1menu_;

  L1Ntuple();
  L1Ntuple(const std::string & fname);

  virtual ~L1Ntuple();

  bool Open(const std::string & fname);
  bool OpenWithList(const std::string & fname);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init();
  //virtual void     Loop();
  void     Test();
  void     Test2();
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

#ifdef l1ntuple_cxx

Long64_t L1Ntuple::GetEntries()
{
  return nentries_;
}

L1Ntuple::L1Ntuple()
{
  doreco=true; domuonreco=true; dol1extra=true; dol1emuextra=true; dol1menu=true;
}

L1Ntuple::L1Ntuple(const std::string & fname)
{
  doreco=true; domuonreco=true; dol1extra=true;  dol1emuextra=true; dol1menu=true;
  Open(fname);
}

bool L1Ntuple::OpenWithList(const std::string & fname)
{
  if (!OpenNtupleList(fname)) exit(0);
  if (!CheckFirstFile())      exit(0);
  if (!OpenWithoutInit())     exit(0);

  std::cout.flush();cout<<"Going to init the available trees..."<<std::endl;std::cout.flush();
  Init();

  return true;
}

bool L1Ntuple::Open(const std::string & fname)
{
  listNtuples.push_back(fname);

  if (!CheckFirstFile())  exit(0);
  if (!OpenWithoutInit()) exit(0);

  std::cout.flush();cout<<"Going to init the available trees..."<<std::endl;std::cout.flush();
  Init();

  return true;
}

bool L1Ntuple::OpenNtupleList(const std::string & fname)
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

bool L1Ntuple::CheckFirstFile()
{
  if (listNtuples.size()==0) return false;

  rf = TFile::Open(listNtuples[0].c_str());

  if (rf==0) return false;
  if (rf->IsOpen()==0) return false;

  TTree * myChain     = (TTree*) rf->Get("l1NtupleProducer/L1Tree");
  TTree * mytreemuon  = (TTree*) rf->Get("l1MuonRecoTreeProducer/MuonRecoTree");
  TTree * mytreejets  = (TTree*) rf->Get("l1RecoTreeProducer/RecoTree");
  TTree * mytreeExtra = (TTree*) rf->Get("l1ExtraTreeProducer/L1ExtraTree");
  TTree * mytreeEmuExtra = (TTree*) rf->Get("l1EmulatorExtraTree/L1ExtraTree");
  TTree * mytreeMenu  = (TTree*) rf->Get("l1MenuTreeProducer/L1MenuTree");

  if (!myChain) {
    std::cout<<"L1Tree not found .... "<<std::endl;
    return false;
  } else {
    std::cout<<"Main tree is found .."<<std::endl;
  }

  if (!mytreejets) {
    std::cout<<"RecoTree not found, it will be skipped..."<<std::endl;
    doreco=false;
  } else
    {
      std::cout << "RecoTree is found ..."<<std::endl;
    }

  if (!mytreemuon) {
    std::cout<<"MuonRecoTree not found, it will be skipped..."<<std::endl;
    domuonreco=false;
  }
else
    {
      std::cout << "MuonRecoTree is found ..."<<std::endl;
    }

  if (!mytreeExtra) {
    std::cout<<"L1ExtraTree not found, it will be skipped..."<<std::endl;
    dol1extra=false;
  }
else
    {
      std::cout << "L1ExtraTree is found ..."<<std::endl;
    }
    if(!mytreeEmuExtra){
    std::cout<<"L1EmuExtraTree not found, it will be skipped..."<<std::endl;
    dol1emuextra=false;
    }
    else{
      std::cout << "L1EmuExtraTree is found ..."<<std::endl;
    }

  if (!mytreeMenu) {
    std::cout<<"L1MenuTree not found, it will be skipped..."<<std::endl;
    dol1menu=false;
  }
else
    {
      std::cout << "L1MenuTree is found ..."<<std::endl;
    }

  return true;
}


bool L1Ntuple::OpenWithoutInit()
{
  fChain     = new TChain("l1NtupleProducer/L1Tree");
  ftreemuon  = new TChain("l1MuonRecoTreeProducer/MuonRecoTree");
  ftreereco  = new TChain("l1RecoTreeProducer/RecoTree");
  ftreeExtra = new TChain("l1ExtraTreeProducer/L1ExtraTree");
  ftreeEmuExtra = new TChain("l1EmulatorExtraTree/L1ExtraTree");
  ftreeMenu  = new TChain("l1MenuTreeProducer/L1MenuTree");

  for (unsigned int i=0;i<listNtuples.size();i++)
  {
    std::cout << " -- Adding " << listNtuples[i] << std::endl;
    fChain->Add(listNtuples[i].c_str());

    if (doreco)     ftreereco  -> Add(listNtuples[i].c_str());
    if (domuonreco) ftreemuon  -> Add(listNtuples[i].c_str());
    if (dol1extra)  ftreeExtra -> Add(listNtuples[i].c_str());
    if (dol1emuextra) ftreeEmuExtra ->Add(listNtuples[i].c_str());
    if (dol1menu)   ftreeMenu  -> Add(listNtuples[i].c_str());

  }

  if (doreco)     fChain->AddFriend(ftreereco);
  if (domuonreco) fChain->AddFriend(ftreemuon);
  if (dol1extra)  fChain->AddFriend(ftreeExtra);
  if (dol1emuextra) fChain->AddFriend(ftreeEmuExtra);
  if (dol1menu)   fChain->AddFriend(ftreeMenu);

  return true;
}

L1Ntuple::~L1Ntuple()
{
  if (ftreemuon)  delete ftreemuon;
  if (ftreereco)  delete ftreereco;
  if (ftreeExtra) delete ftreeExtra;
  if (ftreeEmuExtra) delete ftreeEmuExtra;
  if (ftreeMenu)  delete ftreeMenu;
  if (fChain)     delete fChain;
  if (rf)         delete rf;
}



Int_t L1Ntuple::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}

Long64_t L1Ntuple::LoadTree(Long64_t entry)
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

void L1Ntuple::Init()
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
   std::cout << nentries_ << std::endl;

   event_      = new L1Analysis::L1AnalysisEventDataFormat();
   simulation_ = new L1Analysis::L1AnalysisSimulationDataFormat();

   gct_   = new L1Analysis::L1AnalysisGCTDataFormat();
   gmt_   = new L1Analysis::L1AnalysisGMTDataFormat();
   gt_    = new L1Analysis::L1AnalysisGTDataFormat();
   rct_   = new L1Analysis::L1AnalysisRCTDataFormat();
   dttf_  = new L1Analysis::L1AnalysisDTTFDataFormat();
   csctf_ = new L1Analysis::L1AnalysisCSCTFDataFormat();

   std::cout<<"Setting branch addresses for L1Tree...  "<<std::endl;

   fChain->SetBranchAddress("Event", &event_ );
   if (fChain->GetBranch("Simulation"))
     fChain->SetBranchAddress("Simulation", &simulation_ );
   else
     std::cout<<"Simulation Branch not added..."<<std::endl;

   fChain->SetBranchAddress("GCT",   &gct_   );
   fChain->SetBranchAddress("GMT",   &gmt_   );
   fChain->SetBranchAddress("GT",    &gt_    );
   fChain->SetBranchAddress("RCT",   &rct_   );
   fChain->SetBranchAddress("CSCTF", &csctf_ );
   fChain->SetBranchAddress("DTTF",  &dttf_  );



   if (doreco)
   {
    std::cout<<"Setting branch addresses for reco..."<<std::endl;

    recoMet_          = new L1Analysis::L1AnalysisRecoMetDataFormat();
    recoJet_          = new L1Analysis::L1AnalysisRecoJetDataFormat();
    recoBasicCluster_ = new L1Analysis::L1AnalysisRecoClusterDataFormat();
    recoSuperCluster_ = new L1Analysis::L1AnalysisRecoClusterDataFormat();
    recoVertex_       = new L1Analysis::L1AnalysisRecoVertexDataFormat();
    recoTrack_        = new L1Analysis::L1AnalysisRecoTrackDataFormat();

    ftreereco->SetBranchAddress("Jet", &recoJet_);
    ftreereco->SetBranchAddress("BasicClusters", &recoBasicCluster_);
    ftreereco->SetBranchAddress("SuperClusters", &recoSuperCluster_);
    ftreereco->SetBranchAddress("Met", &recoMet_);
    ftreereco->SetBranchAddress("Tracks", &recoTrack_);
    ftreereco->SetBranchAddress("Vertices", &recoVertex_);
    fChain->AddFriend(ftreereco);
    std::cout << "RecoTree "  << ftreereco->GetEntries() << endl;
  }

   if (domuonreco)
   {
     std::cout<<"Setting branch addresses for muons...   "<<std::endl;
     recoMuon_     = new L1Analysis::L1AnalysisRecoMuonDataFormat() ;
     recoRpcHit_   = new L1Analysis::L1AnalysisRecoRpcHitDataFormat();
     ftreemuon->SetBranchAddress("Muon",&recoMuon_);
     ftreemuon->SetBranchAddress("RpcHit",&recoRpcHit_);
   }

   if (dol1extra)
     {
     std::cout<<"Setting branch addresses for L1Extra... "<<std::endl;
     l1extra_ = new L1Analysis::L1AnalysisL1ExtraDataFormat();
     ftreeExtra->SetBranchAddress("L1Extra",&l1extra_);
     fChain->AddFriend(ftreeExtra);
     std::cout  << "L1ExtraTree: "<< ftreeExtra->GetEntries() << std::endl;
     }
   if ( dol1emuextra){
         std::cout<<"Setting branch addresses for L1EmuExtra... "<<std::endl;
     l1emuextra_ = new L1Analysis::L1AnalysisL1ExtraDataFormat();
     ftreeEmuExtra->SetBranchAddress("L1Extra",&l1emuextra_);
     fChain->AddFriend(ftreeEmuExtra);
     std::cout  << "L1EmuExtraTree: "<< ftreeEmuExtra->GetEntries() << std::endl;
  }

   if (dol1menu)
     {
     std::cout<<"Setting branch addresses for L1Menu... "<<std::endl;
     l1menu_ = new L1Analysis::L1AnalysisL1MenuDataFormat();
     ftreeMenu->SetBranchAddress("L1Menu",&l1menu_);
     }

}





#endif
