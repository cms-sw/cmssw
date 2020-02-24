//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Thu Jul 18 05:42:44 2019 by ROOT version 6.10/09
// from TTree eventTree/event tree for analysis
// found on file: Output_Ntuple_data.root
//////////////////////////////////////////////////////////

#ifndef PhotonClass_h
#define PhotonClass_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
// Header file for the classes stored in the TTree if any.
#include <iostream>
#include <fstream>
#include "vector"
#include <vector>
#include <TH2.h>
#include <TH1.h>
#include <TSystemFile.h>
#include <TSystemDirectory.h>
#include <TChain.h>

using namespace std;


class PhotonClass {
 public :
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain                                                               
  Int_t           fCurrent; //!current Tree number in a TChain                                                                       

  TFile *fileName;
  TTree *tree;

  TH1F *Photon_HoverE;
  TH1F *Photon_phoPFChIso;
  TH1F *Photon_phoPFPhoIso;
  TH1F *Photon_phoPFNeuIso;
  TH1F *Photon_phoEt;
  TH1F *Photon_phoEta;
  TH1F *Photon_phohasPixelSeed;
  TH1F *Photon_phoR9;  
  TH1I *Photon_nPho;


  // Fixed size dimensions of array or collections stored in the TTree if any.                                                              

  // Declaration of leaf types                                                                                                       
  Int_t           nPho;
  vector<float>   *phoE;
  vector<float>   *phoPx;
  vector<float>   *phoPy;
  vector<float>   *phoPz;
  vector<float>   *phoEt;
  vector<float>   *phoEta;
  vector<float>   *phoPhi;
  vector<float>   *phoSCE;
  vector<float>   *phoSCRawE;
  vector<float>   *phoSCEta;
  vector<float>   *phoSCPhi;
  vector<float>   *phoSCEtaWidth;
  vector<float>   *phoSCPhiWidth;
  vector<int>     *phohasPixelSeed;
  vector<int>     *phoEleVeto;
  vector<float>   *phoR9;
  vector<float>   *phoR9Full5x5;
  vector<float>   *phoHoverE;
  vector<float>   *phoPFChIso;
  vector<float>   *phoPFChWorstIso;
  vector<float>   *phoPFPhoIso;
  vector<float>   *phoPFNeuIso;

  // List of branches                                                                                                                
  TBranch        *b_nPho;   //!                                                                                                      
  TBranch        *b_phoE;   //!                                                                                                      
  TBranch        *b_phoPx;   //!                                                                                                     
  TBranch        *b_phoPy;   //!                                                                                                     
  TBranch        *b_phoPz;   //!                                                                                                     
  TBranch        *b_phoEt;   //!                                                                                                     
  TBranch        *b_phoEta;   //!                                                                                                    
  TBranch        *b_phoPhi;   //!                                                                                                    
  TBranch        *b_phoSCE;   //!                                                                                                    
  TBranch        *b_phoSCRawE;   //!                                                                                                 
  TBranch        *b_phoSCEta;   //!                                                                                                  
  TBranch        *b_phoSCPhi;   //!                                                                                                  
  TBranch        *b_phoSCEtaWidth;   //!                                                                                             
  TBranch        *b_phoSCPhiWidth;   //!                                                                                             
  TBranch        *b_phohasPixelSeed;   //!                                                                                           
  TBranch        *b_phoEleVeto;   //!                                                                                                
  TBranch        *b_phoR9;   //!                                                                                                     
  TBranch        *b_phoR9Full5x5;   //!                                                                                              
  TBranch        *b_phoHoverE;   //!                                                                                                 
  TBranch        *b_phoPFChIso;   //!                                                                                                
  TBranch        *b_phoPFChWorstIso;   //!                                                                                           
  TBranch        *b_phoPFPhoIso;   //!           
  TBranch        *b_phoPFNeuIso;   //!                                                                                               

  PhotonClass(const char* file1, const char* file2);
  virtual ~PhotonClass();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TChain *tree);
  virtual void     Loop(Long64_t maxEvents, int reportEvery);
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);
  virtual void Histograms(const char* file2);
};

#endif

#ifdef PhotonClass_cxx
PhotonClass::PhotonClass(const char* file1, const char* file2) 
{

  TChain *chain = new TChain("demo/eventTree");
  TString path = file1;


  TSystemDirectory sourceDir("hi",path);
  TList* fileList = sourceDir.GetListOfFiles();
  TIter next(fileList);
  TSystemFile* filename;
  int fileNumber = 0;
  int maxFiles = -1;
  std::cout<<"path:"<<path<<std::endl;


  // chain->Add("file:/afs/hep.wisc.edu/home/bsahu/Photon_analyzer_2/CMSSW_9_4_9/src/Photon_Analyzer/PhotonAnalyzer/test/Data_Ntuple/Output_Ntuple_data.root");


  chain->Add("file:/afs/hep.wisc.edu/home/bsahu/Photon_analyzer_2/CMSSW_9_4_9/src/Photon_Analyzer/PhotonAnalyzer/test/Signal_Ntuple/Output_Ntuple_Signal_70000_2.root");



  //chain->Add("file:/afs/hep.wisc.edu/home/bsahu/Photon_analyzer_2/CMSSW_9_4_9/src/Photon_Analyzer/PhotonAnalyzer/test/MC_ntuple/Output_Ntuple_mc.root");





//  while ((filename = (TSystemFile*)next()) && fileNumber >  maxFiles)
//    {
//      std::cout<<"filenumber :"<<fileNumber<<std::endl;
//      if(fileNumber > 1)
//        {
//
//	  std::cout<<"filenumber :"<<fileNumber<<std::endl;
//	  //          TString dataset = "";
//          TString  FullPathInputFile = (path+filename->GetName());
//          TString name = filename->GetName();
//          //if(name.Contains(dataset))
//	  //{
//              chain->Add(FullPathInputFile);
//	      //}
//        }
//      fileNumber++;
//    }
//

  Init(chain);
  Histograms(file2);
}


  

PhotonClass::~PhotonClass()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
   fileName->cd();
   fileName->Write();
   fileName->Close();


}

Int_t PhotonClass::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t PhotonClass::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void PhotonClass::Init(TChain *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   phoE = 0;
   phoPx = 0;
   phoPy = 0;
   phoPz = 0;
   phoEt = 0;
   phoEta = 0;
   phoPhi = 0;
   phoSCE = 0;
   phoSCRawE = 0;
   phoSCEta = 0;
   phoSCPhi = 0;
   phoSCEtaWidth = 0;
   phoSCPhiWidth = 0;
   phohasPixelSeed = 0;
   phoEleVeto = 0;
   phoR9 = 0;
   phoR9Full5x5 = 0;
   phoHoverE = 0;
   phoPFChIso = 0;
   phoPFChWorstIso = 0;
   phoPFPhoIso = 0;
   phoPFNeuIso = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("nPho", &nPho, &b_nPho);
   fChain->SetBranchAddress("phoE", &phoE, &b_phoE);
   fChain->SetBranchAddress("phoPx", &phoPx, &b_phoPx);
   fChain->SetBranchAddress("phoPy", &phoPy, &b_phoPy);
   fChain->SetBranchAddress("phoPz", &phoPz, &b_phoPz);
   fChain->SetBranchAddress("phoEt", &phoEt, &b_phoEt);
   fChain->SetBranchAddress("phoEta", &phoEta, &b_phoEta);
   fChain->SetBranchAddress("phoPhi", &phoPhi, &b_phoPhi);
   fChain->SetBranchAddress("phoSCE", &phoSCE, &b_phoSCE);
   fChain->SetBranchAddress("phoSCRawE", &phoSCRawE, &b_phoSCRawE);
   fChain->SetBranchAddress("phoSCEta", &phoSCEta, &b_phoSCEta);
   fChain->SetBranchAddress("phoSCPhi", &phoSCPhi, &b_phoSCPhi);
   fChain->SetBranchAddress("phoSCEtaWidth", &phoSCEtaWidth, &b_phoSCEtaWidth);
   fChain->SetBranchAddress("phoSCPhiWidth", &phoSCPhiWidth, &b_phoSCPhiWidth);
   fChain->SetBranchAddress("phohasPixelSeed", &phohasPixelSeed, &b_phohasPixelSeed);
   fChain->SetBranchAddress("phoEleVeto", &phoEleVeto, &b_phoEleVeto);
   fChain->SetBranchAddress("phoR9", &phoR9, &b_phoR9);
   fChain->SetBranchAddress("phoR9Full5x5", &phoR9Full5x5, &b_phoR9Full5x5);
   fChain->SetBranchAddress("phoHoverE", &phoHoverE, &b_phoHoverE);
   fChain->SetBranchAddress("phoPFChIso", &phoPFChIso, &b_phoPFChIso);
   fChain->SetBranchAddress("phoPFChWorstIso", &phoPFChWorstIso, &b_phoPFChWorstIso);
   fChain->SetBranchAddress("phoPFPhoIso", &phoPFPhoIso, &b_phoPFPhoIso);
   fChain->SetBranchAddress("phoPFNeuIso", &phoPFNeuIso, &b_phoPFNeuIso);
   Notify();
}

Bool_t PhotonClass::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void PhotonClass::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t PhotonClass::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef PhotonClass_cxx
