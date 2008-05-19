#include "UEAnalysisOnRootple.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TVector3.h>

#include <vector>
#include <math.h>

//
#include <TClonesArray.h>
#include <TObjString.h>
//

vector<string> nameList; 

UEAnalysisOnRootple::UEAnalysisOnRootple()
{
  cout << "UEAnalysisOnRootple constructor " <<endl;
  ue = new UEAnalysisUE();
  jets = new UEAnalysisJets();
  mpi = new UEAnalysisMPI();
  cout << "UEAnalysisOnRootple constructor finished initialization" <<endl;

  nameList.reserve(20);
}

void UEAnalysisOnRootple::MultiAnalysis(char* filelist,char* outname,vector<float> weight,Float_t eta,
					  string type,string trigger,string tkpt,Float_t ptCut)
{

  cout << "UEAnalysisOnRootple MultiAnalysis start " <<endl;
  BeginJob(outname,type);
  etaRegion = eta;
  ptThreshold = ptCut/1000.;
  char RootTupleName[255];
  char RootListFileName[255];
  strcpy(RootListFileName,filelist);
  ifstream inFile(RootListFileName);
  int filenumber = 0;
  while(inFile.getline(RootTupleName,255)) {
    if (RootTupleName[0] != '#') {
      cout<<"I'm analyzing file "<<RootTupleName<<endl;

      //TFile *f =  new TFile(RootTupleName);
      f = TFile::Open(RootTupleName);

      // TFileService puts UEAnalysisTree in a directory named after the module
      // which called the EDAnalyzer
      f->cd("UEAnalysisRootple");

      TTree * tree = (TTree*)gDirectory->Get("AnalysisTree");
      Init(tree);
      Loop(weight[filenumber],ptThreshold,type,trigger,tkpt);
    
      f->Close();
    
    } else {
      if (RootTupleName[1] == '#') break;     
    }
    filenumber++;
  }

  EndJob(type);

}

void UEAnalysisOnRootple::Loop(Float_t we,Float_t ptThreshold,string type,string trigger,string tkpt)
{
  
  if (fChain == 0) 
    {
      cout << "fChain == 0 return." << endl;
      return;
    }

  Long64_t nentries = fChain->GetEntriesFast();

  cout << "number of entries: " << nentries << endl;

  
  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;


    int nAcceptedTriggers( 0 );
    nAcceptedTriggers = acceptedTriggers->GetSize();
    for ( int iAcceptedTrigger(0); iAcceptedTrigger<nAcceptedTriggers; ++iAcceptedTrigger )
      {
	std::string filterName( acceptedTriggers->At(iAcceptedTrigger)->GetName() );

	if      ( filterName=="hlt1jet30"  ) h_acceptedTriggers->Fill( 0 );
	else if ( filterName=="hlt1jet60"  ) h_acceptedTriggers->Fill( 1 );
	else if ( filterName=="hlt1jet110" ) h_acceptedTriggers->Fill( 2 );
	else if ( filterName=="hlt1jet150" ) h_acceptedTriggers->Fill( 3 );
	else if ( filterName=="hlt1jet180" ) h_acceptedTriggers->Fill( 4 );
	else if ( filterName=="hlt1jet200" ) h_acceptedTriggers->Fill( 5 );
	else                                 h_acceptedTriggers->Fill( 6 );

	// print out filter name unless we have seen it before
	bool printFilterName( true );

	vector<string>::iterator itname   ( nameList.begin() );
	vector<string>::iterator itnameEnd( nameList.end()   );
	for ( ; itname!=itnameEnd; ++itname )
	  {
	    if ( (*itname).compare( filterName )==0 ) printFilterName = false;
	  }
	if ( printFilterName ) 
	  {
	    cout << "found " << filterName << " filter" << endl;
	    nameList.push_back( filterName );
	  }
      } 
    
    if(type=="Jet"){
      jets->jetCalibAnalysis(we,etaRegion,InclusiveJet,ChargedJet,TracksJet,CalorimeterJet, acceptedTriggers);
    }

    if(type=="MPI"){
      mpi->mpiAnalysisMC(we,etaRegion,ptThreshold,ChargedJet);
      mpi->mpiAnalysisRECO(we,etaRegion,ptThreshold,TracksJet);
    }
    
    if(type=="UE"){

      ue->ueAnalysisMC(we,tkpt,etaRegion,ptThreshold,MonteCarlo,ChargedJet);
      ue->ueAnalysisRECO(we,tkpt,etaRegion,ptThreshold,Track,TracksJet);
    }

  }
}


void UEAnalysisOnRootple::BeginJob(char* outname,string type)
{    
  hFile = new TFile(outname, "RECREATE" );
  
  if(type=="UE")
    ue->Begin(hFile);
  if(type=="Jet")
    jets->Begin(hFile);
  if(type=="MPI")
    mpi->Begin(hFile);

  //
  hFile->cd();
  h_acceptedTriggers = new TH1D("h_acceptedTriggers","h_acceptedTriggers",7,-0.5,6.5);
  //
}

void UEAnalysisOnRootple::EndJob(string type)
{

  if(type=="UE")
    ue->writeToFile(hFile);
  if(type=="Jet")
    jets->writeToFile(hFile);
  if(type=="MPI")
    mpi->writeToFile(hFile);

  hFile->Write();
  hFile->Close();
}

UEAnalysisOnRootple::~UEAnalysisOnRootple()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
   delete ue;
   delete jets;
   delete mpi;
}

Int_t UEAnalysisOnRootple::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t UEAnalysisOnRootple::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (!fChain->InheritsFrom(TChain::Class()))  return centry;
   TChain *chain = (TChain*)fChain;
   if (chain->GetTreeNumber() != fCurrent) {
      fCurrent = chain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void UEAnalysisOnRootple::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normaly not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

  // allocate space for file handle here
  f = new TFile;

   // Set object pointer
   MonteCarlo = 0;
   Track = 0;
   InclusiveJet = 0;
   ChargedJet = 0;
   TracksJet = 0;
   CalorimeterJet = 0;
   acceptedTriggers = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("EventKind", &EventKind, &b_EventKind);
   fChain->SetBranchAddress("MonteCarlo", &MonteCarlo, &b_MonteCarlo);
   fChain->SetBranchAddress("Track", &Track, &b_Track);
   fChain->SetBranchAddress("InclusiveJet", &InclusiveJet, &b_InclusiveJet);
   fChain->SetBranchAddress("ChargedJet", &ChargedJet, &b_ChargedJet);
   fChain->SetBranchAddress("TracksJet", &TracksJet, &b_TracksJet);
   fChain->SetBranchAddress("CalorimeterJet", &CalorimeterJet, &b_CalorimeterJet);
   fChain->SetBranchAddress("acceptedTriggers", &acceptedTriggers, &b_acceptedTriggers);
   Notify();
}

Bool_t UEAnalysisOnRootple::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normaly not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void UEAnalysisOnRootple::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}

Int_t UEAnalysisOnRootple::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
