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

  // UEAnalysisJets
  ueHLTMinBiasPixel = new UEAnalysisUE();
  ueHLTMinBiasHcal  = new UEAnalysisUE();
  ueHLTMinBiasEcal  = new UEAnalysisUE();
  ueHLTMinBias      = new UEAnalysisUE();
  ueHLTZeroBias     = new UEAnalysisUE();
  ueHLT1jet30       = new UEAnalysisUE();
  ueHLT1jet50       = new UEAnalysisUE();
  ueHLT1jet80       = new UEAnalysisUE();
  ueHLT1jet110      = new UEAnalysisUE();
  ueHLT1jet180      = new UEAnalysisUE();
  ueHLT1jet250      = new UEAnalysisUE();
  ueAll             = new UEAnalysisUE();

  // UEAnalysisJets
  jetsHLTMinBiasPixel = new UEAnalysisJets();
  jetsHLTMinBiasHcal  = new UEAnalysisJets();
  jetsHLTMinBiasEcal  = new UEAnalysisJets();
  jetsHLTMinBias      = new UEAnalysisJets();
  jetsHLTZeroBias     = new UEAnalysisJets();
  jetsHLT1jet30       = new UEAnalysisJets();
  jetsHLT1jet50       = new UEAnalysisJets();
  jetsHLT1jet80       = new UEAnalysisJets();
  jetsHLT1jet110      = new UEAnalysisJets();
  jetsHLT1jet180      = new UEAnalysisJets();
  jetsHLT1jet250      = new UEAnalysisJets();
  jetsAll             = new UEAnalysisJets();

  antiKtJets = new UEAnalysisAntiKtJets();
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

      string fileName( RootTupleName );

      //
      // set upper limit on pT of hard interaction to avoid 
      // double-counting when merging datasets
      //
      if ( fileName.compare( 38, 7, "MinBias"  )==0 ) 
	{
	  cout << "choose pthat for minbias range" << endl;
	  pThatMax = 30.;
	}
      else if ( fileName.compare( 38, 7, "JetET20"  )==0 ) 
	{
	  cout << "choose pthat for jetET20 range" << endl;
	  pThatMax = 45.;
	}
      else if ( fileName.compare( 38, 7, "JetET30"  )==0 ) 
	{
	  cout << "choose pthat for jetET30 range" << endl;
	  pThatMax = 75.;
	}
      else if ( fileName.compare( 38, 7, "JetET50"  )==0 ) 
	{
	  cout << "choose pthat for jetET50 range" << endl;
	  pThatMax = 120.;
	}
      else if ( fileName.compare( 38, 7, "JetET80"  )==0 ) 
	{
	  cout << "choose pthat for jetET80 range" << endl;
	  pThatMax = 160.;
	}
      else if ( fileName.compare( 38, 8, "JetET110" )==0 ) 
	{
	  cout << "choose pthat for jetET110 range" << endl;
	  // uncomment if JetET150 is available:
	  // pThatMax = 220.;
	}
      else if ( fileName.compare( 38, 8, "JetET150" )==0 ) 
	{
	  // highest pThat bin: no restriction
	  cout << "choose pthat for jetET150 range" << endl;
	}
      else 
	{
	  cout << "!!! ERROR !!! Cannot determine dataset range (expect MinBias, JetET20, JetET30, ...)" << endl;
	}

      //TFile *f =  new TFile(RootTupleName);
      f = TFile::Open(RootTupleName);

      // TFileService puts UEAnalysisTree in a directory named after the module
      // which called the EDAnalyzer

      if ( TMath::Abs(ptThreshold - 0.9) < 0.001 ) 
	{ 
	  cout << "changing to directory UEAnalysisRootple" << endl;
	  f->cd("UEAnalysisRootple");
	}
      else if ( TMath::Abs(ptThreshold - 0.5) < 0.001 )
	{
	  cout << "changing to directory UEAnalysisRootple500" << endl;
	  f->cd("UEAnalysisRootple500");
	}
      else
	{
	  cout << "please choose 500 or 900 as pT threshold" << endl;
	  break;
	}

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

    // for debugging:
    //for (Long64_t jentry=0; jentry<10; jentry++) {

    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;


    if ( genEventScale >= pThatMax ) continue;

    int nAcceptedTriggers( 0 );
    nAcceptedTriggers = acceptedTriggers->GetSize();
    for ( int iAcceptedTrigger(0); iAcceptedTrigger<nAcceptedTriggers; ++iAcceptedTrigger )
      {
	std::string filterName( acceptedTriggers->At(iAcceptedTrigger)->GetName() );

	if      ( filterName=="HLTMinBiasPixel" ) h_acceptedTriggers->Fill( 0 );
	else if ( filterName=="HLTMinBiasHcal"  ) h_acceptedTriggers->Fill( 1 );
	else if ( filterName=="HLTMinBiasEcal"  ) h_acceptedTriggers->Fill( 2 );
	else if ( filterName=="HLTMinBias"      ) h_acceptedTriggers->Fill( 3 );
	else if ( filterName=="HLTZeroBias"     ) h_acceptedTriggers->Fill( 4 );
	else if ( filterName=="HLT1jet30"       ) h_acceptedTriggers->Fill( 5 );
	else if ( filterName=="HLT1jet50"       ) h_acceptedTriggers->Fill( 6 );
	else if ( filterName=="HLT1jet80"       ) h_acceptedTriggers->Fill( 7 );
	else if ( filterName=="HLT1jet110"      ) h_acceptedTriggers->Fill( 8 );
	else if ( filterName=="HLT1jet180"      ) h_acceptedTriggers->Fill( 9 );
	else if ( filterName=="HLT1jet250"      ) h_acceptedTriggers->Fill( 10 );
	else                                      h_acceptedTriggers->Fill( 11 );

	// fill histos for each HLT bit separately
	if ( filterName=="HLTMinBiasPixel" )
	  {
	    if(type=="Jet") jetsHLTMinBiasPixel->jetCalibAnalysis(we,etaRegion,InclusiveJet,ChargedJet,
								  TracksJet,CalorimeterJet, acceptedTriggers, 
								  hFile, "HLTMinBiasPixel");
	    if(type=="UE") 
	      {
		ueHLTMinBiasPixel->ueAnalysisMC(we,tkpt,etaRegion,ptThreshold,MonteCarlo,ChargedJet, hFile, "HLTMinBiasPixel");
		ueHLTMinBiasPixel->ueAnalysisRECO(we,tkpt,etaRegion,ptThreshold,Track,TracksJet, hFile, "HLTMinBiasPixel");
	      }
	  }
	else if ( filterName=="HLTMinBiasHcal"  )
	  {
	    if(type=="Jet") jetsHLTMinBiasHcal->jetCalibAnalysis(we,etaRegion,InclusiveJet,ChargedJet,
								 TracksJet,CalorimeterJet, acceptedTriggers, 
								 hFile, "HLTMinBiasHcal");
	    if(type=="UE") 
	      {
		ueHLTMinBiasHcal->ueAnalysisMC(we,tkpt,etaRegion,ptThreshold,MonteCarlo,ChargedJet, hFile, "HLTMinBiasHcal");
		ueHLTMinBiasHcal->ueAnalysisRECO(we,tkpt,etaRegion,ptThreshold,Track,TracksJet, hFile, "HLTMinBiasHcal");
	      }
	  }
	else if ( filterName=="HLTMinBiasEcal"  )
	  {
	    if(type=="Jet") jetsHLTMinBiasEcal->jetCalibAnalysis(we,etaRegion,InclusiveJet,ChargedJet,
								 TracksJet,CalorimeterJet, acceptedTriggers, 
								 hFile, "HLTMinBiasEcal");
	    if(type=="UE") 
	      {
		ueHLTMinBiasEcal->ueAnalysisMC(we,tkpt,etaRegion,ptThreshold,MonteCarlo,ChargedJet, hFile, "HLTMinBiasEcal");
		ueHLTMinBiasEcal->ueAnalysisRECO(we,tkpt,etaRegion,ptThreshold,Track,TracksJet, hFile, "HLTMinBiasEcal");
	      }
	  }
	else if ( filterName=="HLTMinBias"      )
	  {
	    if(type=="Jet") jetsHLTMinBias->jetCalibAnalysis(we,etaRegion,InclusiveJet,ChargedJet,
							     TracksJet,CalorimeterJet, acceptedTriggers, 
							     hFile, "HLTMinBias");
	    if(type=="UE") 
	      {
		ueHLTMinBias->ueAnalysisMC(we,tkpt,etaRegion,ptThreshold,MonteCarlo,ChargedJet, hFile, "HLTMinBias");
		ueHLTMinBias->ueAnalysisRECO(we,tkpt,etaRegion,ptThreshold,Track,TracksJet, hFile, "HLTMinBias");
	      }
	  }
	else if ( filterName=="HLTZeroBias"     )
	  {
	    if(type=="Jet") jetsHLTZeroBias->jetCalibAnalysis(we,etaRegion,InclusiveJet,ChargedJet,
							      TracksJet,CalorimeterJet, acceptedTriggers, 
							      hFile, "HLTZeroBias");
	    if(type=="UE") 
	      {
		ueHLTZeroBias->ueAnalysisMC(we,tkpt,etaRegion,ptThreshold,MonteCarlo,ChargedJet, hFile, "HLTZeroBias");
		ueHLTZeroBias->ueAnalysisRECO(we,tkpt,etaRegion,ptThreshold,Track,TracksJet, hFile, "HLTZeroBias");
	      }
	  }
	else if ( filterName=="HLT1jet30"       )
	  {
	    if(type=="Jet") jetsHLT1jet30->jetCalibAnalysis(we,etaRegion,InclusiveJet,ChargedJet,
							    TracksJet,CalorimeterJet, acceptedTriggers, 
							    hFile, "HLT1jet30");
	    if(type=="UE") 
	      {
		ueHLT1jet30->ueAnalysisMC(we,tkpt,etaRegion,ptThreshold,MonteCarlo,ChargedJet, hFile, "HLT1jet30");
		ueHLT1jet30->ueAnalysisRECO(we,tkpt,etaRegion,ptThreshold,Track,TracksJet, hFile, "HLT1jet30");
	      }
	  }
	else if ( filterName=="HLT1jet50"       )
	  {
	    if(type=="Jet") jetsHLT1jet50->jetCalibAnalysis(we,etaRegion,InclusiveJet,ChargedJet,
							    TracksJet,CalorimeterJet, acceptedTriggers, 
							    hFile, "HLT1jet50");
	    if(type=="UE") 
	      {
		ueHLT1jet50->ueAnalysisMC(we,tkpt,etaRegion,ptThreshold,MonteCarlo,ChargedJet, hFile, "HLT1jet50");
		ueHLT1jet50->ueAnalysisRECO(we,tkpt,etaRegion,ptThreshold,Track,TracksJet, hFile, "HLT1jet50");
	      }
	  }
	else if ( filterName=="HLT1jet80"       )
	  {
	    if(type=="Jet") jetsHLT1jet80->jetCalibAnalysis(we,etaRegion,InclusiveJet,ChargedJet,
							    TracksJet,CalorimeterJet, acceptedTriggers, 
							    hFile, "HLT1jet80");
	    if(type=="UE") 
	      {
		ueHLT1jet80->ueAnalysisMC(we,tkpt,etaRegion,ptThreshold,MonteCarlo,ChargedJet, hFile, "HLT1jet80");
		ueHLT1jet80->ueAnalysisRECO(we,tkpt,etaRegion,ptThreshold,Track,TracksJet, hFile, "HLT1jet80");
	      }
	  }
	else if ( filterName=="HLT1jet110"      )
	  {
	    if(type=="Jet") jetsHLT1jet110->jetCalibAnalysis(we,etaRegion,InclusiveJet,ChargedJet,
							     TracksJet,CalorimeterJet, acceptedTriggers, 
							     hFile, "HLT1jet110");
	    if(type=="UE") 
	      {
		ueHLT1jet110->ueAnalysisMC(we,tkpt,etaRegion,ptThreshold,MonteCarlo,ChargedJet, hFile, "HLT1jet110");
		ueHLT1jet110->ueAnalysisRECO(we,tkpt,etaRegion,ptThreshold,Track,TracksJet, hFile, "HLT1jet110");
	      }
	  }
	else if ( filterName=="HLT1jet180"      )
	  {
	    if(type=="Jet") jetsHLT1jet180->jetCalibAnalysis(we,etaRegion,InclusiveJet,ChargedJet,
							     TracksJet,CalorimeterJet, acceptedTriggers, 
							     hFile, "HLT1jet180");
	    if(type=="UE") 
	      {
		ueHLT1jet180->ueAnalysisMC(we,tkpt,etaRegion,ptThreshold,MonteCarlo,ChargedJet, hFile, "HLT1jet180");
		ueHLT1jet180->ueAnalysisRECO(we,tkpt,etaRegion,ptThreshold,Track,TracksJet, hFile, "HLT1jet180");
	      }
	  }
	else if ( filterName=="HLT1jet250"      )
	  {
	    if(type=="Jet") jetsHLT1jet250->jetCalibAnalysis(we,etaRegion,InclusiveJet,ChargedJet,
							     TracksJet,CalorimeterJet, acceptedTriggers, 
							     hFile, "HLT1jet250");
	    if(type=="UE") 
	      {
		ueHLT1jet250->ueAnalysisMC(we,tkpt,etaRegion,ptThreshold,MonteCarlo,ChargedJet, hFile, "HLT1jet250");
		ueHLT1jet250->ueAnalysisRECO(we,tkpt,etaRegion,ptThreshold,Track,TracksJet, hFile, "HLT1jet250");
	      }
	  }
      }
    h_eventScale->Fill( genEventScale );


    if(type=="Jet") {
      jetsAll->jetCalibAnalysis(we,etaRegion,InclusiveJet,ChargedJet,TracksJet,CalorimeterJet, acceptedTriggers, hFile, "All");
    }

    if(type=="AntiKtJet"){
      antiKtJets->jetCalibAnalysis(we,etaRegion,InclusiveJet,ChargedJet,TracksJet,CalorimeterJet, MonteCarlo, Track, acceptedTriggers);
    }

    if(type=="MPI"){
      mpi->mpiAnalysisMC(we,etaRegion,ptThreshold,ChargedJet);
      mpi->mpiAnalysisRECO(we,etaRegion,ptThreshold,TracksJet);
    }
    
    if(type=="UE"){
      ueAll->ueAnalysisMC(we,tkpt,etaRegion,ptThreshold,MonteCarlo,ChargedJet, hFile, "All");
      ueAll->ueAnalysisRECO(we,tkpt,etaRegion,ptThreshold,Track,TracksJet, hFile, "All");
    }

  }
}


void UEAnalysisOnRootple::BeginJob(char* outname,string type)
{    
  hFile = new TFile(outname, "RECREATE" );
  
  hFile->mkdir("HLTMinBiasPixel");
  hFile->mkdir("HLTMinBiasHcal");
  hFile->mkdir("HLTMinBiasEcal");
  hFile->mkdir("HLTMinBias");
  hFile->mkdir("HLTZeroBias");
  hFile->mkdir("HLT1jet30");
  hFile->mkdir("HLT1jet50");
  hFile->mkdir("HLT1jet80");
  hFile->mkdir("HLT1jet110");
  hFile->mkdir("HLT1jet180");
  hFile->mkdir("HLT1jet250");
  hFile->mkdir("All");

  if(type=="UE")
    {
      ueHLTMinBiasPixel->Begin(hFile, "HLTMinBiasPixel");
      ueHLTMinBiasHcal->Begin(hFile, "HLTMinBiasHcal");
      ueHLTMinBiasEcal->Begin(hFile, "HLTMinBiasEcal");
      ueHLTMinBias->Begin(hFile, "HLTMinBias");
      ueHLTZeroBias->Begin(hFile, "HLTZeroBias");
      ueHLT1jet30->Begin(hFile, "HLT1jet30");
      ueHLT1jet50->Begin(hFile, "HLT1jet50");
      ueHLT1jet80->Begin(hFile, "HLT1jet80");
      ueHLT1jet110->Begin(hFile, "HLT1jet110");
      ueHLT1jet180->Begin(hFile, "HLT1jet180");
      ueHLT1jet250->Begin(hFile, "HLT1jet250");
      ueAll->Begin(hFile, "All");
    }
  if(type=="Jet")
    {
      jetsHLTMinBiasPixel->Begin(hFile, "HLTMinBiasPixel");
      jetsHLTMinBiasHcal->Begin(hFile, "HLTMinBiasHcal");
      jetsHLTMinBiasEcal->Begin(hFile, "HLTMinBiasEcal");
      jetsHLTMinBias->Begin(hFile, "HLTMinBias");
      jetsHLTZeroBias->Begin(hFile, "HLTZeroBias");
      jetsHLT1jet30->Begin(hFile, "HLT1jet30");
      jetsHLT1jet50->Begin(hFile, "HLT1jet50");
      jetsHLT1jet80->Begin(hFile, "HLT1jet80");
      jetsHLT1jet110->Begin(hFile, "HLT1jet110");
      jetsHLT1jet180->Begin(hFile, "HLT1jet180");
      jetsHLT1jet250->Begin(hFile, "HLT1jet250");
      jetsAll->Begin(hFile, "All");
    }
  if(type=="AntiKtJet")
    antiKtJets->Begin(hFile);
  if(type=="MPI")
    mpi->Begin(hFile);

  //
  hFile->cd();
  h_acceptedTriggers = new TH1D("h_acceptedTriggers","h_acceptedTriggers",12,-0.5,11.5);
  h_eventScale = new TH1D("h_eventScale", "h_eventScale", 100, 0., 200.);
  //
}

void UEAnalysisOnRootple::EndJob(string type)
{

  if(type=="AntiKtJet")
    antiKtJets->writeToFile(hFile);
  if(type=="MPI")
    mpi->writeToFile(hFile);

  hFile->Write();
  hFile->Close();
}

UEAnalysisOnRootple::~UEAnalysisOnRootple()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();

   delete ueHLTMinBiasPixel;
   delete ueHLTMinBiasHcal;
   delete ueHLTMinBiasEcal;
   delete ueHLTMinBias;
   delete ueHLTZeroBias;
   delete ueHLT1jet30;
   delete ueHLT1jet50;
   delete ueHLT1jet80;
   delete ueHLT1jet110;
   delete ueHLT1jet180;
   delete ueHLT1jet250;
   delete ueAll;

   delete jetsHLTMinBiasPixel;
   delete jetsHLTMinBiasHcal;
   delete jetsHLTMinBiasEcal;
   delete jetsHLTMinBias;
   delete jetsHLTZeroBias;
   delete jetsHLT1jet30;
   delete jetsHLT1jet50;
   delete jetsHLT1jet80;
   delete jetsHLT1jet110;
   delete jetsHLT1jet180;
   delete jetsHLT1jet250;
   delete jetsAll;

   delete antiKtJets;
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
   genEventScale = 0;

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
   fChain->SetBranchAddress("genEventScale", &genEventScale, &b_genEventScale );

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
