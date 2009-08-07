#include "UEAnalysisOnRootple.h"
#include <vector>
#include <math.h>

///
/// ROOT includes
///
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TVector3.h>
#include <TObjString.h>
#include <TClonesArray.h>

vector<string> nameList; 

///
///_____________________________________________________________________________________________
///
UEAnalysisOnRootple::UEAnalysisOnRootple()
{
  //SampleType = "CSA08";
  //  SampleType = "Summer08:Herwig";
  //SampleType = "Pythia8:10TeV";
  SampleType = "Pythia6:10TeV";
  cout << "UEAnalysisOnRootple constructor running on " << SampleType <<endl;

  if ( SampleType == "CSA08" )
    {
      ///
      /// CSA08 HLT names
      ///
      HLTBitNames[0]  = "HLTMinBiasPixel";
      HLTBitNames[1]  = "HLTMinBiasHcal";
      HLTBitNames[2]  = "HLTMinBiasEcal";
      HLTBitNames[3]  = "HLTMinBias";
      HLTBitNames[4]  = "HLTZeroBias";
      HLTBitNames[5]  = "HLT1jet30";
      HLTBitNames[6]  = "HLT1jet50";
      HLTBitNames[7]  = "HLT1jet80";
      HLTBitNames[8]  = "HLT1jet110";
      HLTBitNames[9]  = "HLT1jet180";
      HLTBitNames[10] = "HLT1jet250";
    }
  else
    {
      ///
      /// Summer09 HLT names
      ///
      HLTBitNames[0]  = "HLT_MinBiasPixel";
      HLTBitNames[1]  = "HLT_MinBiasHcal";
      HLTBitNames[2]  = "HLT_MinBiasEcal";
      HLTBitNames[3]  = "HLT_MinBiasPixel_Trk5";
      HLTBitNames[4]  = "HLT_ZeroBias";
      HLTBitNames[5]  = "HLT_Jet30";
      HLTBitNames[6]  = "HLT_Jet50";
      HLTBitNames[7]  = "HLT_Jet80";
      HLTBitNames[8]  = "HLT_Jet110";
      HLTBitNames[9]  = "HLT_Jet180";
      HLTBitNames[10] = "HLT_Jet250";

    }

  nameList.reserve(20);
}

///
///_____________________________________________________________________________________________
///
void 
UEAnalysisOnRootple::MultiAnalysis(char* filelist,char* outname,vector<float> weight,Float_t eta,
				   string type,string trigger,string tkpt,Float_t ptCut)
{
  BeginJob(outname,type);

  /// declare jet algo
  //  jetsWithAreas = new UEAnalysisJetAreas( eta , ptCut , "kT" );
 
   TFile *MpiOutFile = TFile::Open( outname,"recreate" ); //Declare TFile for MPI and Gamma Analysis

   if (type == "Gam" )
     {
       gam = new UEAnalysisGAM();
       gam->Begin(MpiOutFile);
     }

   if ( type == "MPI" || type == "MPIGen" )
     {
      /// declare histogram filler
       mpi = new UEAnalysisMPI();
       mpi->Begin(MpiOutFile);
     }


  if ( type == "Area" || type == "AreaGen" )
    {
      /// declare histogram filler
      areaHistos = new UEJetAreaHistograms( outname, HLTBitNames );
    }

  if ( type == "UE" || type == "UEGen" )
    {
      /// declare histogram filler
      ueHistos = new UEActivityHistograms( outname, HLTBitNames );
    }

  if ( type == "HLT" )
    {
      // declare histogram filler for HLT analysis
      hltHistos = new UETriggerHistograms( outname, HLTBitNames );
    }

  etaRegion   = eta;
  ptThreshold = ptCut/1000.;

  char RootTupleName   [255];
  char RootListFileName[255];

  strcpy ( RootListFileName, filelist );
  ifstream inFile( RootListFileName );
  int filenumber = 0;

  ///
  /// loop on list of file names
  ///
  while ( inFile.getline(RootTupleName, 255) ) 
    {
      if ( RootTupleName[0] != '#') 
	{
	  string fileName( RootTupleName );
	  cout <<"File: "<< fileName << endl;
	  //<< " (" << fileName.size() << " characters)" << endl;
     
	  // no binning of datasets
	  pThatMax = 14000.;
	  
	  ///
	  /// Set upper limit on pT of hard interaction to avoid 
	  /// double-counting when merging datasets
	  ///
	  if ( SampleType == "Summer08:Herwig" || SampleType == "Summer08:Pythia" )
	    {
	      unsigned int startCharacter( 0 );

	      if      ( SampleType == "Summer08:Herwig" ) startCharacter = 74;
	      else if ( SampleType == "Summer08:Pythia" ) startCharacter = 66;

	      if      ( fileName.size() < 45 )                                    pThatMax = 14000.;
	      else if ( fileName.compare( startCharacter, 7, "QCDPt15"   ) == 0 ) pThatMax =    30.;
	      else if ( fileName.compare( startCharacter, 8, "QCDPt170"  ) == 0 ) pThatMax =   300.;
	      else if ( fileName.compare( startCharacter, 9, "QCDPt3000" ) == 0 ) pThatMax = 14000.;
	      else if ( fileName.compare( startCharacter, 8, "QCDPt300"  ) == 0 ) pThatMax =   470.;
	      else if ( fileName.compare( startCharacter, 7, "QCDPt30"   ) == 0 ) pThatMax =    80.;
	      else if ( fileName.compare( startCharacter, 8, "QCDPt470"  ) == 0 ) pThatMax =   800.;
	      else if ( fileName.compare( startCharacter, 8, "QCDPt800"  ) == 0 ) pThatMax =  1400.;
	      else if ( fileName.compare( startCharacter, 7, "QCDPt80"   ) == 0 ) pThatMax =   170.;
	      else if ( fileName.compare( startCharacter, 9, "QCDPt1400" ) == 0 ) pThatMax =  2000.;
	      else                                                                pThatMax = 14000.;
	      
	    }
	  else if ( SampleType == "CSA08" )
	    {
	      if      ( fileName.size() < 45 )                       pThatMax = 14000.;
	      else if ( fileName.compare( 61, 7, "MinBias"  ) == 0 ) pThatMax =    30.;
	      else if ( fileName.compare( 61, 7, "JetET20"  ) == 0 ) pThatMax =    45.;
	      else if ( fileName.compare( 61, 7, "JetET30"  ) == 0 ) pThatMax =    75.;
	      else if ( fileName.compare( 61, 7, "JetET50"  ) == 0 ) pThatMax =   120.;
	      else if ( fileName.compare( 61, 7, "JetET80"  ) == 0 ) pThatMax =   160.;
	      else if ( fileName.compare( 61, 8, "JetET110" ) == 0 ) pThatMax =   220.;
	      else                                                   pThatMax = 14000.;
	    }
	  cout << "Choose maximum pThat for dataset. ptHatMax = " << pThatMax << " GeV/c" << endl;
	  
	  f = TFile::Open(RootTupleName);
	  
	  // TFileService puts UEAnalysisTree in a directory named after the module
	  // which called the EDAnalyzer
	  
	  // different directory names for gen-level only analysis
	  //
	  //    KEY: TDirectoryFile   UEAnalysisRootpleOnlyMC;1       UEAnalysisRootpleOnlyMC (AnalysisRootpleProducerOnlyMC) folder
	  //    KEY: TDirectoryFile   UEAnalysisRootpleOnlyMC500;1    UEAnalysisRootpleOnlyMC500 (AnalysisRootpleProducerOnlyMC) folder

	  char   buffer[200];
	  if ( type == "UEGen" || type == "AreaGen" || type == "MPIGen" || type == "Gam") 
	    {
	      sprintf ( buffer, "UEAnalysisRootpleOnlyMC%i", int(ptCut) );
	    }
	  else if ( int(ptCut)!=900 ) sprintf ( buffer, "UEAnalysisRootple%i", int(ptCut) );
	  else                        sprintf ( buffer, "UEAnalysisRootple"               );
	  cout << endl << "Opening directory " << buffer << endl << endl;
	  f->cd( buffer );

	  TTree * tree = (TTree*)gDirectory->Get("AnalysisTree");
	  Init(tree, type);
	  
	  Loop(weight[filenumber],ptThreshold,type,trigger,tkpt);
	  
	  f->Close();
	} 
      else 
	{
	  if (RootTupleName[1] == '#') break;     
	}
      filenumber++;
    }
  ///
  /// end loop on file names
  ///
  
  EndJob ( type );
  
  if (type == "Gam")
    {
      gam->writeToFile(MpiOutFile);
      delete gam;
    }
  if ( type == "MPI"  || type == "MPIGen"  ) 
    {
      mpi->writeToFile(MpiOutFile);
      delete mpi;
    }
  if ( type == "Area" || type == "AreaGen" ) delete areaHistos;
  if ( type == "UE"   || type == "UEGen"   ) delete ueHistos;
  if ( type == "HLT"                       ) delete hltHistos;
}

///
///_____________________________________________________________________________________________
///
void 
UEAnalysisOnRootple::Loop(Float_t we,Float_t ptThreshold,string type,string trigger,string tkpt)
{
  ///
  /// Sanity check.
  ///
  if (fChain == 0) 
    {
      cout << "fChain == 0 return." << endl;
      return;
    }

  Long64_t nentries = fChain->GetEntriesFast();
  cout << "number of entries: " << nentries << endl;

  Long64_t nbytes = 0, nb = 0;

  ///
  /// Main event loop
  ///
  for ( Long64_t jentry(0); jentry<nentries; ++jentry ) 
    {
      if ( jentry%1000 == 0 ) cout << "/// entry /// " << jentry << " ///" << endl;
 
//         if ( jentry>100 ) 
//    	{
//    	  cout << "[UEAnalysisOnRootple] Stop after " << jentry << " events" << endl;
//    	  break;
//    	}

      ///
      /// Load branches.
      ///
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      
      ///
      /// Veto events with too large pThat to avoid overlap.
      ///
      if ( genEventScale >= pThatMax ) continue;


      ///
      /// save frequency of trigger accepts
      ///
      if ( type == "HLT" ) hltHistos->fill( *acceptedTriggers );


      ///
      /// Area: calculate jet areas
      ///
      if (type=="AreaGen")
	{
	  UEJetAreaFinder *areaFinder = new UEJetAreaFinder( etaRegion , ptThreshold, "kT" );
	  vector<UEJetWithArea> *area = new vector<UEJetWithArea>();

	  if ( areaFinder->find( *MonteCarlo, *area ) ) areaHistos->fill( *area );
	  delete area;
	  delete areaFinder;
	}
      
      if (type=="Area")
	{
 	  ///
 	  /// Hadron level analysis
 	  ///
 	  UEJetAreaFinder *areaGenFinder = new UEJetAreaFinder( etaRegion , ptThreshold, "kT" );
 	  vector<UEJetWithArea> *areaGen = new vector<UEJetWithArea>();
 	  if ( areaGenFinder->find( *MonteCarlo, *areaGen ) ) areaHistos->fill( *areaGen );
 	  delete areaGen;
 	  delete areaGenFinder;
	  
 	  ///
 	  /// Track analysis
 	  ///
	  UEJetAreaFinder *areaFinder = new UEJetAreaFinder( etaRegion , ptThreshold, "kT" );
 	  vector<UEJetWithArea> *area = new vector<UEJetWithArea>();
 	  if ( areaFinder->find( *Track, *area ) ) areaHistos->fill( *area, *acceptedTriggers );
 	  delete area;
 	  delete areaFinder;
	}    
      
      //Gamma
      if (type=="Gam")
	{
	  gam->gammaAnalysisMC(we, etaRegion, ptThreshold, *MCGamma, *ChargedJet);
	}

      ///MPI and MPIGen
      if (type=="MPIGen")
      	{ 
	  mpi->mpiAnalysisMC(we, etaRegion, ptThreshold, *ChargedJet);
	}

      // if (type=="MPI")
      //	{
	  //Not yet tested
      //	  mpi->mpiAnalysisRECO(we, etaRegion, ptThreshold,*TracksJet);
	  //	}

      ///
      /// UE: identify UE densities
      ///
      if (type=="UEGen")
	{
	  UEActivityFinder *activityFinder = new UEActivityFinder( etaRegion , ptThreshold );
	  UEActivity       *activity       = new UEActivity();

	  if ( activityFinder->find( *ChargedJet, *MonteCarlo, *activity ) ) ueHistos->fill( *activity );
	  delete activity;
	  delete activityFinder;
	}
      
      if (type=="UE")
	{
	  ///
	  /// Hadron level analysis
	  ///
	  UEActivityFinder *activityGenFinder = new UEActivityFinder( etaRegion , ptThreshold );
	  UEActivity       *activityGen       = new UEActivity();
	  if ( activityGenFinder->find( *ChargedJet, *MonteCarlo, *activityGen ) ) ueHistos->fill( *activityGen );
	  delete activityGen;
	  delete activityGenFinder;
	  
	  ///
	  /// Track analysis
	  ///
	  UEActivityFinder *activityFinder = new UEActivityFinder( etaRegion , ptThreshold );
	  UEActivity       *activity       = new UEActivity();
	  if ( activityFinder->find( *TracksJet, *Track, *activity ) ) ueHistos->fill( *activity, *acceptedTriggers );
	  delete activity;
	  delete activityFinder;
	}    
      
      //h_eventScale->Fill( genEventScale );
      } 
  /// end reco-level analysis
  
}

void UEAnalysisOnRootple::BeginJob(char* outname,string type)
{    
//   h_acceptedTriggers = new TH1D("h_acceptedTriggers","h_acceptedTriggers",12,-0.5,11.5);
//   h_eventScale = new TH1D("h_eventScale", "h_eventScale", 100, 0., 200.);
}

void UEAnalysisOnRootple::EndJob(string type)
{
  //  hFile->Close();
}

UEAnalysisOnRootple::~UEAnalysisOnRootple()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
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

void UEAnalysisOnRootple::Init(TTree *tree, string type)
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
   MCGamma = 0;
   TracksJet = 0;
   CalorimeterJet = 0;
   acceptedTriggers = 0;
   genEventScale = 0;

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   // gen-level only:
   //
   // *Br    0 :EventKind : EventKind/I                                            *
   // *Br    1 :MonteCarlo :                                                       *
   // *Br    2 :InclusiveJet :                                                     *
   // *Br    3 :ChargedJet :                                                       *

   fChain->SetBranchAddress("EventKind", &EventKind, &b_EventKind);
   fChain->SetBranchAddress("MonteCarlo", &MonteCarlo, &b_MonteCarlo);
   fChain->SetBranchAddress("ChargedJet", &ChargedJet, &b_ChargedJet);
   fChain->SetBranchAddress("MCGamma", &MCGamma, &b_MCGamma);

   if ( type != "UEGen" || type != "MPIGen" || type != "Gam" )
     {
       fChain->SetBranchAddress("InclusiveJet", &InclusiveJet, &b_InclusiveJet);
       fChain->SetBranchAddress("Track", &Track, &b_Track);
       fChain->SetBranchAddress("TracksJet", &TracksJet, &b_TracksJet);
       fChain->SetBranchAddress("CalorimeterJet", &CalorimeterJet, &b_CalorimeterJet);
       fChain->SetBranchAddress("acceptedTriggers", &acceptedTriggers, &b_acceptedTriggers);
       fChain->SetBranchAddress("genEventScale", &genEventScale, &b_genEventScale );
     }
   Notify();
}

Bool_t UEAnalysisOnRootple::Notify()
{
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
