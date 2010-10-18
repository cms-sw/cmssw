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
  bool rejectRunLumi(int run, int lumi);
bool goodBunchCrossing(int run, int bx);
std::vector<std::string> nameList; 

///
///_____________________________________________________________________________________________
///
UEAnalysisOnRootple::UEAnalysisOnRootple()
{
  //SampleType = "CSA08";
  //  SampleType = "Summer08:Herwig";
  //SampleType = "Pythia8:10TeV";
  SampleType = "Pythia6:10TeV";
  std::cout << "UEAnalysisOnRootple constructor running on " << SampleType <<std::endl;

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
      HLTBitNames[0]  = "HLT_MinBiasPixel_SingleTrack";
      HLTBitNames[1]  = "HLT_MinBiasPixel_DoubleTrack";
      HLTBitNames[2]  = "HLT_MinBiasBSC";
      HLTBitNames[3]  = "HLT_MinBiasBSC_OR";
      HLTBitNames[4]  = "HLT_L1_BPTX1kHz";
      HLTBitNames[5]  = "HLT_HFThreshold";
      HLTBitNames[6]  = "HLT_MinBiasPixel";
      HLTBitNames[7]  = "HLT_MinBiasHcal";
      HLTBitNames[8]  = "HLT_MinBiasPixel_Trk5";
      HLTBitNames[9]  = "HLT_ZeroBias";
      HLTBitNames[10]  = "HLT_L1Jet15";
      HLTBitNames[11]  = "HLT_Jet30";

    }

  nameList.reserve(20);
}

///
///_____________________________________________________________________________________________
///
void 
UEAnalysisOnRootple::MultiAnalysis(char* filelist,char* outname,std::vector<float> weight,Float_t eta,
				   std::string type,std::string trigger,std::string tkpt,Float_t ptCut)
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
	  std::string fileName( RootTupleName );
	  std::cout <<"File: "<< fileName << std::endl;
	  //<< " (" << fileName.size() << " characters)" << std::endl;
     
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
	  std::cout << "Choose maximum pThat for dataset. ptHatMax = " << pThatMax << " GeV/c" << std::endl;
	  
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
	  std::cout << std::endl << "Opening directory " << buffer << std::endl << std::endl;
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
UEAnalysisOnRootple::Loop(Float_t we,Float_t ptThreshold,std::string type,std::string trigger,std::string tkpt)
{
  ///
  /// Sanity check.
  ///
  if (fChain == 0) 
    {
      std::cout << "fChain == 0 return." << std::endl;
      return;
    }

  Long64_t nentries = fChain->GetEntriesFast();
  std::cout << "number of entries: " << nentries << std::endl;
 
  Long64_t nbytes = 0, nb = 0;

  ///
  /// Main event loop
  ///
  for ( Long64_t jentry(0); jentry<nentries; ++jentry ) 
    {
      if ( jentry%1000 == 0 ) std::cout << "/// entry /// " << jentry << " ///" << std::endl;
 
//         if ( jentry>100 ) 
//    	{
//    	  std::cout << "[UEAnalysisOnRootple] Stop after " << jentry << " events" << std::endl;
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
      //if ( genEventScale >= pThatMax ) continue;

      if(runNumber!=123596) continue;	 
	 std::cout<<"run"<<runNumber<<std::endl;
	 a2++;
	 std::cout<<"eventi nel run"<<a2<<std::endl;
      ///Selection Run e Lumisection
      if( rejectRunLumi(runNumber,lumiBlock) ) continue;
      if ( !goodBunchCrossing(runNumber,bx) ){ std::cout<<"bunch crossing cattivo"<<std::endl; a1++; continue;}
      else {std::cout<<"bx buono"<<std::endl;}
    

      std::cout<<"selezionati bx"<<a1<<std::endl;
      ///Selezione bx 

	
      ///
      /// save frequency of trigger accepts
      ///
      int a=0;
      if ( type == "HLT" ) hltHistos->fill( *acceptedTriggers,genEventScale);


      ///
      /// Area: calculate jet areas
      ///
      if (type=="AreaGen")
	{
	  UEJetAreaFinder *areaFinder = new UEJetAreaFinder( etaRegion , ptThreshold, "kT" );
	  std::vector<UEJetWithArea> *area = new std::vector<UEJetWithArea>();

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
 	  std::vector<UEJetWithArea> *areaGen = new std::vector<UEJetWithArea>();
 	  if ( areaGenFinder->find( *MonteCarlo, *areaGen ) ) areaHistos->fill( *areaGen );
 	  delete areaGen;
 	  delete areaGenFinder;
	  
 	  ///
 	  /// Track analysis
 	  ///
	  UEJetAreaFinder *areaFinder = new UEJetAreaFinder( etaRegion , ptThreshold, "kT" );
 	  std::vector<UEJetWithArea> *area = new std::vector<UEJetWithArea>();
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
	  /// Hadron level analysis (all trigger)
	  ///
	  UEActivityFinder *activityGenFinder = new UEActivityFinder( etaRegion , ptThreshold );
	  UEActivity       *activityGen       = new UEActivity();
	  if ( activityGenFinder->find( *TracksJet, *Track, *activityGen ) ) ueHistos->fill( *activityGen );
	  delete activityGen;
	  delete activityGenFinder;
	  
	  ///
	  /// Track analysis (different trigger)
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

void UEAnalysisOnRootple::BeginJob(char* outname,std::string type)
{    
//   h_acceptedTriggers = new TH1D("h_acceptedTriggers","h_acceptedTriggers",12,-0.5,11.5);
//   h_eventScale = new TH1D("h_eventScale", "h_eventScale", 100, 0., 200.);
}

void UEAnalysisOnRootple::EndJob(std::string type)
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

void UEAnalysisOnRootple::Init(TTree *tree, std::string type)
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

   runNumber=0;
   lumiBlock=0;
   bx=0;

   a1=0;
   a2=0;
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
  
   fChain->SetBranchAddress("lumiBlock", &lumiBlock, &b_lumiBlock);
   fChain->SetBranchAddress("runNumber", &runNumber, &b_runNumber);
   fChain->SetBranchAddress("bx", &bx, &b_bx);
  
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

bool rejectRunLumi(int run, int lumi){ 
  if ( (run==123592 && (  lumi<3 || lumi>12)) ||
       //    (run==123596 && ( (lumi>2 && lumi<9) || lumi==67 || lumi==68 ) ) ||
       (run==123596 && lumi<=68 ) ||
       (run==123615 &&    lumi<72 ) ||
       (run==123732 && (  lumi<56 || lumi>=62) ) ||
       (run==123815 && (  lumi<7 || lumi>16) ) ||
      !(run==123592 || run==123596 || run==123615 || run==123732 || run==123815 ) )
              return true;
  else        return false;
 
  }

bool goodBunchCrossing(int run, int bx){
 
  if (run==123596 || run==123615) {
    if ((bx==51) || (bx==2724))  return true; 
    }
    else if (run==123732) {
      if ((bx==3487) || (bx==2596))  return true;
      }
      else return false;

  }
