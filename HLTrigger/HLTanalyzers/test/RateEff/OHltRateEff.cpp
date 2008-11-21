/////////////////////////////////////////////////////////////////////////////////////////////////
//
//        Program to calculate rates of trigger paths using variables of OHltTree class,
//
//				Note: OHltTree class needs to be updated if any new variables become available 
//				in OpenHLT (HLTAnalyzer).
//				
//        Author:  Vladimir Rekovic,     Date: 2007/12/10
//					
//
//        Contacts: Jonathan Hollar (LLNL), Chi Nhan Nguyen (TAMU)
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <TMath.h>

#include "OHltTree.h"
#include "OHltMenu.h"

#include "TH1.h"
#include "TH2.h"
#include "TChain.h"
#include "TCut.h"

#include <map>

using namespace std;

// Declaration of different Menus
void BookMenu_Default(OHltMenu *menu, double &iLumi, double &nBunches);
void BookMenu_21XDefault(OHltMenu *menu, double &iLumi, double &nBunches);
void BookMenu_21X_29E30_Default(OHltMenu *menu, double &iLumi, double &nBunches);
void BookMenu_OhltExample(OHltMenu *menu, double &iLumi, double &nBunches);
void BookMenu_L1Default(OHltMenu*  menu, double &iLumi, double &nBunches);

void BookEffHistos(OHltMenu* menu, vector<string> ObjectsToUse,int &MaxMult, int ip 
		   ,std::vector <TH1F*> &Num_pt, std::vector <TH1F*> &Num_eta, std::vector <TH1F*> &Num_phi
		   ,std::vector <TH1F*> &Den_pt, std::vector <TH1F*> &Den_eta, std::vector <TH1F*> &Den_phi
		   ,std::vector <TH1F*> &Eff_pt, std::vector <TH1F*> &Eff_eta, std::vector <TH1F*> &Eff_phi
		   ,std::vector <TH1F*> &DenwrtL1_pt, std::vector <TH1F*> &DenwrtL1_eta, std::vector <TH1F*> &DenwrtL1_phi
		   ,std::vector <TH1F*> &EffwrtL1_pt, std::vector <TH1F*> &EffwrtL1_eta, std::vector <TH1F*> &EffwrtL1_phi
		   );

void FillWriteEffHistos(OHltMenu* menu, vector<string> ObjectsToUse, int ip 
			,std::vector <TH1F*> &Num_pt, std::vector <TH1F*> &Num_eta, std::vector <TH1F*> &Num_phi
			,std::vector <TH1F*> &Den_pt, std::vector <TH1F*> &Den_eta, std::vector <TH1F*> &Den_phi
			,std::vector <TH1F*> &Eff_pt, std::vector <TH1F*> &Eff_eta, std::vector <TH1F*> &Eff_phi
			,std::vector <TH1F*> &DenwrtL1_pt, std::vector <TH1F*> &DenwrtL1_eta, std::vector <TH1F*> &DenwrtL1_phi
			,std::vector <TH1F*> &EffwrtL1_pt, std::vector <TH1F*> &EffwrtL1_eta, std::vector <TH1F*> &EffwrtL1_phi
			);
// Auxiliary functions
Double_t eff(Int_t a, Int_t b){ 
  if (b==0.){return -1.;}
  Double_t af = Double_t(a);
  Double_t bf = Double_t(b);   
  Double_t effi = af/bf;
  return effi;
}
Double_t seff(Int_t a, Int_t b){
  if (b==0.){return -1.;}
  Double_t af = Double_t(a);
  Double_t bf = Double_t(b);   
  Double_t r = af/bf;
  Double_t unc = sqrt(af + (r*r*bf) )/bf;
  return unc;
}
Double_t eff(Double_t a, Double_t b){ 
  if (b==0.){return -1.;}
  Double_t af = Double_t(a);
  Double_t bf = Double_t(b);   
  Double_t effi = af/bf;
  return effi;
}
Double_t seff(Double_t a, Double_t b){
  if (b==0.){return -1.;}
  Double_t af = Double_t(a);
  Double_t bf = Double_t(b);   
  Double_t r = af/bf;
  Double_t unc = sqrt(af + (r*r*bf) )/bf;
  return unc;
}

void ShowUsage() {
  cout << "  Usage:  ./OHltRateEff <nevents> <menu> <conditions> <version tag> <cms energy> <doPrintAll> <RateOnly> <EfficiencyOnly>" << endl;
  cout << "default:  ./OHltRateEff -1 21X startup 20June2008 14 0 1 0" << endl;
}


/* ********************************************** */
// Main
/* ********************************************** */
int main(int argc, char *argv[]){

  //if(argc<3) {ShowUsage();return 0;}

  int NEntries = -1;// -1 means all available
  if (argc>1) {
    if (TString(argv[1])=="-h") {ShowUsage();exit(0);}
    NEntries = atoi(argv[1]);
  }
  //  TString sMenu = "l1default"; // lookup available menus
  //  TString sMenu = "default"; // lookup available menus
  TString sMenu = "21X"; // lookup available menus
  //  TString sMenu = "example"; // lookup available menus
  if (argc>2) {
    sMenu = TString(argv[2]);
  }
  TString sConditions = "startup"; // Available: Ideal, Startup (1pb-1)
  if (argc>3) {
    sConditions = TString(argv[3]);
  }
  TString sVersion = "2008-June-10-v01";
  if (argc>4) {
    sVersion =TString(argv[4]);
  }
  TString sEnergy = "14";
  if(argc>5) {
    sEnergy=TString(argv[5]);
    if ((sEnergy.CompareTo("10") != 0) && (sEnergy.CompareTo("14") != 0))
      {
	cout << "sqrt(s) = " << sEnergy << " TeV is not supported. Options are 10 and 14" << endl;
	exit(0);
      }
  }
  int PrintAll = 0;
  if (argc>6) {
    PrintAll = atoi(argv[6]);
  }
  
  int RateOnly=1;
  if (argc>7) {
    RateOnly=atoi(argv[7]);
  }

  int EfficiencyOnly=0;
  if (argc>8) {
    EfficiencyOnly=atoi(argv[8]);
  }
  if ((EfficiencyOnly ==1 ) && (RateOnly == 1))
      {
	cout << "Cannot run Only Efficiency and Only Rate at the same time!! To get both rate and efficiency information please set both EfficiencyOnly and RateOnly to zero." << endl;
	exit(0);
      }




  ////////////////////////////////////////////////////////////

  /**** Different Beam conditions: ****/ 
  // Fixed LHC defaults
  const double bunchCrossingTime = 75.0E-09; // Design: 25 ns Startup: 75 ns
  const double maxFilledBunches = 3564;

  // Defaults, to be changed in the menu booking
  double ILumi = 1.E27;
  double nFilledBunches = 1;

  /**********************************/


  // Choice of menus
  OHltMenu* menu = new OHltMenu();
  if(sMenu.CompareTo("default") == 0)
    BookMenu_Default(menu,ILumi,nFilledBunches);
  else if(sMenu.CompareTo("21X") == 0)
    BookMenu_21XDefault(menu,ILumi,nFilledBunches);
  else if(sMenu.CompareTo("example") == 0)
    BookMenu_OhltExample(menu,ILumi,nFilledBunches);
  else if(sMenu.CompareTo("l1default") == 0)
    BookMenu_L1Default(menu,ILumi,nFilledBunches);
  else if(sMenu.CompareTo("21X_29E30") == 0)
    BookMenu_21X_29E30_Default(menu,ILumi,nFilledBunches);
  else {
    cout << "No valid menu specified.  Either creat a new menu or use existing one. Exiting!" << endl;
    ShowUsage();
    return 0;
  }
  // This has to be done after menu booking
  double collisionRate = (nFilledBunches / maxFilledBunches) / bunchCrossingTime ;  // Hz
  
  // Setup menu parameters
  vector<TString> trignames = menu->GetHlts(); 
  int Ntrig = (int) trignames.size();
  map<TString,int> map_TrigPrescls = menu->GetTotalPrescaleMap(); 
  map<TString,int> map_L1Prescls = menu->GetL1PrescaleMap(); 
  map<TString,int> map_HLTPrescls = menu->GetHltPrescaleMap(); 
  map<TString,TString> map_HltDesc = menu->GetHltDescriptionMap(); 
  map<TString,double> map_EventSize = menu->GetEventsizeMap();
  map<TString,TString> map_L1Bits = menu->GetHltL1BitMap();
  map<TString,int> map_MultEle = menu->GetMultEleMap(); 
  map<TString,int> map_MultPho = menu->GetMultPhoMap();
  map<TString,int> map_MultMu = menu->GetMultMuMap(); 
  map<TString,int> map_MultJets = menu->GetMultJetsMap(); 
  map<TString,int> map_MultMET = menu->GetMultMETMap(); 

  ////////////////////////////////////////////////////////////
  cout << endl << endl << endl;
  cout << "--------------------------------------------------------------------------" << endl;
  cout << "NEntries = " << NEntries << endl;
  cout << "Menu = " << sMenu << endl;
  if(sConditions.CompareTo("") ==0) sConditions=TString("Ideal");
  cout << "Conditions = " << sConditions << endl;
  cout << "Version = " << sVersion << endl;
  cout << "Inst Luminosity = " << ILumi << ",  Bunches = " << nFilledBunches <<  endl;
  cout << "sqrt(s) = " << sEnergy << " TeV" << endl;
  cout<< "Rate only set to "<< RateOnly <<endl;
  cout<< "Efficiency only set to "<< EfficiencyOnly <<endl;
  cout << "--------------------------------------------------------------------------" << endl;
  cout << endl << endl << endl;

  // Wait 2 sec for user to read announcement
  sleep(2);


  /* **************************************** */
  // Setup files, samples, sigmas and skim efficiencies
  // Each entry in vector correspond to one sample
  // Order is important!
  /* **************************************** */
  
  // Vector for Cross sections [pb] and skim efficiencies
  ////////////////////////////////////////
  vector<Double_t> xsec;
  vector<Double_t> skmeff; // Skim efficiencies
  
  ////////////////////////////////////////
  // In order to handle samples with more than one file
  vector<TChain*> TabChain;
  vector<TString> ProcFil;
  // In order to handle double counting if running minbias, ppEleX and/or ppMuX
  vector<bool> doMuonCut; vector<bool> doElecCut;

  //////////////////////////////////////
  // For Efficiencies
  //For Reco
  std::vector <TH1F*> Num_pt;
  std::vector <TH1F*> Num_eta;
  std::vector <TH1F*> Num_phi;
  std::vector <TH1F*> Den_pt;
  std::vector <TH1F*> Den_eta;
  std::vector <TH1F*> Den_phi;

  std::vector <TH1F*> Eff_pt;
  std::vector <TH1F*> Eff_eta;
  std::vector <TH1F*> Eff_phi;

  std::vector <TH1F*> DenwrtL1_pt;
  std::vector <TH1F*> DenwrtL1_eta;
  std::vector <TH1F*> DenwrtL1_phi;

  std::vector <TH1F*> EffwrtL1_pt;
  std::vector <TH1F*> EffwrtL1_eta;
  std::vector <TH1F*> EffwrtL1_phi;




  /* *************************************************************** */
  // Start filling sample vectors
  if( sConditions.CompareTo("startup") == 0) {
    
    if (RateOnly==0){
      //Signals
 
      TString SamplesDIR = "/castor/cern.ch/user/j/jjhollar/OpenHLT212/signals/";

      //zee
      ProcFil.clear();
      ProcFil.push_back(SamplesDIR+"/zee.relval.10tev.startupv5.root");

      TabChain.push_back(new TChain("HltTree"));
      for (unsigned int ipfile = 0; ipfile < ProcFil.size(); ipfile++){
	TabChain.back()->Add(ProcFil[ipfile]);
      }
      doMuonCut.push_back(false); doElecCut.push_back(false);

      if(sEnergy.CompareTo("10") == 0)
	xsec.push_back(12.32E2); // pp 10 TeV PYTHIA cross-section times filter zmm
      else if(sEnergy.CompareTo("14") == 0)
	xsec.push_back(17.87E2); // pp 14 TeV PYTHIA cross-section times filter zmm
    
      skmeff.push_back(1.);  //

      //zmm
      ProcFil.clear();
      ProcFil.push_back(SamplesDIR+"/zmm.relval.10tev.startupv5.root");

      TabChain.push_back(new TChain("HltTree"));
      for (unsigned int ipfile = 0; ipfile < ProcFil.size(); ipfile++){
	TabChain.back()->Add(ProcFil[ipfile]);
      }
      doMuonCut.push_back(false); doElecCut.push_back(false);

      if(sEnergy.CompareTo("10") == 0)
	xsec.push_back(12.32E2); // pp 10 TeV PYTHIA cross-section times filter zmm
      else if(sEnergy.CompareTo("14") == 0)
	xsec.push_back(17.97E2); // pp 14 TeV PYTHIA cross-section times filter zmm
    
      skmeff.push_back(1.);  //

      //wenu
      ProcFil.push_back(SamplesDIR+"/wenu.relval.10tev.startupv5.root");

      TabChain.push_back(new TChain("HltTree"));
      for (unsigned int ipfile = 0; ipfile < ProcFil.size(); ipfile++){
	TabChain.back()->Add(ProcFil[ipfile]);
      }
      doMuonCut.push_back(false); doElecCut.push_back(false);

      if(sEnergy.CompareTo("10") == 0)
	xsec.push_back(11.865E3); // pp 10 TeV PYTHIA cross-section times filter zmm
      else if(sEnergy.CompareTo("14") == 0)
	xsec.push_back(17.12E3); // pp 14 TeV PYTHIA cross-section times filter zmm
    
      skmeff.push_back(1.);  //

      //wmunu
      ProcFil.clear();
      ProcFil.push_back(SamplesDIR+"/wmunu.relval.10tev.startupv5.root");

      TabChain.push_back(new TChain("HltTree"));
      for (unsigned int ipfile = 0; ipfile < ProcFil.size(); ipfile++){
	TabChain.back()->Add(ProcFil[ipfile]);
      }
      doMuonCut.push_back(false); doElecCut.push_back(false);

      if(sEnergy.CompareTo("10") == 0)
	xsec.push_back(11.865E3); // pp 10 TeV PYTHIA cross-section times filter zmm
      else if(sEnergy.CompareTo("14") == 0)
	xsec.push_back(17.17E3); // pp 14 TeV PYTHIA cross-section times filter zmm
    
      skmeff.push_back(1.);  //
    }

    if (EfficiencyOnly==0){
      cout<< "Reading rate samples "<<endl;
      // ppEleX
      //    TString PPEX_DIR="rfio:/castor/cern.ch/user/j/jjhollar/OpenHLT184/ppex/";
      TString PPEX_DIR="rfio:/castor/cern.ch/user/j/jjhollar/OpenHLT212/ppex/";
      ProcFil.clear();
      //    ProcFil.push_back(PPEX_DIR+"ppex_misAlCa_1.root");
      //    ProcFil.push_back(PPEX_DIR+"ppex_misAlCa_2.root");
      //    ProcFil.push_back(PPEX_DIR+"ppex_misAlCa_3.root");
      //    ProcFil.push_back(PPEX_DIR+"ppex_misAlCa_4.root");
      ProcFil.push_back(PPEX_DIR+"ppex_1.root");
      //      ProcFil.push_back(PPEX_DIR+"ppex*");

      TabChain.push_back(new TChain("HltTree"));
      for (unsigned int ipfile = 0; ipfile < ProcFil.size(); ipfile++){
	TabChain.back()->Add(ProcFil[ipfile]);
      }
      doMuonCut.push_back(true); doElecCut.push_back(false);
    
      if(sEnergy.CompareTo("10") == 0)
	xsec.push_back(4.706E7); // pp 10 TeV PYTHIA cross-section times filter pp->eleX (5.16E10*0.000912) - no diffraction
      else if(sEnergy.CompareTo("14") == 0)
	xsec.push_back(6.427E7); // pp 14 TeV PYTHIA cross-section times filter pp->eleX (5.47E10*0.001175) - no diffraction

      skmeff.push_back(1.);  //

      // ppMuX
      //    TString PPMUX_DIR="rfio:/castor/cern.ch/user/j/jjhollar/OpenHLT184/ppmux/";
      TString PPMUX_DIR="rfio:/castor/cern.ch/user/j/jjhollar/OpenHLT212/ppmux/";    
      ProcFil.clear();
      ProcFil.push_back(PPMUX_DIR+"ppmux_1.root");
      //    ProcFil.push_back(PPMUX_DIR+"ppmux_misAlCa_1.root");
      //    ProcFil.push_back(PPMUX_DIR+"ppmux_misAlCa_2.root");
      //    ProcFil.push_back(PPMUX_DIR+"ppmux_misAlCa_3.root");
      //    ProcFil.push_back(PPMUX_DIR+"ppmux_misAlCa_4.root"); 
      //    ProcFil.push_back(PPMUX_DIR+"ppmux_misAlCa_5.root"); 
      //    ProcFil.push_back(PPMUX_DIR+"ppmux_misAlCa_6.root"); 
      //      ProcFil.push_back(PPMUX_DIR+"ppmux*");
    
      TabChain.push_back(new TChain("HltTree"));
      for (unsigned int ipfile = 0; ipfile < ProcFil.size(); ipfile++){
	TabChain.back()->Add(ProcFil[ipfile]);
      }
      doMuonCut.push_back(false); doElecCut.push_back(true);
    
      if(sEnergy.CompareTo("10") == 0)
	xsec.push_back(3.658E7); // pp 10 TeV PYTHIA cross-section times filter pp->muX (5.16E10*0.000709) - no diffraction
      else if(sEnergy.CompareTo("14") == 0) 
	xsec.push_back(4.671E7); // pp 14 TeV PYTHIA cross-section times filter pp->muX (5.47E10*0.000854) - no diffraction


      skmeff.push_back(1.);  //

      // Minbias
      //    TString MB_DIR="rfio:/castor/cern.ch/user/a/apana/OpenHLT184/MinBias/";
      TString MB_DIR="rfio:/castor/cern.ch/user/j/jjhollar/OpenHLT212/minbias/";
      ProcFil.clear();
      ProcFil.push_back(MB_DIR+"minbias_1.root");
      //    ProcFil.push_back(MB_DIR+"minbias_misAlCa_0.root");
      //    ProcFil.push_back(MB_DIR+"minbias_misAlCa_1.root"); 
      //    ProcFil.push_back(MB_DIR+"minbias_misAlCa_2.root"); 
      //    ProcFil.push_back(MB_DIR+"minbias_misAlCa_3.root"); 
      //    ProcFil.push_back(MB_DIR+"minbias_misAlCa_4.root"); 
      //      ProcFil.push_back(MB_DIR+"minbias*");

    
      TabChain.push_back(new TChain("HltTree"));
      for (unsigned int ipfile = 0; ipfile < ProcFil.size(); ipfile++){
	TabChain.back()->Add(ProcFil[ipfile]);
      }
      doMuonCut.push_back(true); doElecCut.push_back(true);
    
      if(sEnergy.CompareTo("10") == 0)
	xsec.push_back(7.53E10); // pp 10 TeV PYTHIA xsec - includes diffraction
      else if(sEnergy.CompareTo("14") == 0)  
	xsec.push_back(7.923E10); // pp 14 TeV PYTHIA xsec - includes diffraction

      skmeff.push_back(1.);  //

    }


  }
  else if(sConditions.CompareTo("ideal") == 0) {
    
    ////////////////////////////////////////////////////////////////////////////////
    // No samples yet.
    //////////

  }
  else {    
    cout << "No valid Conditions specified.  Either create input files for new Conditions or use existing ones. Exiting!" << endl;
    ShowUsage();
    return 0;
  }
  // End filling sample vectors
  /* *************************************************************** */


  ////////////////////////////////////////////////////////////
  // Do some conversion to appripriate units
  
  // multiply xsec by skim Eff
  cout << "Number of files (datasets) to process " << TabChain.size() << endl;
  for (unsigned int ip = 0; ip < skmeff.size(); ip++) {
    xsec[ip] *= skmeff[ip];
  }
  // Convert cross-sections to cm^2
  for (unsigned int i = 0; i < skmeff.size(); i++){xsec[i] *= 1.E-36;}

  ////////////////////////////////////////////////////////////

  // Vectors for event counting, rates + initialisation
  vector<int> * iCount = new vector<int>();
  vector<int> * sPureCount = new vector<int>();
  vector<int> * pureCount = new vector<int>();
  vector<int> otmp;
  vector< vector<int> > * overlapCount = new vector< vector<int> >();
  for (int it = 0; it < Ntrig; it++){
    iCount->push_back(0);
    sPureCount->push_back(0);
    pureCount->push_back(0);
    otmp.push_back(0);
  }
  for (int it = 0; it < Ntrig; it++){
    overlapCount->push_back(otmp);
  }

  vector<Double_t> Rat,sRat,seqpRat,sseqpRat,pRat,spRat,cRat,cThroughput;
  vector<Double_t> Odenp;
  vector< vector<Double_t> > Onum;
  for (int it = 0; it < Ntrig; it++){
    Rat.push_back(0.);
    sRat.push_back(0.);
    seqpRat.push_back(0.);
    sseqpRat.push_back(0.);
    pRat.push_back(0.);
    spRat.push_back(0.);
    cRat.push_back(0.);
    cThroughput.push_back(0.);
    Odenp.push_back(0.);
  }
  for (int it = 0; it < Ntrig; it++){
    Onum.push_back(Odenp);
  }
  
  /* *************************************************************** */
  // Start calculating rates  
  for (unsigned int ip = 0; ip < TabChain.size(); ip++){
    cout<<"Available sample "<<ip<<", file " << TabChain[ip] <<endl;
    cout<<" xsec = "  << scientific << xsec[ip]/skmeff[ip]/1.E-36 << fixed << ",  skmeff = "<< skmeff[ip] <<", doMuonCut = " << doMuonCut[ip] << ", doElecCut = " << doElecCut[ip] << endl;
  }
  vector<OHltTree*> hltt;
  for (unsigned int ip = 0; ip < TabChain.size(); ip++){
    for (int it = 0; it < Ntrig; it++){
      iCount->at(it) = 0;
      sPureCount->at(it) = 0;
      pureCount->at(it) = 0;
    }
    // For binwise analysis
    vector<Double_t> Rat_bin,sRat_bin,seqpRat_bin,sseqpRat_bin,pRat_bin,spRat_bin,cRat_bin,cThroughput_bin;
    for (int it = 0; it < Ntrig; it++){
      Rat_bin.push_back(0.);
      sRat_bin.push_back(0.);
      seqpRat_bin.push_back(0.);
      sseqpRat_bin.push_back(0.);
      pRat_bin.push_back(0.);
      spRat_bin.push_back(0.);
      cRat_bin.push_back(0.);
      cThroughput_bin.push_back(0.);
	}

    hltt.push_back(new OHltTree((TTree*)TabChain[ip],Ntrig));

    int deno = NEntries; 
    int chainEntries = (int)hltt[ip]->fChain->GetEntries(); 
    if (NEntries <= 0 || NEntries > chainEntries) {
      deno = chainEntries;
    }
    cout<<"---------------------------------------------------------------" << endl;
    cout<<"Processing bin "<<ip<<" ( "<< deno <<" events ) "<<", file " << TabChain[ip] <<" (has "<<hltt[ip]->fChain->GetEntries()<<" events ) "<<endl;
    cout<<scientific;
    cout.precision(5);
    cout<<" xsec = "  << xsec[ip]/skmeff[ip]/1.E-36 << fixed << ",  skmeff = "<< skmeff[ip] <<", doMuonCut = " << doMuonCut[ip] << ", doElecCut = " << doElecCut[ip] << endl;
    cout<<"---------------------------------------------------------------" << endl;

    //Booking histos for eff calculation

    //    BookEffHistos(trignames,map_MultEle,map_MultPho,map_MultMu,map_MultJets,map_MultMET);
    vector<string> ObjectsToUse;
    ObjectsToUse.push_back( "Electron");
    ObjectsToUse.push_back( "Photon");
    ObjectsToUse.push_back( "Muon");
    ObjectsToUse.push_back( "Jet");
    ObjectsToUse.push_back( "Met");
    int NObjectsToUse = (int)ObjectsToUse.size();


    //Root file for efficiency histos

    int MaxMult=0;
    double muonPt=-999.;
    double muonDr=-999;
    if(RateOnly==0){
      char filename[256];
      snprintf(filename,255,"MyEffHist_%d.root",ip);
      TFile*   theFile = new TFile(filename, "RECREATE");
      theFile->cd();
      cout<< "Efficiency root file created: "<<filename <<endl;

      //    cout << "Before booking histos" <<endl;
      BookEffHistos( menu,ObjectsToUse,MaxMult,ip
		     ,Num_pt,Num_eta,Num_phi
		     ,Den_pt,Den_eta,Den_phi
		     ,Eff_pt,Eff_eta,Eff_phi
		     ,DenwrtL1_pt,DenwrtL1_eta,DenwrtL1_phi
		     ,EffwrtL1_pt,EffwrtL1_eta,EffwrtL1_phi
		     );
    
      //    cout << "After book histos: MaxMult " <<MaxMult <<endl;


      hltt[ip]->Loop( iCount,sPureCount,pureCount,overlapCount,trignames,map_TrigPrescls
		      ,map_MultEle,map_MultPho,map_MultMu
		      ,map_MultJets,map_MultMET
		      ,deno,doMuonCut[ip],doElecCut[ip]
		      ,muonPt,muonDr
		      ,NObjectsToUse,MaxMult,ip,RateOnly
		      ,Num_pt,Num_eta,Num_phi,Den_pt,Den_eta,Den_phi
		      ,DenwrtL1_pt,DenwrtL1_eta,DenwrtL1_phi
		      );

      //    cout<< "Before fill write " <<endl;
      FillWriteEffHistos(menu,ObjectsToUse,ip
			 ,Num_pt,Num_eta,Num_phi
			 ,Den_pt,Den_eta,Den_phi
			 ,Eff_pt,Eff_eta,Eff_phi
			 ,DenwrtL1_pt,DenwrtL1_eta,DenwrtL1_phi
			 ,EffwrtL1_pt,EffwrtL1_eta,EffwrtL1_phi
			 );
   
      //    cout<< "after fill write " <<endl;
      theFile->Close();
    }
    else{
      hltt[ip]->Loop( iCount,sPureCount,pureCount,overlapCount,trignames,map_TrigPrescls
		      ,map_MultEle,map_MultPho,map_MultMu
		      ,map_MultJets,map_MultMET
		      ,deno,doMuonCut[ip],doElecCut[ip]
		      ,muonPt,muonDr
		      ,NObjectsToUse,MaxMult,ip,RateOnly
		      ,Num_pt,Num_eta,Num_phi,Den_pt,Den_eta,Den_phi
		      ,DenwrtL1_pt,DenwrtL1_eta,DenwrtL1_phi
		      );
      
    }

    double mu = bunchCrossingTime * xsec[ip] * ILumi * maxFilledBunches / nFilledBunches;
    for (int it = 0; it < Ntrig; it++){
      // Get global overlaps
      for (int jt = 0; jt != Ntrig; ++jt){
	if (jt==it){
	  (Onum.at(it))[jt] = (((double)(iCount->at(it)) * xsec[ip])); 
	} else {
	  (Onum.at(it))[jt] += ( (double)(overlapCount->at(it).at(jt)) * xsec[ip]);     
	}
      }
      Odenp[it] += ((double)(iCount->at(it)) * xsec[ip]); // ovelap denominator
      
      Rat[it] += collisionRate*(1. - exp(- mu * eff(iCount->at(it),deno)));  // Single rates
      sRat[it] += pow(collisionRate*mu * seff(iCount->at(it),deno),2.);    //
      seqpRat[it] += collisionRate*(1. - exp(- mu * eff(sPureCount->at(it),deno)));  // Single rates

      // Debug printouts
      if (PrintAll==1) {
	if(it<6) {			
	  cout << "i=" << it << " Rate=" << Rat[it] << " +/- " << sqrt(sRat[it]) << ", passed evts=" << iCount->at(it) << ", total evts= " << deno << endl;
	}
      }
      
      sseqpRat[it] += pow(collisionRate*mu * seff(sPureCount->at(it),deno),2.);    //
      pRat[it] += collisionRate*(1. - exp(- mu * eff(pureCount->at(it),deno)));  // Single rates
      spRat[it] += pow(collisionRate*mu * seff(pureCount->at(it),deno),2.);    //

      // Binwise
      Rat_bin[it] += collisionRate*(1. - exp(- mu * eff(iCount->at(it),deno)));  // Single rates
      sRat_bin[it] += pow(collisionRate*mu * seff(iCount->at(it),deno),2.);    //
      seqpRat_bin[it] += collisionRate*(1. - exp(- mu * eff(sPureCount->at(it),deno)));  // Single rates
      sseqpRat_bin[it] += pow(collisionRate*mu * seff(sPureCount->at(it),deno),2.);    //
      pRat_bin[it] += collisionRate*(1. - exp(- mu * eff(pureCount->at(it),deno)));  // Single rates
      spRat_bin[it] += pow(collisionRate*mu * seff(pureCount->at(it),deno),2.);    //

    }

    // Print binwise rates:
    // Loop over triggers
    Double_t RTOT_bin = 0.; 
    Double_t sRTOT_bin = 0.; 
    Double_t physRTOT_bin = 0.;  
    Double_t physsRTOT_bin = 0.;  
    Double_t curat_bin = 0.; 
    Double_t cuthroughput_bin = 0.; 
    Double_t physcurat_bin = 0.;  
    Double_t physcuthroughput_bin = 0.;  
    Double_t scuthroughput_bin = 0.; 
    for (int it = 0; it < Ntrig; it++){ 
      curat_bin += seqpRat_bin[it]; 
      cuthroughput_bin += seqpRat_bin[it] * map_EventSize.find(trignames[it])->second; 
      scuthroughput_bin += sseqpRat_bin[it] * map_EventSize.find(trignames[it])->second;  
      cRat_bin[it] = curat_bin; 
      cThroughput_bin[it] = cuthroughput_bin; 
      RTOT_bin += seqpRat_bin[it];                                            // Total Rate 
      sRTOT_bin += sseqpRat_bin[it]; 
 
      // Bookkeeping - store total rate *except* for ALCA triggers 
      if(!(trignames[it].Contains("AlCa"))) 
        { 
          physRTOT_bin  += seqpRat_bin[it];        
          physsRTOT_bin += sseqpRat_bin[it]; 
          physcurat_bin += seqpRat_bin[it] * map_EventSize.find(trignames[it])->second; 
          physcuthroughput_bin += sseqpRat_bin[it] * map_EventSize.find(trignames[it])->second;   
        } 
    } 
 
    sRTOT_bin = sqrt(sRTOT_bin); 
    physsRTOT_bin = sqrt(physsRTOT_bin);  
    scuthroughput_bin = sqrt(scuthroughput_bin); 
    physcuthroughput_bin = sqrt(physcuthroughput_bin); 

    // Print binwise
    cout.setf(ios::floatfield,ios::fixed);
    cout<<setprecision(3);
    for (int it=0; it < Ntrig; it++){
      cout  << setw(3) << it << ")" << setw(26) << trignames[it]  << " (" << setw(7) << map_TrigPrescls.find(trignames[it])->second << ")"
	    << " :   Indiv.: " << setw(7) << Rat_bin[it] << " +/- " << setw(7) << sqrt(sRat_bin[it]) 
	    << "   sPure: " << setw(7) << seqpRat_bin[it]
	    << "   Pure: " << setw(7) << pRat_bin[it] 
	    << "   Cumul: " << setw(7) << cRat_bin[it] << "\n"<<flush;
    }
    cout << "\n"<<flush;
    cout << setw(60) << "TOTAL RATE : " << setw(5) << RTOT_bin << " +- " << sRTOT_bin << " Hz" << "\n";
    cout << "\n"<<flush;
    
  }

  // Loop over triggers
  Double_t RTOT = 0.; 
  Double_t sRTOT = 0.; 
  Double_t physRTOT = 0.; 
  Double_t physsRTOT = 0.;  
  Double_t curat = 0.; 
  Double_t cuthroughput = 0.0; 
  Double_t scuthroughput = 0.0; 
  Double_t physcuthroughput = 0.0; 
  Double_t physscuthroughput = 0.0; 
  for (int it = 0; it < Ntrig; it++){ 
    curat += seqpRat[it]; 
    cuthroughput += seqpRat[it] * map_EventSize.find(trignames[it])->second; 
    scuthroughput += sseqpRat[it] * map_EventSize.find(trignames[it])->second; 
    cRat[it] = curat; 
    cThroughput[it] = cuthroughput; 
    RTOT += seqpRat[it];                                            // Total Rate 
    sRTOT += sseqpRat[it]; 
 
    // Bookkeeping - store total rate *except* for ALCA triggers  
    if(!(trignames[it].Contains("AlCa")))  
      {  
        physRTOT  += seqpRat[it];         
        physsRTOT += sseqpRat[it];  
        physcuthroughput += seqpRat[it] * map_EventSize.find(trignames[it])->second; 
        physscuthroughput += sseqpRat[it] * map_EventSize.find(trignames[it])->second; 
      }  
 
  } 
 
  physsRTOT = sqrt(physsRTOT); 
  sRTOT = sqrt(sRTOT); 
  scuthroughput = sqrt(scuthroughput); 
  physscuthroughput = sqrt(physscuthroughput); 
    
  // End calculating rates  
  /* *************************************************************** */

  char sLumi[10]; 
  sprintf(sLumi,"%1.1e",ILumi); 
  TString hltTableFileName= TString("hltTable_") + + TString(sEnergy) + "TeV_" + TString(sLumi) + TString("_") + sConditions + TString("Conditions") + sVersion; 
  TFile *fr = new TFile(hltTableFileName+TString(".root"),"recreate");
  fr->cd();
  TH1F *individual = new TH1F("individual","individual",Ntrig,1,Ntrig+1);
  TH1F *cumulative = new TH1F("cumulative","cumulative",Ntrig,1,Ntrig+1);
  TH1F *throughput = new TH1F("throughput","throughput",Ntrig,1,Ntrig+1);
  TH1F *eventsize = new TH1F("eventsize","eventsize",Ntrig,1,Ntrig+1);
  TH2F *overlap = new TH2F("overlap","overlap",Ntrig,1,Ntrig+1,Ntrig,1,Ntrig+1);

  ////////////////////////////////////////////////////////////
  // Printout Results
  cout<<setprecision(3);
  cout << endl;
  
  // Printout overlaps
  if (PrintAll==1) {
    cout << "Trigger global overlaps : " << endl;
    for (int it = 0; it != Ntrig; ++it){
      for (int jt = 0; jt != Ntrig; ++jt){
	if (jt>=it) {
	  // Overlap O(ij) = T(i) x T(j) / T(j)
	  cout << "i=" << it << " j=" << jt << "     " << eff((Onum.at(it))[jt],Odenp[jt]) << endl;   
	  overlap->SetBinContent(it+1,jt+1,eff((Onum.at(it))[jt],Odenp[jt]));
	  overlap->GetXaxis()->SetBinLabel(it+1,trignames[it]);
	  overlap->GetYaxis()->SetBinLabel(jt+1,trignames[jt]);
	}
      }
    }
  }

  cout.setf(ios::floatfield,ios::fixed);
  cout<<setprecision(3);

  cout << "\n";
  cout << "Trigger Rates [Hz] : " << "\n";
  cout << "------------------------------------------------------------------------------------------------------------------\n";
  // This is with the accurate formula: 
  for (int it=0; it < Ntrig; it++){
    individual->SetBinContent(it+1,Rat[it]);
    individual->GetXaxis()->SetBinLabel(it+1,trignames[it]);
    cumulative->SetBinContent(it+1,cRat[it]);
    cumulative->GetXaxis()->SetBinLabel(it+1,trignames[it]);

    Double_t theevsize = map_EventSize.find(trignames[it])->second;

    if(map_EventSize.size() > 0)
      {
	throughput->SetBinContent(it+1,cThroughput[it]);
	throughput->GetXaxis()->SetBinLabel(it+1,trignames[it]); 
	eventsize->SetBinContent(it+1,theevsize); 
	eventsize->GetXaxis()->SetBinLabel(it+1,trignames[it]);  
      }
    else
      {
	throughput->SetBinContent(it+1,cRat[it]*1.5); 
        throughput->GetXaxis()->SetBinLabel(it+1,trignames[it]);  
        eventsize->SetBinContent(it+1,1.5);  
        eventsize->GetXaxis()->SetBinLabel(it+1,trignames[it]);   
      }
    
    cout  << setw(3) << it << ")" << setw(26) << trignames[it]  << " (" << setw(7) << map_TrigPrescls.find(trignames[it])->second << ")" 
	  << " :   Indiv.: " << setw(7) << Rat[it] << " +/- " << setw(7) << sqrt(sRat[it]) 
	  << "   sPure: " << setw(7) << seqpRat[it]
	  << "   Pure: " << setw(7) << pRat[it] 
	  << "   Cumul: " << setw(7) << cRat[it] << "\n"<<flush;
  }
  cout << "\n"<<flush;
  cout << setw(60) << "TOTAL RATE : " << setw(5) << RTOT << " +- " << sRTOT << " Hz" << "\n";
  cout << "------------------------------------------------------------------------------------------------------------------\n"<<flush;

  individual->SetStats(0); individual->SetYTitle("Rate (Hz)"); individual->SetTitle("Individual trigger rate");
  cumulative->SetStats(0); cumulative->SetYTitle("Rate (Hz)"); cumulative->SetTitle("Cumulative trigger rate"); 
  overlap->SetStats(0); overlap->SetTitle("Overlap");
  individual->Write();
  cumulative->Write();
  eventsize->Write();
  throughput->Write();
  overlap->Write();
  fr->Close();
  
  ////////////////////////////////////////////////////////////
  // Printout Results to Tex/PDF

  if (PrintAll==1) {
    //    char sLumi[10];
    //    sprintf(sLumi,"%1.1e",ILumi);
    //    TString hltTableFileName= TString("hltTable_") + + TString(sEnergy) + "TeV_" + TString(sLumi) + TString("_") + sConditions + TString("Conditions") + sVersion;
    TString texFile = hltTableFileName + TString(".tex");
    TString dviFile = hltTableFileName + TString(".dvi");
    //TString psFile  = hltTableFileName + TString(".ps");
    TString psFile  = hltTableFileName + TString(".pdf");
    ofstream outFile(texFile.Data());
    if (!outFile){cout<<"Error opening output file"<< endl;}
    outFile <<setprecision(2);
    outFile.setf(ios::floatfield,ios::fixed);
    outFile << "\\documentclass[amsmath,amssymb]{revtex4}" << endl;
    outFile << "\\usepackage{longtable}" << endl;
    outFile << "\\usepackage{color}" << endl;
    outFile << "\\usepackage{lscape}" << endl;
    outFile << "\\begin{document}" << endl;
    outFile << "\\begin{landscape}" << endl;
    outFile << "\\newcommand{\\met}{\\ensuremath{E\\kern-0.6em\\lower-.1ex\\hbox{\\/}\\_T}}" << endl;
    
    
    outFile << "\\begin{footnotesize}" << endl;
    outFile << "\\begin{longtable}{|l|l|c|c|c|c|c|c|}" << endl;
    outFile << "\\caption[Cuts]{New paths are introduced in addition to standard '1e32' paths.  Description of the newly introduced paths is given at the end of the table.  Available HLT bandwith is 150 Hz = ((1 GB/s / 3) - 100 MB/s for AlCa triggers) / 1.5 MB/event. L1 bandwidth is 12 kHz. } \\label{CUTS} \\\\ " << endl;
    
    
    outFile << "\\hline \\multicolumn{8}{|c|}{\\bf \\boldmath HLT for L = "<< sLumi  << "}\\\\  \\hline" << endl;
    //    outFile << "{\\bf Status} & " << endl;
    outFile << "{\\bf Path Name} & " << endl;
    outFile << "{\\bf L1 condtition} & " << endl;
    //    outFile << "\\begin{tabular}{c} {\\bf Threshold} \\\\ {\\bf $[$GeV$]$} \\end{tabular} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf L1} \\\\ {\\bf Prescale} \\end{tabular} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf HLT} \\\\ {\\bf Prescale} \\end{tabular} & " << endl;
    //outFile << "\\begin{tabular}{c} {\\bf Total} \\\\ {\\bf Prescale} \\end{tabular} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf HLT Rate} \\\\ {\\bf $[$Hz$]$} \\end{tabular} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf Total Rate} \\\\ {\\bf $[$Hz$]$} \\end{tabular} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf Avg. Size} \\\\ {\\bf $[$MB$]$} \\end{tabular} & " << endl;  
    outFile << "\\begin{tabular}{c} {\\bf Total} \\\\ {\\bf Throughput} \\\\ {\\bf $[$MB/s$]$} \\end{tabular} \\\\ \\hline" << endl;  
    outFile << "\\endfirsthead " << endl;
    
    outFile << "\\multicolumn{8}{r}{\\bf \\bfseries --continued from previous page (L = " << sLumi << ")"  << "}\\\\ \\hline " << endl;
    //    outFile << "{\\bf Status} & " << endl;
    outFile << "{\\bf Path Name} & " << endl;
    outFile << "{\\bf L1 condtition} & " << endl;
    //    outFile << "\\begin{tabular}{c} {\\bf Threshold} \\\\ {\\bf $[$GeV$]$} \\end{tabular} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf L1} \\\\ {\\bf Prescale} \\end{tabular} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf HLT} \\\\ {\\bf Prescale} \\end{tabular} & " << endl;
    //outFile << "\\begin{tabular}{c} {\\bf Total} \\\\ {\\bf Prescale} \\end{tabular} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf HLT Rate} \\\\ {\\bf $[$Hz$]$} \\end{tabular} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf Total Rate} \\\\ {\\bf $[$Hz$]$} \\end{tabular} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf Avg. Size} \\\\ {\\bf $[$MB$]$} \\end{tabular} & " << endl; 
    outFile << "\\begin{tabular}{c} {\\bf Total} \\\\ {\\bf Throughput} \\\\ {\\bf $[$MB/s$]$} \\end{tabular} \\\\ \\hline" << endl; 
    outFile << "\\endhead " << endl;
    
    outFile << "\\hline \\multicolumn{8}{|r|}{{Continued on next page}} \\\\ \\hline " << endl;
    outFile << "\\endfoot " << endl;
    
    outFile << "\\hline " << endl;
    outFile << "\\endlastfoot " << endl;
    
    for (int it=0; it < Ntrig; it++){
      
      TString tempTrigName = trignames[it];
      TString tempL1BitName = menu->GetHltL1BitMap().find(trignames[it])->second;
      TString tempThreshold = menu->GetHltThresholdMap().find(trignames[it])->second;
      
      if(strlen(tempL1BitName) > 30)
	tempL1BitName = "List Too Long";

      /*
	if(tempTrigName.Contains("Apt")) {
	outFile << "new & " ;
	//tempTrigName.ReplaceAll("AptHLT","");
	tempTrigName.ReplaceAll("AptHLT","HLT");
	}
	else {
	outFile << " 1e32 & " ;
	}
      */
	
      //      outFile << map_HltDesc.find(trignames[it])->second << " & " ;
      
      tempTrigName.ReplaceAll("_","\\_");
      tempL1BitName.ReplaceAll("_","\\_");
      //tempTrigName.ReplaceAll("HLT","");
      outFile << "\\color{blue}"  << tempTrigName << " & " << "${\\it " << tempL1BitName
	//      << "}$ "<< " & " << tempThreshold << " & " <<  map_L1Prescls.find(trignames[it])->second
	      << "}$ " << " & " <<  map_L1Prescls.find(trignames[it])->second 
	      << " & " <<  map_HLTPrescls.find(trignames[it])->second  << " & " << Rat[it]
	      << " {$\\pm$ " << sqrt(sRat[it]) << "} & " << cRat[it] << " & " << map_EventSize.find(trignames[it])->second << " & " 
	      << cThroughput[it] << "\\\\" << endl;
    }

    outFile << "\\hline \\multicolumn{8}{|l|}{\\bf \\boldmath Total Physics HLT rate (Hz), AlCa triggers not included }  "<<  physRTOT << " {$\\pm$ " << physsRTOT << "} \\\\  \\hline" << endl;  
    outFile << "\\hline \\multicolumn{8}{|l|}{\\bf \\boldmath Total Physics HLT throughput (MB/s), AlCa triggers not included }  "<<  physcuthroughput<< " {$\\pm$ " << physscuthroughput << "} \\\\  \\hline" << endl;
    outFile << "\\hline \\multicolumn{8}{|l|}{\\bf \\boldmath Total HLT rate (Hz) }  "<<  RTOT << " {$\\pm$ " << sRTOT << "} \\\\  \\hline" << endl;
    outFile << "\\hline \\multicolumn{8}{|l|}{\\bf \\boldmath Total HLT throughput (MB/s) }  "<<  cuthroughput << " {$\\pm$ " << scuthroughput << "} \\\\  \\hline" << endl; 
    outFile << "\\hline " << endl;
    
    //    outFile << "\\multicolumn{8}{|l|}{ ($\\ast$): Conditions on tracks seeded by L2 muons} \\\\ \\hline  " << endl;
    //    outFile << "\\multicolumn{8}{|l|}{ ($\\star$): {$L1\\_Mu3\\_IsoEG5, L1\\_Mu5\\_IsoEG10, L1\\_Mu3\\_IsoEG12$ }} \\\\ \\hline  " << endl;
    //outFile << "\\multicolumn{8}{|l|}{ ($\\S$): {2JetAve paths use {\\bf uncorrected} jets at HLT }} \\\\ \\hline  " << endl;
    //    outFile << "\\multicolumn{8}{|l|}{ (NI): Not implemented in current version} \\\\ \\hline  " << endl;
    //    outFile << "\\multicolumn{8}{|l|}{ ($\\S$): {$M_{\\mu\\mu} \\in [0.2,3],M_{\\mu\\mu\\mu} \\in [1.2,2.2]$}} \\\\ \\hline  " << endl;
    //    outFile << "\\multicolumn{8}{|l|}{ ($\\circ$): Only Pixel-matching, no track match required} \\\\ \\hline  " << endl;
    //    outFile << "\\multicolumn{8}{|l|}{ ($\\dagger$): {$L1\\_SingleJet150, L1\\_DoubleJet70, L1\\_TripleJet50$}} \\\\ \\hline " << endl; 
    //    outFile << "\\multicolumn{8}{|l|}{ ($\\ddagger$): {$L1\\_SingleJet150, L1\\_DoubleJet70, L1\\_TripleJet50, L1\\_QuadJet30$}} \\\\ \\hline " << endl; 
    
    //    outFile << "\\multicolumn{8}{|l|}{ ($\\Diamond$): {$L1\\_SingleJet100, L1\\_DoubleJet70, L1\\_TripleJet50, L1\\_QuadJet30, L1\\_HTT300$}} \\\\ \\hline " << endl; 
    //    outFile << "\\multicolumn{8}{|l|}{ ($\\Diamond \\Diamond$): {$L1\\_SingleHFTowCount1/12, L1\\_DoubleHFTowCount1/20, L1\\_SingleHFRing0Sum3/20,$}} \\\\   " << endl;
    //    outFile << "\\multicolumn{8}{|l|}{ {$L1\\_DoubleHFRing0Sum3/20, L1\\_SingleHFRing0Sum6/20, L1\\_DoubleHFRing0Sum6/20$}} \\\\\\hline \\hline " << endl; 
    //    outFile << "\\multicolumn{8}{|c|}{ {\\it \\bf HLT Requirements for new introduced paths (in GeV) }  } \\\\ \\hline  " << endl;
    //    outFile << "\\multicolumn{8}{|l|}{ HLTMuons: $|\\eta|<2.5$, $L2Pt+3.9Err<$A, $L3Pt+2.2Err<$A  } \\\\ \\hline  " << endl;
    //    outFile << "\\multicolumn{8}{|l|}{ HLT1jetA: $recoJetCalPt<$A  } \\\\ \\hline  " << endl;
    //    outFile << "\\multicolumn{8}{|l|}{ HLT1ElectronA\\_L1R\\_HI: $Et>$A, $HCAL<3$, $TrkIso<0.06$} \\\\ " << endl; 
    //    outFile << "\\multicolumn{8}{|l|}{ HLT1ElectronA\\_L1R\\_LI: $Et>$A, $HCAL<6$, $TrkIso<0.12$} \\\\   " << endl;
    //    outFile << "\\multicolumn{8}{|l|}{ HLT1ElectronA\\_L1R\\_NI: $Et>$A} \\\\ \\hline  " << endl;
    //    outFile << "\\multicolumn{8}{|l|}{ HLT1PhotonA\\_L1R\\_HI: $Et>$A, $ECAL<1.5$, $HCAL<6(4)$, $TrkIso=0$} \\\\   " << endl;
    //    outFile << "\\multicolumn{8}{|l|}{ HLT1PhotonA\\_L1R\\_LI: $Et>$A, $ECAL<3.0$, $HCAL<12(8)$, $TrkIso\\leq2$} \\\\   " << endl;
    //    outFile << "\\multicolumn{8}{|l|}{ HLT1PhotonA\\_L1R\\_NI: $Et>$A} \\\\  " << endl;
    
    //outFile << "\\end{tabular}" << endl;
    outFile << "\\end{longtable}" << endl;
    outFile << "\\end{footnotesize}" << endl;
    outFile << "\\clearpage" << endl;
    
    outFile << "\\end{landscape}" << endl;
    outFile << "\\end{document}";
    outFile.close();
    
    //TString Command = TString("latex ") + texFile + TString("; dvips -t A4 -f ") + dviFile + TString(" -o ") + psFile;
    TString Command = TString("latex ") + texFile + TString("; latex ") + texFile +
      TString("; dvipdf ") + dviFile + TString(" ") + psFile;
    
    // Latex execution
    cout << "Executing the following latex command: " << endl;
    cout << Command << endl;
    // do it again to fix for column size shift within the table
    for(int i=0;i<2;i++) system(Command.Data());
  }
}

// Forward addition of new names and table for CMSSW 21X
// /* ********** */
// To be effective 2 things have to be updated as well in OHltTree.h:
// - Declaration of branches (use MakeClass() function)
// - OHltTree::SetMapBitOfStandardHLTPath() assignments
// - Prescales are all set to 1!!! Need to be changes!!!
void BookMenu_21XDefault(OHltMenu*  menu, double &iLumi, double &nBunches) {

  iLumi = 2E30;
  nBunches = 43;
  //objects electron,photon,muon,jets,met

  cout <<" Booking menu "<<endl;

  menu->AddHlt("HLT_L1Jet15","L1_SingleJet15",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_Jet30","L1_SingleJet15",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_Jet50","L1_SingleJet30",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_Jet80","L1_SingleJet50",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_Jet110","L1_SingleJet70",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_Jet180","L1_SingleJet70",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_Jet250","L1_SingleJet70",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_FwdJet20","L1_IsoEG10_Jet15_ForJet10",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_DoubleJet150","L1_SingleJet150 OR L1_DoubleJet70",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_DoubleJet125_Aco","L1_SingleJet150 OR L1_DoubleJet70",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_DoubleFwdJet50","L1_SingleJet30",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_DiJetAve15","L1_SingleJet15",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_DiJetAve30","L1_SingleJet30",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_DiJetAve50","L1_SingleJet50",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_DiJetAve70","L1_SingleJet70",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_DiJetAve130","L1_SingleJet70",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_DiJetAve220","L1_SingleJet70",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_TripleJet85","L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50",1,1,"","",1.5,0,0,0,3,0);
  menu->AddHlt("HLT_QuadJet30","L1_QuadJet15",1,1,"","",1.5,0,0,0,4,0);
  menu->AddHlt("HLT_QuadJet60","L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30",1,1,"","",1.5,0,0,0,4,0);
  menu->AddHlt("HLT_SumET120","L1_ETT60",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_L1MET20","L1_ETM20",1,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT_MET25","L1_ETM20",1,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT_MET35","L1_ETM30",1,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT_MET50","L1_ETM40",1,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT_MET65","L1_ETM50",1,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT_MET75","L1_ETM50",1,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT_MET35_HT350","L1_HTT300",1,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT_Jet180_MET60","L1_SingleJet150",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLT_Jet60_MET70_Aco","L1_SingleJet150",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLT_Jet100_MET60_Aco","L1_SingleJet150",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLT_DoubleJet125_MET60","L1_SingleJet150",1,1,"","",1.5,0,0,0,2,1);
  menu->AddHlt("HLT_DoubleFwdJet40_MET60","L1_ETM40",1,1,"","",1.5,0,0,0,2,1);
  menu->AddHlt("HLT_DoubleJet60_MET60_Aco","L1_SingleJet150",1,1,"","",1.5,0,0,0,2,1);
  menu->AddHlt("HLT_DoubleJet50_MET70_Aco","L1_SingleJet150",1,1,"","",1.5,0,0,0,2,1);
  menu->AddHlt("HLT_DoubleJet40_MET70_Aco","L1_SingleJet150",1,1,"","",1.5,0,0,0,2,1);
  menu->AddHlt("HLT_TripleJet60_MET60","L1_SingleJet150",1,1,"","",1.5,0,0,0,3,1);
  menu->AddHlt("HLT_QuadJet35_MET60","L1_SingleJet150",1,1,"","",1.5,0,0,0,4,1);
  menu->AddHlt("HLT_IsoEle15_L1I","L1_SingleIsoEG12",1,1,"","",1.5,1,0,0,0,0);
  menu->AddHlt("HLT_IsoEle18_L1R","L1_SingleEG15",1,1,"","",1.5,1,0,0,0,0);
  menu->AddHlt("HLT_IsoEle15_LW_L1I","L1_SingleIsoEG12",1,1,"","",1.5,1,0,0,0,0);
  menu->AddHlt("OpenHLT_Ele5_SW_L1R","L1_SingleEG5",1,1,"","",1.5,1,0,0,0,0); 
  menu->AddHlt("HLT_LooseIsoEle15_LW_L1R","L1_SingleEG12",1,1,"","",1.5,1,0,0,0,0);
  menu->AddHlt("HLT_Ele10_SW_L1R","L1_SingleEG8",1,1,"","",1.5,1,0,0,0,0);
  menu->AddHlt("HLT_Ele15_SW_L1R","L1_SingleEG12",1,1,"","",1.5,1,0,0,0,0);
  menu->AddHlt("HLT_Ele15_LW_L1R","L1_SingleEG10",1,1,"","",1.5,1,0,0,0,0);
  menu->AddHlt("HLT_EM80","L1_SingleEG15",1,1,"","",1.5,1,1,0,0,0);
  menu->AddHlt("HLT_EM200","L1_SingleEG15",1,1,"","",1.5,1,1,0,0,0);
  menu->AddHlt("HLT_DoubleIsoEle10_L1I","L1_DoubleIsoEG8",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT_DoubleIsoEle12_L1R","L1_DoubleEG10",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT_DoubleIsoEle10_LW_L1I","L1_DoubleIsoEG8",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT_DoubleIsoEle12_LW_L1R","L1_DoubleEG10",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT_DoubleEle5_SW_L1R","L1_DoubleEG5",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT_DoubleEle10_LW_OnlyPixelM_L1R","L1_DoubleEG5",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT_DoubleEle10_Z","L1_DoubleIsoEG8",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT_DoubleEle6_Exclusive","L1_ExclusiveDoubleIsoEG6",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT_IsoPhoton30_L1I","L1_SingleIsoEG12",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT_IsoPhoton10_L1R","L1_SingleEG8",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT_IsoPhoton15_L1R","L1_SingleEG12",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT_IsoPhoton20_L1R","L1_SingleEG15",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT_IsoPhoton25_L1R","L1_SingleEG15",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT_IsoPhoton40_L1R","L1_SingleEG15",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("OpenHLT_L1Photon5","L1_SingleEG5",1,1,"","",1.5,0,1,0,0,0); 
  menu->AddHlt("OpenHLT_Photon10_L1R","L1SingleEG8",1,1,"","",1.5,0,1,0,0,0); 
  menu->AddHlt("HLT_Photon15_L1R","L1_SingleEG10",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT_Photon25_L1R","L1_SingleEG15",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT_DoubleIsoPhoton20_L1I","L1_DoubleIsoEG8",1,1,"","",1.5,0,2,0,0,0);
  menu->AddHlt("HLT_DoubleIsoPhoton20_L1R","L1_DoubleEG10",1,1,"","",1.5,0,2,0,0,0);
  menu->AddHlt("HLT_DoublePhoton10_Exclusive","L1_ExclusiveDoubleIsoEG6",1,1,"","",1.5,0,2,0,0,0);
  menu->AddHlt("HLT_L1Mu","L1_SingleMu7 OR L1_DoubleMu3",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_L1MuOpen","L1_SingleMuOpen OR L1_SingleMu3 OR L1_SingleMu5",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_L2Mu9","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_IsoMu9","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_IsoMu11","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_IsoMu13","L1_SingleMu10",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_IsoMu15","L1_SingleMu10",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_NoTrackerIsoMu15","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu3","L1_SingleMu3",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu5","L1_SingleMu5",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu7","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu9","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu11","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu13","L1_SingleMu10",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu15","L1_SingleMu10",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu15_L1Mu7","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu15_Vtx2cm","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu15_Vtx2mm","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_DoubleIsoMu3","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu3","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu3_Vtx2cm","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu3_Vtx2mm","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu3_JPsi","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu3_Upsilon","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu7_Z","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu3_SameSign","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu3_Psi2S","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_BTagIP_Jet180","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_BTagIP_Jet120_Relaxed","L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_BTagIP_DoubleJet120","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_BTagIP_DoubleJet60_Relaxed","L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_BTagIP_TripleJet70","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,3,0);
  menu->AddHlt("HLT_BTagIP_TripleJet40_Relaxed","L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,3,0);
  menu->AddHlt("HLT_BTagIP_QuadJet40","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,4,0);
  menu->AddHlt("HLT_BTagIP_QuadJet30_Relaxed","L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,4,0);
  menu->AddHlt("HLT_BTagIP_HT470","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_BTagIP_HT320_Relaxed","L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_BTagMu_DoubleJet120","L1_Mu5_Jet15",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_BTagMu_DoubleJet60_Relaxed","L1_Mu5_Jet15",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_BTagMu_TripleJet70","L1_Mu5_Jet15",1,1,"","",1.5,0,0,0,3,0);
  menu->AddHlt("HLT_BTagMu_TripleJet40_Relaxed","L1_Mu5_Jet15",1,1,"","",1.5,0,0,0,3,0);
  menu->AddHlt("HLT_BTagMu_QuadJet40","L1_Mu5_Jet15",1,1,"","",1.5,0,0,0,4,0);
  menu->AddHlt("HLT_BTagMu_QuadJet30_Relaxed","L1_Mu5_Jet15",1,1,"","",1.5,0,0,0,4,0);
  menu->AddHlt("HLT_BTagMu_HT370","L1_HTT300",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_BTagMu_HT250_Relaxed","L1_HTT200",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_DoubleMu3_BJPsi","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu4_BJPsi","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_TripleMu3_TauTo3Mu","L1_DoubleMu3",1,1,"","",1.5,0,0,3,0,0);
  menu->AddHlt("HLT_IsoTau_MET65_Trk20","L1_SingleTauJet80",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLT_IsoTau_MET35_Trk15_L1MET","L1_TauJet30_ETM30",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLT_LooseIsoTau_MET30","L1_SingleTauJet80",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLT_LooseIsoTau_MET30_L1MET","L1_TauJet30_ETM30",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLT_DoubleIsoTau_Trk3","L1_DoubleTauJet40",1,1,"","",1.5,0,0,0,1,0);
  //  menu->AddHlt("HLT_DoubleLooseIsoTau","L1_DoubleTauJet20",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("OpenHLT_DoubleLooseIsoTau","L1_DoubleTauJet40",1,1,"","",1.5,0,0,0,1,0); 
  menu->AddHlt("HLT_IsoEle8_IsoMu7","L1_Mu3_IsoEG5",1,1,"","",1.5,1,0,1,0,0);
  menu->AddHlt("HLT_IsoEle10_Mu10_L1R","L1_Mu3_EG12",1,1,"","",1.5,1,0,1,0,0);
  menu->AddHlt("HLT_IsoEle12_IsoTau_Trk3","L1_IsoEG10_TauJet20",1,1,"","",1.5,1,0,0,1,0);
  menu->AddHlt("HLT_IsoEle10_BTagIP_Jet35","L1_IsoEG10_Jet20",1,1,"","",1.5,1,0,0,1,0);
  menu->AddHlt("HLT_IsoEle12_Jet40","L1_IsoEG10_Jet30",1,1,"","",1.5,1,0,0,1,0);
  menu->AddHlt("HLT_IsoEle12_DoubleJet80","L1_IsoEG10_Jet30",1,1,"","",1.5,1,0,0,1,0);
  menu->AddHlt("HLT_IsoElec5_TripleJet30","L1_EG5_TripleJet15",1,1,"","",1.5,1,0,0,3,0);
  menu->AddHlt("HLT_IsoEle12_TripleJet60","L1_IsoEG10_Jet30",1,1,"","",1.5,1,0,0,3,0);
  menu->AddHlt("HLT_IsoEle12_QuadJet35","L1_IsoEG10_Jet30",1,1,"","",1.5,1,0,0,4,0);
  menu->AddHlt("HLT_IsoMu14_IsoTau_Trk3","L1_Mu5_TauJet20",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLT_IsoMu7_BTagIP_Jet35","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLT_IsoMu7_BTagMu_Jet20","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLT_IsoMu7_Jet40","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLT_NoL2IsoMu8_Jet40","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLT_Mu14_Jet50","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLT_Mu5_TripleJet30","L1_Mu3_TripleJet15",1,1,"","",1.5,0,0,1,3,0);
  menu->AddHlt("HLT_BTagMu_Jet20_Calib","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLT_ZeroBias","L1_ZeroBias",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_MinBias","L1_MinBias_HTT10",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_MinBiasHcal","L1_SingleJetCountsHFTow OR L1_DoubleJetCountsHFTow OR L1_SingleJetCountsHFRing0Sum3 OR L1_DoubleJetCountsHFRing0Sum3 OR L1_SingleJetCountsHFRing0Sum6 OR L1_DoubleJetCountsHFRing0Sum6",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_MinBiasEcal","L1_SingleEG2 OR L1_DoubleEG1",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_MinBiasPixel","L1_ZeroBias",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_MinBiasPixel_Trk5","L1_ZeroBias",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_BackwardBSC","38 OR 39",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_ForwardBSC","36 OR 37",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_CSCBeamHalo","L1_SingleMuBeamHalo",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_CSCBeamHaloOverlapRing1","L1_SingleMuBeamHalo",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_CSCBeamHaloOverlapRing2","L1_SingleMuBeamHalo",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_CSCBeamHaloRing2or3","L1_SingleMuBeamHalo",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_TrackerCosmics","24 OR 25 OR 26 OR 27 OR 28",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("AlCa_IsoTrack","L1_SingleJet30 OR L1_SingleJet50 OR L1_SingleJet70 OR L1_SingleJet100 OR L1_SingleTauJet30 OR L1_SingleTauJet40 OR L1_SingleTauJet60 OR L1_SingleTauJet80",1,1,"","",0.214,0,0,0,0,0);
  menu->AddHlt("AlCa_EcalPhiSym","L1_ZeroBias OR L1_SingleJetCountsHFTow OR L1_DoubleJetCountsHFTow OR L1_SingleEG2 OR L1_DoubleEG1 OR L1_SingleJetCountsHFRing0Sum3 OR L1_DoubleJetCountsHFRing0Sum3 OR L1_SingleJetCountsHFRing0Sum6 OR L1_DoubleJetCountsHFRing0Sum6",1,1,"","",0.001,0,0,0,0,0);
  menu->AddHlt("AlCa_EcalPi0","L1_SingleIsoEG5 OR L1_SingleIsoEG8 OR L1_SingleIsoEG10 OR L1_SingleIsoEG12 OR L1_SingleIsoEG15 OR L1_SingleIsoEG20 OR L1_SingleIsoEG25 OR L1_SingleEG5 OR L1_SingleEG8 OR L1_SingleEG10 OR L1_SingleEG12 OR L1_SingleEG15 OR L1_SingleEG20 OR L1_SingleEG25",1,1,"","",0.007,0,0,0,0,0);

  cout <<" End Booking menu "<<endl;


}

void BookMenu_21X_29E30_Default(OHltMenu*  menu, double &iLumi, double &nBunches) { 
 
  iLumi = 2.9E30; 
  nBunches = 43; 
  //objects electron,photon,muon,jets,met 
 
  cout <<" Booking menu "<<endl; 

  menu->AddHlt("HLT_L1Jet15","L1_SingleJet15",250,20,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_Jet30","L1_SingleJet15",250,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_Jet50","L1_SingleJet30",10,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_Jet80","L1_SingleJet50",5,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_Jet110","L1_SingleJet70",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_Jet180","L1_SingleJet70",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_Jet250","L1_SingleJet70",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_FwdJet20","L1_IsoEG10_Jet15_ForJet10",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_DoubleJet150","L1_SingleJet150 OR L1_DoubleJet70",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_DoubleJet125_Aco","L1_SingleJet150 OR L1_DoubleJet70",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_DoubleFwdJet50","L1_SingleJet30",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_DiJetAve15","L1_SingleJet15",250,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_DiJetAve30","L1_SingleJet30",10,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_DiJetAve50","L1_SingleJet50",5,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_DiJetAve70","L1_SingleJet70",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_DiJetAve130","L1_SingleJet70",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_DiJetAve220","L1_SingleJet70",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_TripleJet85","L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50",1,1,"","",1.5,0,0,0,3,0);
  menu->AddHlt("HLT_QuadJet30","L1_QuadJet15",1,1,"","",1.5,0,0,0,4,0);
  menu->AddHlt("HLT_QuadJet60","L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30",1,1,"","",1.5,0,0,0,4,0);
  menu->AddHlt("HLT_SumET120","L1_ETT60",1500,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_L1MET20","L1_ETM20",250,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT_MET25","L1_ETM20",250,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT_MET35","L1_ETM30",10,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT_MET50","L1_ETM40",1,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT_MET65","L1_ETM50",1,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT_MET75","L1_ETM50",1,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT_MET35_HT350","L1_HTT300",1,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT_Jet180_MET60","L1_SingleJet150",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLT_Jet60_MET70_Aco","L1_SingleJet150",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLT_Jet100_MET60_Aco","L1_SingleJet150",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLT_DoubleJet125_MET60","L1_SingleJet150",1,1,"","",1.5,0,0,0,2,1);
  menu->AddHlt("HLT_DoubleFwdJet40_MET60","L1_ETM40",1,1,"","",1.5,0,0,0,2,1);
  menu->AddHlt("HLT_DoubleJet60_MET60_Aco","L1_SingleJet150",1,1,"","",1.5,0,0,0,2,1);
  menu->AddHlt("HLT_DoubleJet50_MET70_Aco","L1_SingleJet150",1,1,"","",1.5,0,0,0,2,1);
  menu->AddHlt("HLT_DoubleJet40_MET70_Aco","L1_SingleJet150",1,1,"","",1.5,0,0,0,2,1);
  menu->AddHlt("HLT_TripleJet60_MET60","L1_SingleJet150",1,1,"","",1.5,0,0,0,3,1);
  menu->AddHlt("HLT_QuadJet35_MET60","L1_SingleJet150",1,1,"","",1.5,0,0,0,4,1);
  menu->AddHlt("HLT_IsoEle15_L1I","L1_SingleIsoEG12",1,1,"","",1.5,1,0,0,0,0);
  menu->AddHlt("HLT_IsoEle18_L1R","L1_SingleEG15",1,1,"","",1.5,1,0,0,0,0);
  menu->AddHlt("HLT_IsoEle15_LW_L1I","L1_SingleIsoEG12",1,1,"","",1.5,1,0,0,0,0);
  menu->AddHlt("OpenHLT_Ele5_SW_L1R","L1_SingleEG5",1,100,"","",1.5,1,0,0,0,0); 
  menu->AddHlt("HLT_LooseIsoEle15_LW_L1R","L1_SingleEG12",1,5,"","",1.5,1,0,0,0,0);
  menu->AddHlt("HLT_Ele10_SW_L1R","L1_SingleEG8",1,1,"","",1.5,1,0,0,0,0);
  menu->AddHlt("HLT_Ele15_SW_L1R","L1_SingleEG12",5,1,"","",1.5,1,0,0,0,0);
  menu->AddHlt("HLT_Ele15_LW_L1R","L1_SingleEG10",1,1,"","",1.5,1,0,0,0,0);
  menu->AddHlt("HLT_EM80","L1_SingleEG15",1,1,"","",1.5,1,1,0,0,0);
  menu->AddHlt("HLT_EM200","L1_SingleEG15",1,1,"","",1.5,1,1,0,0,0);
  menu->AddHlt("HLT_DoubleIsoEle10_L1I","L1_DoubleIsoEG8",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT_DoubleIsoEle12_L1R","L1_DoubleEG10",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT_DoubleIsoEle10_LW_L1I","L1_DoubleIsoEG8",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT_DoubleIsoEle12_LW_L1R","L1_DoubleEG10",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT_DoubleEle5_SW_L1R","L1_DoubleEG5",1,5,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT_DoubleEle10_LW_OnlyPixelM_L1R","L1_DoubleEG5",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT_DoubleEle10_Z","L1_DoubleIsoEG8",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT_DoubleEle6_Exclusive","L1_ExclusiveDoubleIsoEG6",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT_IsoPhoton30_L1I","L1_SingleIsoEG12",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT_IsoPhoton10_L1R","L1_SingleEG8",1,5,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT_IsoPhoton15_L1R","L1_SingleEG12",5,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT_IsoPhoton20_L1R","L1_SingleEG15",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT_IsoPhoton25_L1R","L1_SingleEG15",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT_IsoPhoton40_L1R","L1_SingleEG15",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("OpenHLT_L1Photon5","L1_SingleEG5",1,500,"","",1.5,0,1,0,0,0); 
  menu->AddHlt("OpenHLT_Photon10_L1R","L1SingleEG8",1,100,"","",1.5,0,1,0,0,0); 
  menu->AddHlt("HLT_Photon15_L1R","L1_SingleEG10",5,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT_Photon25_L1R","L1_SingleEG15",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT_DoubleIsoPhoton20_L1I","L1_DoubleIsoEG8",1,1,"","",1.5,0,2,0,0,0);
  menu->AddHlt("HLT_DoubleIsoPhoton20_L1R","L1_DoubleEG10",1,1,"","",1.5,0,2,0,0,0);
  menu->AddHlt("HLT_DoublePhoton10_Exclusive","L1_ExclusiveDoubleIsoEG6",1,1,"","",1.5,0,2,0,0,0);
  menu->AddHlt("HLT_L1Mu","L1_SingleMu7 OR L1_DoubleMu3",1,20,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_L1MuOpen","L1_SingleMuOpen OR L1_SingleMu3 OR L1_SingleMu5",100,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_L2Mu9","L1_SingleMu7",1,20,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_IsoMu9","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_IsoMu11","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_IsoMu13","L1_SingleMu10",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_IsoMu15","L1_SingleMu10",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_NoTrackerIsoMu15","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu3","L1_SingleMu3",5,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu5","L1_SingleMu5",5,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu7","L1_SingleMu7",1,5,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu9","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu11","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu13","L1_SingleMu10",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu15","L1_SingleMu10",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu15_L1Mu7","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu15_Vtx2cm","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_Mu15_Vtx2mm","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT_DoubleIsoMu3","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu3","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu3_Vtx2cm","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu3_Vtx2mm","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu3_JPsi","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu3_Upsilon","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu7_Z","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu3_SameSign","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu3_Psi2S","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_BTagIP_Jet180","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_BTagIP_Jet120_Relaxed","L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_BTagIP_DoubleJet120","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_BTagIP_DoubleJet60_Relaxed","L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_BTagIP_TripleJet70","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,3,0);
  menu->AddHlt("HLT_BTagIP_TripleJet40_Relaxed","L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,3,0);
  menu->AddHlt("HLT_BTagIP_QuadJet40","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,4,0);
  menu->AddHlt("HLT_BTagIP_QuadJet30_Relaxed","L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,4,0);
  menu->AddHlt("HLT_BTagIP_HT470","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_BTagIP_HT320_Relaxed","L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_BTagMu_DoubleJet120","L1_Mu5_Jet15",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_BTagMu_DoubleJet60_Relaxed","L1_Mu5_Jet15",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT_BTagMu_TripleJet70","L1_Mu5_Jet15",1,1,"","",1.5,0,0,0,3,0);
  menu->AddHlt("HLT_BTagMu_TripleJet40_Relaxed","L1_Mu5_Jet15",1,1,"","",1.5,0,0,0,3,0);
  menu->AddHlt("HLT_BTagMu_QuadJet40","L1_Mu5_Jet15",1,1,"","",1.5,0,0,0,4,0);
  menu->AddHlt("HLT_BTagMu_QuadJet30_Relaxed","L1_Mu5_Jet15",1,1,"","",1.5,0,0,0,4,0);
  menu->AddHlt("HLT_BTagMu_HT370","L1_HTT300",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_BTagMu_HT250_Relaxed","L1_HTT200",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT_DoubleMu3_BJPsi","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_DoubleMu4_BJPsi","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT_TripleMu3_TauTo3Mu","L1_DoubleMu3",1,1,"","",1.5,0,0,3,0,0);
  menu->AddHlt("HLT_IsoTau_MET65_Trk20","L1_SingleTauJet80",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLT_IsoTau_MET35_Trk15_L1MET","L1_TauJet30_ETM30",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLT_LooseIsoTau_MET30","L1_SingleTauJet80",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLT_LooseIsoTau_MET30_L1MET","L1_TauJet30_ETM30",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLT_DoubleIsoTau_Trk3","L1_DoubleTauJet40",1,1,"","",1.5,0,0,0,1,0);
  //  menu->AddHlt("HLT_DoubleLooseIsoTau","L1_DoubleTauJet20",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("OpenHLT_DoubleLooseIsoTau","L1_DoubleTauJet40",1,1,"","",1.5,0,0,0,1,0); 
  menu->AddHlt("HLT_IsoEle8_IsoMu7","L1_Mu3_IsoEG5",1,1,"","",1.5,1,0,1,0,0);
  menu->AddHlt("HLT_IsoEle10_Mu10_L1R","L1_Mu3_EG12",1,1,"","",1.5,1,0,1,0,0);
  menu->AddHlt("HLT_IsoEle12_IsoTau_Trk3","L1_IsoEG10_TauJet20",1,1,"","",1.5,1,0,0,1,0);
  menu->AddHlt("HLT_IsoEle10_BTagIP_Jet35","L1_IsoEG10_Jet20",1,1,"","",1.5,1,0,0,1,0);
  menu->AddHlt("HLT_IsoEle12_Jet40","L1_IsoEG10_Jet30",1,1,"","",1.5,1,0,0,1,0);
  menu->AddHlt("HLT_IsoEle12_DoubleJet80","L1_IsoEG10_Jet30",1,1,"","",1.5,1,0,0,1,0);
  menu->AddHlt("HLT_IsoElec5_TripleJet30","L1_EG5_TripleJet15",1,1,"","",1.5,1,0,0,3,0);
  menu->AddHlt("HLT_IsoEle12_TripleJet60","L1_IsoEG10_Jet30",1,1,"","",1.5,1,0,0,3,0);
  menu->AddHlt("HLT_IsoEle12_QuadJet35","L1_IsoEG10_Jet30",1,1,"","",1.5,1,0,0,4,0);
  menu->AddHlt("HLT_IsoMu14_IsoTau_Trk3","L1_Mu5_TauJet20",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLT_IsoMu7_BTagIP_Jet35","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLT_IsoMu7_BTagMu_Jet20","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLT_IsoMu7_Jet40","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLT_NoL2IsoMu8_Jet40","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLT_Mu14_Jet50","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLT_Mu5_TripleJet30","L1_Mu3_TripleJet15",1,1,"","",1.5,0,0,1,3,0);
  menu->AddHlt("HLT_BTagMu_Jet20_Calib","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLT_ZeroBias","L1_ZeroBias",50000,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_MinBias","L1_MinBias_HTT10",8000,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_MinBiasHcal","L1_SingleJetCountsHFTow OR L1_DoubleJetCountsHFTow OR L1_SingleJetCountsHFRing0Sum3 OR L1_DoubleJetCountsHFRing0Sum3 OR L1_SingleJetCountsHFRing0Sum6 OR L1_DoubleJetCountsHFRing0Sum6",15000,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_MinBiasEcal","L1_SingleEG2 OR L1_DoubleEG1",15000,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_MinBiasPixel","L1_ZeroBias",50000,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_MinBiasPixel_Trk5","L1_ZeroBias",50000,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_BackwardBSC","38 OR 39",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_ForwardBSC","36 OR 37",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_CSCBeamHalo","L1_SingleMuBeamHalo",8,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_CSCBeamHaloOverlapRing1","L1_SingleMuBeamHalo",8,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_CSCBeamHaloOverlapRing2","L1_SingleMuBeamHalo",8,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_CSCBeamHaloRing2or3","L1_SingleMuBeamHalo",8,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT_TrackerCosmics","24 OR 25 OR 26 OR 27 OR 28",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("AlCa_IsoTrack","L1_SingleJet30 OR L1_SingleJet50 OR L1_SingleJet70 OR L1_SingleJet100 OR L1_SingleTauJet30 OR L1_SingleTauJet40 OR L1_SingleTauJet60 OR L1_SingleTauJet80",1,1,"","",0.214,0,0,0,0,0);
  menu->AddHlt("AlCa_EcalPhiSym","L1_ZeroBias OR L1_SingleJetCountsHFTow OR L1_DoubleJetCountsHFTow OR L1_SingleEG2 OR L1_DoubleEG1 OR L1_SingleJetCountsHFRing0Sum3 OR L1_DoubleJetCountsHFRing0Sum3 OR L1_SingleJetCountsHFRing0Sum6 OR L1_DoubleJetCountsHFRing0Sum6",1,1,"","",0.001,0,0,0,0,0);
  menu->AddHlt("AlCa_EcalPi0","L1_SingleIsoEG5 OR L1_SingleIsoEG8 OR L1_SingleIsoEG10 OR L1_SingleIsoEG12 OR L1_SingleIsoEG15 OR L1_SingleIsoEG20 OR L1_SingleIsoEG25 OR L1_SingleEG5 OR L1_SingleEG8 OR L1_SingleEG10 OR L1_SingleEG12 OR L1_SingleEG15 OR L1_SingleEG20 OR L1_SingleEG25",1,1,"","",0.007,0,0,0,0,0);

  cout <<" End Booking menu "<<endl;

}

void BookMenu_Default(OHltMenu*  menu, double &iLumi, double &nBunches) {

  //  iLumi = 2E30; 
  iLumi = 8E29;
  nBunches = 43;
  //objects electron,photon,muon,jets,met

  menu->AddHlt("HLT1jet","L1_SingleJet150",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT2jet","L1_SingleJet150 OR L1_DoubleJet70",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT3jet","L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50",1,1,"","",1.5,0,0,0,3,0);
  menu->AddHlt("HLT4jet","L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet40",1,1,"","",1.5,0,0,0,4,0);
  menu->AddHlt("HLT1MET","L1_ETM40",1,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT2jetAco","L1_SingleJet150 OR L1_DoubleJet70",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT1jet1METAco","L1_SingleJet150",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLT1jet1MET","L1_SingleJet150",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLT2jet1MET","L1_SingleJet150",1,1,"","",1.5,0,0,0,2,1);
  menu->AddHlt("HLT3jet1MET","L1_SingleJet150",1,1,"","",1.5,0,0,0,3,1);
  menu->AddHlt("HLT4jet1MET","L1_SingleJet150",1,1,"","",1.5,0,0,0,4,1);
  menu->AddHlt("HLT1MET1HT","L1_HTT300",1,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("CandHLT1SumET","L1_ETT60",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLT1jetPE1","L1_SingleJet100",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT1jetPE3","L1_SingleJet70",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT1jetPE5","L1_SingleJet30",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT1jetPE7","L1_SingleJet15",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT1METPre1","L1_ETM40",1,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT1METPre2","L1_ETM15",1,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT1METPre3","L1_ETM10",1,1,"","",1.5,0,0,0,0,1);
  menu->AddHlt("HLT2jetAve30","L1_SingleJet15",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT2jetAve60","L1_SingleJet30",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT2jetAve110","L1_SingleJet70",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT2jetAve150","L1_SingleJet100",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT2jetAve200","L1_SingleJet150",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLT2jetvbfMET","L1_ETM40",1,1,"","",1.5,0,0,0,2,1);
  menu->AddHlt("HLTS2jet1METNV","L1_SingleJet150",1,1,"","",1.5,0,0,0,2,1);
  menu->AddHlt("HLTS2jet1METAco","L1_SingleJet150",1,1,"","",1.5,0,0,0,2,1);
  menu->AddHlt("HLTSjet1MET1Aco","L1_SingleJet150",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLTSjet2MET1Aco","L1_SingleJet150",1,1,"","",1.5,0,0,0,2,1);
  menu->AddHlt("HLTS2jetMET1Aco","L1_SingleJet150",1,1,"","",1.5,0,0,0,2,1);
  menu->AddHlt("HLTJetMETRapidityGap","L1_IsoEG10_Jet15_ForJet10",1,1,"","",1.5,0,0,0,1,1);
  menu->AddHlt("HLT1Electron","L1_SingleIsoEG12",1,1,"","",1.5,1,0,0,0,0);
  menu->AddHlt("HLT1ElectronRelaxed","L1_SingleEG15",1,1,"","",1.5,1,0,0,0,0);
  menu->AddHlt("HLT2Electron","L1_DoubleIsoEG8",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT2ElectronRelaxed","L1_DoubleEG10",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT1Photon","L1_SingleIsoEG12",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT1PhotonRelaxed","L1_SingleEG15",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT2Photon","L1_DoubleIsoEG8",1,1,"","",1.5,0,2,0,0,0);
  menu->AddHlt("HLT2PhotonRelaxed","L1_DoubleEG10",1,1,"","",1.5,0,2,0,0,0);
  menu->AddHlt("HLT1EMHighEt","L1_SingleEG15",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT1EMVeryHighEt","L1_SingleEG15",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("HLT2ElectronZCounter","L1_DoubleIsoEG8",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT2ElectronExclusive","L1_ExclusiveDoubleIsoEG6",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT2PhotonExclusive","L1_ExclusiveDoubleIsoEG6",1,1,"","",1.5,0,2,0,0,0);
  menu->AddHlt("HLT1PhotonL1Isolated","L1_SingleIsoEG10",1,1,"","",1.5,0,1,0,0,0);
  menu->AddHlt("CandHLT1ElectronStartup","L1_SingleIsoEG12",1,1,"","",1.5,1,0,0,0,0);
  menu->AddHlt("CandHLT1ElectronRelaxedStartup","L1_SingleEG15",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("CandHLT2ElectronStartup","L1_DoubleIsoEG8",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("CandHLT2ElectronRelaxedStartup","L1_DoubleEG10",1,1,"","",1.5,2,0,0,0,0);
  menu->AddHlt("HLT1MuonIso","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT1MuonNonIso","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT2MuonIso","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT2MuonNonIso","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT2MuonJPsi","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT2MuonUpsilon","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT2MuonZ","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLTNMuonNonIso","L1_TripleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT2MuonSameSign","L1_DoubleMu3",1,1,"","",1.5,0,0,2,0,0);
  menu->AddHlt("HLT1MuonPrescalePt3","L1_SingleMu3",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT1MuonPrescalePt5","L1_SingleMu5",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT1MuonPrescalePt7x7","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT1MuonPrescalePt7x10","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLT1MuonLevel1","L1_SingleMu3 OR L1_SingleMu5 OR L1_SingleMu7 OR L1_DoubleMu3",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("CandHLT1MuonPrescaleVtx2cm","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("CandHLT1MuonPrescaleVtx2mm","L1_SingleMu7",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("CandHLT2MuonPrescaleVtx2cm","L1_DoubleMu3",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("CandHLT2MuonPrescaleVtx2mm","L1_DoubleMu3",1,1,"","",1.5,0,0,1,0,0);
  menu->AddHlt("HLTB1Jet","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet40 OR L1_HTT300",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLTB2Jet","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet40 OR L1_HTT300",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLTB3Jet","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet40 OR L1_HTT300",1,1,"","",1.5,0,0,0,3,0);
  menu->AddHlt("HLTB4Jet","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet40 OR L1_HTT300",1,1,"","",1.5,0,0,0,4,0);
  menu->AddHlt("HLTBHT","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet40 OR L1_HTT300",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLTB1JetMu","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLTB2JetMu","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,2,0);
  menu->AddHlt("HLTB3JetMu","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,3,0);
  menu->AddHlt("HLTB4JetMu","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,4,0);
  menu->AddHlt("HLTBHTMu","L1_HTT300",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLTBJPsiMuMu","L1_DoubleMu3",1,1,"","",1.5,0,0,2,1,0);
  menu->AddHlt("HLT1Tau","L1_SingleTauJet80",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT1Tau1MET","L1_TauJet30_ETM30",1,1,"","",1.5,0,0,0,1,0);
  menu->AddHlt("HLT2TauPixel","L1_DoubleTauJet40",1,1,"","",1.5,0,0,0,2,0);
  menu->AddHlt("HLTXElectronBJet","L1_IsoEG10_Jet20",1,1,"","",1.5,1,0,0,1,0);
  menu->AddHlt("HLTXMuonBJet","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLTXMuonBJetSoftMuon","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLTXElectron1Jet","L1_IsoEG10_Jet30",1,1,"","",1.5,1,0,0,1,0);
  menu->AddHlt("HLTXElectron2Jet","L1_IsoEG10_Jet30",1,1,"","",1.5,1,0,0,2,0);
  menu->AddHlt("HLTXElectron3Jet","L1_IsoEG10_Jet30",1,1,"","",1.5,1,0,0,3,0);
  menu->AddHlt("HLTXElectron4Jet","L1_IsoEG10_Jet30",1,1,"","",1.5,1,0,0,4,0);
  menu->AddHlt("HLTXMuonJets","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("CandHLTXMuonNoL2IsoJets","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("CandHLTXMuonNoIsoJets","L1_Mu5_Jet15",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("HLTXElectronMuon","L1_Mu3_IsoEG5",1,1,"","",1.5,1,0,1,0,0);
  menu->AddHlt("HLTXElectronMuonRelaxed","L1_Mu3_EG12",1,1,"","",1.5,1,0,1,0,0);
  menu->AddHlt("HLTXElectronTau","L1_IsoEG10_TauJet20",1,1,"","",1.5,1,0,0,1,0);
  menu->AddHlt("CandHLTXElectronTauPixel","L1_IsoEG10_TauJet20",1,1,"","",1.5,1,0,0,1,0);
  menu->AddHlt("HLTXMuonTau","L1_Mu5_TauJet20",1,1,"","",1.5,0,0,1,1,0);
  menu->AddHlt("CandHLTEcalPi0","L1_SingleJet15 OR L1_SingleJet20 OR L1_SingleJet30 OR L1_SingleJet50 OR L1_SingleJet70 OR L1_SingleJet100 OR L1_SingleJet150 OR L1_SingleJet200 OR L1_DoubleJet70 OR L1_DoubleJet100",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("CandHLTEcalPhiSym","L1_ZeroBias",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("CandHLTHcalPhiSym","L1_ZeroBias",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLTHcalIsolatedTrack","L1_SingleJet100 OR L1_SingleTauJet100",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("CandHLTHcalIsolatedTrackNoEcalIsol","L1_SingleJet100 OR L1_SingleTauJet100",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLTMinBiasPixel","L1_ZeroBias",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("CandHLTMinBiasForAlignment","L1_ZeroBias",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLTMinBias","L1_MinBias_HTT10",1,1,"","",1.5,0,0,0,0,0);
  menu->AddHlt("HLTZeroBias","L1_ZeroBias",1,1,"","",1.5,0,0,0,0,0);
}



void BookMenu_OhltExample(OHltMenu*  menu,double &iLumi,double &nBunches) {

  iLumi = 2.E31;
  nBunches = 156;

  //
  menu->AddHlt("HLT1Electron","L1_SingleIsoEG12",1,1,"15","1e32",1.5,1,0,0,0,0); 
  menu->AddHlt("OpenHLT1Electron","L1_SingleIsoEG12",1,1,"15","1e32",1.5,1,0,0,0,0);
  
  //
  menu->AddHlt("HLT1Photon","L1_SingleIsoEG12",1,1,"30","1e32",1.5,0,1,0,0,0); 
  menu->AddHlt("OpenHLT1Photon","L1_SingleIsoEG12",1,1,"30","1e32",1.5,0,1,0,0,0); 

  //
  //  menu->AddHlt("HLT2TauPixel","L1_TauJet40",1,1,"15","1e32"); 
  //  menu->AddHlt("OpenHLT2TauPixel","L1_TauJet40",1,1,"15","1e32"); 

  //
  menu->AddHlt("HLT1MuonNonIso","L1_SingleMu7",1,1,"16","1e32",1.5,0,0,1,0,0);
  menu->AddHlt("OpenHLT1MuonNonIso","L1_SingleMu7",1,1,"16","1e32",1.5,0,0,1,0,0);
  menu->AddHlt("HLT1MuonIso","L1_SingleMu7",1,1,"11","1e32",1.5,0,0,1,0,0);
  menu->AddHlt("OpenHLT1MuonIso","L1_SingleMu7",1,1,"11","1e32",1.5,0,0,1,0,0);
  menu->AddHlt("HLT1MuonL1Open","L1_SingleMuOpen",1,1,"-","new",1.5,0,0,1,0,0);  // L1: 150 
}

void BookMenu_L1Default(OHltMenu*  menu, double &iLumi, double &nBunches) {   
   
  iLumi = 2E30;   
  nBunches = 43;   
 
  menu->AddHlt("L1_SingleMuOpen","L1_SingleMuOpen",1,1,"-","1e32",1.,0,0,1,0,0);
  menu->AddHlt("L1_SingleMu3","L1_SingleMu3",1,1,"-","1e32",1.,0,0,1,0,0);
  menu->AddHlt("L1_SingleMu5","L1_SingleMu5",1,1,"-","1e32",1.,0,0,1,0,0);            
  menu->AddHlt("L1_SingleMu7", "L1_SingleMu7",1,1,"-","1e32",1.,0,0,1,0,0);            
  menu->AddHlt("L1_SingleMu10", "L1_SingleMu10",1,1,"-","1e32",1.,0,0,1,0,0);           
  menu->AddHlt("L1_SingleMuBeamHalo", "L1_SingleMuBeamHalo",1,1,"-","1e32",1.,0,0,2,0,0);
  menu->AddHlt("L1_DoubleMu3", "L1_DoubleMu3",1,1,"-","1e32",1.,0,0,2,0,0);            
  menu->AddHlt("L1_TripleMu3", "L1_TripleMu3",1,1,"-","1e32",1.,0,0,3,0,0);            

  menu->AddHlt("L1_SingleIsoEG10", "L1_SingleIsoEG10",1,1,"-","1e32",1.,1,0,0,0,0);      
  menu->AddHlt("L1_SingleIsoEG12", "L1_SingleIsoEG12",1,1,"-","1e32",1.,2,0,0,0,0);      
  menu->AddHlt("L1_DoubleIsoEG8", "L1_DoubleIsoEG8",1,1,"-","1e32",1.,2,0,0,0,0);        
  
  menu->AddHlt("L1_SingleEG2", "L1_SingleEG2",1,1,"-","1e32",1.,1,1,0,0,0);           
  menu->AddHlt("L1_SingleEG5", "L1_SingleEG5",1,1,"-","1e32",1.,1,1,0,0,0);            
  //  menu->AddHlt("L1_SingleEG8", "L1_SingleEG8",1,1,"-","1e32",1.,1,1,0,0,0);            
  menu->AddHlt("L1_SingleEG8", "L1_SingleEG8",1,1,"-","1e32",1.,1,1,0,0,0);
  menu->AddHlt("L1_SingleEG10", "L1_SingleEG10",1,1,"-","1e32",1.,1,1,0,0,0);           
  menu->AddHlt("L1_SingleEG12", "L1_SingleEG12",1,1,"-","1e32",1.,1,1,0,0,0);          
  menu->AddHlt("L1_SingleEG15", "L1_SingleEG15",1,1,"-","1e32",1.,1,1,0,0,0);         
  menu->AddHlt("L1_DoubleEG1", "L1_DoubleEG1",1,1,"-","1e32",1.,2,2,0,0,0);           
  menu->AddHlt("L1_DoubleEG5", "L1_DoubleEG5",1,1,"-","1e32",1.,2,2,0,0,0);            
  menu->AddHlt("L1_DoubleEG10", "L1_DoubleEG10",1,1,"-","1e32",1.,2,2,0,0,0);           
  
  menu->AddHlt("L1_SingleJet15", "L1_SingleJet15",1,1,"-","1e32",1.,0,0,0,1,0); 
  //  menu->AddHlt("L1_SingleJet15", "L1_SingleJet15",1,1,"-","1e32",1.,0,0,0,1,0);  
  menu->AddHlt("L1_SingleJet30", "L1_SingleJet30",1,1,"-","1e32",1.,0,0,0,1,0);
  //  menu->AddHlt("L1_SingleJet30", "L1_SingleJet30",1,1,"-","1e32",1.,0,0,0,1,0);
  menu->AddHlt("L1_SingleJet50", "L1_SingleJet50",1,1,"-","1e32",1.,0,0,0,1,0); 
  menu->AddHlt("L1_SingleJet70", "L1_SingleJet70",1,1,"-","1e32",1.,0,0,0,1,0); 
  menu->AddHlt("L1_SingleJet100", "L1_SingleJet100",1,1,"-","1e32",1.,0,0,0,1,0);
  menu->AddHlt("L1_SingleJet150", "L1_SingleJet150",1,1,"-","1e32",1.,0,0,0,1,0);
  menu->AddHlt("L1_SingleJet200", "L1_SingleJet200",1,1,"-","1e32",1.,0,0,0,1,0);
  menu->AddHlt("L1_DoubleJet70", "L1_DoubleJet70",1,1,"-","1e32",1.,0,0,0,2,0);  
  menu->AddHlt("L1_DoubleJet100", "L1_DoubleJet100",1,1,"-","1e32",1.,0,0,0,2,0);
  menu->AddHlt("L1_TripleJet50", "L1_TripleJet50",1,1,"-","1e32",1.,0,0,0,3,0);  
  menu->AddHlt("L1_QuadJet15", "L1_QuadJet15",1,1,"-","1e32",1.,0,0,0,4,0); 
  menu->AddHlt("L1_QuadJet30", "L1_QuadJet30",1,1,"-","1e32",1.,0,0,0,4,0);  
  menu->AddHlt("L1_HTT200", "L1_HTT200",1,1,"-","1e32",1.,0,0,0,0,0);        
  menu->AddHlt("L1_HTT300", "L1_HTT300",1,1,"-","1e32",1.,0,0,0,0,0);        
  
  menu->AddHlt("L1_ETM20", "L1_ETM20",1,1,"-","1e32",1.,0,0,0,0,0);
  menu->AddHlt("L1_ETM30", "L1_ETM30",1,1,"-","1e32",1.,0,0,0,0,0);  
  menu->AddHlt("L1_ETM40", "L1_ETM40",1,1,"-","1e32",1.,0,0,0,0,0);  
  menu->AddHlt("L1_ETM50", "L1_ETM50",1,1,"-","1e32",1.,0,0,0,0,0);  
  menu->AddHlt("L1_ETT60", "L1_ETT60",1,1,"-","1e32",1.,0,0,0,0,0);
  
  menu->AddHlt("L1_SingleTauJet30", "L1_SingleTauJet30",1,1,"-","1e32",1.,0,0,0,1,0);               
  menu->AddHlt("L1_SingleTauJet40", "L1_SingleTauJet40",1,1,"-","1e32",1.,0,0,0,1,0); 
  menu->AddHlt("L1_SingleTauJet60", "L1_SingleTauJet60",1,1,"-","1e32",1.,0,0,0,1,0);   
  menu->AddHlt("L1_SingleTauJet80", "L1_SingleTauJet80",1,1,"-","1e32",1.,0,0,0,1,0);   
  menu->AddHlt("L1_DoubleTauJet20", "L1_DoubleTauJet20",1,1,"-","1e32",1.,0,0,0,1,0);   
  menu->AddHlt("L1_DoubleTauJet40", "L1_DoubleTauJet40",1,1,"-","1e32",1.,0,0,0,1,0);   
  
  menu->AddHlt("L1_IsoEG10_Jet15_ForJet10", "L1_IsoEG10_Jet15_ForJet10",1,1,"-","1e32",1.,1,0,0,2,0); 
  menu->AddHlt("L1_ExclusiveDoubleIsoEG6", "L1_ExclusiveDoubleIsoEG6",1,1,"-","1e32",1.,2,0,0,0,0);   
  menu->AddHlt("L1_Mu5_Jet15", "L1_Mu5_Jet15",1,1,"-","1e32",1.,0,0,1,1,0);    
  menu->AddHlt("L1_IsoEG10_Jet20", "L1_IsoEG10_Jet20",1,1,"-","1e32",1.,1,0,0,1,0); 
  menu->AddHlt("L1_IsoEG10_Jet30", "L1_IsoEG10_Jet30",1,1,"-","1e32",1.,1,0,0,1,0); 
  menu->AddHlt("L1_Mu3_IsoEG5", "L1_Mu3_IsoEG5",1,1,"-","1e32",1.,0,0,1,1,0);  
  menu->AddHlt("L1_Mu3_EG12", "L1_Mu3_EG12",1,1,"-","1e32",1.,1,1,1,0,0);    
  menu->AddHlt("L1_IsoEG10_TauJet20", "L1_IsoEG10_TauJet20",1,1,"-","1e32",1.,1,0,0,1,0); 
  menu->AddHlt("L1_Mu5_TauJet20", "L1_Mu5_TauJet20",1,1,"-","1e32",1.,0,0,1,1,0); 
  menu->AddHlt("L1_TauJet30_ETM30", "L1_TauJet30_ETM30",1,1,"-","1e32",1.,0,0,0,1,0); 
  menu->AddHlt("L1_EG5_TripleJet15", "L1_EG5_TripleJet15",1,1,"-","1e32",1.,1,0,0,3,0);
  menu->AddHlt("L1_Mu3_TripleJet15", "L1_Mu3_TripleJet15",1,1,"-","1e32",1.,0,0,1,3,0);
  
  menu->AddHlt("L1_ZeroBias", "L1_ZeroBias",1,1,"-","1e32",1.,0,0,0,0,0);  
  menu->AddHlt("L1_MinBias_HTT10", "L1_MinBias_HTT10",1,1,"-","1e32",1.,0,0,0,0,0);  
  //  menu->AddHlt("L1_ZeroBias", "L1_ZeroBias",1,1,"-","1e32",1.,0,0,0,0,0); 
  //  menu->AddHlt("L1_MinBias_HTT10", "L1_MinBias_HTT10",1,1,"-","1e32",1.,0,0,0,0,0); 
  menu->AddHlt("L1_SingleJetCountsHFTow 12","L1_SingleJetCountsHFTow 12",1,1,"-","1e32",1.,0,0,0,0,0);
  menu->AddHlt("L1_DoubleJetCountsHFTow 10","L1_DoubleJetCountsHFTow 10",1,1,"-","1e32",1.,0,0,0,0,0);
  menu->AddHlt("L1_SingleJetCountsHFRing0Sum3", "L1_SingleJetCountsHFRing0Sum3",1,1,"-","1e32",1.,0,0,0,0,0); 
  menu->AddHlt("L1_DoubleJetCountsHFRing0Sum3", "L1_DoubleJetCountsHFRing0Sum3",1,1,"-","1e32",1.,0,0,0,0,0);  
  menu->AddHlt("L1_SingleJetCountsHFRing0Sum6", "L1_SingleJetCountsHFRing0Sum6",1,1,"-","1e32",1.,0,0,0,0,0);  
  menu->AddHlt("L1_DoubleJetCountsHFRing0Sum6", "L1_DoubleJetCountsHFRing0Sum6",1,1,"-","1e32",1.,0,0,0,0,0);  

}


void BookEffHistos (OHltMenu*  menu, vector<string> ObjectsToUse, int &MaxMult,int ip
		    ,std::vector <TH1F*> &Num_pt, std::vector <TH1F*> &Num_eta, std::vector <TH1F*> &Num_phi
		    ,std::vector <TH1F*> &Den_pt, std::vector <TH1F*> &Den_eta, std::vector <TH1F*> &Den_phi
		    ,std::vector <TH1F*> &Eff_pt, std::vector <TH1F*> &Eff_eta, std::vector <TH1F*> &Eff_phi
		    ,std::vector <TH1F*> &DenwrtL1_pt, std::vector <TH1F*> &DenwrtL1_eta, std::vector <TH1F*> &DenwrtL1_phi
		    ,std::vector <TH1F*> &EffwrtL1_pt, std::vector <TH1F*> &EffwrtL1_eta, std::vector <TH1F*> &EffwrtL1_phi
		    )
{
  vector<TString> trignames = menu->GetHlts(); 
  int Ntrig = (int) trignames.size();
  //  map<TString,int> map_TrigPrescls = menu->GetTotalPrescaleMap(); 
  //  map<TString,int> map_L1Prescls = menu->GetL1PrescaleMap(); 
  //  map<TString,int> map_HLTPrescls = menu->GetHltPrescaleMap(); 
  //  map<TString,TString> map_HltDesc = menu->GetHltDescriptionMap(); 

  map<TString,TString> map_L1Bits = menu->GetHltL1BitMap();
  map<TString,int> map_MultEle = menu->GetMultEleMap(); 
  map<TString,int> map_MultPho = menu->GetMultPhoMap();
  map<TString,int> map_MultMu = menu->GetMultMuMap(); 
  map<TString,int> map_MultJets = menu->GetMultJetsMap(); 
  map<TString,int> map_MultMET = menu->GetMultMETMap(); 

  int nbins_pt=500;
  float xmin_pt=0;
  float xmax_pt=500;
  int nbins_eta=50;
  float xmin_eta=-5;
  float xmax_eta=5;
  int nbins_phi=50;
  float xmin_phi=-5;
  float xmax_phi=5;

  int multele=-1;
  int multpho=-1;
  int multmu=-1;
  int multjets=-1;
  int multmet=-1;

  char chname[256];
  char chtitle[256];

  MaxMult=0;
  for (int it = 0; it < Ntrig; it++){
    multele = map_MultEle.find(trignames[it])->second;
    multpho = map_MultPho.find(trignames[it])->second;
    multmu = map_MultMu.find(trignames[it])->second;
    multjets = map_MultJets.find(trignames[it])->second;
    multmet = map_MultMET.find(trignames[it])->second;
    const char* chTrigName=(trignames[it]);

    if(multele>MaxMult)MaxMult=multele;
    if(multpho>MaxMult)MaxMult=multpho;
    if(multmu>MaxMult)MaxMult=multmu;
    if(multjets>MaxMult)MaxMult=multjets;
    if(multmet>MaxMult)MaxMult=multmet;


    int NObjects = (int)ObjectsToUse.size();
    int multObject=-1;

    for(int i=0; i<NObjects;i++){
      if (ObjectsToUse[i].compare("Electron")==0){
	multObject=multele;
      }
      if (ObjectsToUse[i].compare("Photon")==0){
	multObject=multpho;
      }
      if (ObjectsToUse[i].compare("Muon")==0){
	multObject=multmu;
      }
      if (ObjectsToUse[i].compare("Jet")==0){
	multObject=multjets;
      }
      if (ObjectsToUse[i].compare("Met")==0){
	multObject=multmet;
      }
      for (int imult=0;imult<multObject;imult++){

	//Pt
	snprintf(chname,255,"Num_pt_%d_%s_%d_%s",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	snprintf(chtitle,255,"Triggered Events vs %d %s Pt (GeV) HLT trigger bit %d (%s) ",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	Num_pt.push_back(new TH1F(chname, chtitle, nbins_pt, xmin_pt, xmax_pt));
	snprintf(chname,255,"Den_pt_%d_%s_%d_%s",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	snprintf(chtitle,255,"All Events vs %d %s Pt (GeV) HLT trigger bit %d (%s)",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	Den_pt.push_back(new TH1F(chname, chtitle, nbins_pt, xmin_pt, xmax_pt));
	snprintf(chname,255,"Eff_pt_%d_%s_%d_%s",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	snprintf(chtitle,255,"Eff vs %d %s Pt (GeV) HLT trigger bit %d (%s)",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	Eff_pt.push_back(new TH1F( chname, chtitle, nbins_pt, xmin_pt, xmax_pt) );

	snprintf(chname,255,"DenwrtL1_pt_%d_%s_%d_%s",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	snprintf(chtitle,255,"All Events out L1 vs %d %s Pt (GeV) HLT trigger bit %d (%s)",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	DenwrtL1_pt.push_back(new TH1F(chname, chtitle, nbins_pt, xmin_pt, xmax_pt));
	snprintf(chname,255,"EffwrtL1_pt_%d_%s_%d_%s",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	snprintf(chtitle,255,"Eff wrt L1 vs %d %s Pt (GeV) HLT trigger bit %d (%s)",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	EffwrtL1_pt.push_back(new TH1F( chname, chtitle, nbins_pt, xmin_pt, xmax_pt) );

	//Eta
	snprintf(chname,255,"Num_eta_%d_%s_%d_%s",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	snprintf(chtitle,255,"Triggered Events vs %d %s Eta (GeV) HLT trigger bit %d (%s) ",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	Num_eta.push_back(new TH1F(chname, chtitle, nbins_eta, xmin_eta, xmax_eta));
	snprintf(chname,255,"Den_eta_%d_%s_%d_%s",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	snprintf(chtitle,255,"All Events vs %d %s Eta (GeV) HLT trigger bit %d (%s)",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	Den_eta.push_back(new TH1F(chname, chtitle, nbins_eta, xmin_eta, xmax_eta));
	snprintf(chname,255,"Eff_eta_%d_%s_%d_%s",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	snprintf(chtitle,255,"Eff vs %d %s Eta (GeV) HLT trigger bit %d (%s)",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	Eff_eta.push_back(new TH1F( chname, chtitle, nbins_eta, xmin_eta, xmax_eta) );

	snprintf(chname,255,"DenwrtL1_eta_%d_%s_%d_%s",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	snprintf(chtitle,255,"All Events out L1 vs %d %s Eta (GeV) HLT trigger bit %d (%s)",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	DenwrtL1_eta.push_back(new TH1F(chname, chtitle, nbins_eta, xmin_eta, xmax_eta));
	snprintf(chname,255,"EffwrtL1_eta_%d_%s_%d_%s",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	snprintf(chtitle,255,"Eff wrt L1 vs %d %s Eta (GeV) HLT trigger bit %d (%s)",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	EffwrtL1_eta.push_back(new TH1F( chname, chtitle, nbins_eta, xmin_eta, xmax_eta) );

	//Phi
	snprintf(chname,255,"Num_phi_%d_%s_%d_%s",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	snprintf(chtitle,255,"Triggered Events vs %d %s Phi (GeV) HLT trigger bit %d (%s) ",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	Num_phi.push_back(new TH1F(chname, chtitle, nbins_phi, xmin_phi, xmax_phi));
	snprintf(chname,255,"Den_phi_%d_%s_%d_%s",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	snprintf(chtitle,255,"All Events vs %d %s Phi (GeV) HLT trigger bit %d (%s)",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	Den_phi.push_back(new TH1F(chname, chtitle, nbins_phi, xmin_phi, xmax_phi));
	snprintf(chname,255,"Eff_phi_%d_%s_%d_%s",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	snprintf(chtitle,255,"Eff vs %d %s Phi (GeV) HLT trigger bit %d (%s)",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	Eff_phi.push_back(new TH1F( chname, chtitle, nbins_phi, xmin_phi, xmax_phi) );

	snprintf(chname,255,"DenwrtL1_phi_%d_%s_%d_%s",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	snprintf(chtitle,255,"All Events out L1 vs %d %s Phi (GeV) HLT trigger bit %d (%s)",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	DenwrtL1_phi.push_back(new TH1F(chname, chtitle, nbins_phi, xmin_phi, xmax_phi));
	snprintf(chname,255,"EffwrtL1_phi_%d_%s_%d_%s",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	snprintf(chtitle,255,"Eff wrt L1 vs %d %s Phi (GeV) HLT trigger bit %d (%s)",it,chTrigName,imult+1,ObjectsToUse[i].c_str());
	EffwrtL1_phi.push_back(new TH1F( chname, chtitle, nbins_phi, xmin_phi, xmax_phi) );

      }
    }
  } 
}

void FillWriteEffHistos ( OHltMenu*  menu, vector<string> ObjectsToUse, int ip
			 ,std::vector <TH1F*> &Num_pt, std::vector <TH1F*> &Num_eta, std::vector <TH1F*> &Num_phi
			 ,std::vector <TH1F*> &Den_pt, std::vector <TH1F*> &Den_eta, std::vector <TH1F*> &Den_phi
			 ,std::vector <TH1F*> &Eff_pt, std::vector <TH1F*> &Eff_eta, std::vector <TH1F*> &Eff_phi
			 ,std::vector <TH1F*> &DenwrtL1_pt, std::vector <TH1F*> &DenwrtL1_eta, std::vector <TH1F*> &DenwrtL1_phi
			 ,std::vector <TH1F*> &EffwrtL1_pt, std::vector <TH1F*> &EffwrtL1_eta, std::vector <TH1F*> &EffwrtL1_phi
			 )
{
  vector<TString> trignames = menu->GetHlts(); 
  int Ntrig = (int) trignames.size();
  //  map<TString,int> map_TrigPrescls = menu->GetTotalPrescaleMap(); 
  //  map<TString,int> map_L1Prescls = menu->GetL1PrescaleMap(); 
  //  map<TString,int> map_HLTPrescls = menu->GetHltPrescaleMap(); 
  //  map<TString,TString> map_HltDesc = menu->GetHltDescriptionMap(); 

  map<TString,TString> map_L1Bits = menu->GetHltL1BitMap();
  map<TString,int> map_MultEle = menu->GetMultEleMap(); 
  map<TString,int> map_MultPho = menu->GetMultPhoMap();
  map<TString,int> map_MultMu = menu->GetMultMuMap(); 
  map<TString,int> map_MultJets = menu->GetMultJetsMap(); 
  map<TString,int> map_MultMET = menu->GetMultMETMap(); 

  int multele=-1;
  int multpho=-1;
  int multmu=-1;
  int multjets=-1;
  int multmet=-1;

  int CountObjects=0;
  int numbhist=Num_pt.size()/(ip+1);
  for (int it = 0; it < Ntrig; it++){
    multele = map_MultEle.find(trignames[it])->second;
    multpho = map_MultPho.find(trignames[it])->second;
    multmu = map_MultMu.find(trignames[it])->second;
    multjets = map_MultJets.find(trignames[it])->second;
    multmet = map_MultMET.find(trignames[it])->second;

    int NObjects = (int)ObjectsToUse.size();
    int multObject=-1;

    for(int i=0; i<NObjects;i++){
      if (ObjectsToUse[i].compare("Electron")==0){
	multObject=multele;
      }
      if (ObjectsToUse[i].compare("Photon")==0){
	multObject=multpho;
      }
      if (ObjectsToUse[i].compare("Muon")==0){
	multObject=multmu;
      }
      if (ObjectsToUse[i].compare("Jet")==0){
	multObject=multjets;
      }
      if (ObjectsToUse[i].compare("Met")==0){
	multObject=multmet;
      }
      for (int imult=0;imult<multObject;imult++){
        Eff_pt[CountObjects+imult+ip*numbhist]->Divide(Num_pt[CountObjects+imult+ip*numbhist],Den_pt[CountObjects+imult+ip*numbhist],1.,1.);
        Eff_eta[CountObjects+imult+ip*numbhist]->Divide(Num_eta[CountObjects+imult+ip*numbhist],Den_eta[CountObjects+imult+ip*numbhist],1.,1.);
        Eff_phi[CountObjects+imult+ip*numbhist]->Divide(Num_phi[CountObjects+imult+ip*numbhist],Den_phi[CountObjects+imult+ip*numbhist],1.,1.);
        EffwrtL1_pt[CountObjects+imult+ip*numbhist]->Divide(Num_pt[CountObjects+imult+ip*numbhist],DenwrtL1_pt[CountObjects+imult+ip*numbhist],1.,1.);
        EffwrtL1_eta[CountObjects+imult+ip*numbhist]->Divide(Num_eta[CountObjects+imult+ip*numbhist],DenwrtL1_eta[CountObjects+imult+ip*numbhist],1.,1.);
        EffwrtL1_phi[CountObjects+imult+ip*numbhist]->Divide(Num_phi[CountObjects+imult+ip*numbhist],DenwrtL1_phi[CountObjects+imult+ip*numbhist],1.,1.);
	int nbins_pt=Num_pt[ip*numbhist]->GetNbinsX();
        for ( int i=1; i<=nbins_pt; i++ ) {
          double a = Num_pt[CountObjects+imult+ip*numbhist]->GetBinContent( i );
          double n = Den_pt[CountObjects+imult+ip*numbhist]->GetBinContent( i );
          double nl1 = DenwrtL1_pt[CountObjects+imult+ip*numbhist]->GetBinContent( i );
          if ( n != 0. )
            Eff_pt[CountObjects+imult+ip*numbhist]->SetBinError( i, sqrt( 1./n * a/n * ( 1. - a/n ) ) );
          if ( nl1 != 0. )
            EffwrtL1_pt[CountObjects+imult+ip*numbhist]->SetBinError( i, sqrt( 1./nl1 * a/nl1 * ( 1. - a/nl1 ) ) );
        }
	int nbins_eta=Num_eta[ip*numbhist]->GetNbinsX();
	for ( int i=1; i<=nbins_eta; i++ ) {
          double a = Num_eta[CountObjects+imult+ip*numbhist]->GetBinContent( i );
          double n = Den_eta[CountObjects+imult+ip*numbhist]->GetBinContent( i );
          double nl1 = DenwrtL1_eta[CountObjects+imult+ip*numbhist]->GetBinContent( i );
          if ( n != 0. )
            Eff_eta[CountObjects+imult+ip*numbhist]->SetBinError( i, sqrt( 1./n * a/n * ( 1. - a/n ) ) );
          if ( nl1 != 0. )
            EffwrtL1_eta[CountObjects+imult+ip*numbhist]->SetBinError( i, sqrt( 1./nl1 * a/nl1 * ( 1. - a/nl1 ) ) );
        }
	int nbins_phi=Num_phi[ip*numbhist]->GetNbinsX();
        for ( int i=1; i<=nbins_phi; i++ ) {
          double a = Num_phi[CountObjects+imult+ip*numbhist]->GetBinContent( i );
          double n = Den_phi[CountObjects+imult+ip*numbhist]->GetBinContent( i );
          double nl1 = DenwrtL1_phi[CountObjects+imult+ip*numbhist]->GetBinContent( i );
          if ( n != 0. )
            Eff_phi[CountObjects+imult+ip*numbhist]->SetBinError( i, sqrt( 1./n * a/n * ( 1. - a/n ) ) );
          if ( nl1 != 0. )
            EffwrtL1_phi[CountObjects+imult+ip*numbhist]->SetBinError( i, sqrt( 1./nl1 * a/nl1 * ( 1. - a/nl1 ) ) );
        }
        Eff_pt[CountObjects+imult+ip*numbhist]->Write();
        Eff_eta[CountObjects+imult+ip*numbhist]->Write();
        Eff_phi[CountObjects+imult+ip*numbhist]->Write();
        EffwrtL1_pt[CountObjects+imult+ip*numbhist]->Write();
        EffwrtL1_eta[CountObjects+imult+ip*numbhist]->Write();
        EffwrtL1_phi[CountObjects+imult+ip*numbhist]->Write();

	Num_pt[CountObjects+imult+ip*numbhist]->Write();
	Den_pt[CountObjects+imult+ip*numbhist]->Write();
	DenwrtL1_pt[CountObjects+imult+ip*numbhist]->Write();
	Num_eta[CountObjects+imult+ip*numbhist]->Write();
	Den_eta[CountObjects+imult+ip*numbhist]->Write();
	DenwrtL1_eta[CountObjects+imult+ip*numbhist]->Write();
	Num_phi[CountObjects+imult+ip*numbhist]->Write();
	Den_phi[CountObjects+imult+ip*numbhist]->Write();
	DenwrtL1_phi[CountObjects+imult+ip*numbhist]->Write();
	
      }

      CountObjects= CountObjects+multObject;
    }

    bool CountBitEff=true;
    if(CountBitEff){
      if(Den_pt[CountObjects-1+ip*numbhist]->GetEntries() != 0){
	double hlt_eff_pt=eff(Num_pt[CountObjects-1+ip*numbhist]->GetEntries(),Den_pt[CountObjects-1+ip*numbhist]->GetEntries());
	double shlt_eff_pt=seff(Num_pt[CountObjects-1+ip*numbhist]->GetEntries(),Den_pt[CountObjects-1+ip*numbhist]->GetEntries());
	cout << "HLT Overall Eff for Trigger Bit " << it << "("<<trignames[it] <<"): " << hlt_eff_pt <<"+-"<< shlt_eff_pt<<endl;
      }
      if(DenwrtL1_pt[CountObjects-1+ip*numbhist]->GetEntries() != 0){
	double hlt_eff_wrtL1_pt=eff(Num_pt[CountObjects-1+ip*numbhist]->GetEntries(),DenwrtL1_pt[CountObjects-1+ip*numbhist]->GetEntries());
	double shlt_eff_wrtL1_pt=seff(Num_pt[CountObjects-1+ip*numbhist]->GetEntries(),DenwrtL1_pt[CountObjects-1+ip*numbhist]->GetEntries());
	cout << "HLT Eff wrt L1 for Trigger Bit " << it << "("<<trignames[it] <<"): " << hlt_eff_wrtL1_pt <<"+-"<< shlt_eff_wrtL1_pt<<endl;
      }
    }
      if(Den_pt[CountObjects-1+ip*numbhist]->GetEntries() != 0){
	int  nbins = Num_pt[CountObjects-1+ip*numbhist]->GetXaxis()->GetNbins();
	double hlt_eff_pt=eff(Num_pt[CountObjects-1+ip*numbhist]->Integral(1,nbins),Den_pt[CountObjects-1+ip*numbhist]->Integral(1,nbins));
	double shlt_eff_pt=seff(Num_pt[CountObjects-1+ip*numbhist]->Integral(1,nbins),Den_pt[CountObjects-1+ip*numbhist]->Integral(1,nbins));
	cout << "RECO: HLT Overall Eff for Reco Objects for Trigger Bit " << it << "("<<trignames[it] <<"): " << hlt_eff_pt <<"+-"<< shlt_eff_pt<<endl;
	
      }
      if(DenwrtL1_pt[CountObjects-1+ip*numbhist]->GetEntries() != 0){
	int  nbins = Num_pt[CountObjects-1+ip*numbhist]->GetXaxis()->GetNbins();
	double hlt_eff_wrtL1_pt=eff(Num_pt[CountObjects-1+ip*numbhist]->Integral(1,nbins),DenwrtL1_pt[CountObjects-1+ip*numbhist]->Integral(1,nbins));
	double shlt_eff_wrtL1_pt=seff(Num_pt[CountObjects-1+ip*numbhist]->Integral(1,nbins),DenwrtL1_pt[CountObjects-1+ip*numbhist]->Integral(1,nbins));
	cout << "RECO: HLT Eff wrt L1 for Reco Objects for Trigger Bit " << it << "("<<trignames[it] <<"): " << hlt_eff_wrtL1_pt <<"+-"<< shlt_eff_wrtL1_pt<<endl;
    }
  }
}
