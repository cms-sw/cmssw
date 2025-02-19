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
void BookMenu_OhltExample(OHltMenu *menu, double &iLumi, double &nBunches);
void BookMenu_TauStudy(OHltMenu *menu, double &iLumi, double &nBunches);
void BookMenu_L1Default(OHltMenu*  menu, double &iLumi, double &nBunches);

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
  cout << "  Usage:  ./OHltRates <nevents> <menu> <conditions> <version tag> <cms energy> <doPrintAll> " << endl;
  cout << "default:  ./OHltRates -1 default startup 20June2008 14 0 " << endl;
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
  TString sMenu = "default"; // lookup available menus
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


  ////////////////////////////////////////////////////////////

  /**** Different Beam conditions: ****/ 
  // Fixed LHC defaults
  const double bunchCrossingTime = 50.0E-09;  // Design: 25 ns Startup: 25, 50 or 75 ns?
  const double maxFilledBunches = 3557;
    
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
  else if(sMenu.CompareTo("tau") == 0)
    BookMenu_TauStudy(menu,ILumi,nFilledBunches);
  else if(sMenu.CompareTo("l1default") == 0)
    BookMenu_L1Default(menu,ILumi,nFilledBunches);
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
  
  /* *************************************************************** */
  // Start filling sample vectors
  if( sConditions.CompareTo("startup") == 0) {

    // ppEleX
    TString PPEX_DIR="rfio:/castor/cern.ch/user/j/jjhollar/OpenHLT184/ppex/";
    ProcFil.clear();
    ProcFil.push_back(PPEX_DIR+"ppex_misAlCa_1.root");
    ProcFil.push_back(PPEX_DIR+"ppex_misAlCa_2.root");
    ProcFil.push_back(PPEX_DIR+"ppex_misAlCa_3.root");
    ProcFil.push_back(PPEX_DIR+"ppex_misAlCa_4.root");

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
    TString PPMUX_DIR="rfio:/castor/cern.ch/user/j/jjhollar/OpenHLT184/ppmux/";
    
    ProcFil.clear();
    ProcFil.push_back(PPMUX_DIR+"ppmux_misAlCa_1.root");
    ProcFil.push_back(PPMUX_DIR+"ppmux_misAlCa_2.root");
    ProcFil.push_back(PPMUX_DIR+"ppmux_misAlCa_3.root");
    ProcFil.push_back(PPMUX_DIR+"ppmux_misAlCa_4.root"); 
    ProcFil.push_back(PPMUX_DIR+"ppmux_misAlCa_5.root"); 
    ProcFil.push_back(PPMUX_DIR+"ppmux_misAlCa_6.root"); 
    
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
    TString MB_DIR="rfio:/castor/cern.ch/user/a/apana/OpenHLT184/MinBias/";
    ProcFil.clear();

    ProcFil.push_back(MB_DIR+"minbias_misAlCa_0.root");
    ProcFil.push_back(MB_DIR+"minbias_misAlCa_1.root"); 
    ProcFil.push_back(MB_DIR+"minbias_misAlCa_2.root"); 
    ProcFil.push_back(MB_DIR+"minbias_misAlCa_3.root"); 
    ProcFil.push_back(MB_DIR+"minbias_misAlCa_4.root"); 

   
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

  vector<Double_t> Rat,sRat,seqpRat,sseqpRat,pRat,spRat,cRat;
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
    vector<Double_t> Rat_bin,sRat_bin,seqpRat_bin,sseqpRat_bin,pRat_bin,spRat_bin,cRat_bin;
    for (int it = 0; it < Ntrig; it++){
      Rat_bin.push_back(0.);
      sRat_bin.push_back(0.);
      seqpRat_bin.push_back(0.);
      sseqpRat_bin.push_back(0.);
      pRat_bin.push_back(0.);
      spRat_bin.push_back(0.);
      cRat_bin.push_back(0.);
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

    hltt[ip]->Loop(iCount,sPureCount,pureCount,overlapCount,trignames,map_TrigPrescls,deno,doMuonCut[ip],doElecCut[ip]);
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
    Double_t curat_bin = 0.;
    for (int it = 0; it < Ntrig; it++){
      curat_bin += seqpRat_bin[it];
      cRat_bin[it] = curat_bin;
      RTOT_bin += seqpRat_bin[it];                                            // Total Rate
      sRTOT_bin += sseqpRat_bin[it];
    }
    sRTOT_bin = sqrt(sRTOT_bin);

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
  Double_t curat = 0.;
  for (int it = 0; it < Ntrig; it++){
    curat += seqpRat[it];
    cRat[it] = curat;
    RTOT += seqpRat[it];                                            // Total Rate
    sRTOT += sseqpRat[it];
  }

  sRTOT = sqrt(sRTOT);
    
  // End calculating rates  
  /* *************************************************************** */

  char sLumi[10]; 
  snprintf(sLumi, 10, "%1.1e",ILumi); 
  TString hltTableFileName= TString("hltTable_") + + TString(sEnergy) + "TeV_" + TString(sLumi) + TString("_") + sConditions + TString("Conditions") + sVersion; 
  TFile *fr = new TFile(hltTableFileName+TString(".root"),"recreate");
  fr->cd();
  TH1F *individual = new TH1F("individual","individual",Ntrig,1,Ntrig+1);
  TH1F *cumulative = new TH1F("cumulative","cumulative",Ntrig,1,Ntrig+1);
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
  overlap->Write();
  fr->Close();
  
  ////////////////////////////////////////////////////////////
  // Printout Results to Tex/PDF

  if (PrintAll==1) {
    //    char sLumi[10];
    //    snprintf(sLumi, 10, "%1.1e",ILumi);
    //    TString hltTableFileName= TString("hltTable_") + + TString(sEnergy) + "TeV_" + TString(sLumi) + TString("_") + sConditions + TString("Conditions") + sVersion;
    TString texFile = hltTableFileName + TString(".tex");
    TString dviFile = hltTableFileName + TString(".dvi");
    //TString psFile  = hltTableFileName + TString(".ps");
    TString psFile  = hltTableFileName + TString(".pdf");
    ofstream outFile(texFile.Data());
    if (!outFile){cout<<"Error opening output file"<< endl;}
    outFile <<setprecision(1);
    outFile.setf(ios::floatfield,ios::fixed);
    outFile << "\\documentclass[amsmath,amssymb]{revtex4}" << endl;
    outFile << "\\usepackage{longtable}" << endl;
    outFile << "\\usepackage{color}" << endl;
    outFile << "\\usepackage{lscape}" << endl;
    outFile << "\\begin{document}" << endl;
    outFile << "\\begin{landscape}" << endl;
    outFile << "\\newcommand{\\met}{\\ensuremath{E\\kern-0.6em\\lower-.1ex\\hbox{\\/}\\_T}}" << endl;
    
    
    outFile << "\\begin{footnotesize}" << endl;
    outFile << "\\begin{longtable}{|c|l|c|c|c|c|c|c|}" << endl;
    outFile << "\\caption[Cuts]{New paths are introduced in addition to standard '1e32' paths.  Description of the newly introduced paths is given at the end of the table.  Available HLT bandwith is 150 Hz = ((1 GB/s / 3) - 100 MB/s for AlCa triggers) / 1.5 MB/event. L1 bandwidth is 12 kHz. } \\label{CUTS} \\\\ " << endl;
    
    
    outFile << "\\hline \\multicolumn{8}{|c|}{\\bf \\boldmath HLT for L = "<< sLumi  << "}\\\\  \\hline" << endl;
    outFile << "{\\bf Status} & " << endl;
    outFile << "{\\bf Path Name} & " << endl;
    outFile << "{\\bf L1 condtition} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf Threshold} \\\\ {\\bf $[$GeV$]$} \\end{tabular} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf L1} \\\\ {\\bf Prescale} \\end{tabular} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf HLT} \\\\ {\\bf Prescale} \\end{tabular} & " << endl;
    //outFile << "\\begin{tabular}{c} {\\bf Total} \\\\ {\\bf Prescale} \\end{tabular} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf HLT Rate} \\\\ {\\bf $[$Hz$]$} \\end{tabular} &" << endl;
    outFile << "\\begin{tabular}{c} {\\bf Total Rate} \\\\ {\\bf $[$Hz$]$} \\end{tabular} \\\\ \\hline" << endl;
    outFile << "\\endfirsthead " << endl;
    
    outFile << "\\multicolumn{8}{r}{\\bf \\bfseries --continued from previous page (L = " << sLumi << ")"  << "}\\\\ \\hline " << endl;
    outFile << "{\\bf Status} & " << endl;
    outFile << "{\\bf Path Name} & " << endl;
    outFile << "{\\bf L1 condtition} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf Threshold} \\\\ {\\bf $[$GeV$]$} \\end{tabular} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf L1} \\\\ {\\bf Prescale} \\end{tabular} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf HLT} \\\\ {\\bf Prescale} \\end{tabular} & " << endl;
    //outFile << "\\begin{tabular}{c} {\\bf Total} \\\\ {\\bf Prescale} \\end{tabular} & " << endl;
    outFile << "\\begin{tabular}{c} {\\bf HLT Rate} \\\\ {\\bf $[$Hz$]$} \\end{tabular} &" << endl;
    outFile << "\\begin{tabular}{c} {\\bf Total Rate} \\\\ {\\bf $[$Hz$]$} \\end{tabular} \\\\ \\hline" << endl;
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
	
      outFile << map_HltDesc.find(trignames[it])->second << " & " ;
      
      tempTrigName.ReplaceAll("_","\\_");
      tempL1BitName.ReplaceAll("_","\\_");
      //tempTrigName.ReplaceAll("HLT","");
      outFile << "\\color{blue}"  << tempTrigName << " & " << "${\\it " << tempL1BitName
	      << "}$ "<< " & " << tempThreshold << " & " <<  map_L1Prescls.find(trignames[it])->second
	      << " & " <<  map_HLTPrescls.find(trignames[it])->second  << " & " << Rat[it]
	      << " {$\\pm$ " << sqrt(sRat[it]) << "} & " << cRat[it] << "\\\\" << endl;
    }
    
    outFile << "\\hline \\multicolumn{7}{|c|}{\\bf \\boldmath Total HLT rate (Hz) } & "<<  RTOT << " {$\\pm$ " << sRTOT << "} \\\\  \\hline" << endl;
    outFile << "\\hline " << endl;
    
    
    outFile << "\\multicolumn{8}{|l|}{ ($\\ast$): Conditions on tracks seeded by L2 muons} \\\\ \\hline  " << endl;
    outFile << "\\multicolumn{8}{|l|}{ ($\\star$): {$L1\\_Mu3\\_IsoEG5, L1\\_Mu5\\_IsoEG10, L1\\_Mu3\\_IsoEG12$ }} \\\\ \\hline  " << endl;
    //outFile << "\\multicolumn{8}{|l|}{ ($\\S$): {2JetAve paths use {\\bf uncorrected} jets at HLT }} \\\\ \\hline  " << endl;
    outFile << "\\multicolumn{8}{|l|}{ (NI): Not implemented in current version} \\\\ \\hline  " << endl;
    outFile << "\\multicolumn{8}{|l|}{ ($\\S$): {$M_{\\mu\\mu} \\in [0.2,3],M_{\\mu\\mu\\mu} \\in [1.2,2.2]$}} \\\\ \\hline  " << endl;
    outFile << "\\multicolumn{8}{|l|}{ ($\\circ$): Only Pixel-matching, no track match required} \\\\ \\hline  " << endl;
    outFile << "\\multicolumn{8}{|l|}{ ($\\dagger$): {$L1\\_SingleJet150, L1\\_DoubleJet70, L1\\_TripleJet50$}} \\\\ \\hline " << endl; 
    outFile << "\\multicolumn{8}{|l|}{ ($\\ddagger$): {$L1\\_SingleJet150, L1\\_DoubleJet70, L1\\_TripleJet50, L1\\_QuadJet30$}} \\\\ \\hline " << endl; 
    
    outFile << "\\multicolumn{8}{|l|}{ ($\\Diamond$): {$L1\\_SingleJet100, L1\\_DoubleJet70, L1\\_TripleJet50, L1\\_QuadJet30, L1\\_HTT300$}} \\\\ \\hline " << endl; 
    outFile << "\\multicolumn{8}{|l|}{ ($\\Diamond \\Diamond$): {$L1\\_SingleHFTowCount1/12, L1\\_DoubleHFTowCount1/20, L1\\_SingleHFRing0Sum3/20,$}} \\\\   " << endl;
    outFile << "\\multicolumn{8}{|l|}{ {$L1\\_DoubleHFRing0Sum3/20, L1\\_SingleHFRing0Sum6/20, L1\\_DoubleHFRing0Sum6/20$}} \\\\\\hline \\hline " << endl; 
    outFile << "\\multicolumn{8}{|c|}{ {\\it \\bf HLT Requirements for new introduced paths (in GeV) }  } \\\\ \\hline  " << endl;
    outFile << "\\multicolumn{8}{|l|}{ HLTMuons: $|\\eta|<2.5$, $L2Pt+3.9Err<$A, $L3Pt+2.2Err<$A  } \\\\ \\hline  " << endl;
    outFile << "\\multicolumn{8}{|l|}{ HLT1jetA: $recoJetCalPt<$A  } \\\\ \\hline  " << endl;
    outFile << "\\multicolumn{8}{|l|}{ HLT1ElectronA\\_L1R\\_HI: $Et>$A, $HCAL<3$, $TrkIso<0.06$} \\\\ " << endl; 
    outFile << "\\multicolumn{8}{|l|}{ HLT1ElectronA\\_L1R\\_LI: $Et>$A, $HCAL<6$, $TrkIso<0.12$} \\\\   " << endl;
    outFile << "\\multicolumn{8}{|l|}{ HLT1ElectronA\\_L1R\\_NI: $Et>$A} \\\\ \\hline  " << endl;
    outFile << "\\multicolumn{8}{|l|}{ HLT1PhotonA\\_L1R\\_HI: $Et>$A, $ECAL<1.5$, $HCAL<6(4)$, $TrkIso=0$} \\\\   " << endl;
    outFile << "\\multicolumn{8}{|l|}{ HLT1PhotonA\\_L1R\\_LI: $Et>$A, $ECAL<3.0$, $HCAL<12(8)$, $TrkIso\\leq2$} \\\\   " << endl;
    outFile << "\\multicolumn{8}{|l|}{ HLT1PhotonA\\_L1R\\_NI: $Et>$A} \\\\  " << endl;
    
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

  menu->AddHlt("HLT_L1Jet15","L1_SingleJet15",1,1,"","");
  menu->AddHlt("HLT_Jet30","L1_SingleJet15",1,1,"","");
  menu->AddHlt("HLT_Jet50","L1_SingleJet30",1,1,"","");
  menu->AddHlt("HLT_Jet80","L1_SingleJet50",1,1,"","");
  menu->AddHlt("HLT_Jet110","L1_SingleJet70",1,1,"","");
  menu->AddHlt("HLT_Jet180","L1_SingleJet70",1,1,"","");
  menu->AddHlt("HLT_Jet250","L1_SingleJet70",1,1,"","");
  menu->AddHlt("HLT_FwdJet20","L1_IsoEG10_Jet15_ForJet10",1,1,"","");
  menu->AddHlt("HLT_DoubleJet150","L1_SingleJet150 OR L1_DoubleJet70",1,1,"","");
  menu->AddHlt("HLT_DoubleJet125_Aco","L1_SingleJet150 OR L1_DoubleJet70",1,1,"","");
  menu->AddHlt("HLT_DoubleFwdJet50","L1_SingleJet30",1,1,"","");
  menu->AddHlt("HLT_DiJetAve15","L1_SingleJet15",1,1,"","");
  menu->AddHlt("HLT_DiJetAve30","L1_SingleJet30",1,1,"","");
  menu->AddHlt("HLT_DiJetAve50","L1_SingleJet50",1,1,"","");
  menu->AddHlt("HLT_DiJetAve70","L1_SingleJet70",1,1,"","");
  menu->AddHlt("HLT_DiJetAve130","L1_SingleJet70",1,1,"","");
  menu->AddHlt("HLT_DiJetAve220","L1_SingleJet70",1,1,"","");
  menu->AddHlt("HLT_TripleJet85","L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50",1,1,"","");
  menu->AddHlt("HLT_QuadJet30","L1_QuadJet15",1,1,"","");
  menu->AddHlt("HLT_QuadJet60","L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30",1,1,"","");
  menu->AddHlt("HLT_SumET120","L1_ETT60",1,1,"","");
  menu->AddHlt("HLT_L1MET20","L1_ETM20",1,1,"","");
  menu->AddHlt("HLT_MET25","L1_ETM20",1,1,"","");
  menu->AddHlt("HLT_MET35","L1_ETM30",1,1,"","");
  menu->AddHlt("HLT_MET50","L1_ETM40",1,1,"","");
  menu->AddHlt("HLT_MET65","L1_ETM50",1,1,"","");
  menu->AddHlt("HLT_MET75","L1_ETM50",1,1,"","");
  menu->AddHlt("HLT_MET35_HT350","L1_HTT300",1,1,"","");
  menu->AddHlt("HLT_Jet180_MET60","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLT_Jet60_MET70_Aco","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLT_Jet100_MET60_Aco","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLT_DoubleJet125_MET60","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLT_DoubleFwdJet40_MET60","L1_ETM40",1,1,"","");
  menu->AddHlt("HLT_DoubleJet60_MET60_Aco","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLT_DoubleJet50_MET70_Aco","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLT_DoubleJet40_MET70_Aco","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLT_TripleJet60_MET60","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLT_QuadJet35_MET60","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLT_IsoEle15_L1I","L1_SingleIsoEG12",1,1,"","");
  menu->AddHlt("HLT_IsoEle18_L1R","L1_SingleEG15",1,1,"","");
  menu->AddHlt("HLT_IsoEle15_LW_L1I","L1_SingleIsoEG12",1,1,"","");
  menu->AddHlt("HLT_LooseIsoEle15_LW_L1R","L1_SingleEG12",1,1,"","");
  menu->AddHlt("HLT_Ele10_SW_L1R","L1_SingleEG8",1,1,"","");
  menu->AddHlt("HLT_Ele15_SW_L1R","L1_SingleEG12",1,1,"","");
  menu->AddHlt("HLT_Ele15_LW_L1R","L1_SingleEG10",1,1,"","");
  menu->AddHlt("HLT_EM80","L1_SingleEG15",1,1,"","");
  menu->AddHlt("HLT_EM200","L1_SingleEG15",1,1,"","");
  menu->AddHlt("HLT_DoubleIsoEle10_L1I","L1_DoubleIsoEG8",1,1,"","");
  menu->AddHlt("HLT_DoubleIsoEle12_L1R","L1_DoubleEG10",1,1,"","");
  menu->AddHlt("HLT_DoubleIsoEle10_LW_L1I","L1_DoubleIsoEG8",1,1,"","");
  menu->AddHlt("HLT_DoubleIsoEle12_LW_L1R","L1_DoubleEG10",1,1,"","");
  menu->AddHlt("HLT_DoubleEle5_SW_L1R","L1_DoubleEG5",1,1,"","");
  menu->AddHlt("HLT_DoubleEle10_LW_OnlyPixelM_L1R","L1_DoubleEG5",1,1,"","");
  menu->AddHlt("HLT_DoubleEle10_Z","L1_DoubleIsoEG8",1,1,"","");
  menu->AddHlt("HLT_DoubleEle6_Exclusive","L1_ExclusiveDoubleIsoEG6",1,1,"","");
  menu->AddHlt("HLT_IsoPhoton30_L1I","L1_SingleIsoEG12",1,1,"","");
  menu->AddHlt("HLT_IsoPhoton10_L1R","L1_SingleEG8",1,1,"","");
  menu->AddHlt("HLT_IsoPhoton15_L1R","L1_SingleEG12",1,1,"","");
  menu->AddHlt("HLT_IsoPhoton20_L1R","L1_SingleEG15",1,1,"","");
  menu->AddHlt("HLT_IsoPhoton25_L1R","L1_SingleEG15",1,1,"","");
  menu->AddHlt("HLT_IsoPhoton40_L1R","L1_SingleEG15",1,1,"","");
  menu->AddHlt("HLT_Photon15_L1R","L1_SingleEG10",1,1,"","");
  menu->AddHlt("HLT_Photon25_L1R","L1_SingleEG15",1,1,"","");
  menu->AddHlt("HLT_DoubleIsoPhoton20_L1I","L1_DoubleIsoEG8",1,1,"","");
  menu->AddHlt("HLT_DoubleIsoPhoton20_L1R","L1_DoubleEG10",1,1,"","");
  menu->AddHlt("HLT_DoublePhoton10_Exclusive","L1_ExclusiveDoubleIsoEG6",1,1,"","");
  menu->AddHlt("HLT_L1Mu","L1_SingleMu7 OR L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT_L1MuOpen","L1_SingleMuOpen OR L1_SingleMu3 OR L1_SingleMu5",1,1,"","");
  menu->AddHlt("HLT_L2Mu9","L1_SingleMu7",1,1,"","");
  menu->AddHlt("HLT_IsoMu9","L1_SingleMu7",1,1,"","");
  menu->AddHlt("HLT_IsoMu11","L1_SingleMu7",1,1,"","");
  menu->AddHlt("HLT_IsoMu13","L1_SingleMu10",1,1,"","");
  menu->AddHlt("HLT_IsoMu15","L1_SingleMu10",1,1,"","");
  menu->AddHlt("HLT_NoTrackerIsoMu15","L1_SingleMu7",1,1,"","");
  menu->AddHlt("HLT_Mu3","L1_SingleMu3",1,1,"","");
  menu->AddHlt("HLT_Mu5","L1_SingleMu5",1,1,"","");
  menu->AddHlt("HLT_Mu7","L1_SingleMu7",1,1,"","");
  menu->AddHlt("HLT_Mu9","L1_SingleMu7",1,1,"","");
  menu->AddHlt("HLT_Mu11","L1_SingleMu7",1,1,"","");
  menu->AddHlt("HLT_Mu13","L1_SingleMu10",1,1,"","");
  menu->AddHlt("HLT_Mu15","L1_SingleMu10",1,1,"","");
  menu->AddHlt("HLT_Mu15_L1Mu7","L1_SingleMu7",1,1,"","");
  menu->AddHlt("HLT_Mu15_Vtx2cm","L1_SingleMu7",1,1,"","");
  menu->AddHlt("HLT_Mu15_Vtx2mm","L1_SingleMu7",1,1,"","");
  menu->AddHlt("HLT_DoubleIsoMu3","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT_DoubleMu3","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT_DoubleMu3_Vtx2cm","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT_DoubleMu3_Vtx2mm","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT_DoubleMu3_JPsi","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT_DoubleMu3_Upsilon","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT_DoubleMu7_Z","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT_DoubleMu3_SameSign","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT_DoubleMu3_Psi2S","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT_BTagIP_Jet180","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","");
  menu->AddHlt("HLT_BTagIP_Jet120_Relaxed","L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","");
  menu->AddHlt("HLT_BTagIP_DoubleJet120","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","");
  menu->AddHlt("HLT_BTagIP_DoubleJet60_Relaxed","L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","");
  menu->AddHlt("HLT_BTagIP_TripleJet70","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","");
  menu->AddHlt("HLT_BTagIP_TripleJet40_Relaxed","L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","");
  menu->AddHlt("HLT_BTagIP_QuadJet40","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","");
  menu->AddHlt("HLT_BTagIP_QuadJet30_Relaxed","L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","");
  menu->AddHlt("HLT_BTagIP_HT470","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","");
  menu->AddHlt("HLT_BTagIP_HT320_Relaxed","L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300",1,1,"","");
  menu->AddHlt("HLT_BTagMu_DoubleJet120","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLT_BTagMu_DoubleJet60_Relaxed","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLT_BTagMu_TripleJet70","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLT_BTagMu_TripleJet40_Relaxed","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLT_BTagMu_QuadJet40","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLT_BTagMu_QuadJet30_Relaxed","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLT_BTagMu_HT370","L1_HTT300",1,1,"","");
  menu->AddHlt("HLT_BTagMu_HT250_Relaxed","L1_HTT200",1,1,"","");
  menu->AddHlt("HLT_DoubleMu3_BJPsi","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT_DoubleMu4_BJPsi","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT_TripleMu3_TauTo3Mu","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT_IsoTau_MET65_Trk20","L1_SingleTauJet80",1,1,"","");
  menu->AddHlt("HLT_IsoTau_MET35_Trk15_L1MET","L1_TauJet30_ETM30",1,1,"","");
  menu->AddHlt("HLT_LooseIsoTau_MET30","L1_SingleTauJet80",1,1,"","");
  menu->AddHlt("HLT_LooseIsoTau_MET30_L1MET","L1_TauJet30_ETM30",1,1,"","");
  menu->AddHlt("HLT_DoubleIsoTau_Trk3","L1_DoubleTauJet40",1,1,"","");
  menu->AddHlt("HLT_DoubleLooseIsoTau","L1_DoubleTauJet20",1,1,"","");
  menu->AddHlt("HLT_IsoEle8_IsoMu7","L1_Mu3_IsoEG5",1,1,"","");
  menu->AddHlt("HLT_IsoEle10_Mu10_L1R","L1_Mu3_EG12",1,1,"","");
  menu->AddHlt("HLT_IsoEle12_IsoTau_Trk3","L1_IsoEG10_TauJet20",1,1,"","");
  menu->AddHlt("HLT_IsoEle10_BTagIP_Jet35","L1_IsoEG10_Jet20",1,1,"","");
  menu->AddHlt("HLT_IsoEle12_Jet40","L1_IsoEG10_Jet30",1,1,"","");
  menu->AddHlt("HLT_IsoEle12_DoubleJet80","L1_IsoEG10_Jet30",1,1,"","");
  menu->AddHlt("HLT_IsoElec5_TripleJet30","L1_EG5_TripleJet15",1,1,"","");
  menu->AddHlt("HLT_IsoEle12_TripleJet60","L1_IsoEG10_Jet30",1,1,"","");
  menu->AddHlt("HLT_IsoEle12_QuadJet35","L1_IsoEG10_Jet30",1,1,"","");
  menu->AddHlt("HLT_IsoMu14_IsoTau_Trk3","L1_Mu5_TauJet20",1,1,"","");
  menu->AddHlt("HLT_IsoMu7_BTagIP_Jet35","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLT_IsoMu7_BTagMu_Jet20","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLT_IsoMu7_Jet40","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLT_NoL2IsoMu8_Jet40","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLT_Mu14_Jet50","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLT_Mu5_TripleJet30","L1_Mu3_TripleJet15",1,1,"","");
  menu->AddHlt("HLT_BTagMu_Jet20_Calib","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLT_ZeroBias","L1_ZeroBias",1,1,"","");
  menu->AddHlt("HLT_MinBias","L1_MinBias_HTT10",1,1,"","");
  menu->AddHlt("HLT_MinBiasHcal","L1_SingleJetCountsHFTow OR L1_DoubleJetCountsHFTow OR L1_SingleJetCountsHFRing0Sum3 OR L1_DoubleJetCountsHFRing0Sum3 OR L1_SingleJetCountsHFRing0Sum6 OR L1_DoubleJetCountsHFRing0Sum6",1,1,"","");
  menu->AddHlt("HLT_MinBiasEcal","L1_SingleEG2 OR L1_DoubleEG1",1,1,"","");
  menu->AddHlt("HLT_MinBiasPixel","L1_ZeroBias",1,1,"","");
  menu->AddHlt("HLT_MinBiasPixel_Trk5","L1_ZeroBias",1,1,"","");
  menu->AddHlt("HLT_BackwardBSC","38 OR 39",1,1,"","");
  menu->AddHlt("HLT_ForwardBSC","36 OR 37",1,1,"","");
  menu->AddHlt("HLT_CSCBeamHalo","L1_SingleMuBeamHalo",1,1,"","");
  menu->AddHlt("HLT_CSCBeamHaloOverlapRing1","L1_SingleMuBeamHalo",1,1,"","");
  menu->AddHlt("HLT_CSCBeamHaloOverlapRing2","L1_SingleMuBeamHalo",1,1,"","");
  menu->AddHlt("HLT_CSCBeamHaloRing2or3","L1_SingleMuBeamHalo",1,1,"","");
  menu->AddHlt("HLT_TrackerCosmics","24 OR 25 OR 26 OR 27 OR 28",1,1,"","");
  menu->AddHlt("AlCa_IsoTrack","L1_SingleJet30 OR L1_SingleJet50 OR L1_SingleJet70 OR L1_SingleJet100 OR L1_SingleTauJet30 OR L1_SingleTauJet40 OR L1_SingleTauJet60 OR L1_SingleTauJet80",1,1,"","");
  menu->AddHlt("AlCa_EcalPhiSym","L1_ZeroBias OR L1_SingleJetCountsHFTow OR L1_DoubleJetCountsHFTow OR L1_SingleEG2 OR L1_DoubleEG1 OR L1_SingleJetCountsHFRing0Sum3 OR L1_DoubleJetCountsHFRing0Sum3 OR L1_SingleJetCountsHFRing0Sum6 OR L1_DoubleJetCountsHFRing0Sum6",1,1,"","");
  menu->AddHlt("AlCa_EcalPi0","L1_SingleIsoEG5 OR L1_SingleIsoEG8 OR L1_SingleIsoEG10 OR L1_SingleIsoEG12 OR L1_SingleIsoEG15 OR L1_SingleIsoEG20 OR L1_SingleIsoEG25 OR L1_SingleEG5 OR L1_SingleEG8 OR L1_SingleEG10 OR L1_SingleEG12 OR L1_SingleEG15 OR L1_SingleEG20 OR L1_SingleEG25",1,1,"","");

}

void BookMenu_Default(OHltMenu*  menu, double &iLumi, double &nBunches) {

  iLumi = 2E30;
  nBunches = 43;

  menu->AddHlt("HLT1jet","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLT2jet","L1_SingleJet150 OR L1_DoubleJet70",1,1,"","");
  menu->AddHlt("HLT3jet","L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50",1,1,"","");
  menu->AddHlt("HLT4jet","L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet40",1,1,"","");
  menu->AddHlt("HLT1MET","L1_ETM40",1,1,"","");
  menu->AddHlt("HLT2jetAco","L1_SingleJet150 OR L1_DoubleJet70",1,1,"","");
  menu->AddHlt("HLT1jet1METAco","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLT1jet1MET","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLT2jet1MET","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLT3jet1MET","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLT4jet1MET","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLT1MET1HT","L1_HTT300",1,1,"","");
  menu->AddHlt("CandHLT1SumET","L1_ETT60",1,1,"","");
  menu->AddHlt("HLT1jetPE1","L1_SingleJet100",1,1,"","");
  menu->AddHlt("HLT1jetPE3","L1_SingleJet70",1,1,"","");
  menu->AddHlt("HLT1jetPE5","L1_SingleJet30",1,1,"","");
  menu->AddHlt("HLT1jetPE7","L1_SingleJet15",1,1,"","");
  menu->AddHlt("HLT1METPre1","L1_ETM40",1,1,"","");
  menu->AddHlt("HLT1METPre2","L1_ETM15",1,1,"","");
  menu->AddHlt("HLT1METPre3","L1_ETM10",1,1,"","");
  menu->AddHlt("HLT2jetAve30","L1_SingleJet15",1,1,"","");
  menu->AddHlt("HLT2jetAve60","L1_SingleJet30",1,1,"","");
  menu->AddHlt("HLT2jetAve110","L1_SingleJet70",1,1,"","");
  menu->AddHlt("HLT2jetAve150","L1_SingleJet100",1,1,"","");
  menu->AddHlt("HLT2jetAve200","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLT2jetvbfMET","L1_ETM40",1,1,"","");
  menu->AddHlt("HLTS2jet1METNV","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLTS2jet1METAco","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLTSjet1MET1Aco","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLTSjet2MET1Aco","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLTS2jetMET1Aco","L1_SingleJet150",1,1,"","");
  menu->AddHlt("HLTJetMETRapidityGap","L1_IsoEG10_Jet15_ForJet10",1,1,"","");
  menu->AddHlt("HLT1Electron","L1_SingleIsoEG12",1,1,"","");
  menu->AddHlt("HLT1ElectronRelaxed","L1_SingleEG15",1,1,"","");
  menu->AddHlt("HLT2Electron","L1_DoubleIsoEG8",1,1,"","");
  menu->AddHlt("HLT2ElectronRelaxed","L1_DoubleEG10",1,1,"","");
  menu->AddHlt("HLT1Photon","L1_SingleIsoEG12",1,1,"","");
  menu->AddHlt("HLT1PhotonRelaxed","L1_SingleEG15",1,1,"","");
  menu->AddHlt("HLT2Photon","L1_DoubleIsoEG8",1,1,"","");
  menu->AddHlt("HLT2PhotonRelaxed","L1_DoubleEG10",1,1,"","");
  menu->AddHlt("HLT1EMHighEt","L1_SingleEG15",1,1,"","");
  menu->AddHlt("HLT1EMVeryHighEt","L1_SingleEG15",1,1,"","");
  menu->AddHlt("HLT2ElectronZCounter","L1_DoubleIsoEG8",1,1,"","");
  menu->AddHlt("HLT2ElectronExclusive","L1_ExclusiveDoubleIsoEG6",1,1,"","");
  menu->AddHlt("HLT2PhotonExclusive","L1_ExclusiveDoubleIsoEG6",1,1,"","");
  menu->AddHlt("HLT1PhotonL1Isolated","L1_SingleIsoEG10",1,1,"","");
  menu->AddHlt("CandHLT1ElectronStartup","L1_SingleIsoEG12",1,1,"","");
  menu->AddHlt("CandHLT1ElectronRelaxedStartup","L1_SingleEG15",1,1,"","");
  menu->AddHlt("CandHLT2ElectronStartup","L1_DoubleIsoEG8",1,1,"","");
  menu->AddHlt("CandHLT2ElectronRelaxedStartup","L1_DoubleEG10",1,1,"","");
  menu->AddHlt("HLT1MuonIso","L1_SingleMu7",1,1,"","");
  menu->AddHlt("HLT1MuonNonIso","L1_SingleMu7",1,1,"","");
  menu->AddHlt("HLT2MuonIso","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT2MuonNonIso","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT2MuonJPsi","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT2MuonUpsilon","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT2MuonZ","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLTNMuonNonIso","L1_TripleMu3",1,1,"","");
  menu->AddHlt("HLT2MuonSameSign","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT1MuonPrescalePt3","L1_SingleMu3",1,1,"","");
  menu->AddHlt("HLT1MuonPrescalePt5","L1_SingleMu5",1,1,"","");
  menu->AddHlt("HLT1MuonPrescalePt7x7","L1_SingleMu7",1,1,"","");
  menu->AddHlt("HLT1MuonPrescalePt7x10","L1_SingleMu7",1,1,"","");
  menu->AddHlt("HLT1MuonLevel1","L1_SingleMu3 OR L1_SingleMu5 OR L1_SingleMu7 OR L1_DoubleMu3",1,1,"","");
  menu->AddHlt("CandHLT1MuonPrescaleVtx2cm","L1_SingleMu7",1,1,"","");
  menu->AddHlt("CandHLT1MuonPrescaleVtx2mm","L1_SingleMu7",1,1,"","");
  menu->AddHlt("CandHLT2MuonPrescaleVtx2cm","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("CandHLT2MuonPrescaleVtx2mm","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLTB1Jet","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet40 OR L1_HTT300",1,1,"","");
  menu->AddHlt("HLTB2Jet","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet40 OR L1_HTT300",1,1,"","");
  menu->AddHlt("HLTB3Jet","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet40 OR L1_HTT300",1,1,"","");
  menu->AddHlt("HLTB4Jet","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet40 OR L1_HTT300",1,1,"","");
  menu->AddHlt("HLTBHT","L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet40 OR L1_HTT300",1,1,"","");
  menu->AddHlt("HLTB1JetMu","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLTB2JetMu","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLTB3JetMu","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLTB4JetMu","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLTBHTMu","L1_HTT300",1,1,"","");
  menu->AddHlt("HLTBJPsiMuMu","L1_DoubleMu3",1,1,"","");
  menu->AddHlt("HLT1Tau","L1_SingleTauJet80",1,1,"","");
  menu->AddHlt("HLT1Tau1MET","L1_TauJet30_ETM30",1,1,"","");
  menu->AddHlt("HLT2TauPixel","L1_DoubleTauJet40",1,1,"","");
  menu->AddHlt("HLTXElectronBJet","L1_IsoEG10_Jet20",1,1,"","");
  menu->AddHlt("HLTXMuonBJet","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLTXMuonBJetSoftMuon","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLTXElectron1Jet","L1_IsoEG10_Jet30",1,1,"","");
  menu->AddHlt("HLTXElectron2Jet","L1_IsoEG10_Jet30",1,1,"","");
  menu->AddHlt("HLTXElectron3Jet","L1_IsoEG10_Jet30",1,1,"","");
  menu->AddHlt("HLTXElectron4Jet","L1_IsoEG10_Jet30",1,1,"","");
  menu->AddHlt("HLTXMuonJets","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("CandHLTXMuonNoL2IsoJets","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("CandHLTXMuonNoIsoJets","L1_Mu5_Jet15",1,1,"","");
  menu->AddHlt("HLTXElectronMuon","L1_Mu3_IsoEG5",1,1,"","");
  menu->AddHlt("HLTXElectronMuonRelaxed","L1_Mu3_EG12",1,1,"","");
  menu->AddHlt("HLTXElectronTau","L1_IsoEG10_TauJet20",1,1,"","");
  menu->AddHlt("CandHLTXElectronTauPixel","L1_IsoEG10_TauJet20",1,1,"","");
  menu->AddHlt("HLTXMuonTau","L1_Mu5_TauJet20",1,1,"","");
  menu->AddHlt("CandHLTEcalPi0","L1_SingleJet15 OR L1_SingleJet20 OR L1_SingleJet30 OR L1_SingleJet50 OR L1_SingleJet70 OR L1_SingleJet100 OR L1_SingleJet150 OR L1_SingleJet200 OR L1_DoubleJet70 OR L1_DoubleJet100",1,1,"","");
  menu->AddHlt("CandHLTEcalPhiSym","L1_ZeroBias",1,1,"","");
  menu->AddHlt("CandHLTHcalPhiSym","L1_ZeroBias",1,1,"","");
  menu->AddHlt("HLTHcalIsolatedTrack","L1_SingleJet100 OR L1_SingleTauJet100",1,1,"","");
  menu->AddHlt("CandHLTHcalIsolatedTrackNoEcalIsol","L1_SingleJet100 OR L1_SingleTauJet100",1,1,"","");
  menu->AddHlt("HLTMinBiasPixel","L1_ZeroBias",1,1,"","");
  menu->AddHlt("CandHLTMinBiasForAlignment","L1_ZeroBias",1,1,"","");
  menu->AddHlt("HLTMinBias","L1_MinBias_HTT10",1,1,"","");
  menu->AddHlt("HLTZeroBias","L1_ZeroBias",1,1,"","");
}



void BookMenu_OhltExample(OHltMenu*  menu,double &iLumi,double &nBunches) {

  iLumi = 2.E31;
  nBunches = 156;

  //
  menu->AddHlt("HLT1Electron","L1_SingleIsoEG12",1,1,"15","1e32"); 
  menu->AddHlt("OpenHLT1Electron","L1_SingleIsoEG12",1,1,"15","1e32");
  
  //
  menu->AddHlt("HLT1Photon","L1_SingleIsoEG12",1,1,"30","1e32"); 
  menu->AddHlt("OpenHLT1Photon","L1_SingleIsoEG12",1,1,"30","1e32"); 

  //
  //  menu->AddHlt("HLT2TauPixel","L1_TauJet40",1,1,"15","1e32"); 
  //  menu->AddHlt("OpenHLT2TauPixel","L1_TauJet40",1,1,"15","1e32"); 

  //
  menu->AddHlt("HLT1MuonNonIso","L1_SingleMu7",1,1,"16","1e32");
  menu->AddHlt("OpenHLT1MuonNonIso","L1_SingleMu7",1,1,"16","1e32");
  menu->AddHlt("HLT1MuonIso","L1_SingleMu7",1,1,"11","1e32");
  menu->AddHlt("OpenHLT1MuonIso","L1_SingleMu7",1,1,"11","1e32");
  menu->AddHlt("HLT1MuonL1Open","L1_SingleMuOpen",1,1,"-","new");  // L1: 150 
}


void BookMenu_TauStudy(OHltMenu*  menu, double &iLumi, double &nBunches) {

  iLumi = 2E30;
  nBunches = 43;

  menu->AddHlt("HLT1MuonL1Open","L1_SingleMuOpen",150,1,"-","new");  // L1: 150
  menu->AddHlt("HLT1MuonLevel1","L1_SingleMu7,L1_DoubleMu3",1,20,"-","new"); 

  menu->AddHlt("HLT1MuonPrescalePt3","L1_SingleMu3",80,1,"3","1e32");  // L1: 80
  menu->AddHlt("HLT1MuonPrescalePt5","L1_SingleMu5",80,1,"5","1e32");  // L1: 80
  menu->AddHlt("HLT1MuonPrescalePt7x7","L1_SingleMu7",1,1,"7","1e32"); 
  menu->AddHlt("HLT1MuonPrescalePt7x10","L1_SingleMu7",1,1,"10","1e32"); 

  menu->AddHlt("HLT1MuonLevel2","L1_SingleMu7",1,1,"16","new");             
  menu->AddHlt("CandHLT1MuonPrescaleVtx2cm","L1_SingleMu7",1,1,"16","1e32"); 
  menu->AddHlt("CandHLT1MuonPrescaleVtx2mm","L1_SingleMu7",1,1,"16","1e32"); 

  menu->AddHlt("HLT1MuonIso9","L1_SingleMu7",1,1,"9","new");             
  menu->AddHlt("HLT1MuonIso13","L1_SingleMu10",1,1,"13","new");          
  menu->AddHlt("HLT1MuonIso15","L1_SingleMu10",1,1,"15","new");

  menu->AddHlt("HLT1MuonIso","L1_SingleMu7",1,1,"11","1e32"); 

  menu->AddHlt("HLT1MuonNonIso9","L1_SingleMu7",1,1,"9","new");             
  menu->AddHlt("HLT1MuonNonIso11","L1_SingleMu7",1,1,"11","new");           
  menu->AddHlt("HLT1MuonNonIso13","L1_SingleMu10",1,1,"13","new");          
  menu->AddHlt("HLT1MuonNonIso15","L1_SingleMu10",1,1,"15","new");          
  menu->AddHlt("HLT1MuonNonIso","L1_SingleMu7",1,1,"16","new");            

  menu->AddHlt("HLT2MuonIso","L1_DoubleMu3",1,1,"(3,3)","1e32"); 
  menu->AddHlt("HLT2MuonNonIso","L1_DoubleMu3",1,1,"(3,3)","1e32");
  menu->AddHlt("HLT2MuonJPsi","L1_DoubleMu3",1,1,"(3,3)","1e32"); 
  menu->AddHlt("HLT2MuonUpsilon","L1_DoubleMu3",1,1,"(3,3)","1e32"); 
  menu->AddHlt("HLT2MuonZ","L1_DoubleMu3",1,1,"(7,7)","1e32"); 
  menu->AddHlt("HLTNMuonNonIso","L1_DoubleMu3",1,1,"(3,3,3) ","1e32"); 
  menu->AddHlt("HLT2MuonSameSign","L1_DoubleMu3",1,1,"(3,3)","1e32");
  menu->AddHlt("CandHLT2MuonPrescaleVtx2cm","L1_DoubleMu3",1,1,"(3,3)","1e32"); 
  menu->AddHlt("CandHLT2MuonPrescaleVtx2mm","L1_DoubleMu3",1,1,"(3,3)","1e32"); 

  menu->AddHlt("HLTB1JetMu","L1_Mu5_Jet15",1,1,"20 ","1e32"); 
  menu->AddHlt("HLTB2JetMu","L1_Mu5_Jet15",1,1,"120","1e32"); 
  menu->AddHlt("HLTB3JetMu","L1_Mu5_Jet15",1,1,"70","1e32"); 
  menu->AddHlt("HLTB4JetMu","L1_Mu5_Jet15",1,1,"40","1e32"); 
  menu->AddHlt("HLTBHTMu","L1_HTT250",1,1,"300","1e32"); 

  menu->AddHlt("HLTB2JetMu60","L1_Mu5_Jet15",1,1,"60","new");                   
  menu->AddHlt("HLTB3JetMu40","L1_Mu5_Jet15",1,1,"40","new");                   
  menu->AddHlt("HLTB4JetMu30","L1_Mu5_Jet15",1,1,"30","new");                   
  menu->AddHlt("HLTBHTMu250","L1_HTT200",1,1,"250","new");                      

  menu->AddHlt("HLTBJPsiMuMu","L1_DoubleMu3",1,1,"$\\ast$ (4,4) $M_{\\mu\\mu} \\in [1,6]$","1e32");
  menu->AddHlt("HLTBJPsiMuMuRelaxed","L1_DoubleMu3",1,1,"$\\ast$ (3,3) $M_{\\mu\\mu} \\in [1,6]$","new");   
  menu->AddHlt("HLTTauTo3Mu","L1_DoubleMu3",1,1,"NI $\\ast$ (3,3,3)$^{\\S}$","new");                      //
  menu->AddHlt("HLTXMuonBJet","L1_Mu5_Jet15",1,1,"(7,35)","1e32"); 
  menu->AddHlt("HLTXMuonBJetSoftMuon","L1_Mu5_Jet15",1,1,"(7,20)","1e32"); 
  menu->AddHlt("HLTXMuonJets","L1_Mu5_Jet15",1,1,"(7,40)","1e32"); 

  menu->AddHlt("CandHLTXMuonNoL2IsoJets","L1_Mu5_Jet15",1,1,"NI (8,40)","1e32");         //        
  menu->AddHlt("CandHLTXMuonNoIsoJets","L1_Mu5_Jet15",1,1,"NI (14,40)","1e32");          //

  menu->AddHlt("HLT1ElectronStartup","L1_SingleIsoEG12",1,1,"-","new");
  menu->AddHlt("HLT1ElectronRelaxedStartup","L1_SingleIsoEG15",1,1,"-","new");
  menu->AddHlt("HLT2ElectronStartup","L1_DoubleIsoEG8",1,1,"-","new");
  menu->AddHlt("HLT2ElectronRelaxedStartup","L1_DoubleIsoEG10",1,1,"-","new");

  menu->AddHlt("HLTXElectronMuon","\\star",1,1,"(8,7)","1e32"); 
  menu->AddHlt("HLTXElectronMuonRelaxed","\\star",1,1,"(10,10)","1e32"); 
  menu->AddHlt("HLTXMuonTau","L1_Mu5_TauJet20",1,1,"(15,20)","1e32");
  
  menu->AddHlt("HLT1Level1jet15","L1_SingleJet15",10,100,"-","new"); // L1: 10!
  menu->AddHlt("HLT1jet30","L1_SingleJet15",10,20,"30","new");  // L1: 10
  menu->AddHlt("HLT1jet50","L1_SingleJet30",1,10,"50","new"); 
  menu->AddHlt("HLT1jet80","L1_SingleJet50",1,5,"80","new"); 
  menu->AddHlt("HLT1jet110","L1_SingleJet70",1,1,"110","new"); 
  menu->AddHlt("HLT1jet180","L1_SingleJet70",1,1,"180","new"); 
  menu->AddHlt("HLT1jet250","L1_SingleJet70",1,1,"250","new");

  menu->AddHlt("HLT2jetAve15","L1_SingleJet15",10,20,"15","new");  // L1: 10
  menu->AddHlt("HLT2jetAve30","L1_SingleJet30",1,10,"30","new"); 
  menu->AddHlt("HLT2jetAve50","L1_SingleJet50",1,1,"50","new"); 
  menu->AddHlt("HLT2jetAve70","L1_SingleJet70",1,1,"70","new"); 
  menu->AddHlt("HLT2jetAve130","L1_SingleJet130",1,1,"130","new"); 
  menu->AddHlt("HLT2jetAve220","L1_SingleJet220",1,1,"220","new"); 

  menu->AddHlt("HLT2jet","L1_SingleJet150,L1_DoubleJet70",1,1,"150","1e32"); 
  menu->AddHlt("HLT3jet","\\dagger",1,1,"85","1e32"); 
  menu->AddHlt("HLT4jet","\\ddagger",1,1,"60","1e32"); 

  menu->AddHlt("HLT2jetAco","L1_SingleJet150, DoubleJet70",1,1,"125","1e32"); 
  menu->AddHlt("HLT1jet1METAco","L1_SingleJet150",1,1,"(100,60)","1e32"); 

  menu->AddHlt("HLT1jet1MET","L1_SingleJet150",1,1,"(180,60)","1e32"); 
  menu->AddHlt("HLT2jet1MET","L1_SingleJet150",1,1,"(125,60)","1e32"); 
  menu->AddHlt("HLT3jet1MET","L1_SingleJet150",1,1,"(60,60)","1e32"); 
  menu->AddHlt("HLT4jet1MET","L1_SingleJet150",1,1,"(35,60)","1e32"); 

  menu->AddHlt("HLT1MET1HT","L1_HTT300 ",1,1,"(350,65)","1e32"); 
  menu->AddHlt("HLT1SumET","L1_ETT60",1,500,"120","1e32"); 
  menu->AddHlt("HLT1Level1MET20","L1_ETM20",1,500,"20","new");                
  menu->AddHlt("HLT1MET25","L1_ETM20",1,50,"25","new");                      
  menu->AddHlt("HLT1MET35","L1_ETM30",1,1,"35","new"); 
  menu->AddHlt("HLT1MET50","L1_ETM40",1,1,"50","new"); 
  menu->AddHlt("HLT1MET65","L1_ETM50",1,1,"65","new"); 
  menu->AddHlt("HLT1MET75","L1_ETM50",1,1,"75","new"); 
  
  menu->AddHlt("HLT2jetvbfMET","L1_ETM30",1,1,"(40,60)","1e32"); 
  menu->AddHlt("HLTS2jet1METNV","L1_SingleJet150",1,1,"(-,60)","1e32"); 
  menu->AddHlt("HLTS2jet1METAco","L1_SingleJet150",1,1,"(-,70)","1e32"); 
  menu->AddHlt("HLTSjet1MET1Aco","L1_SingleJet150",1,1,"(-,70)","1e32"); 
  menu->AddHlt("HLTSjet2MET1Aco","L1_SingleJet150",1,1,"(-,70)","1e32"); 
  menu->AddHlt("HLTS2jetAco","L1_SingleJet150",1,1,"NI (-,-)","1e32");                            //
  menu->AddHlt("HLTJetMETRapidityGap","L1_IsoEG10_Jet20_ForJet10",1,1,"20","1e32");
  menu->AddHlt("HLT4jet30","L1_QuadJet15",10,1,"30","new"); // L1: 10!

  menu->AddHlt("CandHLTXMuonNoIso3Jets30","L1_Mu3_TripleJet20",1,1,"(3,30)","new");       
  menu->AddHlt("CandHLTXElectron3Jet30","L1_EG5_TripleJet20",1,1,"(5,30)","new");         


  menu->AddHlt("HLT1ElectronEt12_L1R_HI","L1_SingleEG8",1,5,"12","new");       
  menu->AddHlt("HLT1Electron8_L1R_NI","L1_SingleEG5",1,10,"8","new");       
  menu->AddHlt("HLT1Electron10_L1R_NI","L1_SingleEG8",1,10,"10","new");       
  menu->AddHlt("HLT1ElectronEt15_L1R_NI","L1_SingleEG12",1,5,"15","new");       
  menu->AddHlt("HLT1ElectronEt18_L1R_NI","L1_SingleEG15",1,1,"18","new");
  menu->AddHlt("HLT1ElectronLWEt12_L1R_NI","L1_SingleEG8",1,10,"12","new");       
  menu->AddHlt("HLT1ElectronLWEt15_L1R_NI","L1_SingleEG10",1,10,"15","new");       
  menu->AddHlt("HLT2Electron5_L1R_NI","L1_DoubleEG5",1,10,"(5,5)","new");       
  menu->AddHlt("HLT1ElectronEt15_L1R_LI","L1_SingleEG12",1,1,"15","new");       
  menu->AddHlt("HLT1ElectronLWEt15_L1R_LI","L1_SingleEG12",1,5,"15","new");       
  menu->AddHlt("HLT1ElectronLWEt18_L1R_LI","L1_SingleEG15",1,5,"18","new");       

  //menu->AddHlt("HLT2ElectronLWonlyPMEt8_L1R_NI","L1_DoubleEG5",1,1,"$8^\\circ$","new");       
  //menu->AddHlt("HLT2ElectronLWonlyPMEt10_L1R_NI","L1_DoubleEG5",1,1,"$10^\\circ$","new");       
  //menu->AddHlt("HLT2ElectronLWonlyPMEt12_L1R_NI","L1_DoubleEG10",1,1,"$12^\\circ$","new");       
  // Just shorten names
  menu->AddHlt("2ElecLWonlyPMEt8_L1R_NI","L1_DoubleEG5",1,1,"$8^\\circ$","new");       
  menu->AddHlt("2ElecLWonlyPMEt10_L1R_NI","L1_DoubleEG5",1,1,"$10^\\circ$","new");       
  menu->AddHlt("2ElecLWonlyPMEt12_L1R_NI","L1_DoubleEG10",1,1,"$12^\\circ$","new");       
  
  menu->AddHlt("HLT1Electron","L1_SingleIsoEG12",1,1,"15","1e32"); 
  menu->AddHlt("HLT1ElectronRelaxed","L1_SingleEG15",1,1,"18","1e32"); 
  menu->AddHlt("HLT2Electron","L1_DoubleIsoEG8",1,1,"10","1e32"); 
  menu->AddHlt("HLT2ElectronRelaxed","L1_DoubleEG10",1,1,"12","1e32"); 

  
  menu->AddHlt("HLT1Photon10_L1R","L1_SingleEG8",1,10,"10","new");       
  menu->AddHlt("HLT1PhotonEt15_L1R_HI","L1_SingleEG10",1,1,"15","new");
  menu->AddHlt("HLT1PhotonEt25_L1R_HI","L1_SingleEG12",1,1,"25","new");       
  menu->AddHlt("HLT1PhotonEt20_L1R_LI","L1_SingleEG12",1,1,"20","new");       
  menu->AddHlt("HLT1PhotonEt30_L1R_LI","L1_SingleEG12",1,1,"30","new");       
  menu->AddHlt("HLT1PhotonEt40_L1R_LI","L1_SingleEG12",1,1,"40","new");       
  menu->AddHlt("HLT1PhotonEt45_L1R_LI","L1_SingleEG12",1,1,"45","new");       
  menu->AddHlt("HLT1PhotonEt15_L1R_NI","L1_SingleEG10",1,10,"15","new");       
  menu->AddHlt("HLT1PhotonEt25_L1R_NI","L1_SingleEG10",1,1,"25","new");       
  menu->AddHlt("HLT1PhotonEt30_L1R_NI","L1_SingleEG15",1,1,"30","new");       
  menu->AddHlt("HLT1PhotonEt40_L1R_NI","L1_SingleEG15",1,1,"40","new");       
  menu->AddHlt("HLT2PhotonEt20_L1R_LI","L1_DoubleEG10",1,1,"20","new");         
  menu->AddHlt("HLT2PhotonEt8_L1R_NI","L1_DoubleEG5",1,10,"8","new");       
  menu->AddHlt("HLT2PhotonEt10_L1R_NI","L1_DoubleEG5",1,10,"10","new");       
  menu->AddHlt("HLT2PhotonEt20_L1R_NI","L1_DoubleEG10",1,1,"20","new");       
  
  menu->AddHlt("HLT1Photon","L1_SingleIsoEG12",1,1,"30","1e32"); 
  menu->AddHlt("HLT1PhotonRelaxed","L1_SingleEG15",1,1,"40","1e32"); 
  menu->AddHlt("HLT2Photon","L1_DoubleIsoEG8",1,1,"(20,20)","1e32"); 
  menu->AddHlt("HLT2PhotonRelaxed","L1_DoubleEG10",1,1,"(20,20)","1e32"); 

  menu->AddHlt("HLT1EMHighEt","L1_SingleEG15",1,1,"80","1e32"); 
  menu->AddHlt("HLT1EMVeryHighEt","L1_SingleEG15",1,1,"200","1e32"); 
  menu->AddHlt("HLT2ElectronZCounter","L1_DoubleIsoEG8",1,1,"(10,10)","1e32"); 
  menu->AddHlt("HLT2ElectronExclusive","L1_ExclusiveDoubleIsoEG6",1,1,"(6,6)","1e32"); 
  menu->AddHlt("HLT2PhotonExclusive","L1_ExclusiveDoubleIsoEG6",1,1,"(10,10)","1e32"); 
  menu->AddHlt("HLT1PhotonL1Isolated","L1_SingleIsoEG10",1,1,"12","1e32"); 

  menu->AddHlt("HLTB1Jet","\\Diamond ",1,1,"180","1e32"); 
  menu->AddHlt("HLTB2Jet","\\Diamond ",1,1,"120","1e32"); 
  menu->AddHlt("HLTB3Jet","\\Diamond ",1,1,"70","1e32"); 
  menu->AddHlt("HLTB4Jet","\\Diamond ",1,1,"40","1e32"); 
  menu->AddHlt("HLTBHT","\\Diamond ",1,1,"470","1e32"); 

  menu->AddHlt("HLTB1Jet120","\\Diamond ",1,1,"120","new");    
  menu->AddHlt("HLTB2Jet60","\\Diamond ",1,1,"60","new");       
  menu->AddHlt("HLTB3Jet40","\\Diamond ",1,1,"40","new");      
  menu->AddHlt("HLTB4Jet30","\\Diamond ",1,1,"30","new");      
  menu->AddHlt("HLTBHT320","\\Diamond ",1,1,"320","new");      

  menu->AddHlt("HLTB1Jet160","\\Diamond ",1,1,"NI","new");
  menu->AddHlt("HLTB2Jet100","\\Diamond ",1,1,"NI","new");
  menu->AddHlt("HLTB2JetMu100","\\Diamond ",1,1,"NI","new");
  menu->AddHlt("HLTB3Jet60","\\Diamond ",1,1,"NI","new");
  menu->AddHlt("HLTB3JetMu60","\\Diamond ",1,1,"NI","new");
  menu->AddHlt("HLTB4Jet35","\\Diamond ",1,1,"NI","new");
  menu->AddHlt("HLTB4JetMu35","\\Diamond ",1,1,"NI","new");
  menu->AddHlt("HLTBHT420","\\Diamond ",1,1,"NI","new");
  menu->AddHlt("HLTBHTMu330","\\Diamond ",1,1,"NI","new");
    
  menu->AddHlt("HLT1Tau","L1_SingleTauJet80",1,1,"15","1e32"); 
  menu->AddHlt("HLT1Tau1MET","L1_TauJet30_ETM30",1,1,"15","1e32"); 
  menu->AddHlt("HLT2TauPixel","L1_TauJet40",1,1,"15","1e32"); 

  menu->AddHlt("HLT1TauRelaxed","L1_SingleTauJet80",1,1,"15","new");          
  menu->AddHlt("HLT1Tau1METRelaxed","L1_TauJet30_ETM30",1,1,"15","new");      
  menu->AddHlt("HLT2TauPixelRelaxed","L1_DoubleTau20",1,1,"15","new");        

  menu->AddHlt("HLTXElectronBJet","L1_IsoEG10_Jet20",1,1,"(10,35)","1e32"); 
  menu->AddHlt("HLTXElectron1Jet","L1_IsoEG10_Jet30",1,1,"(12,40)","1e32"); 
  menu->AddHlt("HLTXElectron2Jet","L1_IsoEG10_Jet30",1,1,"(12,80)","1e32"); 
  menu->AddHlt("HLTXElectron3Jet","L1_IsoEG10_Jet30",1,1,"(12,60)","1e32"); 
  menu->AddHlt("HLTXElectron4Jet","L1_IsoEG10_Jet30",1,1,"(12,35)","1e32"); 
  menu->AddHlt("HLTXElectronTau","L1_IsoEG10_TauJet20",1,1,"(12,20)","1e32"); 

  menu->AddHlt("HLTMinBias","L1_MinBias_HTT10",300000,1,"-","1e32");  // L1: 300000
  menu->AddHlt("HLTMinBiasPixel","L1_ZeroBias",300000,1,"- ","1e32");  // L1: 300000
  menu->AddHlt("HLTMinBiasHcal","\\Diamond\\Diamond",12,1000,"-","new");          // L1: 12
  menu->AddHlt("HLTMinBiasEcal","L1_SingleEG2,L1_DoubleEG1",20,1000,"-","new"); // L1: 20
  menu->AddHlt("HLTZeroBias","L1_ZeroBias",300000,1,"-","1e32"); // L1: 300000


  // Testing
  menu->AddHlt("OpenHLT1Electron","L1_SingleIsoEG12",1,1,"15","1e32");
  menu->AddHlt("OpenHLT1Photon","L1_SingleIsoEG12",1,1,"30","1e32"); 
  menu->AddHlt("OpenHLT1jet","L1_SingleJet159",1,1,"200","1e32"); 
  menu->AddHlt("OpenHLT1MuonNonIso","L1_SingleMu7",1,1,"16","1e32");
  menu->AddHlt("OpenHLT1MuonIso","L1_SingleMu7",1,1,"11","1e32");
  menu->AddHlt("OpenHLT2MuonNonIso","L1_DoubleMu3",1,1,"(3,3)","1e32"); 


  // New Tau Triggers
  menu->AddHlt("HighPtTauMET","L1_SingleTauJet80",1,1,"20","new"); 
  menu->AddHlt("TauMET","L1_TauJet30_ETM30",1,1,"20","new"); 
  menu->AddHlt("DiTau","L1_TauJet40",1,1,"15","new"); 
  menu->AddHlt("MuonTau","L1_Mu5_TauJet20",1,1,"15","new"); 
  menu->AddHlt("ElectronTau","L1_IsoEG10_TauJet20",1,1,"15","new"); 

  menu->AddHlt("TauMET_NoSi","L1_TauJet30_ETM30",1,1,"20","new"); 
  menu->AddHlt("DiTau_NoSi","L1_TauJet40",1,1,"15","new"); 
  menu->AddHlt("MuonTau_NoSi","L1_Mu5_TauJet20",1,1,"15","new"); 
  menu->AddHlt("ElectronTau_NoSi","L1_IsoEG10_TauJet20",1,1,"15","new");

  menu->AddHlt("MuonTau_NoL1","L1_Mu5_Jet15",1,1,"15","new"); 
  menu->AddHlt("MuonTau_NoL2","L1_Mu5_TauJet20",1,1,"15","new"); 
  menu->AddHlt("MuonTau_NoL25","L1_Mu5_TauJet20",1,1,"15","new"); 

  menu->AddHlt("ElectronTau_NoL1","L1_IsoEG10_Jet15",1,1,"15","new");
  menu->AddHlt("ElectronTau_NoL2","L1_IsoEG10_TauJet20",1,1,"15","new");
  menu->AddHlt("ElectronTau_NoL25","L1_IsoEG10_TauJet20",1,1,"15","new");

  menu->AddHlt("ElectronMET","L1_SingleIsoEG12",1,1,"10","new");

  

}

void BookMenu_L1Default(OHltMenu*  menu, double &iLumi, double &nBunches) {   
   
  iLumi = 2E30;   
  nBunches = 43;   
 
  menu->AddHlt("L1_SingleMuOpen","L1_SingleMuOpen",1,1,"-","1e32");
  menu->AddHlt("L1_SingleMu3","L1_SingleMu3",1,1,"-","1e32");
  menu->AddHlt("L1_SingleMu5","L1_SingleMu5",1,1,"-","1e32");            
  menu->AddHlt("L1_SingleMu7", "L1_SingleMu7",1,1,"-","1e32");            
  menu->AddHlt("L1_SingleMu10", "L1_SingleMu10",1,1,"-","1e32");           
  menu->AddHlt("L1_SingleMuBeamHalo", "L1_SingleMuBeamHalo",1,1,"-","1e32");
  menu->AddHlt("L1_DoubleMu3", "L1_DoubleMu3",1,1,"-","1e32");            
  menu->AddHlt("L1_TripleMu3", "L1_TripleMu3",1,1,"-","1e32");            

  menu->AddHlt("L1_SingleIsoEG10", "L1_SingleIsoEG10",1,1,"-","1e32");      
  menu->AddHlt("L1_SingleIsoEG12", "L1_SingleIsoEG12",1,1,"-","1e32");      
  menu->AddHlt("L1_DoubleIsoEG8", "L1_DoubleIsoEG8",1,1,"-","1e32");        
  
  menu->AddHlt("L1_SingleEG2", "L1_SingleEG2",1,1,"-","1e32");           
  menu->AddHlt("L1_SingleEG5", "L1_SingleEG5",1,1,"-","1e32");            
  //  menu->AddHlt("L1_SingleEG8", "L1_SingleEG8",1,1,"-","1e32");            
  menu->AddHlt("L1_SingleEG8", "L1_SingleEG8",1,1,"-","1e32");
  menu->AddHlt("L1_SingleEG10", "L1_SingleEG10",1,1,"-","1e32");           
  menu->AddHlt("L1_SingleEG12", "L1_SingleEG12",1,1,"-","1e32");          
  menu->AddHlt("L1_SingleEG15", "L1_SingleEG15",1,1,"-","1e32");         
  menu->AddHlt("L1_DoubleEG1", "L1_DoubleEG1",1,1,"-","1e32");           
  menu->AddHlt("L1_DoubleEG5", "L1_DoubleEG5",1,1,"-","1e32");            
  menu->AddHlt("L1_DoubleEG10", "L1_DoubleEG10",1,1,"-","1e32");           
  
  menu->AddHlt("L1_SingleJet15", "L1_SingleJet15",1,1,"-","1e32"); 
  //  menu->AddHlt("L1_SingleJet15", "L1_SingleJet15",1,1,"-","1e32");  
  menu->AddHlt("L1_SingleJet30", "L1_SingleJet30",1,1,"-","1e32");
  //  menu->AddHlt("L1_SingleJet30", "L1_SingleJet30",1,1,"-","1e32");
  menu->AddHlt("L1_SingleJet50", "L1_SingleJet50",1,1,"-","1e32"); 
  menu->AddHlt("L1_SingleJet70", "L1_SingleJet70",1,1,"-","1e32"); 
  menu->AddHlt("L1_SingleJet100", "L1_SingleJet100",1,1,"-","1e32");
  menu->AddHlt("L1_SingleJet150", "L1_SingleJet150",1,1,"-","1e32");
  menu->AddHlt("L1_SingleJet200", "L1_SingleJet200",1,1,"-","1e32");
  menu->AddHlt("L1_DoubleJet70", "L1_DoubleJet70",1,1,"-","1e32");  
  menu->AddHlt("L1_DoubleJet100", "L1_DoubleJet100",1,1,"-","1e32");
  menu->AddHlt("L1_TripleJet50", "L1_TripleJet50",1,1,"-","1e32");  
  menu->AddHlt("L1_QuadJet15", "L1_QuadJet15",1,1,"-","1e32"); 
  menu->AddHlt("L1_QuadJet30", "L1_QuadJet30",1,1,"-","1e32");  
  menu->AddHlt("L1_HTT200", "L1_HTT200",1,1,"-","1e32");        
  menu->AddHlt("L1_HTT300", "L1_HTT300",1,1,"-","1e32");        
  
  menu->AddHlt("L1_ETM20", "L1_ETM20",1,1,"-","1e32");
  menu->AddHlt("L1_ETM30", "L1_ETM30",1,1,"-","1e32");  
  menu->AddHlt("L1_ETM40", "L1_ETM40",1,1,"-","1e32");  
  menu->AddHlt("L1_ETM50", "L1_ETM50",1,1,"-","1e32");  
  menu->AddHlt("L1_ETT60", "L1_ETT60",1,1,"-","1e32");
  
  menu->AddHlt("L1_SingleTauJet30", "L1_SingleTauJet30",1,1,"-","1e32");               
  menu->AddHlt("L1_SingleTauJet40", "L1_SingleTauJet40",1,1,"-","1e32"); 
  menu->AddHlt("L1_SingleTauJet60", "L1_SingleTauJet60",1,1,"-","1e32");   
  menu->AddHlt("L1_SingleTauJet80", "L1_SingleTauJet80",1,1,"-","1e32");   
  menu->AddHlt("L1_DoubleTauJet20", "L1_DoubleTauJet20",1,1,"-","1e32");   
  menu->AddHlt("L1_DoubleTauJet40", "L1_DoubleTauJet40",1,1,"-","1e32");   
  
  menu->AddHlt("L1_IsoEG10_Jet15_ForJet10", "L1_IsoEG10_Jet15_ForJet10",1,1,"-","1e32"); 
  menu->AddHlt("L1_ExclusiveDoubleIsoEG6", "L1_ExclusiveDoubleIsoEG6",1,1,"-","1e32");   
  menu->AddHlt("L1_Mu5_Jet15", "L1_Mu5_Jet15",1,1,"-","1e32");    
  menu->AddHlt("L1_IsoEG10_Jet20", "L1_IsoEG10_Jet20",1,1,"-","1e32"); 
  menu->AddHlt("L1_IsoEG10_Jet30", "L1_IsoEG10_Jet30",1,1,"-","1e32"); 
  menu->AddHlt("L1_Mu3_IsoEG5", "L1_Mu3_IsoEG5",1,1,"-","1e32");  
  menu->AddHlt("L1_Mu3_EG12", "L1_Mu3_EG12",1,1,"-","1e32");    
  menu->AddHlt("L1_IsoEG10_TauJet20", "L1_IsoEG10_TauJet20",1,1,"-","1e32"); 
  menu->AddHlt("L1_Mu5_TauJet20", "L1_Mu5_TauJet20",1,1,"-","1e32"); 
  menu->AddHlt("L1_TauJet30_ETM30", "L1_TauJet30_ETM30",1,1,"-","1e32"); 
  menu->AddHlt("L1_EG5_TripleJet15", "L1_EG5_TripleJet15",1,1,"-","1e32");
  menu->AddHlt("L1_Mu3_TripleJet15", "L1_Mu3_TripleJet15",1,1,"-","1e32");
  
  menu->AddHlt("L1_ZeroBias", "L1_ZeroBias",1,1,"-","1e32");  
  menu->AddHlt("L1_MinBias_HTT10", "L1_MinBias_HTT10",1,1,"-","1e32");  
  //  menu->AddHlt("L1_ZeroBias", "L1_ZeroBias",1,1,"-","1e32"); 
  //  menu->AddHlt("L1_MinBias_HTT10", "L1_MinBias_HTT10",1,1,"-","1e32"); 
  menu->AddHlt("L1_SingleJetCountsHFTow 12","L1_SingleJetCountsHFTow 12",1,1,"-","1e32");
  menu->AddHlt("L1_DoubleJetCountsHFTow 10","L1_DoubleJetCountsHFTow 10",1,1,"-","1e32");
  menu->AddHlt("L1_SingleJetCountsHFRing0Sum3", "L1_SingleJetCountsHFRing0Sum3",1,1,"-","1e32"); 
  menu->AddHlt("L1_DoubleJetCountsHFRing0Sum3", "L1_DoubleJetCountsHFRing0Sum3",1,1,"-","1e32");  
  menu->AddHlt("L1_SingleJetCountsHFRing0Sum6", "L1_SingleJetCountsHFRing0Sum6",1,1,"-","1e32");  
  menu->AddHlt("L1_DoubleJetCountsHFRing0Sum6", "L1_DoubleJetCountsHFRing0Sum6",1,1,"-","1e32");  

}
