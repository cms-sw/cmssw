/////////////////////////////////////////////////////////////////////////////////////////////////
//
//   Program to calculate HLT rates
//
//   Contacts: Jonathan Hollar (LLNL), Chi Nhan Nguyen (TAMU)
//
//   2008 June 10
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
#include "TChain.h"
#include "TCut.h"

#include <map>

using namespace std;

// Declaration of different Menus
void BookMenu_Default(OHltMenu *menu, double &iLumi, double &nBunches);
void BookMenu_OhltExample(OHltMenu *menu, double &iLumi, double &nBunches);
void BookMenu_TauStudy(OHltMenu *menu, double &iLumi, double &nBunches);

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
  cout << "  Usage:  ./OHltRates <nevents> <menu> <conditions> <version tag> <doPrintAll> " << endl;
  cout << "default:  ./OHltRates -1 default startup 20June2008 0 " << endl;
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
  int PrintAll = 0;
  if (argc>5) {
    PrintAll = atoi(argv[5]);
  }


  ////////////////////////////////////////////////////////////

  /**** Different Beam conditions: ****/ 
  // Fixed LHC defaults
  const double bunchCrossingTime = 25.0E-09;  // 25 ns
  const double maxFilledBunches = 3557;
    
  // Defaults, to be changed in the menu booking
  double ILumi = 1.E27;
  double nFilledBunches = 1;

  /**********************************/


  // Choice of menus
  OHltMenu* menu = new OHltMenu();
  if(sMenu.CompareTo("default") == 0)
    BookMenu_Default(menu,ILumi,nFilledBunches);
  else if(sMenu.CompareTo("example") == 0)
    BookMenu_OhltExample(menu,ILumi,nFilledBunches);
  else if(sMenu.CompareTo("tau") == 0)
    BookMenu_TauStudy(menu,ILumi,nFilledBunches);
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
    TString PPEX_DIR="dcap://cmsdcap.hep.wisc.edu:22125/pnfs/hep.wisc.edu/data5/uscms01/cnhan/ppEleX_hltana_184bckprt-hltana/";
    ProcFil.clear();
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-042D3420-1DE3-DC11-A822-0030482374D6.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-04CFF518-24DA-DC11-98E0-0030482C93AC.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-062D143B-B7D8-DC11-A644-003048770BB4.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-0649E97C-13DB-DC11-9CC6-0030487721F4.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-0EFE947A-38E1-DC11-AE83-003048770C64.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-18FE6933-32E1-DC11-9D61-003048770C30.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-20F4C41B-80DE-DC11-A237-003048770BBC.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-22FF9459-FFDF-DC11-A629-003048770C5A.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-266965F2-2BE1-DC11-A854-003048770BB8.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-281369D8-D2DB-DC11-BDD2-003048770BB4.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-28D76F67-96DC-DC11-9BFF-003048770E30.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-2ABEBB32-9BDC-DC11-AAB5-003048770E2E.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-30355F84-B8DE-DC11-94B7-003048772384.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-34AAD171-96DC-DC11-8AED-003048770BBC.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-3A07AE0A-68DD-DC11-B4BB-003048341AFA.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-40DB37C2-22DA-DC11-84C8-0030482CC25A.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-42A89670-91DF-DC11-872F-00304824249B.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-547C4EFB-90DF-DC11-B399-003048770C6A.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-581A772B-65D9-DC11-9038-0030485941DC.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-648E49DB-54DD-DC11-808F-00304824243A.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-6A910896-5DE3-DC11-AF28-003048770B8E.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-6CE74EA8-A1DF-DC11-B05A-003048770B8C.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-6E2FA0F5-85E0-DC11-85CE-003048770BA8.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-7091831A-96DC-DC11-AB93-003048770C44.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-7C856A7A-42E4-DC11-B3BD-003048770BAA.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-80C66601-D3DB-DC11-9517-003048770DB8.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-82718E20-6EDE-DC11-95A6-00304823F5ED.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-82FC31A3-15DB-DC11-A947-0002B3E92671.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-884AFC24-55DD-DC11-86D7-0030482CD996.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-8A61A9FC-2BE1-DC11-8B9B-0030482CDA6A.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-8ACFAE1C-6EDE-DC11-A279-003048237499.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-8E2FB660-2FE1-DC11-97A9-003048770C44.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-8EACF622-B8D8-DC11-A4E9-003048770DCA.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-90928226-55DD-DC11-AAB9-0030482CD97A.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-9AE97EEA-54E4-DC11-9506-003048770DB8.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-9E9455C4-17DB-DC11-97D2-0030482CDA42.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-A0B9FAFF-D2DB-DC11-A7AA-003048770C64.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-A0E3ACEC-00E5-DC11-89C5-0030482CD98E.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-A8B4D5CF-C5DB-DC11-9D64-003048770B8A.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-AA34C2B5-22DA-DC11-9440-0030482C9332.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-AA66DC62-42E4-DC11-BDF0-003048770DBC.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-AAB3D52B-00E5-DC11-BA32-0030482374D6.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-AC1EB9D9-96DC-DC11-A269-003048770DCC.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-BA9577E7-D2DB-DC11-A26E-003048770DBE.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-BCD368F8-2BE1-DC11-BFFD-0030482CDA5E.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-C0414A3E-33E1-DC11-98A7-003048770DB6.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-C6CA427B-97E0-DC11-B17A-003048770C5A.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-C828A545-96DC-DC11-85F6-003048770D66.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-CA8D9F8E-5DE3-DC11-B342-003048770DCA.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-CCD9B22D-9BDC-DC11-9075-003048770E2E.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-D808D25B-67DD-DC11-B851-003048770DBA.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-DAD57760-48DD-DC11-80B3-000347FF4517.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-E213B7BD-22DA-DC11-B2A1-0030482C92EE.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-E23339E7-D2DB-DC11-A18D-003048770BA8.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-E437DBC5-B7DE-DC11-9A03-003048770C6A.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-E4ACD56F-70DA-DC11-A85B-003048770BB4.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-EAD8AE2D-9BDC-DC11-9886-003048770E2E.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-EE83DAD1-30E1-DC11-8398-003048770D68.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-F0262F97-2BE1-DC11-B512-003048770DB6.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-F07D71BA-96DC-DC11-AADA-003048770BA8.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-F652D857-65D9-DC11-93B4-003048770D6A.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-F8399A8B-21DA-DC11-B973-000347FF44AD.root");
    ProcFil.push_back(PPEX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-F84A2C02-55DD-DC11-A7D2-0030482E5B8C.root");
    
    TabChain.push_back(new TChain("HltTree"));
    for (unsigned int ipfile = 0; ipfile < ProcFil.size(); ipfile++){
      TabChain.back()->Add(ProcFil[ipfile]);
    }
    doMuonCut.push_back(true); doElecCut.push_back(false);
    
    xsec.push_back(9.51E7); // PYTHIA cross-section times filter pp->eleX (7.923E10*0.0012)
    skmeff.push_back(1.);  //


    // ppMuX
    TString PPMUX_DIR="dcap://cmsdcap.hep.wisc.edu:22125/pnfs/hep.wisc.edu/data5/uscms01/cnhan/ppMuX_hltana_184bckprt-hltana/";
    ProcFil.clear();
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-90B6686E-52ED-DC11-8509-003048770C44.root");
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-9E919C9E-DCED-DC11-87D5-00E08132879C.root");
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-A04BD1F2-B7EB-DC11-B4CA-001731AF6783.root");
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-A8E4311A-0AEE-DC11-94F1-00E08133D50E.root");
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-B06EB10C-52ED-DC11-B209-00304875652B.root");
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-B6F8952E-DFEC-DC11-8DC0-001731A28BE1.root");
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-C60E2A0C-9DE7-DC11-B4CD-001731A28EDB.root");
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-C80EC50E-5DE7-DC11-9EC6-003048726DCB.root");
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-CE2D32DE-3CEC-DC11-8454-003048770C64.root");
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-CED392D6-1EED-DC11-BC10-001A92971B64.root");
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-CEDCD9A1-4EED-DC11-A70C-00304875AAE7.root");
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-D23CBAA3-4EED-DC11-A78E-00304875AA6F.root");
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-DAF32AA9-22EE-DC11-9CE1-001C23C105FC.root");
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-E20D367C-90EE-DC11-8111-00145E552564.root");
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-E4295EB2-6EEE-DC11-B9AE-001C23C0D109.root");
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-E8422574-57E7-DC11-A1B6-001A92810AD0.root");
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-EED21CF2-77ED-DC11-B919-00304876A0D9.root");
    ProcFil.push_back(PPMUX_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw-F2532987-DDED-DC11-8F4F-00E08134C270.root");
  
    TabChain.push_back(new TChain("HltTree"));
    for (unsigned int ipfile = 0; ipfile < ProcFil.size(); ipfile++){
      TabChain.back()->Add(ProcFil[ipfile]);
    }
    doMuonCut.push_back(false); doElecCut.push_back(true);
    
    xsec.push_back(6.8375E7); // PYTHIA cross-section times filter pp->muX (7.923E10*0.000863)
    skmeff.push_back(1.);  //

    // Minbias
    TString MB_DIR="dcap://cmsdcap.hep.wisc.edu:22125/pnfs/hep.wisc.edu/data5/uscms01/cnhan/Minbias_hltana_184bckprt-hltana/";
    ProcFil.clear();
    //ProcFil.push_back("../hltana_test.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-001D4178-11AD-DC11-87AF-001617E30CA4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-00298B16-22AD-DC11-A004-000423D98B6C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-0223EBA9-74AC-DC11-998E-000423D944FC.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-047FB65B-F7AD-DC11-B163-000423D98DD4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-062B20DB-ECAD-DC11-9106-000423D9865C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-06EC0352-73AC-DC11-960F-0030485617EA.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-081191F9-95AB-DC11-9BC9-001617C3B5F0.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-0A038E9B-8FAB-DC11-8030-000423D944F8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-0A45B96C-13AD-DC11-9F6F-000423D65DFE.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-0C496271-90AE-DC11-A27E-001617C3B6E2.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-0C9935F6-11AD-DC11-893A-000423D6C8E6.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-0CA12819-8CAB-DC11-B0E8-001617C3B76A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-0E318FB6-B7AD-DC11-83BD-000423D6CA72.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-0E73FE2E-76AC-DC11-B94E-000423D986A8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-0E9865B0-CCAC-DC11-B268-000E0C3E6D5B.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-10138975-39AD-DC11-9A65-0030485618A6.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-122B2BE2-47AD-DC11-AE81-000423D98B5C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-123FCAC5-6AAD-DC11-B2EE-000423D65FF2.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-12F6A386-6FAC-DC11-A641-001617C3B69C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-14302FDA-87AB-DC11-B7E1-001617E30F56.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-1438E990-77AC-DC11-928F-001617E30D38.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-14D01065-75AC-DC11-A615-000423D9870C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-14ECF5EF-0FAE-DC11-BA2F-0030485628D4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-14F4017C-E8AD-DC11-8E4B-001617E30D52.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-16636FAE-C3AD-DC11-BCC0-001617E30D4A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-168ACC2E-8EAC-DC11-99ED-001617C3B69C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-1694DEDD-C6AD-DC11-88C3-0030485627E2.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-1AC65EF6-3EAD-DC11-8EEE-00304885AA18.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-1C72F74A-A0AB-DC11-92C5-000423D992A4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-1CBB2249-E0AD-DC11-85F2-001617DBCEDC.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-1CBFAA0E-B2AD-DC11-AF6B-003048836638.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-1CDA44CB-A4AD-DC11-B120-000423D94700.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-1E08B7C4-A4AD-DC11-8469-00304885B16E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-1E83CF60-9DAD-DC11-B121-000E0C3F095B.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-1EDB141E-7BAC-DC11-B7E1-000423D6A3E4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-202C004C-81AD-DC11-983A-000423D9890C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-20305DF4-78AC-DC11-9C06-000423D674FE.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-2087F6A0-31AC-DC11-AA13-003048560F10.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-20A79BD3-A6AD-DC11-81C5-001617C3B726.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-22526CD7-7BAC-DC11-AAA6-000423D674BE.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-22B7777F-7CAC-DC11-951B-000423D98B08.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-242E43BB-C8AD-DC11-A221-000423D6C8E6.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-245585A4-D4AE-DC11-8880-000423D992DC.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-26D0F9E7-46AD-DC11-9659-000423D662EE.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-2823F8AD-E3AB-DC11-B14A-0030488318E0.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-28720F50-43AD-DC11-AB18-000423D6CA42.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-2A35FBE6-E2AC-DC11-8C67-00304885B400.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-2A5EB20A-8EAD-DC11-928B-00304856291E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-2C2B99C2-0CAD-DC11-941C-000423D94AA8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-2C312059-71AC-DC11-9CCA-003048562890.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-301386B7-93AD-DC11-95AD-00304856289E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-304F35DB-3EAD-DC11-8AB9-000423D98B6C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-30D41865-7AAC-DC11-9C8D-000423D951D4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-32AAB703-CEAC-DC11-A3D2-001617DBD558.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-341E4FF5-B9AD-DC11-B2E9-001617E30D38.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-345FAE51-78AC-DC11-A94C-000423D6006E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-34B58F93-63AC-DC11-917C-003048561110.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-34C52BAB-3FAD-DC11-9F82-00304855D4C8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-34EEC13E-70AC-DC11-B47A-001617DBD33C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-36ADF716-19AD-DC11-A787-001617C3B762.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-36E8E84F-73AC-DC11-8B01-0019DB29C5FC.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-3AA510A8-72AD-DC11-87C5-00304856284C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-3AAC3925-71AC-DC11-886C-000423D98930.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-3AE8D67D-48AD-DC11-8845-00304885B028.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-3C8212BA-D4AD-DC11-BC8E-001617DC1F70.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-3CF1F30C-74AC-DC11-A6C5-00304885AA5A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-3ECE3783-25AD-DC11-AF35-001617DBD57E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-4009D178-3BAD-DC11-B50E-00304855D4B8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-40415EDD-12AC-DC11-9F91-001617DBD22A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-426B774C-C0AE-DC11-AAAE-001617E30E28.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-426F0801-CEAC-DC11-A27F-000423D6B358.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-42791AD8-D0AC-DC11-AD24-000423D9863C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-42ADE157-14AD-DC11-ADFE-000E0C3F062F.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-442FB3FA-95AB-DC11-B974-000423D944F8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-4603D2EA-65AB-DC11-9E71-001617E30E2C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-466254BB-93AD-DC11-BED5-00304885AA26.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-480646B6-93AC-DC11-BFFC-001617DBCF46.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-485915F4-33AC-DC11-94FD-000423D99660.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-48A552D2-05AF-DC11-8CFA-003048562908.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-48AD76ED-70AC-DC11-821A-000423D662EE.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-48BFE321-72AC-DC11-B8F7-000423D944FC.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-48CA517E-41AD-DC11-A825-001617DBD288.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-48E0ADAE-B7AD-DC11-859D-0030485628C6.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-48FFFBAA-3FAD-DC11-95EB-00304885AD68.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-4A04CC72-E3AB-DC11-839D-000423D94BCC.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-4A4A1581-77AC-DC11-8DC9-000423D99BF2.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-4A4C9075-4BAD-DC11-9179-000423D98B5C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-4A87E7CE-89AB-DC11-814D-000423D9870C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-4AE4EE71-81AC-DC11-B35E-000423D6CA72.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-4C605D4F-43AD-DC11-97EF-00304885AF3A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-4CDB0B50-84AB-DC11-874E-00304885AEE2.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-4EFDCF2A-8EAC-DC11-9388-0016177CA7A0.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-50249460-7AAC-DC11-836F-001617E30CEA.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-5044AE6D-61AB-DC11-88C5-00304885AA2A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-504F8D89-3BAD-DC11-B19D-00304855D55A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-525DC3FC-78AC-DC11-9B92-001617C3B73A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-52996140-4BAD-DC11-A6A9-000423D952D8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-54B1B3D0-C6AD-DC11-B90F-00304885B4C2.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-5804F8E9-E4AD-DC11-B2AF-000E0C3F0478.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-580BEE02-71AC-DC11-9163-000423D6B1CC.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-584610AA-11AD-DC11-A290-001617C3B79A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-589CCCC6-46AD-DC11-9A73-00304885AEF2.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-5A7513E4-20AD-DC11-A23F-000423D98800.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-5C7626BB-70AC-DC11-8FAA-001617DBD22A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-5C9D8175-5DAB-DC11-A641-001617DBD258.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-5E203C72-75AC-DC11-9D51-0030485628E2.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-5E3934C5-10AD-DC11-8E42-000423D94D68.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-5EB354EF-97AC-DC11-B13E-001617C3B70E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-6014D9FC-47AD-DC11-A327-000E0C3F08F1.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-60CBFBE7-93AB-DC11-B8E8-001617C3B69C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-60D249E7-65AD-DC11-A72F-00304856114E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-60E92F9F-8FAB-DC11-B33D-001617C3B6E8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-60F862DD-85AD-DC11-8F82-000E0C3E6D2E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-6225098E-41AD-DC11-9075-00304885A9D2.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-626908DC-46AD-DC11-910A-000E0C3F0E47.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-627BE0F3-82AB-DC11-8217-000423D98750.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-62955555-73AC-DC11-B3C3-000423D99896.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-6435F623-7BAC-DC11-B5CD-000423D9989E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-6445E327-70AC-DC11-BE4D-000423D999CA.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-64532350-99AD-DC11-96DC-00304885B024.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-64B20674-75AC-DC11-9271-000423D99658.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-64C0DB22-24AD-DC11-A0F7-001617E30D4A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-64EF68D4-6FAD-DC11-8F1C-00304885A74E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-66CB35C8-CCAC-DC11-A5F5-00304885B24A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-6A248D61-7AAC-DC11-A989-001617E30CC8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-6A276C84-79AD-DC11-9F21-001617DBD556.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-6AAD90E9-50AD-DC11-B697-000423D6628E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-6ACB7A16-12AD-DC11-836E-000423D98930.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-6C4BC48D-F6AD-DC11-A90F-000E0C3F0630.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-6CBEA9B3-C8AD-DC11-A5DC-00304885AC4E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-6E01B249-78AC-DC11-A4D9-000423D6CA42.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-6E515FE9-9AAB-DC11-BE7B-0019DB29C620.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-70524516-11AC-DC11-A5AD-000423D95244.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-70BCB0AC-12AD-DC11-A4A3-0030485629BE.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-72B5F0DA-49AD-DC11-8ADF-000423D98C20.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-72BBF1A9-79AC-DC11-A35F-001617C3B654.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-74744A9B-C3AD-DC11-83B5-000E0C3F08BB.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-766FB25F-0AAC-DC11-A3B1-003048562870.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-78611C8D-41AD-DC11-B137-00304885AC7C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-78D6B655-19AD-DC11-A6E8-000423D995FC.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-78EF7248-84AC-DC11-826A-000423D6B328.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-7A260A92-83AC-DC11-B66C-001617E30CD4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-7A4A5F4E-43AD-DC11-8D3B-001617C3B79A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-7A892ADD-A9AD-DC11-A14A-0030485611D0.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-7E0F19D0-43AD-DC11-A42F-00304885AE3E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-7E1171A6-F6AB-DC11-A0B2-000423D98E6C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-7E5A8A19-5EAB-DC11-8DA6-00304885AA7A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-7E5F4E50-99AC-DC11-BCA8-001617C3B5D8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-7EB4ACBC-22AE-DC11-BAB5-000423D98DC4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-8035790A-6AAB-DC11-AC1B-000423D98FBC.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-82384661-44AD-DC11-AE21-000423D987FC.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-84A6DE6E-70AC-DC11-9BBB-000423D6B3C8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-860D38F1-40AB-DC11-8992-000423D992A4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-86214ED1-3EAD-DC11-9C07-00304855D622.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-86905A6B-75AC-DC11-BB19-000423D9997E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-869E2D0A-E0AD-DC11-85A7-000423D99AAE.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-86DB3217-19AD-DC11-8DFB-000423D995FC.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-88735914-70AC-DC11-8477-003048562A78.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-88D01282-3BAD-DC11-8B14-003048562902.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-8A8DB6B9-F5AD-DC11-AEFC-000423D99660.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-8A96E3E7-6FAC-DC11-8CE2-001617E30F4A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-8AA9FAD1-46AD-DC11-9A89-00304885AA7C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-8ACA432A-6DAB-DC11-BBC1-000423D6006E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-8CFA71D3-7BAC-DC11-97CE-001617DBCF6A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-8E5CEE6A-53AD-DC11-93EE-000423D952C0.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-8E7C08FA-95AB-DC11-8EB5-000423D99A2A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-8EA9B8DB-11AD-DC11-AEDF-001617DBD342.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-8EC03D01-CEAC-DC11-907C-000423D985E4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-90567EED-67AD-DC11-9D94-00304885AEA0.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-905D1466-70AD-DC11-BC92-00304858DF62.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-92326586-55AD-DC11-B6C8-000423D952C0.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-9246D18B-61AD-DC11-9742-000423D59C4E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-94F4F757-4BAD-DC11-8BCD-001617C3B70A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-961527BA-74AC-DC11-9587-001617C3B5E4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-962DE9D8-C5AD-DC11-912F-000423D99614.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-96C35CD4-47AE-DC11-BA27-000423D94524.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-9831C1DA-6EAC-DC11-94A0-001617C3B78C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-98D3B716-44AD-DC11-8182-000423D951D4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-98F86549-43AD-DC11-A26E-001617C3B670.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-9A5B15F3-40AB-DC11-9705-001617DBCF90.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-9C8F2972-0DAC-DC11-B141-000423D95198.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-A0380D65-75AC-DC11-B778-001617C3B6B8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-A077AA2B-BAAB-DC11-97D5-000423D987E0.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-A0B7CB59-70AC-DC11-929E-00304855D4BC.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-A26D06CF-76AC-DC11-8647-000423D6C9E2.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-A4376829-76AC-DC11-92FF-003048562928.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-A451770A-71AC-DC11-B21C-001617DBD25C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-A8AE741F-75AD-DC11-9CBF-000423D64922.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-AC3544C2-F1AB-DC11-A767-000423D6A3E4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-AE34BAA2-57AD-DC11-8D4A-000423D65FF2.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-AE78BC8C-3BAD-DC11-9AD5-0030485610C0.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-B04914E5-87AB-DC11-A47D-0016177CA778.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-B075397E-41AD-DC11-89B3-003048560F10.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-B0BCE646-89AD-DC11-B3EF-003048562874.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-B0F74C71-7EAB-DC11-BADC-00304885A902.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-B216925E-10AD-DC11-8795-000423D94BE4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-B277E5A9-DEAC-DC11-BB0A-000423D99A2A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-B2798477-7AAC-DC11-9741-000423D998BA.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-B486D309-C0AD-DC11-9675-0030488318E0.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-B4935C5B-7AAC-DC11-927B-001617C3B6DE.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-B63BAA00-74AC-DC11-B772-000423D9863C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-B641B94E-B2AB-DC11-9978-003048562752.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-B6515775-03AE-DC11-9168-0030485629A0.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-B68FF557-42AB-DC11-B2E2-000423D6C8EE.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-B6E22A48-78AC-DC11-903D-000423D662EE.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-B83883AA-6BAB-DC11-9613-001617C3B76A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-B8E15FB6-E3AB-DC11-A345-000423D98634.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-BA4FD85F-64AD-DC11-8888-001617DBD258.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-BC0643BC-93AD-DC11-ACCE-00304856141E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-BC71F418-7BAC-DC11-AEEA-001617C3B654.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-BC8BEE38-84AB-DC11-B678-000423D98804.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-BCE86426-44AD-DC11-B358-001617DBCF44.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-BE5844EE-C0AD-DC11-A01D-000423D9853C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-BED06938-40AB-DC11-9B85-000423D6C8EE.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-C210CAFE-AAAD-DC11-B485-000423D94A04.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-C27D6C6D-64AD-DC11-A82C-001617C3B708.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-C2E4854D-6EAC-DC11-99C5-000423D944D4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-C43D9D30-F4AD-DC11-8261-00304885AA5A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-C478C471-75AC-DC11-AF42-000423D99660.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-C4A61D79-E1AC-DC11-B312-001617C3B6E8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-C603A823-75AD-DC11-AA6E-001617E30F4A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-C6ED727F-69AD-DC11-9DB4-000423D6B444.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-C8BBDC3E-71AC-DC11-934A-000423D951D4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-C8D1AF1E-7BAC-DC11-AC5C-000423D99F1E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-C8D3ED81-3BAD-DC11-B83F-003048560F30.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-CA19BE1D-7AAB-DC11-ADF3-000423D8F63C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-CA27AECC-7BAC-DC11-A603-001617C3B66C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-CC43E200-EFAD-DC11-870C-00304885AA8A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-CCD3AC8B-E3AB-DC11-82B0-0030488318E0.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-CE07C823-44AD-DC11-9B85-000423D995FC.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-CEE44E6E-13AD-DC11-B41B-000423D996B4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-CEE94E46-D2AC-DC11-BC35-003048562822.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-D04EE43F-78AC-DC11-BEB9-001617DBCF6A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-D0530A1F-9FAD-DC11-A78E-00304885AA98.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-D06332EC-87AB-DC11-AD00-00304885AA28.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-D0AC7D73-75AC-DC11-909F-003048562956.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-D0CFFFD4-6EAC-DC11-A289-001617DBD224.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-D2409089-10AD-DC11-92A3-000423D8F63C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-D43A1D90-77AC-DC11-A6B4-000423D98930.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-D49767AF-79AC-DC11-B769-000423D951D4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-D61EC7F0-6CAD-DC11-A49D-00304885B042.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-D68EA5D7-E7AC-DC11-8579-00304885B4C8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-D696B675-8AAD-DC11-9CD8-001617C3B6E8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-D6B9BFB0-79AC-DC11-9979-000423D94700.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-D81129AB-DEAC-DC11-B3EB-001617DBCF1E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-D8318775-13AD-DC11-A1D0-000423D8F63C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-D840B48B-0DAC-DC11-8CE4-00304885AC15.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-D8617A29-A4AE-DC11-BB59-000423D6C8EE.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-D8F4C1D5-6FAC-DC11-8A93-001617C3B78C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-DA38E4C1-ECAD-DC11-B681-00304885AA50.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-DC53E5D8-F3AD-DC11-BEC3-00304885B3F8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-DCB7EEA0-91AD-DC11-929F-00304885AAA6.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-DCB9986D-E3AB-DC11-973D-000423D6A6D8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-DE59D932-D2AB-DC11-B6DF-000423D95220.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-DE6F08F1-73AC-DC11-BAC8-001617DBD49E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-DEC76D86-61AD-DC11-A55F-001617C3B77E.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-DED5AEBB-74AC-DC11-AD0B-000423D996B4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-E0E7F309-6AAB-DC11-AB96-000423D999CA.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-E46A993E-43AD-DC11-B254-003048562876.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-E6F48D65-0FAD-DC11-9A33-000E0C3F06E3.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-E8914540-97AD-DC11-B6A4-00304855D4D0.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-E89E2832-9EAB-DC11-92BA-000423DD2F34.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-E8CD638B-70AC-DC11-84A3-000423D98EB4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-EA3F14AF-F4AD-DC11-90D5-00304885A9D6.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-EA5B49AC-79AD-DC11-9136-001617DBD5B2.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-EA63B242-73AC-DC11-9CF6-000423D950D8.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-F073F16A-81AD-DC11-8094-000423D98E30.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-F213E397-DEAD-DC11-A8EA-000423D94A04.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-F230AC6C-CFAC-DC11-BDC5-000423D99660.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-F2CD5A02-8EAD-DC11-81B1-003048836648.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-F47129D4-6AAD-DC11-B9A1-00304885A55A.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-F49864AD-ADAC-DC11-81FC-000E0C3F0864.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-F4AF54E2-45AB-DC11-9834-000423D9853C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-F6010DCD-65AD-DC11-81CB-00304885AEE2.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-F6339FF3-71AC-DC11-9FF1-000423D33970.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-F69D9702-4BAD-DC11-A5A7-000423D944FC.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-F6C6D45C-99AD-DC11-BB07-00304855D524.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-F86D3DF5-82AB-DC11-8C6F-000423D6B42C.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-F898870B-79AC-DC11-B709-000423D999CA.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-FAE678BB-05AC-DC11-8589-001617E30CA4.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-FC0A227B-39AD-DC11-94C4-003048562846.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-FC35EE3C-DAAD-DC11-A7F0-001617DBD230.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-FC81F19B-60AB-DC11-9896-000423D99E46.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-FC863127-48AD-DC11-A72B-000423D98C20.root");
    ProcFil.push_back(MB_DIR+"hltana-runHLT-Mis-ALCA-RelVal_Digi_Digi2Raw_mb-FC924BE2-76AC-DC11-BC82-00304856298A.root");

    TabChain.push_back(new TChain("HltTree"));
    for (unsigned int ipfile = 0; ipfile < ProcFil.size(); ipfile++){
      TabChain.back()->Add(ProcFil[ipfile]);
    }
    doMuonCut.push_back(true); doElecCut.push_back(true);
    
    xsec.push_back(7.923E10); // 
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
    cout  << setw(3) << it << ")" << setw(26) << trignames[it]  << " (" << setw(7) << map_TrigPrescls.find(trignames[it])->second << ")" 
	  << " :   Indiv.: " << setw(7) << Rat[it] << " +/- " << setw(7) << sqrt(sRat[it]) 
	  << "   sPure: " << setw(7) << seqpRat[it]
	  << "   Pure: " << setw(7) << pRat[it] 
	  << "   Cumul: " << setw(7) << cRat[it] << "\n"<<flush;
  }
  cout << "\n"<<flush;
  cout << setw(60) << "TOTAL RATE : " << setw(5) << RTOT << " +- " << sRTOT << " Hz" << "\n";
  cout << "------------------------------------------------------------------------------------------------------------------\n"<<flush;
  
  ////////////////////////////////////////////////////////////
  // Printout Results to Tex/PDF

  if (PrintAll==1) {
    char sLumi[10];
    sprintf(sLumi,"%1.1e",ILumi);
    TString hltTableFileName= TString("hltTable_") + TString(sLumi) + TString("_") + sConditions + TString("Conditions") + sVersion;
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


void BookMenu_Default(OHltMenu*  menu, double &iLumi, double &nBunches) {

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
  menu->AddHlt("HLT2TauPixel","L1_TauJet40",1,1,"15","1e32"); 
  menu->AddHlt("OpenHLT2TauPixel","L1_TauJet40",1,1,"15","1e32"); 

  //
  menu->AddHlt("HLT1MuonNonIso","L1_SingleMu7",1,1,"16","1e32");
  menu->AddHlt("OpenHLT1MuonNonIso","L1_SingleMu7",1,1,"16","1e32");
  menu->AddHlt("HLT1MuonIso","L1_SingleMu7",1,1,"11","1e32");
  menu->AddHlt("OpenHLT1MuonIso","L1_SingleMu7",1,1,"11","1e32");
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
