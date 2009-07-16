// Standard Includes
#include <string>
#include <vector>
#include <map>
#include <iostream>

using namespace std;

// ROOT Includes
#include <TH1D.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TError.h>

// Local Includes
#include "EffPullCalculator.cc"
#include "CompHisto1D.cc"
#include "JetMETComp.cc"

void printhelp() {
  cout << "triggerComparison inputs:" << endl;
  cout << " -File1       -> Name of first root file" << endl;
  cout << " -File2       -> Name of second root file" << endl;
  cout << " -OutputFile  -> Name of the output root file" << endl;
  cout << "                 (default trigger_validation.root)" << endl;
  cout << " --correlated -> Use 100% correlated error for efficiency difference"   << endl;
  cout << " --oldL1names -> Use A_ rather than L1_ for L1 paths"   << endl;
  cout << " --useL1objets -> Use L1 objects for shape comparison" << endl;
  cout << " --help    -> Print this output"   << endl;
  cout << "Example: ./triggerComparison.x -File1=fil1.root -File2=file2.root" << endl;
}

void WriteEffEps(vector<TH1D*> effhistos, string label1, string label2) {
  TCanvas* c1 = new TCanvas("c1","c1",300,300);
  c1->SetLeftMargin(0.25);
  c1->SetRightMargin(0.20);
  c1->SetFillColor(0);
  c1->SetHighLightColor(0);
  c1->SetBorderMode(0);
  for(int i=0; i< int(effhistos.size())/2; i++) {
    effhistos[i]->SetMaximum(1.);
    effhistos[i]->Draw("hbar2");
    effhistos[i+effhistos.size()/2]->Draw("hbar2same");
    TLegend* legend = new TLegend(0.81,0.75,0.99,0.95);
    legend->AddEntry(effhistos[i],label1.c_str(),"f");
    legend->AddEntry(effhistos[i+effhistos.size()/2],label2.c_str(),"f");
    legend->Draw();
    c1->SaveAs((string(effhistos[i]->GetName())+".eps").c_str());
  }
}

void WriteEffEps(TH1D* pullhisto, TH1D* reshisto) {
  TCanvas* c1 = new TCanvas("c1","c1",300,300);
  c1->SetLeftMargin(0.20);
  c1->SetFillColor(0);
  c1->SetHighLightColor(0);
  c1->SetBorderMode(0);
  c1->SetBottomMargin(0.25);
  pullhisto->Draw();
  c1->SaveAs((string(pullhisto->GetName())+".eps").c_str());
  
  c1 = new TCanvas("c1","c1",900,300);
  c1->SetLeftMargin(0.20);
  c1->SetFillColor(0);
  c1->SetHighLightColor(0);
  c1->SetBorderMode(0);
  c1->SetBottomMargin(0.35);
  reshisto->Draw();
  c1->SaveAs((string(reshisto->GetName())+".eps").c_str());
}

int main(int argc, char *argv[]){
  if (argc<3) {
    printhelp();
    return 0;
  }

  gErrorIgnoreLevel = 2001;
  
  cout << "aaaaaaaaaaaaaaaaaaaa" << endl;

  string file1name, file2name;
  string outputfilename("trigger_validation.root");
  char inputstr[256];
  bool oldl1name = false;
  bool usel1obj = false;
  string error = "uncorrelated";
  for (int i=1;i<argc;i++){
    if (strncmp(argv[i],"-",1)==0){
      if (strncmp(argv[i],"--correlated",12)==0) error = "correlated";
      if (strncmp(argv[i],"--useL1objets",13)==0) usel1obj = true;
      if (strncmp(argv[i],"--oldL1names",12)==0) oldl1name = true;
      if (strncmp(argv[i],"-help",4)==0) {printhelp(); return 0;}
      if (strncmp(argv[i],"-File1",6)==0) {
	sscanf(argv[i],"-File1=%s",inputstr);
	file1name = string(inputstr,strlen(argv[i]));
      }
      if (strncmp(argv[i],"-File2",6)==0) {
	sscanf(argv[i],"-File2=%s",inputstr);
	file2name = string(inputstr,strlen(argv[i]));
      }
      if (strncmp(argv[i],"-OutputFile",11)==0) {
	sscanf(argv[i],"-OutputFile=%s",inputstr);
	outputfilename = string(inputstr,strlen(argv[i]));
      }
    }
  }

  cout << "bbbbbbbbbbbbbbbbbbbbbbb" << endl;
  vector<TFile*> files;
  files.push_back(new TFile(file1name.c_str()));
  files.push_back(new TFile(file2name.c_str()));

  TH1D* L1effhisto1; 
  TH1D* L1effhisto2; 
  TH1D* L1pullhisto; 
  TH1D* L1reshisto; 

  TH1D* HLTeffhisto1; 
  TH1D* HLTeffhisto2; 
  TH1D* HLTpullhisto; 
  TH1D* HLTreshisto; 

  vector<TH1D*> L1effhistos;
  vector<TH1D*> HLTeffhistos;

  string label1 = file1name.erase(file1name.find("/"),file1name.size()-file1name.find("/"));
  string label2 = file2name.erase(file2name.find("/"),file2name.size()-file2name.find("/"));

  cout << "cccccccccccccccc" << endl;
   for (int i=1;i<argc;i++){
     if (strncmp(argv[i],"-",1)==0){
       if (strncmp(argv[i],"-label1",7)==0) {
       sscanf(argv[i],"-label1=%s",inputstr);
       label1 = string(inputstr,strlen(argv[i]));
       }
       if (strncmp(argv[i],"-label2",7)==0) {
       sscanf(argv[i],"-label2=%s",inputstr);
       label2 = string(inputstr,strlen(argv[i]));
       }
     }
   }

  cout << "ddddddddddddddddddddd" << endl;
  // Get TriggerBit histogram 
  L1effhisto1  = (TH1D*) files[0]->Get("DQMData/Run 1/HLT/Run summary/SusyExo/TriggerBits/L1Paths"); 
  L1effhisto2  = (TH1D*) files[1]->Get("DQMData/Run 1/HLT/Run summary/SusyExo/TriggerBits/L1Paths"); 
  HLTeffhisto1 = (TH1D*) files[0]->Get("DQMData/Run 1/HLT/Run summary/SusyExo/TriggerBits/HltPaths"); 
  HLTeffhisto2 = (TH1D*) files[1]->Get("DQMData/Run 1/HLT/Run summary/SusyExo/TriggerBits/HltPaths"); 

  cout << "eeeeeeeeeeeeeeeee" << endl;
  // pull and residual histograms
  EffPullcalculator* HLTpullcal = new EffPullcalculator(HLTeffhisto1,HLTeffhisto2,error);
  HLTpullcal->CalculatePulls();
  HLTpullhisto = HLTpullcal->GetPullHisto();
  HLTreshisto  = HLTpullcal->GetResidualHisto();
  HLTeffhistos = HLTpullcal->GetEffHistos();
  HLTpullcal->WriteLogFile("HLTcomparison.log");

  cout << "ffffffffffffffff" << endl;
  EffPullcalculator* L1pullcal = new EffPullcalculator(L1effhisto1,L1effhisto2,error);
  L1pullcal->CalculatePulls();
  L1pullhisto = L1pullcal->GetPullHisto();
  L1reshisto  = L1pullcal->GetResidualHisto();
  L1effhistos = L1pullcal->GetEffHistos();

  cout << "ggggggggggggggg" << endl;
  L1pullcal->WriteLogFile("L1comparison.log");

  //////////////////////
  // Open Output file //
  //////////////////////
  TFile* outfile = new TFile(outputfilename.c_str(),"recreate");

  /////////////////////////////////////////
  // Efficiency pull and residual histos //
  /////////////////////////////////////////

  L1pullhisto->Write();
  L1reshisto->Write();
  HLTpullhisto->Write();
  HLTreshisto->Write();
  cout << "hhhhhhhhhhhhhhhhhhh" << endl;

  for(int ih=0; ih<int(L1effhistos.size()); ih++) 
    L1effhistos[ih]->Write();
  for(int ih=0; ih<int(HLTeffhistos.size()); ih++) 
    HLTeffhistos[ih]->Write();

//   WriteEffEps(L1pullhisto,  L1reshisto);
//   WriteEffEps(HLTpullhisto, HLTreshisto);

  cout << "iiiiiiiiiiiiiiiii" << endl;
  WriteEffEps(L1effhistos,label1,label2);
  WriteEffEps(HLTeffhistos,label1,label2);

  // close
  outfile->Close();

  cout << "lllllllllllllllllll" << endl;

  /////////////////////////////////////////////
  // comparison histograms for each HLT path //
  /////////////////////////////////////////////

  vector<TH1D*> inputhisto; 
  vector<TCanvas*> compcanvas; 
  map<string,double> compatibility;

  string obj = "Reco";
  if(usel1obj == true) obj = "L1";

  cout << "mmmmmmmmmmmmmmmmmmmmm" << endl;
  for(int ifile=0; ifile<2; ifile++) {  
    // general plots
    if(usel1obj == false) {
      inputhisto.push_back((TH1D*) files[ifile]->Get("DQMData/Run 1/HLT/Run summary/SusyExo/RecoJets/General/JetMult")); 
      inputhisto.push_back((TH1D*) files[ifile]->Get("DQMData/Run 1/HLT/Run summary/SusyExo/RecoMuons/General/MuonMult"));
      inputhisto.push_back((TH1D*) files[ifile]->Get("DQMData/Run 1/HLT/Run summary/SusyExo/RecoElectrons/General/ElecMult"));
      inputhisto.push_back((TH1D*) files[ifile]->Get("DQMData/Run 1/HLT/Run summary/SusyExo/RecoPhotons/General/PhotonMult"));
      inputhisto.push_back((TH1D*) files[ifile]->Get("DQMData/Run 1/HLT/Run summary/SusyExo/RecoMET/General/MET"));
    } else {
      inputhisto.push_back((TH1D*) files[ifile]->Get("DQMData/Run 1/HLT/Run summary/SusyExo/L1Jets/General/JetMult")); 
      inputhisto.push_back((TH1D*) files[ifile]->Get("DQMData/Run 1/HLT/Run summary/SusyExo/L1Muons/General/MuonMult"));
      inputhisto.push_back((TH1D*) files[ifile]->Get("DQMData/Run 1/HLT/Run summary/SusyExo/L1Em/General/ElecMult"));
      inputhisto.push_back((TH1D*) files[ifile]->Get("DQMData/Run 1/HLT/Run summary/SusyExo/L1MET/General/MET"));      
    }

  cout << "nnnnnnnnnnnnnnnnnnnnnn" << endl;
    // L1 plots
    TAxis* axis = L1reshisto->GetXaxis();
    for(int i=1; i<L1reshisto->GetNbinsX(); i++) { //we skip the last bin, which is the total
      string label = axis->GetBinLabel(i);
      // Maria&Massimiliano >30% eff condition
      if(L1pullcal->GetEff(label,0) > 0.3 &&
	 L1pullcal->GetEff(label,1) > 0.3) {
	if(usel1obj == false) {
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/RecoJets/L1/JetMult_"+label).c_str()));
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/RecoMuons/L1/MuonMult_"+label).c_str()));
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/RecoElectrons/L1/ElecMult_"+label).c_str()));
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/RecoPhotons/L1/PhotonMult_"+label).c_str()));
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/RecoMET/L1/MET_"+label).c_str()));
	} else {
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/L1Jets/L1/JetMult_"+label).c_str()));
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/L1Muons/L1/MuonMult_"+label).c_str()));
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/L1Em/L1/ElecMult_"+label).c_str()));
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/L1MET/L1/MET_"+label).c_str()));
	}
      }
    }

  cout << "ooooooooooooooooooooooo" << endl;
    // HLT plots
    axis = HLTreshisto->GetXaxis();
    for(int i=1; i<HLTreshisto->GetNbinsX(); i++) { //we skip the last bin, which is the total
      string label = axis->GetBinLabel(i);
      // Maria&Massimiliano >30% eff condition
      if(HLTpullcal->GetEff(label,0) > 0.3 &&
	 HLTpullcal->GetEff(label,1) > 0.3) {
	if(usel1obj == false) {
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/RecoJets/HLT/JetMult_"+label).c_str()));
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/RecoMuons/HLT/MuonMult_"+label).c_str()));
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/RecoElectrons/HLT/ElecMult_"+label).c_str()));
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/RecoPhotons/HLT/PhotonMult_"+label).c_str()));
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/RecoMET/HLT/MET_"+label).c_str()));
	} else {
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/L1Jets/HLT/JetMult_"+label).c_str()));
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/L1Muons/HLT/MuonMult_"+label).c_str()));
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/L1Em/HLT/ElecMult_"+label).c_str()));
	  inputhisto.push_back((TH1D*) files[ifile]->Get(("DQMData/Run 1/HLT/Run summary/SusyExo/L1MET/HLT/MET_"+label).c_str()));	  
	}
      }
    }
  }

  cout << "pppppppppppppppppp" << endl;
  for(int i=0; i<int(inputhisto.size()/2); i++) {
    cout << "loop CompHisto 1" << endl;
    CompHisto1D* compare = new CompHisto1D(inputhisto[i],inputhisto[i+inputhisto.size()/2]);
    cout << "loop CompHisto 2" << endl;
    compare->SetLabel1(label1);
    cout << "loop CompHisto 3" << endl;
   compare->SetLabel2(label2);
    cout << "loop CompHisto 4" << endl;
   cout << "inputhisto[i]->GetName() = " << endl;
    cout << "inputhisto[i]->GetName() = " << inputhisto[i]->GetName() << endl;
    compatibility.insert(std::make_pair(string(inputhisto[i]->GetName()),compare->Compare()));
    cout << "loop CompHisto 5" << endl;
     compare->SaveAsEps();
    cout << "loop CompHisto 6" << endl;
  }
  
  cout << "qqqqqqqqqqqqqqqqqqqq" << endl;
  // Make plot of compatibilities
  JetMETComp* jetmetcomp = new JetMETComp(compatibility);
  // L1
  if (oldl1name == true) {
    jetmetcomp->MakePlot("ElecMult_A");
    jetmetcomp->MakePlot("MuonMult_A");
    jetmetcomp->MakePlot("PhotonMult_A");
    jetmetcomp->MakePlot("JetMult_A");
    jetmetcomp->MakePlot("MET_A");
  } else {
    jetmetcomp->MakePlot("ElecMult_L1");
    jetmetcomp->MakePlot("MuonMult_L1");
    jetmetcomp->MakePlot("PhotonMult_L1");
    jetmetcomp->MakePlot("JetMult_L1");
    jetmetcomp->MakePlot("MET_L1");
  }
  // HLT
  jetmetcomp->MakePlot("ElecMult_HLT");
  jetmetcomp->MakePlot("MuonMult_HLT");
  jetmetcomp->MakePlot("PhotonMult_HLT");
  jetmetcomp->MakePlot("JetMult_HLT");
  jetmetcomp->MakePlot("MET_HLT");
  jetmetcomp->WriteFile();
  cout << "rrrrrrrrrrrrrrrrrr" << endl;

}

