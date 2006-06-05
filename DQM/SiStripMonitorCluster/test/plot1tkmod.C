//#include "z1.C"

// 2006.06.05 dkcira - plot all histograms from 1 Tk Module
plot1tkmod(TString dirpath="TIB/layer_1/backward_strings/internal_strings/string_1", TString mod="369164550")
{
  gStyle->SetOptStat(1110);
  gStyle->SetCanvasColor(0);
  gStyle->SetStatBorderSize(1);
  gStyle->SetHistFillColor(0);
  gStyle->SetHistLineColor(1);
  gStyle->SetMarkerStyle(23);
  gStyle->SetMarkerSize(2.5);
  gStyle->SetMarkerColor(1);
  gStyle->SetTitleSize(1.0);
//  zc->SetGrid();

//TString dirpath="TIB/layer_1/backward_strings/internal_strings/string_1", TString mod="369164550"
//TString dirpath="TIB/layer_1/backward_strings/internal_strings/string_17",TString mod="369168654"


  // read in file with histograms
  TString histo_file = "test_digi_cluster.root";
  TFile dqmf(histo_file); // dqmf.ls();
  TString fullpath="DQMData/SiStrip/MechanicalView/"+dirpath+"/module_"+mod+"/";
//  dqmf.cd(fullpath);
   cout<<"fullpath="<<fullpath<<endl;

  TString S_DigisPerDetector = fullpath+"DigisPerDetector__det__"+mod;
  TString S_ADCsCoolestStrip = fullpath+"ADCsCoolestStrip__det__"+mod;
  TString S_ADCsHottestStrip = fullpath+"ADCsHottestStrip__det__"+mod;
  TString S_NrOfClusterizedStrips = fullpath+"NrOfClusterizedStrips__det__"+mod;
  TString S_ClusterCharge = fullpath+"ClusterCharge__det__"+mod;
  TString S_ClusterPosition = fullpath+"ClusterPosition__det__"+mod;
  TString S_ClusterWidth = fullpath+"ClusterWidth__det__"+mod;
  TString S_ClustersPerDetector = fullpath+"ClustersPerDetector__det__"+mod;
  TString S_ModuleLocalOccupancy = fullpath+"ModuleLocalOccupancy__det__"+mod;

  TH1F * DigisPerDetector = (TH1F*)dqmf.Get(S_DigisPerDetector);
  TH1F * ADCsCoolestStrip = (TH1F*)dqmf.Get(S_ADCsCoolestStrip);
  TH1F * ADCsHottestStrip = (TH1F*)dqmf.Get(S_ADCsHottestStrip);
  TH1F * NrOfClusterizedStrips = (TH1F*)dqmf.Get(S_NrOfClusterizedStrips);
  TH1F * ClusterCharge = (TH1F*)dqmf.Get(S_ClusterCharge);
  TH1F * ClusterPosition = (TH1F*)dqmf.Get(S_ClusterPosition);
  TH1F * ClusterWidth = (TH1F*)dqmf.Get(S_ClusterWidth);
  TH1F * ClustersPerDetector = (TH1F*)dqmf.Get(S_ClustersPerDetector);
  TH1F * ModuleLocalOccupancy = (TH1F*)dqmf.Get(S_ModuleLocalOccupancy);

  z1(DigisPerDetector, dirpath);
  z1(ADCsCoolestStrip, dirpath);
  TString lokt = "ADCs__det__"+mod; ADCsHottestStrip->SetTitle(lokt);
  cout<<lokt<<endl;
  z1(ADCsHottestStrip, dirpath);
  z1(ClustersPerDetector, dirpath);
  z1(ClusterCharge, dirpath);
  z1(ClusterPosition, dirpath);
  z1(ClusterWidth, dirpath);
  z1(ClustersPerDetector, dirpath);
  z1(ModuleLocalOccupancy, dirpath);
}

void z1(TH1F *ah1, TString dirpath){
//2006.06.05 - plot 1 of modules histograms and print plot to file
TCanvas *myCanvas = new TCanvas("zc","zC",500,0,700,700);
//  mode = "ourmen" o = number of entries in overflow box, u = number of entries in underflow box r = rms of distribution m = mean of distribution e = total number of entries n = histogram name
  //
  ah1->SetXTitle(ah1->GetTitle()); ah1->SetTitle(dirpath);
  TString thetitle = ah1->GetXaxis()->GetTitle();
  TString filenamewheretoprint = thetitle+".gif";
  cout<<"filenamewheretoprint="<<filenamewheretoprint<<endl;
  ah1->Draw();
  char *s = new char[1];
//  cout<<"hit return to continue"<<endl; gets(s);  // "return" is enough, root waits until you hit a key
  zc->Print(filenamewheretoprint);
//  zc->Close();
}

