////////
// Macro that provide to plot multivalidation varibles for DAFValidator class
//////////

#include <iomanip>

std::string folderName = "/afs/cern.ch/user/e/ebrondol/CMSSWarea/CMSSW_7_1_0_pre5/src/DAFTest/DAFValidator/MultiTrackValidator/";
std::string FileName1 = "DAFValidator_SingleMuPt10_100evts_v3_TrackAssocByPull.root";
std::string FileName2 = "DAFValidator_TTbar_10evts_v3_TrackAssocByPull.root";
std::string baseFolderRootName = "demo/";
std::string HistoName = "Weight";

//void PlotComparisonMaker_v2(const char* HistoName)
void PlotComparisonMaker_v2()
{
  std::cout << "Plotting " << HistoName << " variable..." << std::endl;
  std::string inFileTotalName1 = folderName + FileName1;
  std::string inFileTotalName2 = folderName + FileName2;

  std::cout << "InputFile1: " << inFileTotalName1 << std::endl;
  std::cout << "InputFile2: " << inFileTotalName2 << std::endl;

  TFile* f1 = TFile::Open(inFileTotalName1.c_str(), "READ");  
  TFile* f2 = TFile::Open(inFileTotalName2.c_str(), "READ");  
  
  TH1F* histo1 = (TH1F*)( f1->Get((baseFolderRootName+HistoName).c_str()) ); 
  TH1F* histo2 = (TH1F*)( f2->Get((baseFolderRootName+HistoName).c_str()) ); 

  TCanvas* c1 = new TCanvas();
  c1 -> cd();
  c1 -> SetGridx();
  c1 -> SetGridy();

  histo1 -> SetMarkerColor(2);
  histo1 -> SetLineColor(2);
  histo1 -> SetLineWidth(2);
  histo2 -> SetMarkerColor(4);
  histo2 -> SetLineColor(4);
  histo1 -> GetXaxis() -> SetTitle(HistoName.c_str());
  histo1 -> GetYaxis() -> SetTitle("Number of events");
  histo1 -> GetXaxis() -> SetTitleSize(0.03);
  histo1 -> GetYaxis() -> SetTitleSize(0.03);
  histo1 -> GetXaxis() -> SetLabelSize(0.03);
  histo1 -> GetYaxis() -> SetLabelSize(0.03);

  TLegend* legend = new TLegend(0.16, 0.77, 0.30, 0.92);
  legend -> SetFillColor(kWhite);
  legend -> SetFillStyle(1001);
  legend -> SetTextFont(42);
  legend -> SetTextSize(0.03);

  legend -> AddEntry(histo1,"SingleMuPt10","PL");
  legend -> AddEntry(histo2,"TTbar","PL");

  histo1 -> DrawNormalized();
  histo2 -> DrawNormalized("same");
//  histo1 -> Draw("Histo");
//  histo2 -> Draw("Histosame");
//  histo1 -> Draw("P");
//  histo2 -> Draw("Psame");
  legend -> Draw("same");

  c1 -> Print((string(HistoName)+".pdf").c_str(), "pdf");
  c1 -> Print((string(HistoName)+".png").c_str(), "png");
//  c1 -> SaveAs((string(HistoName)+"Comparison.C").c_str());
}

