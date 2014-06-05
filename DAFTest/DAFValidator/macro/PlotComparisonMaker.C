///////////
// Macro that provide to plot multivalidation varibles for recoTrack and DAF
//      variables are: effic,fakerate,dzres_vs_eta_Sigma,pull*,chi2_prob ...
//////////

#include <iomanip>

std::string folderName = "/afs/cern.ch/user/e/ebrondol/CMSSWarea/CMSSW_7_1_0_pre5/src/DAFTest/DAFValidator/";
std::string FileName1 = "multiprova.root";
std::string FileName2 = "multiprova.root";
//std::string FileName1 = "multitrackvalidator_SingleMuPt10_100evts_AllAssociators.root";
//std::string FileName2 = "multitrackvalidator_DAF_SingleMuPt10_100evts_v3_AllAssociators.root";
//std::string FileName1 = "multitrackvalidator_TTbar_10evts_AllAssociators.root";
//std::string FileName2 = "multitrackvalidator_DAF_TTbar_10evts_v3_AllAssociators.root";
std::string baseFolderRootName = "DQMData/Tracking/Track/";
//std::string Associator = "quickAssociatorByHits";
std::string Associator = "AssociatorByChi2";
//std::string Associator = "AssociatorByPull";
std::string MultivalLabelTracks1 = "cutsReco";
std::string MultivalLabelTracks2 = "ctfWithMaterialDAF";
std::string FolderRootName1 = baseFolderRootName + MultivalLabelTracks2 + "_" + Associator + "/"; 
std::string FolderRootName2 = baseFolderRootName + MultivalLabelTracks2 + "_" + Associator + "/"; 
//std::string HistoName = "dzres_vs_eta_Sigma";

//std::string HistoName = "chi2";
//std::string HistoName = "assocSharedHit";
//std::string HistoName = "tracks";
//std::string HistoName = "pullDz";
//std::string HistoName = "hits";

void PlotComparisonMaker(const char* HistoName)
{
  std::cout << "Plotting " << HistoName << " variable..." << std::endl;
  std::string inFileTotalName1 = folderName + FileName1;
  std::string inFileTotalName2 = folderName + FileName2;

  std::cout << "InputFile1: " << inFileTotalName1 << std::endl;
  std::cout << "InputFile2: " << inFileTotalName2 << std::endl;

  TFile* f1 = TFile::Open(inFileTotalName1.c_str(), "READ");  
  TFile* f2 = TFile::Open(inFileTotalName2.c_str(), "READ");  
  
  std::cout << "InputRootFile1: " << FolderRootName1 << std::endl;
  std::cout << "InputRootFile2: " << FolderRootName2 << std::endl;

  TH1F* histo1 = (TH1F*)( f1->Get((FolderRootName1+HistoName).c_str()) ); 
  TH1F* histo2 = (TH1F*)( f2->Get((FolderRootName2+HistoName).c_str()) ); 

  TCanvas* c1 = new TCanvas();
  c1 -> cd();
  c1 -> SetGridx();
  c1 -> SetGridy();

  histo1 -> SetMarkerColor(2);
  histo1 -> SetLineColor(2);
  histo1 -> SetLineWidth(2);
  histo2 -> SetMarkerColor(4);
  histo2 -> SetLineColor(4);
  histo1 -> GetYaxis() -> SetTitle("Number of Events");
  histo1 -> GetXaxis() -> SetTitle("#chi^{2}");
  histo1 -> GetXaxis() -> SetTitleSize(0.03);
  histo1 -> GetYaxis() -> SetTitleSize(0.03);
  histo1 -> GetXaxis() -> SetLabelSize(0.03);
  histo1 -> GetYaxis() -> SetLabelSize(0.03);

  TLegend* legend = new TLegend(0.16, 0.77, 0.30, 0.92);
  legend -> SetFillColor(kWhite);
  legend -> SetFillStyle(1001);
  legend -> SetTextFont(42);
  legend -> SetTextSize(0.03);

  legend -> AddEntry(histo1,("cutsRecoTracks - "+Associator).c_str(),"PL");
  legend -> AddEntry(histo2,("ctfWithMaterialTracksDAF - "+Associator).c_str(),"PL");

//  histo1 -> Draw("Histo");
//  histo2 -> Draw("Histosame");
  histo1 -> Draw("P");
  histo2 -> Draw("Psame");
  legend -> Draw("same");

  c1 -> Print((string(HistoName)+"Comparison.pdf").c_str(), "pdf");
  c1 -> Print((string(HistoName)+"Comparison.png").c_str(), "png");
//  c1 -> SaveAs((string(HistoName)+"Comparison.C").c_str());
}
