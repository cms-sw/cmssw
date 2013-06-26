void gctAnalyzerBasicPlotter()
{

  setStyle();

  // Open the file
  TFile *data = new TFile("raw_gctAnalyzer.root");
  
  // Canvas
  TCanvas *c1 = new TCanvas("c1","c1",800,800);

  PlotErrors(data,"analyzer/ErrorHistograms_Flags/isoEg_errorFlag"); c1->Print("isoEg_errorFlag.png");
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/nonIsoEg_errorFlag"); c1->Print("nonIsoEg_errorFlag.png");
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/cenJet_errorFlag"); c1->Print("cenJet_errorFlag.png");
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/tauJet_errorFlag"); c1->Print("tauJet_errorFlag.png");
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/forJet_errorFlag"); c1->Print("forJet_errorFlag.png");
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/hfRingSum_errorFlag"); c1->Print("hfRingSum_errorFlag.png");
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/hfBitCount_errorFlag"); c1->Print("hfBitCount_errorFlag.png");
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/totalEt_errorFlag"); c1->Print("totalEt_errorFlag.png");  
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/totalHt_errorFlag"); c1->Print("totalHt_errorFlag.png");  
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/missingEt_errorFlag"); c1->Print("missingEt_errorFlag.png");  
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/missingHt_errorFlag"); c1->Print("missingHt_errorFlag.png");  
  
  data->Close();

}

void PlotErrors(TFile* data, TString Hist, TString Opt="")
{

  // Get the histograms from the files
  TH1D *Data = (TH1D*)data->Get(Hist);

  //check to make sure there are some events for log scale
  if(Data->Integral() == 0 ) c1->SetLogy(0);
  else  c1->SetLogy(1);

  // Fill for histogram
  Data->SetFillColor(kBlue);

  // plot them
  Data->DrawCopy("hist");

  gPad->RedrawAxis();

}

void setStyle() {

  TStyle *setStyle = new TStyle("setStyle","Style for GCTAnalyzer");

  // Stuff from plain style
  setStyle->SetFrameBorderMode(0);
  setStyle->SetCanvasBorderMode(0);
  setStyle->SetCanvasColor(kWhite);
  setStyle->SetPadBorderMode(0);
  setStyle->SetPadColor(kWhite); 

  setStyle->SetOptStat(0);
  setStyle->SetOptTitle(1);
  setStyle->SetOptFit(0);
  setStyle->SetOptDate(0);

  setStyle->cd();
}
