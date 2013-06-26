void plotter()
{

  //set the style
  setStyle();

  // Open the file
  TFile *data = new TFile("gctAnalyzer_1.root");

  // Canvas
  TCanvas *c1 = new TCanvas("c1","c1",800,700);

  c1->SetLogy(1);
  //Iso e/gamma
  Plot(data,"analyzer/DataHistograms/IsoEm/isoEgD_Rank","analyzer/EmulatorHistograms/IsoEm/isoEgE_Rank","E_{T}","Iso EM candidates",0,0); c1->Print("TriggerMeetingTalk171109/isoEgRank.png");
  Plot(data,"analyzer/DataHistograms/IsoEm/isoEgD_EtEtaPhi","analyzer/EmulatorHistograms/IsoEm/isoEgE_EtEtaPhi","#eta","Iso EM candidates",0,0,"eta"); c1->Print("TriggerMeetingTalk171109/isoEgEta.png");
  Plot(data,"analyzer/DataHistograms/IsoEm/isoEgD_EtEtaPhi","analyzer/EmulatorHistograms/IsoEm/isoEgE_EtEtaPhi","#phi","Iso EM candidates",0,0,"phi"); c1->Print("TriggerMeetingTalk171109/isoEgPhi.png");
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/isoEg_errorFlag"); c1->Print("TriggerMeetingTalk171109/isoEg_errorFlag.png");

  //Non Iso e/gamma
  Plot(data,"analyzer/DataHistograms/NonIsoEM/nonIsoEgD_Rank","analyzer/EmulatorHistograms/NonIsoEM/nonIsoEgE_Rank","E_{T}","Non Iso EM candidates",0,0); c1->Print("TriggerMeetingTalk171109/nonIsoEgRank.png");
  Plot(data,"analyzer/DataHistograms/NonIsoEM/nonIsoEgD_EtEtaPhi","analyzer/EmulatorHistograms/NonIsoEM/nonIsoEgE_EtEtaPhi","#eta","Non Iso EM candidates",0,0,"eta"); c1->Print("TriggerMeetingTalk171109/nonIsoEgEta.png");
  Plot(data,"analyzer/DataHistograms/NonIsoEM/nonIsoEgD_EtEtaPhi","analyzer/EmulatorHistograms/NonIsoEM/nonIsoEgE_EtEtaPhi","#phi","Non Iso EM candidates",0,0,"phi"); c1->Print("TriggerMeetingTalk171109/nonIsoEgPhi.png");
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/nonIsoEg_errorFlag"); c1->Print("TriggerMeetingTalk171109/nonIsoEg_errorFlag.png");

  //Central Jets
  Plot(data,"analyzer/DataHistograms/CenJets/cenJetD_Rank","analyzer/EmulatorHistograms/CenJets/cenJetE_Rank","E_{T}","Jet candidates",0,0); c1->Print("TriggerMeetingTalk171109/cenJetsRank.png");
  Plot(data,"analyzer/DataHistograms/CenJets/cenJetD_EtEtaPhi","analyzer/EmulatorHistograms/CenJets/cenJetE_EtEtaPhi","#eta","Jet candidates",0,0,"eta"); c1->Print("TriggerMeetingTalk171109/cenJetsEta.png");
  Plot(data,"analyzer/DataHistograms/CenJets/cenJetD_EtEtaPhi","analyzer/EmulatorHistograms/CenJets/cenJetE_EtEtaPhi","#phi","Jet candidates",0,0,"phi"); c1->Print("TriggerMeetingTalk171109/cenJetsPhi.png");
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/cenJet_errorFlag"); c1->Print("TriggerMeetingTalk171109/cenJet_errorFlag.png");

  //Tau Jets
  Plot(data,"analyzer/DataHistograms/TauJets/tauJetD_Rank","analyzer/EmulatorHistograms/TauJets/tauJetE_Rank","E_{T}","Jet candidates",0,0); c1->Print("TriggerMeetingTalk171109/tauJetsRank.png");
  Plot(data,"analyzer/DataHistograms/TauJets/tauJetD_EtEtaPhi","analyzer/EmulatorHistograms/TauJets/tauJetE_EtEtaPhi","#eta","Jet candidates",0,0,"eta"); c1->Print("TriggerMeetingTalk171109/tauJetsEta.png");
  Plot(data,"analyzer/DataHistograms/TauJets/tauJetD_EtEtaPhi","analyzer/EmulatorHistograms/TauJets/tauJetE_EtEtaPhi","#phi","Jet candidates",0,0,"phi"); c1->Print("TriggerMeetingTalk171109/tauJetsPhi.png");
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/tauJet_errorFlag"); c1->Print("TriggerMeetingTalk171109/tauJet_errorFlag.png");

  //Forward Jets
  //There were no forward jets in CRUZET'09...
  //Plot(data,"analyzer/DataHistograms/ForJets/forJetD_Rank","analyzer/EmulatorHistograms/ForJets/forJetE_Rank","E_{T}","Jet candidates",0,0); c1->Print("TriggerMeetingTalk171109/forJetsRank.png");
  //Plot(data,"analyzer/DataHistograms/ForJets/forJetD_EtEtaPhi","analyzer/EmulatorHistograms/ForJets/forJetE_EtEtaPhi","#eta","Jet candidates",0,0,"eta"); c1->Print("TriggerMeetingTalk171109/forJetsEta.png");
  //Plot(data,"analyzer/DataHistograms/ForJets/forJetD_EtEtaPhi","analyzer/EmulatorHistograms/ForJets/forJetE_EtEtaPhi","#phi","Jet candidates",0,0,"phi"); c1->Print("TriggerMeetingTalk171109/forJetsPhi.png");
  //PlotErrors(data,"analyzer/ErrorHistograms_Flags/forJet_errorFlag"); c1->Print("TriggerMeetingTalk171109/forJet_errorFlag.png");

  //HF Ring Sums
  //...and hence no Ring Sums
  //Plot(data,"analyzer/DataHistograms/HFRingSums/hfRingSumD_1+","analyzer/EmulatorHistograms/HFRingSums/hfRingSumE_1+","E_{T}","Ring Sums",0,0); c1->Print("TriggerMeetingTalk171109/hfRingSum1+.png");
  //Plot(data,"analyzer/DataHistograms/HFRingSums/hfRingSumD_1-","analyzer/EmulatorHistograms/HFRingSums/hfRingSumE_1-","E_{T}","Ring Sums",0,0); c1->Print("TriggerMeetingTalk171109/hfRingSum1-.png");
  //Plot(data,"analyzer/DataHistograms/HFRingSums/hfRingSumD_2+","analyzer/EmulatorHistograms/HFRingSums/hfRingSumE_2+","E_{T}","Ring Sums",0,0); c1->Print("TriggerMeetingTalk171109/hfRingSum2+.png");
  //Plot(data,"analyzer/DataHistograms/HFRingSums/hfRingSumD_2-","analyzer/EmulatorHistograms/HFRingSums/hfRingSumE_2-","E_{T}","Ring Sums",0,0); c1->Print("TriggerMeetingTalk171109/hfRingSum2-.png");
  //PlotErrors(data,"analyzer/ErrorHistograms_Flags/hfRingSum_errorFlag"); c1->Print("TriggerMeetingTalk171109/hfRingSum_errorFlag.png");

  //HF Bit Counts
  Plot(data,"analyzer/DataHistograms/HFBitCounts/hfBitCountD_1+","analyzer/EmulatorHistograms/HFBitCounts/hfBitCountE_1+","E_{T}","Ring Sums",0,0); c1->Print("TriggerMeetingTalk171109/hfBitCount1+.png");
  Plot(data,"analyzer/DataHistograms/HFBitCounts/hfBitCountD_1-","analyzer/EmulatorHistograms/HFBitCounts/hfBitCountE_1-","E_{T}","Ring Sums",0,0); c1->Print("TriggerMeetingTalk171109/hfBitCount1-.png");
  Plot(data,"analyzer/DataHistograms/HFBitCounts/hfBitCountD_2+","analyzer/EmulatorHistograms/HFBitCounts/hfBitCountE_2+","E_{T}","Ring Sums",0,0); c1->Print("TriggerMeetingTalk171109/hfBitCount2+.png");
  Plot(data,"analyzer/DataHistograms/HFBitCounts/hfBitCountD_2-","analyzer/EmulatorHistograms/HFBitCounts/hfBitCountE_2-","E_{T}","Ring Sums",0,0); c1->Print("TriggerMeetingTalk171109/hfBitCount2-.png");
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/hfBitCount_errorFlag"); c1->Print("TriggerMeetingTalk171109/hfBitCount_errorFlag.png");

  //Total ET
  Plot(data,"analyzer/DataHistograms/TotalESums/totalEtD","analyzer/EmulatorHistograms/TotalESums/totalEtE","E_{T}","Events",16,0); c1->Print("TriggerMeetingTalk171109/totalEt.png");
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/totalEt_errorFlag"); c1->Print("TriggerMeetingTalk171109/totalEt_errorFlag.png");  

  //Total HT
  Plot(data,"analyzer/DataHistograms/TotalESums/totalHtD","analyzer/EmulatorHistograms/TotalESums/totalHtE","H_{T}","Events",16,0); c1->Print("TriggerMeetingTalk171109/totalHt.png");
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/totalHt_errorFlag"); c1->Print("TriggerMeetingTalk171109/totalHt_errorFlag.png");  

  //Missing ET
  Plot(data,"analyzer/DataHistograms/MissingESums/missingEtD","analyzer/EmulatorHistograms/MissingESums/missingEtE","ME_{T}","Events",16,0); c1->Print("TriggerMeetingTalk171109/missingEt.png");
  Plot(data,"analyzer/DataHistograms/MissingESums/missingEtD_Phi","analyzer/EmulatorHistograms/MissingESums/missingEtE_Phi","ME_{T} #phi","Events",0,0); c1->Print("TriggerMeetingTalk171109/missingEt_phi.png");
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/missingEt_errorFlag"); c1->Print("TriggerMeetingTalk171109/missingEt_errorFlag.png");  

  //Missing HT
  Plot(data,"analyzer/DataHistograms/MissingESums/missingHtD","analyzer/EmulatorHistograms/MissingESums/missingHtE","MH_{T}","Events",2,128); c1->Print("TriggerMeetingTalk171109/missingHt.png");
  Plot(data,"analyzer/DataHistograms/MissingESums/missingHtD_Phi","analyzer/EmulatorHistograms/MissingESums/missingHtE_Phi","MH_{T} #phi","Events",0,17); c1->Print("TriggerMeetingTalk171109/missingHt_phi.png");
  PlotErrors(data,"analyzer/ErrorHistograms_Flags/missingHt_errorFlag"); c1->Print("TriggerMeetingTalk171109/missingHt_errorFlag.png");
  
  data->Close();

}

void Plot(TFile* data, TString Hist_data, TString Hist_emu, TString XAxisLabel, TString YAxisLabel="Events", int rebin, int xAxisRange, TString Opt="")
{

  // Get the histograms from the files

  if(Opt == "eta" || Opt == "phi")
  {
	if(Opt == "eta") {
	TH2F *Data2 = (TH2F*)data->Get(Hist_data);	
	TH1D *Data = Data2->ProjectionX();
	if(rebin>0) Data->Rebin(rebin);
	if(xAxisRange>0) Data->GetXaxis()->SetRangeUser(0,xAxisRange);

	TH2F *Emu2 = (TH2F*)data->Get(Hist_emu);
	TH1D *Emu = Emu2->ProjectionX();
	if(rebin>0) Emu->Rebin(rebin);
	if(xAxisRange>0) Emu->GetXaxis()->SetRangeUser(0,xAxisRange);
	}
	if(Opt == "phi") {
	TH2F *Data2 = (TH2F*)data->Get(Hist_data);	
	TH1D *Data = Data2->ProjectionY();
	if(rebin>0) Data->Rebin(rebin);
    if(xAxisRange>0) Data->GetXaxis()->SetRangeUser(0,xAxisRange);

	TH2F *Emu2 = (TH2F*)data->Get(Hist_emu);
	TH1D *Emu = Emu2->ProjectionY();
	if(rebin>0) Emu->Rebin(rebin);
	if(xAxisRange>0) Emu->GetXaxis()->SetRangeUser(0,xAxisRange);
	}
  }
  else
  {
	TH1D *Data = (TH1D*)data->Get(Hist_data);
	if(rebin>0) Data->Rebin(rebin);
	if(xAxisRange>0) Data->GetXaxis()->SetRangeUser(0,xAxisRange);

	TH1D *Emu = (TH1D*)data->Get(Hist_emu);
	if(rebin>0) Emu->Rebin(rebin);
	if(xAxisRange>0) Emu->GetXaxis()->SetRangeUser(0,xAxisRange);
  }

  // Add the X axis label
  Emu->GetXaxis()->SetTitle(XAxisLabel);
  Emu->GetYaxis()->SetTitle(YAxisLabel);
  Emu->SetTitleOffset(1.5,"y");

  // Marker type for data
  Data->SetMarkerStyle(20);
  Data->SetMarkerColor(kRed);

  // plot them
  if (gPad->GetLogy()){
    Emu->SetMaximum(TMath::Max(Emu->GetMaximum(),Data->GetMaximum())*5);
  } else {
    Emu->SetMaximum(TMath::Max(Emu->GetMaximum(),Data->GetMaximum())*1.75);
  }
  Emu->DrawCopy("hist");
  Data->DrawCopy("psame");

  gPad->RedrawAxis();
}

void PlotErrors(TFile* data, TString Hist, TString Opt="")
{

  // Get the histograms from the files
  TH1D *Data = (TH1D*)data->Get(Hist);

  //check to make sure there are some events before setting the log scale
  if(Data->Integral() == 0 ) c1->SetLogy(0);
  else  c1->SetLogy(1);

  // Fill for histogram
  Data->SetFillColor(kBlue);

  // plot them
  Data->DrawCopy("hist");

  gPad->RedrawAxis();

}

void setStyle() {

  TStyle *setStyle = new TStyle("setStyle","Style for GCT Analyzer");

  // Stuff from plain style
  setStyle->SetCanvasColor(kWhite);
  setStyle->SetFrameBorderMode(0);
  setStyle->SetCanvasBorderMode(0);
  setStyle->SetFrameFillColor(0);
  setStyle->SetPadBorderMode(0);
  setStyle->SetPadColor(kWhite); 

  setStyle->SetOptStat(0);
  setStyle->SetOptTitle(0);
  setStyle->SetOptFit(0);
  setStyle->SetOptDate(0);

  // Labels and borders
  
  setStyle->SetLabelSize(0.055,"x");
  setStyle->SetLabelSize(0.06,"y");
  setStyle->SetLabelOffset(0.00,"x");
  setStyle->SetLabelOffset(0.00,"y");
  setStyle->SetTitleOffset(0.05,"x");
  setStyle->SetTitleOffset(0.50,"y");
  //setStyle->SetLabelFont(22,"x");
  //setStyle->SetLabelFont(22,"y");
  //setStyle->SetErrorX(0.0000);
  //setStyle->SetTickLength(0.05,"x");
  //setStyle->SetTickLength(0.05,"y");
  //setStyle->SetLineWidth(0.8);
  //setStyle->SetPadTickX(1);
  //setStyle->SetPadTickY(1);
  setStyle->SetPadLeftMargin(0.15);
  //setStyle->SetPadBottomMargin(0.2);
  
  setStyle->cd();
}
