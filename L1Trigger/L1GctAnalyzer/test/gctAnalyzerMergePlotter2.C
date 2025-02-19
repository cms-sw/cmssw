void gctAnalyzerMergePlotter2()
{

  //set the style
  setStyle();

  //Create the file you want to write the output canvasses to
  TFile *output = new TFile("SUSY_pattern_results_rawgctAnalyzer.root", "RECREATE");

  //Open the file
  TFile *input = new TFile("raw_gctAnalyzer.root");

  //Canvas
  //TCanvas *c1 = new TCanvas("c1","c1",800,700);

  //Set the output folder
  TString outfolder = "/afs/cern.ch/user/j/jad/scratch0/SUSY_pattern_results010410/";
  TString basedata = "gctErrorAnalyzer/DataHistograms/";
  TString baseemu = "gctErrorAnalyzer/EmulatorHistograms/";
  TString baseerrorhist = "gctErrorAnalyzer/ErrorHistograms_Flags/";
  TString isoem = "IsoEm/";
  TString nisoem = "NonIsoEM/";
  TString cenjet = "CenJets/";
  TString taujet = "TauJets/";
  TString forjet = "ForJets/";
  TString hfrings = "HFRingSums/";
  TString hfringb = "HFBitCounts/";
  TString totale = "TotalESums/";
  TString missinge = "MissingESums/";

  //Iso e/gamma
  PlotWrite(input,output,outfolder,TString("isoEgRank"),TString(basedata+isoem+"isoEgD_Rank"),TString(baseemu+isoem+"isoEgE_Rank"),"E_{T}","Iso EM candidates",0,0);
  PlotWrite(input,output,outfolder,TString("isoEgEta"),TString(basedata+isoem+"isoEgD_EtEtaPhi"),TString(baseemu+isoem+"isoEgE_EtEtaPhi"),"#eta","Iso EM candidates",0,0,"eta");
  PlotWrite(input,output,outfolder,TString("isoEgPhi"),TString(basedata+isoem+"isoEgD_EtEtaPhi"),TString(baseemu+isoem+"isoEgE_EtEtaPhi"),"#phi","Iso EM candidates",0,0,"phi");
  PlotWriteErrors(input,output,outfolder,TString("isoEg_errorFlag"),TString(baseerrorhist+"isoEg_errorFlag"));

  //Non Iso e/gamma
  PlotWrite(input,output,outfolder,TString("nonIsoEgRank"),TString(basedata+nisoem+"nonIsoEgD_Rank"),TString(baseemu+nisoem+"nonIsoEgE_Rank"),"E_{T}","Non-Iso EM candidates",0,0);
  PlotWrite(input,output,outfolder,TString("nonIsoEgEta"),TString(basedata+nisoem+"nonIsoEgD_EtEtaPhi"),TString(baseemu+nisoem+"nonIsoEgE_EtEtaPhi"),"#eta","Non-Iso EM candidates",0,0,"eta");
  PlotWrite(input,output,outfolder,TString("nonIsoEgPhi"),TString(basedata+nisoem+"nonIsoEgD_EtEtaPhi"),TString(baseemu+nisoem+"nonIsoEgE_EtEtaPhi"),"#phi","Non-Iso EM candidates",0,0,"phi");
  PlotWriteErrors(input,output,outfolder,TString("nonIsoEg_errorFlag"),TString(baseerrorhist+"nonIsoEg_errorFlag"));

  //Central Jets
  PlotWrite(input,output,outfolder,TString("cenJetsRank"),TString(basedata+cenjet+"cenJetD_Rank"),TString(baseemu+cenjet+"cenJetE_Rank"),"E_{T}","Central Jet candidates",0,0);
  PlotWrite(input,output,outfolder,TString("cenJetsEta"),TString(basedata+cenjet+"cenJetD_EtEtaPhi"),TString(baseemu+cenjet+"cenJetE_EtEtaPhi"),"#eta","Central Jet candidates",0,0,"eta");
  PlotWrite(input,output,outfolder,TString("cenJetsPhi"),TString(basedata+cenjet+"cenJetD_EtEtaPhi"),TString(baseemu+cenjet+"cenJetE_EtEtaPhi"),"#phi","Central Jet candidates",0,0,"phi");
  PlotWriteErrors(input,output,outfolder,TString("cenJet_errorFlag"),TString(baseerrorhist+"cenJet_errorFlag"));

  //Tau Jets
  PlotWrite(input,output,outfolder,TString("tauJetsRank"),TString(basedata+taujet+"tauJetD_Rank"),TString(baseemu+taujet+"tauJetE_Rank"),"E_{T}","Tau-Jet candidates",0,0);
  PlotWrite(input,output,outfolder,TString("tauJetsEta"),TString(basedata+taujet+"tauJetD_EtEtaPhi"),TString(baseemu+taujet+"tauJetE_EtEtaPhi"),"#eta","Tau-Jet candidates",0,0,"eta");
  PlotWrite(input,output,outfolder,TString("tauJetsPhi"),TString(basedata+taujet+"tauJetD_EtEtaPhi"),TString(baseemu+taujet+"tauJetE_EtEtaPhi"),"#phi","Tau-Jet candidates",0,0,"phi");
  PlotWriteErrors(input,output,outfolder,TString("tauJet_errorFlag"),TString(baseerrorhist+"tauJet_errorFlag"));

  //Forward Jets
  PlotWrite(input,output,outfolder,TString("forJetsRank"),TString(basedata+forjet+"forJetD_Rank"),TString(baseemu+forjet+"forJetE_Rank"),"E_{T}","Forward Jet candidates",0,0);
  PlotWrite(input,output,outfolder,TString("forJetsEta"),TString(basedata+forjet+"forJetD_EtEtaPhi"),TString(baseemu+forjet+"forJetE_EtEtaPhi"),"#eta","Forward Jet candidates",0,0,"eta");
  PlotWrite(input,output,outfolder,TString("forJetsPhi"),TString(basedata+forjet+"forJetD_EtEtaPhi"),TString(baseemu+forjet+"forJetE_EtEtaPhi"),"#phi","Forward Jet candidates",0,0,"phi");
  PlotWriteErrors(input,output,outfolder,TString("forJet_errorFlag"),TString(baseerrorhist+"forJet_errorFlag"));

  //HF Ring Sums
  PlotWrite(input,output,outfolder,TString("hfRingSum1+"),TString(basedata+hfrings+"hfRingSumD_1+"),TString(baseemu+hfrings+"hfRingSumE_1+"),"Ring 1+ E_{T} Sum","Events",0,0);
  PlotWrite(input,output,outfolder,TString("hfRingSum1-"),TString(basedata+hfrings+"hfRingSumD_1-"),TString(baseemu+hfrings+"hfRingSumE_1-"),"Ring 1- E_{T} Sum","Events",0,0);
  PlotWrite(input,output,outfolder,TString("hfRingSum2+"),TString(basedata+hfrings+"hfRingSumD_2+"),TString(baseemu+hfrings+"hfRingSumE_2+"),"Ring 2+ E_{T} Sum","Events",0,0);
  PlotWrite(input,output,outfolder,TString("hfRingSum2-"),TString(basedata+hfrings+"hfRingSumD_2-"),TString(baseemu+hfrings+"hfRingSumE_2-"),"Ring 2- E_{T} Sum","Events",0,0);
  PlotWriteErrors(input,output,outfolder,TString("hfRingSum_errorFlag"),TString(baseerrorhist+"hfRingSum_errorFlag"));
  
  //HF Bit Counts
  PlotWrite(input,output,outfolder,TString("hfBitCount1+"),TString(basedata+hfringb+"hfBitCountD_1+"),TString(baseemu+hfringb+"hfBitCountE_1+"),"Ring 1+ Bit Counts","Events",0,0);
  PlotWrite(input,output,outfolder,TString("hfBitCount1-"),TString(basedata+hfringb+"hfBitCountD_1-"),TString(baseemu+hfringb+"hfBitCountE_1-"),"Ring 1- Bit Counts","Events",0,0);
  PlotWrite(input,output,outfolder,TString("hfBitCount2+"),TString(basedata+hfringb+"hfBitCountD_2+"),TString(baseemu+hfringb+"hfBitCountE_2+"),"Ring 2+ Bit Counts","Events",0,0);
  PlotWrite(input,output,outfolder,TString("hfBitCount2-"),TString(basedata+hfringb+"hfBitCountD_2-"),TString(baseemu+hfringb+"hfBitCountE_2-"),"Ring 2- Bit Counts","Events",0,0);
  PlotWriteErrors(input,output,outfolder,TString("hfBitCount_errorFlag"),TString(baseerrorhist+"hfBitCount_errorFlag"));

  //Total ET
  PlotWrite(input,output,outfolder,TString("totalEt"),TString(basedata+totale+"totalEtD"),TString(baseemu+totale+"totalEtE"),"E_{T}","Events",32,0);
  PlotWriteErrors(input,output,outfolder,TString("totalEt_errorFlag"),TString(baseerrorhist+"totalEt_errorFlag"));

  //Total HT
  PlotWrite(input,output,outfolder,TString("totalHt"),TString(basedata+totale+"totalHtD"),TString(baseemu+totale+"totalHtE"),"H_{T}","Events",32,0);
  PlotWriteErrors(input,output,outfolder,TString("totalHt_errorFlag"),TString(baseerrorhist+"totalHt_errorFlag"));

  //Missing ET
  PlotWrite(input,output,outfolder,TString("missingEt"),TString(basedata+missinge+"missingEtD"),TString(baseemu+missinge+"missingEtE"),"|E_{T}^{miss}|","Events",32,0);
  PlotWrite(input,output,outfolder,TString("missingEt_phi"),TString(basedata+missinge+"missingEtD_Phi"),TString(baseemu+missinge+"missingEtE_Phi"),"E_{T}^{miss} #phi","Events",0,0);
  PlotWriteErrors(input,output,outfolder,TString("missingEt_errorFlag"),TString(baseerrorhist+"missingEt_errorFlag"));

  //Missing HT
  PlotWrite(input,output,outfolder,TString("missingHt"),TString(basedata+missinge+"missingHtD"),TString(baseemu+missinge+"missingHtE"),"|H_{T}^{miss}|","Events",2,128);
  PlotWrite(input,output,outfolder,TString("missingHt_phi"),TString(basedata+missinge+"missingHtD_Phi"),TString(baseemu+missinge+"missingHtE_Phi"),"H_{T}^{miss} #phi","Events",0,17);
  PlotWriteErrors(input,output,outfolder,TString("missingHt_errorFlag"),TString(baseerrorhist+"missingHt_errorFlag"));

  input->Close();
  output->Close();

  return;
  
}

void PlotWrite(TFile* input, TFile* output, TString output_folder, TString canvas, TString Hist_data, TString Hist_emu, TString XAxisLabel, TString YAxisLabel="Events", int rebin, int xAxisRange, TString Opt="")
{

  // Setup the canvas
  TCanvas *c1= new TCanvas(canvas,canvas,800,700);
  c1->SetLogy(1);

  // Get the histograms from the files
  if(Opt == "eta" || Opt == "phi")
  {
	if(Opt == "eta") {
	TH2F *Data2 = (TH2F*)input->Get(Hist_data);
	TH1D *Data = Data2->ProjectionX();
	if(rebin>0) Data->Rebin(rebin);
	if(xAxisRange>0) Data->GetXaxis()->SetRangeUser(0,xAxisRange);

	TH2F *Emu2 = (TH2F*)input->Get(Hist_emu);
	TH1D *Emu = Emu2->ProjectionX();
	if(rebin>0) Emu->Rebin(rebin);
	if(xAxisRange>0) Emu->GetXaxis()->SetRangeUser(0,xAxisRange);
	}
	if(Opt == "phi") {
	TH2F *Data2 = (TH2F*)input->Get(Hist_data);	
	TH1D *Data = Data2->ProjectionY();
	if(rebin>0) Data->Rebin(rebin);
    if(xAxisRange>0) Data->GetXaxis()->SetRangeUser(0,xAxisRange);

	TH2F *Emu2 = (TH2F*)input->Get(Hist_emu);
	TH1D *Emu = Emu2->ProjectionY();
	if(rebin>0) Emu->Rebin(rebin);
	if(xAxisRange>0) Emu->GetXaxis()->SetRangeUser(0,xAxisRange);
	}
  }
  else
  {
	TH1D *Data = (TH1D*)input->Get(Hist_data);
	if(rebin>0) Data->Rebin(rebin);
	if(xAxisRange>0) Data->GetXaxis()->SetRangeUser(0,xAxisRange);

	TH1D *Emu = (TH1D*)input->Get(Hist_emu);
	if(rebin>0) Emu->Rebin(rebin);
	if(xAxisRange>0) Emu->GetXaxis()->SetRangeUser(0,xAxisRange);
  }

  // Add the X axis label
  Emu->GetXaxis()->SetTitle(XAxisLabel);
  Emu->GetYaxis()->SetTitle(YAxisLabel);
  Emu->GetYaxis()->SetTitleSize(0.06);
  Emu->GetXaxis()->SetTitleSize(0.06);
  Emu->SetLineWidth(2);
  Emu->SetTitleOffset(1.10,"y");
  Emu->SetTitleOffset(0.80,"x");

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

  //make Legend
  TLegend * aLegend = new TLegend(0.646,0.768,0.746,0.868,NULL,"brNDC");
  aLegend->SetFillColor(0);
  aLegend->SetLineColor(0);
  aLegend->SetTextSize(0.05);
  aLegend->AddEntry(Emu,TString("Emulator"), "L");
  aLegend->AddEntry(Data,TString("Hardware"), "P");
  aLegend->DrawClone("same");
  gPad->RedrawAxis();

  //write canvas as png
  c1->Print(TString(output_folder+canvas+".png"));  

  //write canvas to output file
  output->cd();
  c1->Write();
  return;
}

void PlotWriteErrors(TFile* input, TFile* output, TString output_folder, TString canvas, TString Hist, TString Opt="")
{

  // Setup the canvas
  TCanvas *c1= new TCanvas(canvas,canvas,800,700);

  // Get the histograms from the files
  TH1D *Data = (TH1D*)input->Get(Hist);

  //check to make sure there are some events before setting the log scale
  if(Data->Integral() == 0 ) c1->SetLogy(0);
  else  c1->SetLogy(1);

  // Fill for histogram
  Data->SetFillColor(kBlue);
  Data->GetXaxis()->SetTitleSize(0.06);
  Data->GetXaxis()->SetTitleOffset(0.75);
  Data->GetYaxis()->SetTitleSize(0.05);
  Data->GetYaxis()->SetTitleOffset(1.00);

  // plot them
  Data->DrawCopy("hist");

  gPad->RedrawAxis();

  //write canvas as png
  c1->Print(TString(output_folder+canvas+".png"));  

  //write canvas to output file
  output->cd();
  c1->Write();
  return;
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
  setStyle->SetPadTopMargin(0.077);
  setStyle->SetPadBottomMargin(0.123);
  
  setStyle->cd();
}
