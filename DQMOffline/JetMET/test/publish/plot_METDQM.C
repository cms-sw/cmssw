//-------------------------------
// Usage: .L plot_METDQM.C
//        plot_METDQM("DQM_V0001_R000123575__JetMET__CMSSW_3_3_4__Harvesting.root",123575,"CaloMET");
//        plot_METDQM("DQM_V0001_R000123575__JetMET__CMSSW_3_3_4__Harvesting.root",123575,"CaloMETNoHF");
//        plot_METDQM("DQM_V0001_R000123575__JetMET__CMSSW_3_3_4__Harvesting.root",123575,"TcMET");
//        plot_METDQM("DQM_V0001_R000123575__JetMET__CMSSW_3_3_4__Harvesting.root",123575,"PfMET");
//-------------------------------

#include <stdio.h>
#include <stdlib.h>

void plot_METDQM(std::string filename, int run, std::string METName="CaloMET"){

  //-------------------------------

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetFrameLineWidth(2);
  gStyle->SetPadColor(0);
  gStyle->SetTitleFillColor(0);
  gStyle->SetStatColor(0);

  gStyle->SetOptStat(111110);
  gStyle->SetOptFit(1100);

  gStyle->SetStatX(0.92);
  gStyle->SetStatY(0.86);
  gStyle->SetStatW(0.60);
  gStyle->SetStatH(0.20);

  gStyle->SetTitleX(0.15);
  gStyle->SetTitleY(0.98);
  gStyle->SetTitleW(0.5);
  gStyle->SetTitleH(0.06);

  //-------------------------------

  std::cout << filename << std::endl;
  TFile *_file    = TFile::Open(filename.c_str());
  TFile *_fileRef = new TFile("DQM_V0001_R000000001__JetMET__CMSSW_3_3_4__Harvesting.root");

  char crun[10];
  sprintf(crun,"%d",run);

  char cprefix[3000];
  sprintf(cprefix,"DQMData/Run %d/JetMET/Run summary",run);  
  printf("%s\n",cprefix);

  char cprefixRef[3000];
  sprintf(cprefixRef,"DQMData/Run 1/JetMET/Run summary");
  printf("%s\n",cprefixRef);

  char ctitle[3000];
  char ctitleRef[3000];
  char cjpgname[3000];

  std::string METClass;
  if (METName=="CaloMET")     METClass="Calo";
  if (METName=="CaloMETNoHF") METClass="Calo";
  if (METName=="PfMET")       METClass="Pf";
  if (METName=="TcMET")       METClass="";

  //--------------------------------
  //--- METRate
  //--------------------------------

  sprintf(ctitle,"%s/MET/%s/All/METTask_%sMETRate",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METRate = (TH1F*) _file->Get(ctitle);    //
  METRate->SetTitleSize(0.065,"X");
  METRate->SetTitleSize(0.065,"Y");
  METRate->SetTitle("METRate");
  METRate->GetXaxis()->SetTitle("MET Threshold (GeV)");
  METRate->GetYaxis()->SetTitle("Rate (Hz)");

  sprintf(ctitle,"%s/MET/%s/JetIDMinimal/METTask_%sMETRate",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METRate_JetID                = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/JetIDTight/METTask_%sMETRate",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METRate_JetIDTight           = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilter/METTask_%sMETRate",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METRate_HcalNoiseFilter      = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilterTight/METTask_%sMETRate",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METRate_HcalNoiseFilterTight = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDLoosePass/METTask_%sMETRate",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METRate_BeamHaloIDLoosePass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDTightPass/METTask_%sMETRate",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METRate_BeamHaloIDTightPass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/Cleanup/METTask_%sMETRate",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METRate_Cleanup              = (TH1F*) _file->Get(ctitle);    //

  sprintf(ctitle,"%s/MET/%s/All/METTask_%sMETRate",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METRate_IonFeedbck = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/All/METTask_%sMETRate",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METRate_RBXNoise   = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/All/METTask_%sMETRate",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METRate_HPDNoise   = (TH1F*) _file->Get(ctitle);    //

  const int n = 3;
  Double_t x[n]  = {35., 60., 100.};
  Double_t y[n]  = {6.6, 0.5, 0.00001};
  METRate_QCD = new TGraphErrors(n,x,y);
  METRate_QCD->SetTitle("MET rate QCD 1E31");
  METRate_QCD->SetMarkerColor(4);
  METRate_QCD->SetMarkerStyle(21);

  const int n2 = 1;
  Double_t x2[n]  = {100.};
  Double_t y2[n]  = {0.2};
  METRate_QCD2 = new TGraphErrors(n2,x2,y2);
  METRate_QCD2->SetTitle("MET rate QCD 10E33");
  METRate_QCD2->SetMarkerColor(2);
  METRate_QCD2->SetMarkerStyle(21);

  //-------------------------------

  char title[100];
  char name[100];
  sprintf(title,"%s_Rate_run=%d",METName.c_str(),run);
  sprintf(name,"%s_Rate_%d",METName.c_str(),run);
  TCanvas *c_METRate = new TCanvas(title,name,600.,400.);

  gPad->SetBottomMargin(0.15);
  gPad->SetLeftMargin(0.15);
  gPad->SetLogy(1);
  gPad->SetGrid(1);

  METRate->SetLineColor(1);
  METRate->SetLineWidth(3);
  METRate_JetID->SetLineColor(2);
  METRate_JetID->SetLineWidth(3);
  METRate_JetIDTight->SetLineColor(9);
  METRate_JetIDTight->SetLineStyle(2);
  METRate_HcalNoiseFilter->SetLineColor(4);
  METRate_HcalNoiseFilterTight->SetLineColor(6);
  METRate_Cleanup->SetLineColor(8);
  METRate_BeamHaloIDLoosePass->SetLineColor(3);
  METRate_BeamHaloIDTightPass->SetLineColor(11);

  METRate->SetStats(kFALSE);
  METRate->SetTitle(title);
  METRate->GetXaxis()->SetRange(1,75);
  METRate->SetMaximum(100.);
  METRate->SetMinimum(0.0001);
  METRate->DrawCopy();
  METRate_JetID->Draw("same");
  METRate_JetIDTight->Draw("same");
  METRate_HcalNoiseFilter->Draw("same");
  METRate_HcalNoiseFilterTight->Draw("same");
  METRate_BeamHaloIDTightPass->Draw("same");
  METRate_BeamHaloIDLoosePass->Draw("same");
  METRate_Cleanup->Draw("same");
  //METRate_QCD->Draw("P,same");
  //METRate_QCD2->Draw("P,same");

  TLegend *tl_METRate = new TLegend(0.5,0.6,0.88,0.86);
  tl_METRate->AddEntry(METRate,"All triggers (data)","l");
  tl_METRate->AddEntry(METRate_JetID,"JetID cuts Minimal","l");
  tl_METRate->AddEntry(METRate_JetIDTight,"JetID cuts Tight","l");
  tl_METRate->AddEntry(METRate_HcalNoiseFilter,"Hcal Noise Filter","l");
  tl_METRate->AddEntry(METRate_HcalNoiseFilterTight,"Hcal Noise Filter Tight","l");
  tl_METRate->AddEntry(METRate_BeamHaloIDLoosePass,"BeamHalo ID Loose Pass","l");
  tl_METRate->AddEntry(METRate_BeamHaloIDTightPass,"BeamHalo ID Tight Pass","l");
  tl_METRate->AddEntry(METRate_Cleanup,"JetID + Hcal Noise + Halo Filter","l");
  //tl_METRate->AddEntry(METRate_QCD,"QCD 1E31","p");
  //tl_METRate->AddEntry(METRate_QCD2,"QCD 10E33","p");
  tl_METRate->SetFillColor(0);
  tl_METRate->Draw();

  sprintf(cjpgname,"%s/METRate.gif",METName.c_str());
  c_METRate->SaveAs(cjpgname);

  //-------------------------------
  // MET
  //-------------------------------

  sprintf(ctitle,"%s/MET/%s/All/METTask_%sMET",cprefix,METName.c_str(),METClass.c_str());
  std::cout << ctitle << std::endl;
  TH1F *MET = (TH1F*) _file->Get(ctitle);    //
  MET->SetTitleSize(0.065,"X");
  MET->SetTitleSize(0.065,"Y");
  MET->SetTitle("MET");
  MET->GetXaxis()->SetTitle("MET (GeV)");
  MET->GetYaxis()->SetTitle("Events");

  sprintf(ctitle,"%s/MET/%s/JetIDMinimal/METTask_%sMET",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MET_JetID                = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/JetIDTight/METTask_%sMET",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MET_JetIDTight           = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilter/METTask_%sMET",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MET_HcalNoiseFilter      = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilterTight/METTask_%sMET",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MET_HcalNoiseFilterTight = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDLoosePass/METTask_%sMET",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MET_BeamHaloIDLoosePass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDTightPass/METTask_%sMET",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MET_BeamHaloIDTightPass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/Cleanup/METTask_%sMET",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MET_Cleanup              = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/CleanupPV/METTask_%sMET",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MET_CleanupPV              = (TH1F*) _file->Get(ctitle);    //

  sprintf(ctitleRef,"%s/MET/%s/Cleanup/METTask_%sMET",cprefixRef,METName.c_str(),METClass.c_str());
  TH1F *MET_Ref = _fileRef->Get(ctitleRef);

  double SF=1.;
  if (METClass=="Calo") {
    if (MET_CleanupPV->GetEntries()>5.)  SF = (MET_CleanupPV->GetEntries())/(MET_Ref->GetEntries());
    else                                 SF = (MET_Cleanup->GetEntries())/(MET_Ref->GetEntries());
  } else {
    SF = (MET_Cleanup->GetEntries())/(MET_Ref->GetEntries());
  } 
  MET_Ref->Scale(SF);

  //-------------------------------

  sprintf(title,"%s_run=%d",METName.c_str(),run);
  sprintf(name,"%s_%d",METName.c_str(),run);
  TCanvas *c_MET = new TCanvas(title,name,600.,400.);

  gPad->SetBottomMargin(0.15);
  gPad->SetLeftMargin(0.15);
  gPad->SetLogy(1);
  gPad->SetGrid(1);

  MET->SetLineColor(1);
  MET_JetID->SetLineColor(2);
  MET_JetID->SetLineWidth(3);
  MET_JetIDTight->SetLineColor(9);
  MET_HcalNoiseFilter->SetLineColor(4);
  MET_HcalNoiseFilterTight->SetLineColor(6);
  MET_Cleanup->SetLineColor(8);
  if (METClass=="Calo") MET_CleanupPV->SetLineColor(4);
  MET_BeamHaloIDLoosePass->SetLineColor(2);
  MET_BeamHaloIDTightPass->SetLineColor(9);
  MET_BeamHaloIDLoosePass->SetLineWidth(2);
  MET_BeamHaloIDTightPass->SetLineWidth(2);

  MET->SetStats(kFALSE);
  MET->SetTitle(title);
  MET->GetXaxis()->SetRange(1,25);
  //MET->SetMaximum(1000.);
  MET->SetMinimum(0.5);
  MET->DrawCopy();
  //MET_JetID->Draw("same");
  //MET_JetIDTight->Draw("same");
  //MET_HcalNoiseFilter->Draw("same");
  //MET_HcalNoiseFilterTight->Draw("same");
  //MET_BeamHaloIDTightPass->Draw("same");
  //MET_BeamHaloIDLoosePass->Draw("same");
  MET_Cleanup->DrawCopy("same");

  MET_Ref->SetLineColor(2);
  MET_Ref->SetFillColor(2);
  MET_Ref->SetLineWidth(3);
  MET_Ref->DrawCopy("same");

  MET->DrawCopy("same,s");
  MET->DrawCopy("same,e");
  MET_Cleanup->DrawCopy("same,s");
  MET_Cleanup->DrawCopy("same,e");
  if (METClass=="Calo") MET_CleanupPV->DrawCopy("same,s");
  if (METClass=="Calo") MET_CleanupPV->DrawCopy("same,e");

  TLegend *tl_MET = new TLegend(0.5,0.65,0.88,0.86);
  tl_MET->AddEntry(MET,"All triggers (data)","l");
  //tl_MET->AddEntry(MET_JetID,"JetID cuts Minimal","l");
  //tl_MET->AddEntry(MET_JetIDTight,"JetID cuts Tight","l");
  //tl_MET->AddEntry(MET_HcalNoiseFilter,"Hcal Noise Filter","l");
  //tl_MET->AddEntry(MET_HcalNoiseFilterTight,"Hcal Noise Filter Tight","l");
  //tl_MET->AddEntry(MET_BeamHaloIDLoosePass,"BeamHalo ID Loose Pass","l");
  //tl_MET->AddEntry(MET_BeamHaloIDTightPass,"BeamHalo ID Tight Pass","l");
  tl_MET->AddEntry(MET_Cleanup,"JetID + Hcal Noise + Halo Filter","l");
  if (METClass=="Calo") tl_MET->AddEntry(MET_CleanupPV,"+ Primary Vertex","l");
  tl_MET->AddEntry(MET_Ref,"MinBias MC","l");
  tl_MET->SetFillColor(0);
  tl_MET->Draw();

  sprintf(cjpgname,"%s/MET.gif",METName.c_str());
  c_MET->SaveAs(cjpgname);

  //-------------------------------
  // MET_logx
  //-------------------------------

  sprintf(ctitle,"%s/MET/%s/All/METTask_%sMET_logx",cprefix,METName.c_str(),METClass.c_str());
  std::cout << ctitle << std::endl;
  TH1F *MET_logx = (TH1F*) _file->Get(ctitle);    //
  MET_logx->SetTitleSize(0.065,"X");
  MET_logx->SetTitleSize(0.065,"Y");
  MET_logx->SetTitle("MET");
  MET_logx->GetXaxis()->SetTitle("log10(MET [GeV])");
  MET_logx->GetYaxis()->SetTitle("Events");

  sprintf(ctitle,"%s/MET/%s/JetIDMinimal/METTask_%sMET_logx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MET_JetID_logx                = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/JetIDTight/METTask_%sMET_logx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MET_JetIDTight_logx           = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilter/METTask_%sMET_logx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MET_HcalNoiseFilter_logx      = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilterTight/METTask_%sMET_logx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MET_HcalNoiseFilterTight_logx = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDLoosePass/METTask_%sMET_logx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MET_BeamHaloIDLoosePass_logx  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDTightPass/METTask_%sMET_logx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MET_BeamHaloIDTightPass_logx  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/Cleanup/METTask_%sMET_logx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MET_Cleanup_logx              = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/CleanupPV/METTask_%sMET_logx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MET_CleanupPV_logx            = (TH1F*) _file->Get(ctitle);    //

  sprintf(ctitleRef,"%s/MET/%s/Cleanup/METTask_%sMET_logx",cprefixRef,METName.c_str(),METClass.c_str());
  TH1F *MET_Ref_logx = _fileRef->Get(ctitleRef);
  MET_Ref_logx->Scale(SF);

  //-------------------------------

  sprintf(title,"%s_logx_run=%d",METName.c_str(),run);
  sprintf(name,"%s_logx_%d",METName.c_str(),run);
  TCanvas *c_MET_logx = new TCanvas(title,name,600.,400.);

  gPad->SetBottomMargin(0.15);
  gPad->SetLeftMargin(0.15);
  gPad->SetLogy(1);
  gPad->SetGrid(1);

  MET_logx->SetLineColor(1);
  MET_JetID_logx->SetLineColor(2);
  MET_JetID_logx->SetLineWidth(3);
  MET_JetIDTight_logx->SetLineColor(9);
  MET_HcalNoiseFilter_logx->SetLineColor(4);
  MET_HcalNoiseFilterTight_logx->SetLineColor(6);
  MET_Cleanup_logx->SetLineColor(8);
  if (METClass=="Calo") MET_CleanupPV_logx->SetLineColor(4);
  MET_BeamHaloIDLoosePass_logx->SetLineColor(2);
  MET_BeamHaloIDTightPass_logx->SetLineColor(9);
  MET_BeamHaloIDLoosePass_logx->SetLineWidth(2);
  MET_BeamHaloIDTightPass_logx->SetLineWidth(2);

  MET_logx->SetStats(kFALSE);
  MET_logx->SetTitle(title);
  MET_logx->GetXaxis()->SetRange(1,25);
  //MET_logx->SetMaximum(1000.);
  MET_logx->SetMinimum(0.5);
  MET_logx->DrawCopy();
  //MET_JetID_logx->Draw("same");
  //MET_JetIDTight_logx->Draw("same");
  //MET_HcalNoiseFilter_logx->Draw("same");
  //MET_HcalNoiseFilterTight_logx->Draw("same");
  //MET_BeamHaloIDTightPass_logx->Draw("same");
  //MET_BeamHaloIDLoosePass_logx->Draw("same");
  MET_Cleanup_logx->DrawCopy("same");

  MET_Ref_logx->SetLineColor(2);
  MET_Ref_logx->SetFillColor(2);
  MET_Ref_logx->SetLineWidth(3);
  MET_Ref_logx->DrawCopy("same");

  MET_logx->DrawCopy("same,s");
  MET_logx->DrawCopy("same,e");
  MET_Cleanup_logx->DrawCopy("same,s");
  MET_Cleanup_logx->DrawCopy("same,e");
  if (METClass=="Calo") MET_CleanupPV_logx->DrawCopy("same,s");
  if (METClass=="Calo") MET_CleanupPV_logx->DrawCopy("same,e");

  TLegend *tl_MET_logx = new TLegend(0.5,0.65,0.88,0.86);
  tl_MET_logx->AddEntry(MET,"All triggers (data)","l");
  //tl_MET_logx->AddEntry(MET_JetID,"JetID cuts Minimal","l");
  //tl_MET_logx->AddEntry(MET_JetIDTight,"JetID cuts Tight","l");
  //tl_MET_logx->AddEntry(MET_HcalNoiseFilter,"Hcal Noise Filter","l");
  //tl_MET_logx->AddEntry(MET_HcalNoiseFilterTight,"Hcal Noise Filter Tight","l");
  //tl_MET_logx->AddEntry(MET_BeamHaloIDLoosePass,"BeamHalo ID Loose Pass","l");
  //tl_MET_logx->AddEntry(MET_BeamHaloIDTightPass,"BeamHalo ID Tight Pass","l");
  tl_MET_logx->AddEntry(MET_Cleanup,"JetID + Hcal Noise + Halo Filter","l");
  if (METClass=="Calo") tl_MET_logx->AddEntry(MET_CleanupPV,"+ Primary Vertex","l");
  tl_MET_logx->AddEntry(MET_Ref,"MinBias MC","l");
  tl_MET_logx->SetFillColor(0);
  tl_MET_logx->Draw();

  sprintf(cjpgname,"%s/MET_logx.gif",METName.c_str());
  c_MET_logx->SaveAs(cjpgname);

  //-------------------------------
  // SumET
  //-------------------------------

  sprintf(ctitle,"%s/MET/%s/All/METTask_%sSumET",cprefix,METName.c_str(),METClass.c_str());
  std::cout << ctitle << std::endl;
  TH1F *SumET = (TH1F*) _file->Get(ctitle);    //
  SumET->SetTitleSize(0.065,"X");
  SumET->SetTitleSize(0.065,"Y");
  SumET->SetTitle("SumET");
  SumET->GetXaxis()->SetTitle("SumET (GeV)");
  SumET->GetYaxis()->SetTitle("Events");

  sprintf(ctitle,"%s/MET/%s/JetIDMinimal/METTask_%sSumET",cprefix,METName.c_str(),METClass.c_str());
  TH1F *SumET_JetID                = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/JetIDTight/METTask_%sSumET",cprefix,METName.c_str(),METClass.c_str());
  TH1F *SumET_JetIDTight           = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilter/METTask_%sSumET",cprefix,METName.c_str(),METClass.c_str());
  TH1F *SumET_HcalNoiseFilter      = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilterTight/METTask_%sSumET",cprefix,METName.c_str(),METClass.c_str());
  TH1F *SumET_HcalNoiseFilterTight = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDLoosePass/METTask_%sSumET",cprefix,METName.c_str(),METClass.c_str());
  TH1F *SumET_BeamHaloIDLoosePass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDTightPass/METTask_%sSumET",cprefix,METName.c_str(),METClass.c_str());
  TH1F *SumET_BeamHaloIDTightPass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/Cleanup/METTask_%sSumET",cprefix,METName.c_str(),METClass.c_str());
  TH1F *SumET_Cleanup              = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/CleanupPV/METTask_%sSumET",cprefix,METName.c_str(),METClass.c_str());
  TH1F *SumET_CleanupPV              = (TH1F*) _file->Get(ctitle);    //

  sprintf(ctitleRef,"%s/MET/%s/Cleanup/METTask_%sSumET",cprefixRef,METName.c_str(),METClass.c_str());
  TH1F *SumET_Ref = _fileRef->Get(ctitleRef);
  SumET_Ref->Scale(SF);

  //-------------------------------

  sprintf(title,"%s_SumET_run=%d",METName.c_str(),run);
  sprintf(name,"%s_SumET_%d",METName.c_str(),run);
  TCanvas *c_SumET = new TCanvas(title,name,600.,400.);

  gPad->SetBottomMargin(0.15);
  gPad->SetLeftMargin(0.15);
  gPad->SetLogy(1);
  gPad->SetGrid(1);

  SumET->SetLineColor(1);
  SumET_JetID->SetLineColor(2);
  SumET_JetID->SetLineWidth(3);
  SumET_JetIDTight->SetLineColor(9);
  SumET_HcalNoiseFilter->SetLineColor(4);
  SumET_HcalNoiseFilterTight->SetLineColor(6);
  SumET_Cleanup->SetLineColor(8);
  if (METClass=="Calo") SumET_CleanupPV->SetLineColor(4);
  SumET_BeamHaloIDLoosePass->SetLineColor(2);
  SumET_BeamHaloIDTightPass->SetLineColor(9);
  SumET_BeamHaloIDLoosePass->SetLineWidth(2);
  SumET_BeamHaloIDTightPass->SetLineWidth(2);

  SumET->SetStats(kFALSE);
  SumET->SetTitle(title);
  SumET->GetXaxis()->SetRange(1,25);
  //SumET->SetMaximum(1000.);
  SumET->SetMinimum(0.5);
  SumET->DrawCopy();
  //SumET_JetID->Draw("same");
  //SumET_JetIDTight->Draw("same");
  //SumET_HcalNoiseFilter->Draw("same");
  //SumET_HcalNoiseFilterTight->Draw("same");
  //SumET_BeamHaloIDTightPass->Draw("same");
  //SumET_BeamHaloIDLoosePass->Draw("same");
  SumET_Cleanup->DrawCopy("same");

  SumET_Ref->SetLineColor(2);
  SumET_Ref->SetFillColor(2);
  SumET_Ref->SetLineWidth(3);
  SumET_Ref->DrawCopy("same");

  SumET->DrawCopy("same,s");
  SumET->DrawCopy("same,e");
  SumET_Cleanup->DrawCopy("same,s");
  SumET_Cleanup->DrawCopy("same,e");
  if (METClass=="Calo") SumET_CleanupPV->DrawCopy("same,s");
  if (METClass=="Calo") SumET_CleanupPV->DrawCopy("same,e");

  TLegend *tl_SumET = new TLegend(0.5,0.65,0.88,0.86);
  tl_SumET->AddEntry(SumET,"All triggers (data)","l");
  //tl_SumET->AddEntry(SumET_JetID,"JetID cuts Minimal","l");
  //tl_SumET->AddEntry(SumET_JetIDTight,"JetID cuts Tight","l");
  //tl_SumET->AddEntry(SumET_HcalNoiseFilter,"Hcal Noise Filter","l");
  //tl_SumET->AddEntry(SumET_HcalNoiseFilterTight,"Hcal Noise Filter Tight","l");
  //tl_SumET->AddEntry(SumET_BeamHaloIDLoosePass,"BeamHalo ID Loose Pass","l");
  //tl_SumET->AddEntry(SumET_BeamHaloIDTightPass,"BeamHalo ID Tight Pass","l");
  tl_SumET->AddEntry(SumET_Cleanup,"JetID + Hcal Noise + Halo Filter","l");
  if (METClass=="Calo") tl_SumET->AddEntry(SumET_CleanupPV,"+ Primary Vertex","l");
  tl_SumET->AddEntry(SumET_Ref,"MinBias MC","l");
  tl_SumET->SetFillColor(0);
  tl_SumET->Draw();

  sprintf(cjpgname,"%s/SumET.gif",METName.c_str());
  c_SumET->SaveAs(cjpgname);

  //-------------------------------
  // SumET_logx
  //-------------------------------

  sprintf(ctitle,"%s/MET/%s/All/METTask_%sSumET_logx",cprefix,METName.c_str(),METClass.c_str());
  std::cout << ctitle << std::endl;
  TH1F *SumET_logx = (TH1F*) _file->Get(ctitle);    //
  SumET_logx->SetTitleSize(0.065,"X");
  SumET_logx->SetTitleSize(0.065,"Y");
  SumET_logx->SetTitle("SumET_logx");
  SumET_logx->GetXaxis()->SetTitle("log10(SumET [GeV])");
  SumET_logx->GetYaxis()->SetTitle("Events");

  sprintf(ctitle,"%s/MET/%s/JetIDMinimal/METTask_%sSumET_logx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *SumET_JetID_logx                = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/JetIDTight/METTask_%sSumET_logx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *SumET_JetIDTight_logx           = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilter/METTask_%sSumET_logx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *SumET_HcalNoiseFilter_logx      = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilterTight/METTask_%sSumET_logx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *SumET_HcalNoiseFilterTight_logx = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDLoosePass/METTask_%sSumET_logx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *SumET_BeamHaloIDLoosePass_logx  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDTightPass/METTask_%sSumET_logx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *SumET_BeamHaloIDTightPass_logx  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/Cleanup/METTask_%sSumET_logx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *SumET_Cleanup_logx              = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/CleanupPV/METTask_%sSumET_logx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *SumET_CleanupPV_logx              = (TH1F*) _file->Get(ctitle);    //

  sprintf(ctitleRef,"%s/MET/%s/Cleanup/METTask_%sSumET_logx",cprefixRef,METName.c_str(),METClass.c_str());
  TH1F *SumET_Ref_logx = _fileRef->Get(ctitleRef);
  SumET_Ref_logx->Scale(SF);

  //-------------------------------

  sprintf(title,"%s_SumET_logx_run=%d",METName.c_str(),run);
  sprintf(name,"%s_SumET_logx_%d",METName.c_str(),run);
  TCanvas *c_SumET_logx = new TCanvas(title,name,600.,400.);

  gPad->SetBottomMargin(0.15);
  gPad->SetLeftMargin(0.15);
  gPad->SetLogy(1);
  gPad->SetGrid(1);

  SumET_logx->SetLineColor(1);
  SumET_JetID_logx->SetLineColor(2);
  SumET_JetID_logx->SetLineWidth(3);
  SumET_JetIDTight_logx->SetLineColor(9);
  SumET_HcalNoiseFilter_logx->SetLineColor(4);
  SumET_HcalNoiseFilterTight_logx->SetLineColor(6);
  SumET_Cleanup_logx->SetLineColor(8);
  if (METClass=="Calo") SumET_CleanupPV_logx->SetLineColor(4);
  SumET_BeamHaloIDLoosePass_logx->SetLineColor(2);
  SumET_BeamHaloIDTightPass_logx->SetLineColor(9);
  SumET_BeamHaloIDLoosePass_logx->SetLineWidth(2);
  SumET_BeamHaloIDTightPass_logx->SetLineWidth(2);

  SumET_logx->SetStats(kFALSE);
  SumET_logx->SetTitle(title);
  SumET_logx->GetXaxis()->SetRange(1,25);
  //SumET_logx->SetMaximum(1000.);
  SumET_logx->SetMinimum(0.5);
  SumET_logx->DrawCopy();
  //SumET_JetID_logx->Draw("same");
  //SumET_JetIDTight_logx->Draw("same");
  //SumET_HcalNoiseFilter_logx->Draw("same");
  //SumET_HcalNoiseFilterTight_logx->Draw("same");
  //SumET_BeamHaloIDTightPass_logx->Draw("same");
  //SumET_BeamHaloIDLoosePass_logx->Draw("same");
  SumET_Cleanup_logx->DrawCopy("same");

  SumET_Ref_logx->SetLineColor(2);
  SumET_Ref_logx->SetFillColor(2);
  SumET_Ref_logx->SetLineWidth(3);
  SumET_Ref_logx->DrawCopy("same");

  SumET_logx->DrawCopy("same,s");
  SumET_logx->DrawCopy("same,e");
  SumET_Cleanup_logx->DrawCopy("same,s");
  SumET_Cleanup_logx->DrawCopy("same,e");
  if (METClass=="Calo") SumET_CleanupPV_logx->DrawCopy("same,s");
  if (METClass=="Calo") SumET_CleanupPV_logx->DrawCopy("same,e");

  TLegend *tl_SumET_logx = new TLegend(0.5,0.65,0.88,0.86);
  tl_SumET_logx->AddEntry(SumET_logx,"All triggers (data)","l");
  //tl_SumET_logx->AddEntry(SumET_JetID_logx,"JetID cuts Minimal","l");
  //tl_SumET_logx->AddEntry(SumET_JetIDTight_logx,"JetID cuts Tight","l");
  //tl_SumET_logx->AddEntry(SumET_HcalNoiseFilter_logx,"Hcal Noise Filter","l");
  //tl_SumET_logx->AddEntry(SumET_HcalNoiseFilterTight_logx,"Hcal Noise Filter Tight","l");
  //tl_SumET_logx->AddEntry(SumET_BeamHaloIDLoosePass_logx,"BeamHalo ID Loose Pass","l");
  //tl_SumET_logx->AddEntry(SumET_BeamHaloIDTightPass_logx,"BeamHalo ID Tight Pass","l");
  tl_SumET_logx->AddEntry(SumET_Cleanup,"JetID + Hcal Noise + Halo Filter","l");
  if (METClass=="Calo") tl_SumET_logx->AddEntry(SumET_CleanupPV_logx,"+ Primary Vertex","l");
  tl_SumET_logx->AddEntry(SumET_Ref_logx,"MinBias MC","l");
  tl_SumET_logx->SetFillColor(0);
  tl_SumET_logx->Draw();

  sprintf(cjpgname,"%s/SumET_logx.gif",METName.c_str());
  c_SumET_logx->SaveAs(cjpgname);

  //-------------------------------
  // MEx
  //-------------------------------

  sprintf(ctitle,"%s/MET/%s/All/METTask_%sMEx",cprefix,METName.c_str(),METClass.c_str());
  std::cout << ctitle << std::endl;
  TH1F *MEx = (TH1F*) _file->Get(ctitle);    //
  MEx->SetTitleSize(0.065,"X");
  MEx->SetTitleSize(0.065,"Y");
  MEx->SetTitle("MEx");
  MEx->GetXaxis()->SetTitle("MEx (GeV)");
  MEx->GetYaxis()->SetTitle("Events");

  sprintf(ctitle,"%s/MET/%s/JetIDMinimal/METTask_%sMEx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MEx_JetID                = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/JetIDTight/METTask_%sMEx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MEx_JetIDTight           = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilter/METTask_%sMEx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MEx_HcalNoiseFilter      = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilterTight/METTask_%sMEx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MEx_HcalNoiseFilterTight = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDLoosePass/METTask_%sMEx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MEx_BeamHaloIDLoosePass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDTightPass/METTask_%sMEx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MEx_BeamHaloIDTightPass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/Cleanup/METTask_%sMEx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MEx_Cleanup              = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/CleanupPV/METTask_%sMEx",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MEx_CleanupPV              = (TH1F*) _file->Get(ctitle);    //

  sprintf(ctitleRef,"%s/MET/%s/Cleanup/METTask_%sMEx",cprefixRef,METName.c_str(),METClass.c_str());
  TH1F *MEx_Ref = _fileRef->Get(ctitleRef);
  MEx_Ref->Scale(SF);

  //-------------------------------

  sprintf(title,"%s_MEx_run=%d",METName.c_str(),run);
  sprintf(name,"%s_MEy_%d",METName.c_str(),run);
  TCanvas *c_MEx = new TCanvas(title,name,600.,400.);

  gPad->SetBottomMargin(0.15);
  gPad->SetLeftMargin(0.15);
  gPad->SetLogy(1);
  gPad->SetGrid(1);

  MEx->SetLineColor(1);
  MEx_JetID->SetLineColor(2);
  MEx_JetID->SetLineWidth(3);
  MEx_JetIDTight->SetLineColor(9);
  MEx_HcalNoiseFilter->SetLineColor(4);
  MEx_HcalNoiseFilterTight->SetLineColor(6);
  MEx_Cleanup->SetLineColor(8);
  if (METClass=="Calo") MEx_CleanupPV->SetLineColor(4);
  MEx_BeamHaloIDLoosePass->SetLineColor(2);
  MEx_BeamHaloIDTightPass->SetLineColor(9);
  MEx_BeamHaloIDLoosePass->SetLineWidth(2);
  MEx_BeamHaloIDTightPass->SetLineWidth(2);

  MEx->SetStats(kFALSE);
  MEx->SetTitle(title);
  MEx->GetXaxis()->SetRange(221,280);
  //MEx->SetMaximum(1000.);
  MEx->SetMinimum(0.5);
  MEx->DrawCopy();
  //MEx_JetID->Draw("same");
  //MEx_JetIDTight->Draw("same");
  //MEx_HcalNoiseFilter->Draw("same");
  //MEx_HcalNoiseFilterTight->Draw("same");
  //MEx_BeamHaloIDTightPass->Draw("same");
  //MEx_BeamHaloIDLoosePass->Draw("same");
  MEx_Cleanup->DrawCopy("same");

  MEx_Ref->SetLineColor(2);
  MEx_Ref->SetFillColor(2);
  MEx_Ref->SetLineWidth(3);
  MEx_Ref->DrawCopy("same");

  MEx->DrawCopy("same,s");
  MEx->DrawCopy("same,e");
  MEx_Cleanup->DrawCopy("same,s");
  MEx_Cleanup->DrawCopy("same,e");
  if (METClass=="Calo") MEx_CleanupPV->DrawCopy("same,s");
  if (METClass=="Calo") MEx_CleanupPV->DrawCopy("same,e");

  TLegend *tl_MEx = new TLegend(0.60,0.65,0.88,0.86);
  tl_MEx->AddEntry(MEx,"All triggers (data)","l");
  //tl_MEx->AddEntry(MEx_JetID,"JetID cuts Minimal","l");
  //tl_MEx->AddEntry(MEx_JetIDTight,"JetID cuts Tight","l");
  //tl_MEx->AddEntry(MEx_HcalNoiseFilter,"Hcal Noise Filter","l");
  //tl_MEx->AddEntry(MEx_HcalNoiseFilterTight,"Hcal Noise Filter Tight","l");
  //tl_MEx->AddEntry(MEx_BeamHaloIDLoosePass,"BeamHalo ID Loose Pass","l");
  //tl_MEx->AddEntry(MEx_BeamHaloIDTightPass,"BeamHalo ID Tight Pass","l");
  tl_MEx->AddEntry(MEx_Cleanup,"JetID + Hcal Noise + Halo Filter","l");
  if (METClass=="Calo") tl_MEx->AddEntry(MEx_CleanupPV,"+ Primary Vertex","l");
  tl_MEx->AddEntry(MEx_Ref,"MinBias MC","l");
  tl_MEx->SetFillColor(0);
  tl_MEx->Draw();

  sprintf(cjpgname,"%s/MEx.gif",METName.c_str());
  c_MEx->SaveAs(cjpgname);

  //-------------------------------
  // MEy
  //-------------------------------

  sprintf(ctitle,"%s/MET/%s/All/METTask_%sMEy",cprefix,METName.c_str(),METClass.c_str());
  std::cout << ctitle << std::endl;
  TH1F *MEy = (TH1F*) _file->Get(ctitle);    //
  MEy->SetTitleSize(0.065,"X");
  MEy->SetTitleSize(0.065,"Y");
  MEy->SetTitle("MEy");
  MEy->GetXaxis()->SetTitle("MEy (GeV)");
  MEy->GetYaxis()->SetTitle("Events");

  sprintf(ctitle,"%s/MET/%s/JetIDMinimal/METTask_%sMEy",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MEy_JetID                = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/JetIDTight/METTask_%sMEy",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MEy_JetIDTight           = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilter/METTask_%sMEy",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MEy_HcalNoiseFilter      = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilterTight/METTask_%sMEy",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MEy_HcalNoiseFilterTight = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDLoosePass/METTask_%sMEy",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MEy_BeamHaloIDLoosePass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDTightPass/METTask_%sMEy",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MEy_BeamHaloIDTightPass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/Cleanup/METTask_%sMEy",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MEy_Cleanup              = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/CleanupPV/METTask_%sMEy",cprefix,METName.c_str(),METClass.c_str());
  TH1F *MEy_CleanupPV              = (TH1F*) _file->Get(ctitle);    //

  sprintf(ctitleRef,"%s/MET/%s/Cleanup/METTask_%sMEy",cprefixRef,METName.c_str(),METClass.c_str());
  TH1F *MEy_Ref = _fileRef->Get(ctitleRef);
  MEy_Ref->Scale(SF);

  //-------------------------------

  sprintf(title,"%s_MEy_run=%d",METName.c_str(),run);
  sprintf(name,"%s_MEy_%d",METName.c_str(),run);
  TCanvas *c_MEy = new TCanvas(title,name,600.,400.);

  gPad->SetBottomMargin(0.15);
  gPad->SetLeftMargin(0.15);
  gPad->SetLogy(1);
  gPad->SetGrid(1);

  MEy->SetLineColor(1);
  MEy_JetID->SetLineColor(2);
  MEy_JetID->SetLineWidth(3);
  MEy_JetIDTight->SetLineColor(9);
  MEy_HcalNoiseFilter->SetLineColor(4);
  MEy_HcalNoiseFilterTight->SetLineColor(6);
  MEy_Cleanup->SetLineColor(8);
  if (METClass=="Calo") MEy_CleanupPV->SetLineColor(4);
  MEy_BeamHaloIDLoosePass->SetLineColor(2);
  MEy_BeamHaloIDTightPass->SetLineColor(9);
  MEy_BeamHaloIDLoosePass->SetLineWidth(2);
  MEy_BeamHaloIDTightPass->SetLineWidth(2);

  MEy->SetStats(kFALSE);
  MEy->SetTitle(title);
  MEy->GetXaxis()->SetRange(221,280);
  //MEy->SetMaximum(1000.);
  MEy->SetMinimum(0.5);
  MEy->DrawCopy();
  //MEy_JetID->Draw("same");
  //MEy_JetIDTight->Draw("same");
  //MEy_HcalNoiseFilter->Draw("same");
  //MEy_HcalNoiseFilterTight->Draw("same");
  //MEy_BeamHaloIDTightPass->Draw("same");
  //MEy_BeamHaloIDLoosePass->Draw("same");
  MEy_Cleanup->DrawCopy("same");

  MEy_Ref->SetLineColor(2);
  MEy_Ref->SetFillColor(2);
  MEy_Ref->SetLineWidth(3);
  MEy_Ref->DrawCopy("same");

  MEy->DrawCopy("same,s");
  MEy->DrawCopy("same,e");
  MEy_Cleanup->DrawCopy("same,s");
  MEy_Cleanup->DrawCopy("same,e");
  if (METClass=="Calo") MEy_CleanupPV->DrawCopy("same,s");
  if (METClass=="Calo") MEy_CleanupPV->DrawCopy("same,e");

  TLegend *tl_MEy = new TLegend(0.60,0.65,0.88,0.86);
  tl_MEy->AddEntry(MEy,"All triggers (data)","l");
  //tl_MEy->AddEntry(MEy_JetID,"JetID cuts Minimal","l");
  //tl_MEy->AddEntry(MEy_JetIDTight,"JetID cuts Tight","l");
  //tl_MEy->AddEntry(MEy_HcalNoiseFilter,"Hcal Noise Filter","l");
  //tl_MEy->AddEntry(MEy_HcalNoiseFilterTight,"Hcal Noise Filter Tight","l");
  //tl_MEy->AddEntry(MEy_BeamHaloIDLoosePass,"BeamHalo ID Loose Pass","l");
  //tl_MEy->AddEntry(MEy_BeamHaloIDTightPass,"BeamHalo ID Tight Pass","l");
  tl_MEy->AddEntry(MEy_Cleanup,"JetID + Hcal Noise + Halo Filter","l");
  if (METClass=="Calo") tl_MEy->AddEntry(MEy_CleanupPV,"+ Primary Vertex","l");
  tl_MEy->AddEntry(MEy_Ref,"MinBias MC","l");
  tl_MEy->SetFillColor(0);
  tl_MEy->Draw();

  sprintf(cjpgname,"%s/MEy.gif",METName.c_str());
  c_MEy->SaveAs(cjpgname);

  //-------------------------------

  /*
  sprintf(title,"%s_Rate_HcalNoise_run=%d",METName.c_str(),run);
  sprintf(name,"%s_Rate_HcalNoise_%d",METName.c_str(),run);
  TCanvas *c_METRateHcalNoise = new TCanvas(title,title,600.,400.);

  gPad->SetBottomMargin(0.15);
  gPad->SetLeftMargin(0.15);
  gPad->SetLogy(1);
  gPad->SetGrid(1);

  METRate->SetLineColor(1);
  METRate_IonFeedbck->SetLineColor(2);
  METRate_IonFeedbck->SetLineWidth(3);
  METRate_RBXNoise->SetLineColor(4);
  METRate_HPDNoise->SetLineColor(6);

  METRate->SetStats(kFALSE);
  METRate->SetTitle(title);
  METRate->GetXaxis()->SetRange(1,75);
  METRate->SetMinimum(0.001);
  METRate->DrawCopy();
  METRate_IonFeedbck->Draw("same");
  METRate_RBXNoise->Draw("same");
  METRate_HPDNoise->Draw("same");

  TLegend *tl_METRate = new TLegend(0.5,0.6,0.88,0.86);
  tl_METRate->AddEntry(METRate,"All triggers","l");
  tl_METRate->AddEntry(METRate_IonFeedbck,"Ion Feedback","l");
  tl_METRate->AddEntry(METRate_RBXNoise,"HPD noise","l");
  tl_METRate->AddEntry(METRate_HPDNoise,"RBX noise","l");
  tl_METRate->SetFillColor(0);
  tl_METRate->Draw();

  sprintf(cjpgname,"%s/METRateHcalNoise.gif",METName.c_str());
  //c_METRateHcalNoise->SaveAs(cjpgname);
  */

  //-------------------------------
  // METPhi
  //-------------------------------

  sprintf(ctitle,"%s/MET/%s/All/METTask_%sMETPhi",cprefix,METName.c_str(),METClass.c_str());
  std::cout << ctitle << std::endl;
  TH1F *METPhi = (TH1F*) _file->Get(ctitle);    //
  METPhi->SetTitleSize(0.065,"X");
  METPhi->SetTitleSize(0.065,"Y");
  METPhi->SetTitle("METPhi");
  METPhi->GetXaxis()->SetTitle("METPhi (rad)");
  METPhi->GetYaxis()->SetTitle("Events");

  sprintf(ctitle,"%s/MET/%s/JetIDMinimal/METTask_%sMETPhi",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi_JetID                = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/JetIDTight/METTask_%sMETPhi",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi_JetIDTight           = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilter/METTask_%sMETPhi",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi_HcalNoiseFilter      = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilterTight/METTask_%sMETPhi",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi_HcalNoiseFilterTight = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDLoosePass/METTask_%sMETPhi",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi_BeamHaloIDLoosePass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDTightPass/METTask_%sMETPhi",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi_BeamHaloIDTightPass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/Cleanup/METTask_%sMETPhi",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi_Cleanup              = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/CleanupPV/METTask_%sMETPhi",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi_CleanupPV              = (TH1F*) _file->Get(ctitle);    //

  sprintf(ctitleRef,"%s/MET/%s/Cleanup/METTask_%sMETPhi",cprefixRef,METName.c_str(),METClass.c_str());
  TH1F *METPhi_Ref = _fileRef->Get(ctitleRef);
  METPhi_Ref->Scale(SF);

  //-------------------------------

  sprintf(title,"%s_METPhi_run=%d",METName.c_str(),run);
  sprintf(name,"%s_METPhi_%d",METName.c_str(),run);
  TCanvas *c_METPhi = new TCanvas(title,name,600.,400.);

  gPad->SetBottomMargin(0.15);
  gPad->SetLeftMargin(0.15);
  //gPad->SetLogy(1);
  gPad->SetGrid(1);

  METPhi->SetLineColor(1);
  METPhi_JetID->SetLineColor(2);
  METPhi_JetID->SetLineWidth(3);
  METPhi_JetIDTight->SetLineColor(9);
  METPhi_HcalNoiseFilter->SetLineColor(4);
  METPhi_HcalNoiseFilterTight->SetLineColor(6);
  METPhi_Cleanup->SetLineColor(8);
  if (METClass=="Calo") METPhi_CleanupPV->SetLineColor(4);
  METPhi_BeamHaloIDLoosePass->SetLineColor(2);
  METPhi_BeamHaloIDTightPass->SetLineColor(9);
  METPhi_BeamHaloIDLoosePass->SetLineWidth(2);
  METPhi_BeamHaloIDTightPass->SetLineWidth(2);

  METPhi->SetStats(kFALSE);
  METPhi->SetTitle(title);
  METPhi->GetXaxis()->SetRange(221,280);
  //METPhi->SetMaximum(1000.);
  METPhi->SetMinimum(0.);
  METPhi->DrawCopy();
  //METPhi_JetID->Draw("same");
  //METPhi_JetIDTight->Draw("same");
  //METPhi_HcalNoiseFilter->Draw("same");
  //METPhi_HcalNoiseFilterTight->Draw("same");
  //METPhi_BeamHaloIDTightPass->Draw("same");
  //METPhi_BeamHaloIDLoosePass->Draw("same");
  METPhi_Cleanup->DrawCopy("same");

  METPhi_Ref->SetLineColor(2);
  METPhi_Ref->SetFillColor(2);
  METPhi_Ref->SetLineWidth(3);
  METPhi_Ref->DrawCopy("same");

  METPhi->DrawCopy("same,s");
  METPhi->DrawCopy("same,e");
  METPhi_Cleanup->DrawCopy("same,s");
  METPhi_Cleanup->DrawCopy("same,e");
  if (METClass=="Calo") METPhi_CleanupPV->DrawCopy("same,s");
  if (METClass=="Calo") METPhi_CleanupPV->DrawCopy("same,e");

  TLegend *tl_METPhi = new TLegend(0.20,0.20,0.55,0.40);
  tl_METPhi->AddEntry(METPhi,"All triggers (data)","l");
  //tl_METPhi->AddEntry(METPhi_JetID,"JetID cuts Minimal","l");
  //tl_METPhi->AddEntry(METPhi_JetIDTight,"JetID cuts Tight","l");
  //tl_METPhi->AddEntry(METPhi_HcalNoiseFilter,"Hcal Noise Filter","l");
  //tl_METPhi->AddEntry(METPhi_HcalNoiseFilterTight,"Hcal Noise Filter Tight","l");
  //tl_METPhi->AddEntry(METPhi_BeamHaloIDLoosePass,"BeamHalo ID Loose Pass","l");
  //tl_METPhi->AddEntry(METPhi_BeamHaloIDTightPass,"BeamHalo ID Tight Pass","l");
  tl_METPhi->AddEntry(METPhi_Cleanup,"JetID + Hcal Noise + Halo Filter","l");
  if (METClass=="Calo") tl_METPhi->AddEntry(METPhi_CleanupPV,"+ Primary Vertex","l");
  tl_METPhi->AddEntry(METPhi_Ref,"MinBias MC","l");
  tl_METPhi->SetFillColor(0);
  tl_METPhi->Draw();

  sprintf(cjpgname,"%s/METPhi.gif",METName.c_str());
  c_METPhi->SaveAs(cjpgname);

  if (METClass=="Calo") {

  //-------------------------------
  // METPhi002
  //-------------------------------

  sprintf(ctitle,"%s/MET/%s/All/METTask_%sMETPhi002",cprefix,METName.c_str(),METClass.c_str());
  std::cout << ctitle << std::endl;
  TH1F *METPhi002 = (TH1F*) _file->Get(ctitle);    //
  METPhi002->SetTitleSize(0.065,"X");
  METPhi002->SetTitleSize(0.065,"Y");
  METPhi002->SetTitle("METPhi");
  METPhi002->GetXaxis()->SetTitle("METPhi (rad)");
  METPhi002->GetYaxis()->SetTitle("Events");

  sprintf(ctitle,"%s/MET/%s/JetIDMinimal/METTask_%sMETPhi002",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi002_JetID                = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/JetIDTight/METTask_%sMETPhi002",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi002_JetIDTight           = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilter/METTask_%sMETPhi002",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi002_HcalNoiseFilter      = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilterTight/METTask_%sMETPhi002",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi002_HcalNoiseFilterTight = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDLoosePass/METTask_%sMETPhi002",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi002_BeamHaloIDLoosePass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDTightPass/METTask_%sMETPhi002",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi002_BeamHaloIDTightPass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/Cleanup/METTask_%sMETPhi002",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi002_Cleanup              = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/CleanupPV/METTask_%sMETPhi002",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi002_CleanupPV              = (TH1F*) _file->Get(ctitle);    //

  sprintf(ctitleRef,"%s/MET/%s/Cleanup/METTask_%sMETPhi002",cprefixRef,METName.c_str(),METClass.c_str());
  TH1F *METPhi002_Ref = _fileRef->Get(ctitleRef);
  METPhi002_Ref->Scale(SF);

  //-------------------------------

  sprintf(title,"%s_METPhi_run=%d (MET>2GeV)",METName.c_str(),run);
  sprintf(name,"%s_METPhi002_%d",METName.c_str(),run);
  TCanvas *c_METPhi002 = new TCanvas(title,name,600.,400.);

  gPad->SetBottomMargin(0.15);
  gPad->SetLeftMargin(0.15);
  //gPad->SetLogy(1);
  gPad->SetGrid(1);

  METPhi002->SetLineColor(1);
  METPhi002_JetID->SetLineColor(2);
  METPhi002_JetID->SetLineWidth(3);
  METPhi002_JetIDTight->SetLineColor(9);
  METPhi002_HcalNoiseFilter->SetLineColor(4);
  METPhi002_HcalNoiseFilterTight->SetLineColor(6);
  METPhi002_Cleanup->SetLineColor(8);
  if (METClass=="Calo") METPhi002_CleanupPV->SetLineColor(4);
  METPhi002_BeamHaloIDLoosePass->SetLineColor(2);
  METPhi002_BeamHaloIDTightPass->SetLineColor(9);
  METPhi002_BeamHaloIDLoosePass->SetLineWidth(2);
  METPhi002_BeamHaloIDTightPass->SetLineWidth(2);

  METPhi002->SetStats(kFALSE);
  METPhi002->SetTitle(title);
  METPhi002->GetXaxis()->SetRange(221,280);
  //METPhi002->SetMaximum(1000.);
  METPhi002->SetMinimum(0.0);
  METPhi002->DrawCopy();
  //METPhi002_JetID->Draw("same");
  //METPhi002_JetIDTight->Draw("same");
  //METPhi002_HcalNoiseFilter->Draw("same");
  //METPhi002_HcalNoiseFilterTight->Draw("same");
  //METPhi002_BeamHaloIDTightPass->Draw("same");
  //METPhi002_BeamHaloIDLoosePass->Draw("same");
  METPhi002_Cleanup->DrawCopy("same");

  METPhi002_Ref->SetLineColor(2);
  METPhi002_Ref->SetFillColor(2);
  METPhi002_Ref->SetLineWidth(3);
  METPhi002_Ref->DrawCopy("same");

  METPhi002->DrawCopy("same,s");
  METPhi002->DrawCopy("same,e");
  METPhi002_Cleanup->DrawCopy("same,s");
  METPhi002_Cleanup->DrawCopy("same,e");
  if (METClass=="Calo") METPhi002_CleanupPV->DrawCopy("same,s");
  if (METClass=="Calo") METPhi002_CleanupPV->DrawCopy("same,e");

  TLegend *tl_METPhi002 = new TLegend(0.20,0.20,0.55,0.40);
  tl_METPhi002->AddEntry(METPhi002,"All triggers (data)","l");
  //tl_METPhi002->AddEntry(METPhi002_JetID,"JetID cuts Minimal","l");
  //tl_METPhi002->AddEntry(METPhi002_JetIDTight,"JetID cuts Tight","l");
  //tl_METPhi002->AddEntry(METPhi002_HcalNoiseFilter,"Hcal Noise Filter","l");
  //tl_METPhi002->AddEntry(METPhi002_HcalNoiseFilterTight,"Hcal Noise Filter Tight","l");
  //tl_METPhi002->AddEntry(METPhi002_BeamHaloIDLoosePass,"BeamHalo ID Loose Pass","l");
  //tl_METPhi002->AddEntry(METPhi002_BeamHaloIDTightPass,"BeamHalo ID Tight Pass","l");
  tl_METPhi002->AddEntry(METPhi002_Cleanup,"JetID + Hcal Noise + Halo Filter","l");
  if (METClass=="Calo") tl_METPhi002->AddEntry(METPhi002_CleanupPV,"+ Primary Vertex","l");
  tl_METPhi002->AddEntry(METPhi002_Ref,"MinBias MC","l");
  tl_METPhi002->SetFillColor(0);
  tl_METPhi002->Draw();

  sprintf(cjpgname,"%s/METPhi002.gif",METName.c_str());
  c_METPhi002->SaveAs(cjpgname);

  //-------------------------------
  // METPhi010
  //-------------------------------

  sprintf(ctitle,"%s/MET/%s/All/METTask_%sMETPhi010",cprefix,METName.c_str(),METClass.c_str());
  std::cout << ctitle << std::endl;
  TH1F *METPhi010 = (TH1F*) _file->Get(ctitle);    //
  METPhi010->SetTitleSize(0.065,"X");
  METPhi010->SetTitleSize(0.065,"Y");
  METPhi010->SetTitle("METPhi");
  METPhi010->GetXaxis()->SetTitle("METPhi (rad)");
  METPhi010->GetYaxis()->SetTitle("Events");

  sprintf(ctitle,"%s/MET/%s/JetIDMinimal/METTask_%sMETPhi010",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi010_JetID                = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/JetIDTight/METTask_%sMETPhi010",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi010_JetIDTight           = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilter/METTask_%sMETPhi010",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi010_HcalNoiseFilter      = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilterTight/METTask_%sMETPhi010",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi010_HcalNoiseFilterTight = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDLoosePass/METTask_%sMETPhi010",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi010_BeamHaloIDLoosePass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDTightPass/METTask_%sMETPhi010",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi010_BeamHaloIDTightPass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/Cleanup/METTask_%sMETPhi010",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi010_Cleanup              = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/CleanupPV/METTask_%sMETPhi010",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi010_CleanupPV              = (TH1F*) _file->Get(ctitle);    //

  sprintf(ctitleRef,"%s/MET/%s/Cleanup/METTask_%sMETPhi010",cprefixRef,METName.c_str(),METClass.c_str());
  TH1F *METPhi010_Ref = _fileRef->Get(ctitleRef);
  METPhi010_Ref->Scale(SF);

  //-------------------------------

  sprintf(title,"%s_METPhi_run=%d (MET>10GeV)",METName.c_str(),run);
  sprintf(name,"%s_METPhi010_%d",METName.c_str(),run);
  TCanvas *c_METPhi010 = new TCanvas(title,name,600.,400.);

  gPad->SetBottomMargin(0.15);
  gPad->SetLeftMargin(0.15);
  //gPad->SetLogy(1);
  gPad->SetGrid(1);

  METPhi010->SetLineColor(1);
  METPhi010_JetID->SetLineColor(2);
  METPhi010_JetID->SetLineWidth(3);
  METPhi010_JetIDTight->SetLineColor(9);
  METPhi010_HcalNoiseFilter->SetLineColor(4);
  METPhi010_HcalNoiseFilterTight->SetLineColor(6);
  METPhi010_Cleanup->SetLineColor(8);
  if (METClass=="Calo") METPhi010_CleanupPV->SetLineColor(4);
  METPhi010_BeamHaloIDLoosePass->SetLineColor(2);
  METPhi010_BeamHaloIDTightPass->SetLineColor(9);
  METPhi010_BeamHaloIDLoosePass->SetLineWidth(2);
  METPhi010_BeamHaloIDTightPass->SetLineWidth(2);

  METPhi010->SetStats(kFALSE);
  METPhi010->SetTitle(title);
  METPhi010->GetXaxis()->SetRange(221,280);
  //METPhi010->SetMaximum(1000.);
  METPhi010->SetMinimum(0.0);
  METPhi010->DrawCopy();
  //METPhi010_JetID->Draw("same");
  //METPhi010_JetIDTight->Draw("same");
  //METPhi010_HcalNoiseFilter->Draw("same");
  //METPhi010_HcalNoiseFilterTight->Draw("same");
  //METPhi010_BeamHaloIDTightPass->Draw("same");
  //METPhi010_BeamHaloIDLoosePass->Draw("same");
  METPhi010_Cleanup->DrawCopy("same");

  METPhi010_Ref->SetLineColor(2);
  METPhi010_Ref->SetFillColor(2);
  METPhi010_Ref->SetLineWidth(3);
  METPhi010_Ref->DrawCopy("same");

  METPhi010->DrawCopy("same,s");
  METPhi010->DrawCopy("same,e");
  METPhi010_Cleanup->DrawCopy("same,s");
  METPhi010_Cleanup->DrawCopy("same,e");
  if (METClass=="Calo") METPhi010_CleanupPV->DrawCopy("same,s");
  if (METClass=="Calo") METPhi010_CleanupPV->DrawCopy("same,e");

  TLegend *tl_METPhi010 = new TLegend(0.20,0.65,0.55,0.85);
  tl_METPhi010->AddEntry(METPhi010,"All triggers (data)","l");
  //tl_METPhi010->AddEntry(METPhi010_JetID,"JetID cuts Minimal","l");
  //tl_METPhi010->AddEntry(METPhi010_JetIDTight,"JetID cuts Tight","l");
  //tl_METPhi010->AddEntry(METPhi010_HcalNoiseFilter,"Hcal Noise Filter","l");
  //tl_METPhi010->AddEntry(METPhi010_HcalNoiseFilterTight,"Hcal Noise Filter Tight","l");
  //tl_METPhi010->AddEntry(METPhi010_BeamHaloIDLoosePass,"BeamHalo ID Loose Pass","l");
  //tl_METPhi010->AddEntry(METPhi010_BeamHaloIDTightPass,"BeamHalo ID Tight Pass","l");
  tl_METPhi010->AddEntry(METPhi010_Cleanup,"JetID + Hcal Noise + Halo Filter","l");
  if (METClass=="Calo") tl_METPhi010->AddEntry(METPhi010_CleanupPV,"+ Primary Vertex","l");
  tl_METPhi010->AddEntry(METPhi010_Ref,"MinBias MC","l");
  tl_METPhi010->SetFillColor(0);
  tl_METPhi010->Draw();

  sprintf(cjpgname,"%s/METPhi010.gif",METName.c_str());
  c_METPhi010->SaveAs(cjpgname);

  //-------------------------------
  // METPhi020
  //-------------------------------

  sprintf(ctitle,"%s/MET/%s/All/METTask_%sMETPhi020",cprefix,METName.c_str(),METClass.c_str());
  std::cout << ctitle << std::endl;
  TH1F *METPhi020 = (TH1F*) _file->Get(ctitle);    //
  METPhi020->SetTitleSize(0.065,"X");
  METPhi020->SetTitleSize(0.065,"Y");
  METPhi020->SetTitle("METPhi");
  METPhi020->GetXaxis()->SetTitle("METPhi (rad)");
  METPhi020->GetYaxis()->SetTitle("Events");

  sprintf(ctitle,"%s/MET/%s/JetIDMinimal/METTask_%sMETPhi020",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi020_JetID                = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/JetIDTight/METTask_%sMETPhi020",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi020_JetIDTight           = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilter/METTask_%sMETPhi020",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi020_HcalNoiseFilter      = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilterTight/METTask_%sMETPhi020",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi020_HcalNoiseFilterTight = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDLoosePass/METTask_%sMETPhi020",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi020_BeamHaloIDLoosePass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDTightPass/METTask_%sMETPhi020",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi020_BeamHaloIDTightPass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/Cleanup/METTask_%sMETPhi020",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi020_Cleanup              = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/CleanupPV/METTask_%sMETPhi020",cprefix,METName.c_str(),METClass.c_str());
  TH1F *METPhi020_CleanupPV              = (TH1F*) _file->Get(ctitle);    //

  sprintf(ctitleRef,"%s/MET/%s/Cleanup/METTask_%sMETPhi020",cprefixRef,METName.c_str(),METClass.c_str());
  TH1F *METPhi020_Ref = _fileRef->Get(ctitleRef);
  METPhi020_Ref->Scale(SF);

  //-------------------------------

  sprintf(title,"%s_METPhi_run=%d (MET>20GeV)",METName.c_str(),run);
  sprintf(name,"%s_METPhi020_%d",METName.c_str(),run);
  TCanvas *c_METPhi020 = new TCanvas(title,name,600.,400.);

  gPad->SetBottomMargin(0.15);
  gPad->SetLeftMargin(0.15);
  //gPad->SetLogy(1);
  gPad->SetGrid(1);

  METPhi020->SetLineColor(1);
  METPhi020_JetID->SetLineColor(2);
  METPhi020_JetID->SetLineWidth(3);
  METPhi020_JetIDTight->SetLineColor(9);
  METPhi020_HcalNoiseFilter->SetLineColor(4);
  METPhi020_HcalNoiseFilterTight->SetLineColor(6);
  METPhi020_Cleanup->SetLineColor(8);
  if (METClass=="Calo") METPhi020_CleanupPV->SetLineColor(4);
  METPhi020_BeamHaloIDLoosePass->SetLineColor(2);
  METPhi020_BeamHaloIDTightPass->SetLineColor(9);
  METPhi020_BeamHaloIDLoosePass->SetLineWidth(2);
  METPhi020_BeamHaloIDTightPass->SetLineWidth(2);

  METPhi020->SetStats(kFALSE);
  METPhi020->SetTitle(title);
  METPhi020->GetXaxis()->SetRange(221,280);
  //METPhi020->SetMaximum(1000.);
  METPhi020->SetMinimum(0.0);
  METPhi020->DrawCopy();
  //METPhi020_JetID->Draw("same");
  //METPhi020_JetIDTight->Draw("same");
  //METPhi020_HcalNoiseFilter->Draw("same");
  //METPhi020_HcalNoiseFilterTight->Draw("same");
  //METPhi020_BeamHaloIDTightPass->Draw("same");
  //METPhi020_BeamHaloIDLoosePass->Draw("same");
  METPhi020_Cleanup->DrawCopy("same");

  METPhi020_Ref->SetLineColor(2);
  METPhi020_Ref->SetFillColor(2);
  METPhi020_Ref->SetLineWidth(3);
  METPhi020_Ref->DrawCopy("same");

  METPhi020->DrawCopy("same,s");
  METPhi020->DrawCopy("same,e");
  METPhi020_Cleanup->DrawCopy("same,s");
  METPhi020_Cleanup->DrawCopy("same,e");
  if (METClass=="Calo") METPhi020_CleanupPV->DrawCopy("same,s");
  if (METClass=="Calo") METPhi020_CleanupPV->DrawCopy("same,e");

  TLegend *tl_METPhi020 = new TLegend(0.20,0.65,0.55,0.85);
  tl_METPhi020->AddEntry(METPhi020,"All triggers (data)","l");
  //tl_METPhi020->AddEntry(METPhi020_JetID,"JetID cuts Minimal","l");
  //tl_METPhi020->AddEntry(METPhi020_JetIDTight,"JetID cuts Tight","l");
  //tl_METPhi020->AddEntry(METPhi020_HcalNoiseFilter,"Hcal Noise Filter","l");
  //tl_METPhi020->AddEntry(METPhi020_HcalNoiseFilterTight,"Hcal Noise Filter Tight","l");
  //tl_METPhi020->AddEntry(METPhi020_BeamHaloIDLoosePass,"BeamHalo ID Loose Pass","l");
  //tl_METPhi020->AddEntry(METPhi020_BeamHaloIDTightPass,"BeamHalo ID Tight Pass","l");
  tl_METPhi020->AddEntry(METPhi020_Cleanup,"JetID + Hcal Noise + Halo Filter","l");
  if (METClass=="Calo") tl_METPhi020->AddEntry(METPhi020_CleanupPV,"+ Primary Vertex","l");
  tl_METPhi020->AddEntry(METPhi020_Ref,"MinBias MC","l");
  tl_METPhi020->SetFillColor(0);
  tl_METPhi020->Draw();

  sprintf(cjpgname,"%s/METPhi020.gif",METName.c_str());
  c_METPhi020->SaveAs(cjpgname);

  }

  if (METName=="CaloMET") {

  //-------------------------------
  // EmEtFraction
  //-------------------------------

  sprintf(ctitle,"%s/MET/%s/All/METTask_%sEmEtFraction",cprefix,METName.c_str(),METClass.c_str());
  std::cout << ctitle << std::endl;
  TH1F *EmEtFraction = (TH1F*) _file->Get(ctitle);    //
  EmEtFraction->SetTitleSize(0.065,"X");
  EmEtFraction->SetTitleSize(0.065,"Y");
  EmEtFraction->SetTitle("EmEtFraction");
  EmEtFraction->GetXaxis()->SetTitle("EmEtFraction");
  EmEtFraction->GetYaxis()->SetTitle("Events");

  sprintf(ctitle,"%s/MET/%s/JetIDMinimal/METTask_%sEmEtFraction",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction_JetID                = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/JetIDTight/METTask_%sEmEtFraction",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction_JetIDTight           = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilter/METTask_%sEmEtFraction",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction_HcalNoiseFilter      = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilterTight/METTask_%sEmEtFraction",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction_HcalNoiseFilterTight = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDLoosePass/METTask_%sEmEtFraction",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction_BeamHaloIDLoosePass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDTightPass/METTask_%sEmEtFraction",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction_BeamHaloIDTightPass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/Cleanup/METTask_%sEmEtFraction",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction_Cleanup              = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/CleanupPV/METTask_%sEmEtFraction",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction_CleanupPV              = (TH1F*) _file->Get(ctitle);    //

  sprintf(ctitleRef,"%s/MET/%s/Cleanup/METTask_%sEmEtFraction",cprefixRef,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction_Ref = _fileRef->Get(ctitleRef);
  EmEtFraction_Ref->Scale(SF);

  //-------------------------------

  sprintf(title,"%s_EmEtFraction_run=%d",METName.c_str(),run);
  sprintf(name,"%s_EmEtFraction_%d",METName.c_str(),run);
  TCanvas *c_EmEtFraction = new TCanvas(title,name,600.,400.);

  gPad->SetBottomMargin(0.15);
  gPad->SetLeftMargin(0.15);
  //gPad->SetLogy(1);
  gPad->SetGrid(1);

  EmEtFraction->SetLineColor(1);
  EmEtFraction_JetID->SetLineColor(2);
  EmEtFraction_JetID->SetLineWidth(3);
  EmEtFraction_JetIDTight->SetLineColor(9);
  EmEtFraction_HcalNoiseFilter->SetLineColor(4);
  EmEtFraction_HcalNoiseFilterTight->SetLineColor(6);
  EmEtFraction_Cleanup->SetLineColor(8);
  if (METClass=="Calo") EmEtFraction_CleanupPV->SetLineColor(4);
  EmEtFraction_BeamHaloIDLoosePass->SetLineColor(2);
  EmEtFraction_BeamHaloIDTightPass->SetLineColor(9);
  EmEtFraction_BeamHaloIDLoosePass->SetLineWidth(2);
  EmEtFraction_BeamHaloIDTightPass->SetLineWidth(2);

  EmEtFraction->SetStats(kFALSE);
  EmEtFraction->SetTitle(title);
  EmEtFraction->GetXaxis()->SetRange(221,280);
  //EmEtFraction->SetMaximum(1000.);
  EmEtFraction->SetMinimum(0.0);
  EmEtFraction->DrawCopy();
  //EmEtFraction_JetID->Draw("same");
  //EmEtFraction_JetIDTight->Draw("same");
  //EmEtFraction_HcalNoiseFilter->Draw("same");
  //EmEtFraction_HcalNoiseFilterTight->Draw("same");
  //EmEtFraction_BeamHaloIDTightPass->Draw("same");
  //EmEtFraction_BeamHaloIDLoosePass->Draw("same");
  EmEtFraction_Cleanup->DrawCopy("same");

  EmEtFraction_Ref->SetLineColor(2);
  EmEtFraction_Ref->SetFillColor(2);
  EmEtFraction_Ref->SetLineWidth(3);
  EmEtFraction_Ref->DrawCopy("same");

  EmEtFraction->DrawCopy("same,s");
  EmEtFraction->DrawCopy("same,e");
  EmEtFraction_Cleanup->DrawCopy("same,s");
  EmEtFraction_Cleanup->DrawCopy("same,e");
  if (METClass=="Calo") EmEtFraction_CleanupPV->DrawCopy("same,s");
  if (METClass=="Calo") EmEtFraction_CleanupPV->DrawCopy("same,e");

  //TLegend *tl_EmEtFraction = new TLegend(0.20,0.20,0.55,0.40);
  TLegend *tl_EmEtFraction = new TLegend(0.50,0.65,0.85,0.85);
  tl_EmEtFraction->AddEntry(EmEtFraction,"All triggers (data)","l");
  //tl_EmEtFraction->AddEntry(EmEtFraction_JetID,"JetID cuts Minimal","l");
  //tl_EmEtFraction->AddEntry(EmEtFraction_JetIDTight,"JetID cuts Tight","l");
  //tl_EmEtFraction->AddEntry(EmEtFraction_HcalNoiseFilter,"Hcal Noise Filter","l");
  //tl_EmEtFraction->AddEntry(EmEtFraction_HcalNoiseFilterTight,"Hcal Noise Filter Tight","l");
  //tl_EmEtFraction->AddEntry(EmEtFraction_BeamHaloIDLoosePass,"BeamHalo ID Loose Pass","l");
  //tl_EmEtFraction->AddEntry(EmEtFraction_BeamHaloIDTightPass,"BeamHalo ID Tight Pass","l");
  tl_EmEtFraction->AddEntry(EmEtFraction_Cleanup,"JetID + Hcal Noise + Halo Filter","l");
  if (METClass=="Calo") tl_EmEtFraction->AddEntry(EmEtFraction_CleanupPV,"+ Primary Vertex","l");
  tl_EmEtFraction->AddEntry(EmEtFraction_Ref,"MinBias MC","l");
  tl_EmEtFraction->SetFillColor(0);
  tl_EmEtFraction->Draw();

  sprintf(cjpgname,"%s/EmEtFraction.gif",METName.c_str());
  c_EmEtFraction->SaveAs(cjpgname);

  //-------------------------------
  // EmEtFraction002
  //-------------------------------

  sprintf(ctitle,"%s/MET/%s/All/METTask_%sEmEtFraction002",cprefix,METName.c_str(),METClass.c_str());
  std::cout << ctitle << std::endl;
  TH1F *EmEtFraction002 = (TH1F*) _file->Get(ctitle);    //
  EmEtFraction002->SetTitleSize(0.065,"X");
  EmEtFraction002->SetTitleSize(0.065,"Y");
  EmEtFraction002->SetTitle("EmEtFraction");
  EmEtFraction002->GetXaxis()->SetTitle("EmEtFraction");
  EmEtFraction002->GetYaxis()->SetTitle("Events");

  sprintf(ctitle,"%s/MET/%s/JetIDMinimal/METTask_%sEmEtFraction002",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction002_JetID                = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/JetIDTight/METTask_%sEmEtFraction002",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction002_JetIDTight           = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilter/METTask_%sEmEtFraction002",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction002_HcalNoiseFilter      = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilterTight/METTask_%sEmEtFraction002",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction002_HcalNoiseFilterTight = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDLoosePass/METTask_%sEmEtFraction002",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction002_BeamHaloIDLoosePass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDTightPass/METTask_%sEmEtFraction002",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction002_BeamHaloIDTightPass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/Cleanup/METTask_%sEmEtFraction002",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction002_Cleanup              = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/CleanupPV/METTask_%sEmEtFraction002",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction002_CleanupPV              = (TH1F*) _file->Get(ctitle);    //

  sprintf(ctitleRef,"%s/MET/%s/Cleanup/METTask_%sEmEtFraction002",cprefixRef,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction002_Ref = _fileRef->Get(ctitleRef);
  EmEtFraction002_Ref->Scale(SF);

  //-------------------------------

  sprintf(title,"%s_EmEtFraction_run=%d (MET>2GeV)",METName.c_str(),run);
  sprintf(name,"%s_EmEtFraction002_%d",METName.c_str(),run);
  TCanvas *c_EmEtFraction002 = new TCanvas(title,name,600.,400.);

  gPad->SetBottomMargin(0.15);
  gPad->SetLeftMargin(0.15);
  //gPad->SetLogy(1);
  gPad->SetGrid(1);

  EmEtFraction002->SetLineColor(1);
  EmEtFraction002_JetID->SetLineColor(2);
  EmEtFraction002_JetID->SetLineWidth(3);
  EmEtFraction002_JetIDTight->SetLineColor(9);
  EmEtFraction002_HcalNoiseFilter->SetLineColor(4);
  EmEtFraction002_HcalNoiseFilterTight->SetLineColor(6);
  EmEtFraction002_Cleanup->SetLineColor(8);
  if (METClass=="Calo") EmEtFraction002_CleanupPV->SetLineColor(4);
  EmEtFraction002_BeamHaloIDLoosePass->SetLineColor(2);
  EmEtFraction002_BeamHaloIDTightPass->SetLineColor(9);
  EmEtFraction002_BeamHaloIDLoosePass->SetLineWidth(2);
  EmEtFraction002_BeamHaloIDTightPass->SetLineWidth(2);

  EmEtFraction002->SetStats(kFALSE);
  EmEtFraction002->SetTitle(title);
  EmEtFraction002->GetXaxis()->SetRange(221,280);
  //EmEtFraction002->SetMaximum(1000.);
  EmEtFraction002->SetMinimum(0.0);
  EmEtFraction002->DrawCopy();
  //EmEtFraction002_JetID->Draw("same");
  //EmEtFraction002_JetIDTight->Draw("same");
  //EmEtFraction002_HcalNoiseFilter->Draw("same");
  //EmEtFraction002_HcalNoiseFilterTight->Draw("same");
  //EmEtFraction002_BeamHaloIDTightPass->Draw("same");
  //EmEtFraction002_BeamHaloIDLoosePass->Draw("same");
  EmEtFraction002_Cleanup->DrawCopy("same");

  EmEtFraction002_Ref->SetLineColor(2);
  EmEtFraction002_Ref->SetFillColor(2);
  EmEtFraction002_Ref->SetLineWidth(3);
  EmEtFraction002_Ref->DrawCopy("same");

  EmEtFraction002->DrawCopy("same,s");
  EmEtFraction002->DrawCopy("same,e");
  EmEtFraction002_Cleanup->DrawCopy("same,s");
  EmEtFraction002_Cleanup->DrawCopy("same,e");
  if (METClass=="Calo") EmEtFraction002_CleanupPV->DrawCopy("same,s");
  if (METClass=="Calo") EmEtFraction002_CleanupPV->DrawCopy("same,e");

  TLegend *tl_EmEtFraction002 = new TLegend(0.50,0.65,0.85,0.85);
  tl_EmEtFraction002->AddEntry(EmEtFraction002,"All triggers (data)","l");
  //tl_EmEtFraction002->AddEntry(EmEtFraction002_JetID,"JetID cuts Minimal","l");
  //tl_EmEtFraction002->AddEntry(EmEtFraction002_JetIDTight,"JetID cuts Tight","l");
  //tl_EmEtFraction002->AddEntry(EmEtFraction002_HcalNoiseFilter,"Hcal Noise Filter","l");
  //tl_EmEtFraction002->AddEntry(EmEtFraction002_HcalNoiseFilterTight,"Hcal Noise Filter Tight","l");
  //tl_EmEtFraction002->AddEntry(EmEtFraction002_BeamHaloIDLoosePass,"BeamHalo ID Loose Pass","l");
  //tl_EmEtFraction002->AddEntry(EmEtFraction002_BeamHaloIDTightPass,"BeamHalo ID Tight Pass","l");
  tl_EmEtFraction002->AddEntry(EmEtFraction002_Cleanup,"JetID + Hcal Noise + Halo Filter","l");
  if (METClass=="Calo") tl_EmEtFraction002->AddEntry(EmEtFraction002_CleanupPV,"+ Primary Vertex","l");
  tl_EmEtFraction002->AddEntry(EmEtFraction002_Ref,"MinBias MC","l");
  tl_EmEtFraction002->SetFillColor(0);
  tl_EmEtFraction002->Draw();

  sprintf(cjpgname,"%s/EmEtFraction002.gif",METName.c_str());
  c_EmEtFraction002->SaveAs(cjpgname);

  //-------------------------------
  // EmEtFraction010
  //-------------------------------

  sprintf(ctitle,"%s/MET/%s/All/METTask_%sEmEtFraction010",cprefix,METName.c_str(),METClass.c_str());
  std::cout << ctitle << std::endl;
  TH1F *EmEtFraction010 = (TH1F*) _file->Get(ctitle);    //
  EmEtFraction010->SetTitleSize(0.065,"X");
  EmEtFraction010->SetTitleSize(0.065,"Y");
  EmEtFraction010->SetTitle("EmEtFraction");
  EmEtFraction010->GetXaxis()->SetTitle("EmEtFraction");
  EmEtFraction010->GetYaxis()->SetTitle("Events");

  sprintf(ctitle,"%s/MET/%s/JetIDMinimal/METTask_%sEmEtFraction010",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction010_JetID                = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/JetIDTight/METTask_%sEmEtFraction010",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction010_JetIDTight           = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilter/METTask_%sEmEtFraction010",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction010_HcalNoiseFilter      = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilterTight/METTask_%sEmEtFraction010",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction010_HcalNoiseFilterTight = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDLoosePass/METTask_%sEmEtFraction010",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction010_BeamHaloIDLoosePass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDTightPass/METTask_%sEmEtFraction010",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction010_BeamHaloIDTightPass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/Cleanup/METTask_%sEmEtFraction010",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction010_Cleanup              = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/CleanupPV/METTask_%sEmEtFraction010",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction010_CleanupPV              = (TH1F*) _file->Get(ctitle);    //

  sprintf(ctitleRef,"%s/MET/%s/Cleanup/METTask_%sEmEtFraction010",cprefixRef,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction010_Ref = _fileRef->Get(ctitleRef);
  EmEtFraction010_Ref->Scale(SF);

  //-------------------------------

  sprintf(title,"%s_EmEtFraction_run=%d (MET>10GeV)",METName.c_str(),run);
  sprintf(name,"%s_EmEtFraction010_%d",METName.c_str(),run);
  TCanvas *c_EmEtFraction010 = new TCanvas(title,name,600.,400.);

  gPad->SetBottomMargin(0.15);
  gPad->SetLeftMargin(0.15);
  //gPad->SetLogy(1);
  gPad->SetGrid(1);

  EmEtFraction010->SetLineColor(1);
  EmEtFraction010_JetID->SetLineColor(2);
  EmEtFraction010_JetID->SetLineWidth(3);
  EmEtFraction010_JetIDTight->SetLineColor(9);
  EmEtFraction010_HcalNoiseFilter->SetLineColor(4);
  EmEtFraction010_HcalNoiseFilterTight->SetLineColor(6);
  EmEtFraction010_Cleanup->SetLineColor(8);
  if (METClass=="Calo") EmEtFraction010_CleanupPV->SetLineColor(4);
  EmEtFraction010_BeamHaloIDLoosePass->SetLineColor(2);
  EmEtFraction010_BeamHaloIDTightPass->SetLineColor(9);
  EmEtFraction010_BeamHaloIDLoosePass->SetLineWidth(2);
  EmEtFraction010_BeamHaloIDTightPass->SetLineWidth(2);

  EmEtFraction010->SetStats(kFALSE);
  EmEtFraction010->SetTitle(title);
  EmEtFraction010->GetXaxis()->SetRange(221,280);
  //EmEtFraction010->SetMaximum(1000.);
  EmEtFraction010->SetMinimum(0.0);
  EmEtFraction010->DrawCopy();
  //EmEtFraction010_JetID->Draw("same");
  //EmEtFraction010_JetIDTight->Draw("same");
  //EmEtFraction010_HcalNoiseFilter->Draw("same");
  //EmEtFraction010_HcalNoiseFilterTight->Draw("same");
  //EmEtFraction010_BeamHaloIDTightPass->Draw("same");
  //EmEtFraction010_BeamHaloIDLoosePass->Draw("same");
  EmEtFraction010_Cleanup->DrawCopy("same");

  EmEtFraction010_Ref->SetLineColor(2);
  EmEtFraction010_Ref->SetFillColor(2);
  EmEtFraction010_Ref->SetLineWidth(3);
  EmEtFraction010_Ref->DrawCopy("same");

  EmEtFraction010->DrawCopy("same,s");
  EmEtFraction010->DrawCopy("same,e");
  EmEtFraction010_Cleanup->DrawCopy("same,s");
  EmEtFraction010_Cleanup->DrawCopy("same,e");
  if (METClass=="Calo") EmEtFraction010_CleanupPV->DrawCopy("same,s");
  if (METClass=="Calo") EmEtFraction010_CleanupPV->DrawCopy("same,e");

  TLegend *tl_EmEtFraction010 = new TLegend(0.50,0.65,0.85,0.85);
  tl_EmEtFraction010->AddEntry(EmEtFraction010,"All triggers (data)","l");
  //tl_EmEtFraction010->AddEntry(EmEtFraction010_JetID,"JetID cuts Minimal","l");
  //tl_EmEtFraction010->AddEntry(EmEtFraction010_JetIDTight,"JetID cuts Tight","l");
  //tl_EmEtFraction010->AddEntry(EmEtFraction010_HcalNoiseFilter,"Hcal Noise Filter","l");
  //tl_EmEtFraction010->AddEntry(EmEtFraction010_HcalNoiseFilterTight,"Hcal Noise Filter Tight","l");
  //tl_EmEtFraction010->AddEntry(EmEtFraction010_BeamHaloIDLoosePass,"BeamHalo ID Loose Pass","l");
  //tl_EmEtFraction010->AddEntry(EmEtFraction010_BeamHaloIDTightPass,"BeamHalo ID Tight Pass","l");
  tl_EmEtFraction010->AddEntry(EmEtFraction010_Cleanup,"JetID + Hcal Noise + Halo Filter","l");
  if (METClass=="Calo") tl_EmEtFraction010->AddEntry(EmEtFraction010_CleanupPV,"+ Primary Vertex","l");
  tl_EmEtFraction010->AddEntry(EmEtFraction010_Ref,"MinBias MC","l");
  tl_EmEtFraction010->SetFillColor(0);
  tl_EmEtFraction010->Draw();

  sprintf(cjpgname,"%s/EmEtFraction010.gif",METName.c_str());
  c_EmEtFraction010->SaveAs(cjpgname);

  //-------------------------------
  // EmEtFraction020
  //-------------------------------

  sprintf(ctitle,"%s/MET/%s/All/METTask_%sEmEtFraction020",cprefix,METName.c_str(),METClass.c_str());
  std::cout << ctitle << std::endl;
  TH1F *EmEtFraction020 = (TH1F*) _file->Get(ctitle);    //
  EmEtFraction020->SetTitleSize(0.065,"X");
  EmEtFraction020->SetTitleSize(0.065,"Y");
  EmEtFraction020->SetTitle("EmEtFraction");
  EmEtFraction020->GetXaxis()->SetTitle("EmEtFraction");
  EmEtFraction020->GetYaxis()->SetTitle("Events");

  sprintf(ctitle,"%s/MET/%s/JetIDMinimal/METTask_%sEmEtFraction020",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction020_JetID                = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/JetIDTight/METTask_%sEmEtFraction020",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction020_JetIDTight           = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilter/METTask_%sEmEtFraction020",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction020_HcalNoiseFilter      = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/HcalNoiseFilterTight/METTask_%sEmEtFraction020",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction020_HcalNoiseFilterTight = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDLoosePass/METTask_%sEmEtFraction020",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction020_BeamHaloIDLoosePass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/BeamHaloIDTightPass/METTask_%sEmEtFraction020",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction020_BeamHaloIDTightPass  = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/Cleanup/METTask_%sEmEtFraction020",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction020_Cleanup              = (TH1F*) _file->Get(ctitle);    //
  sprintf(ctitle,"%s/MET/%s/CleanupPV/METTask_%sEmEtFraction020",cprefix,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction020_CleanupPV              = (TH1F*) _file->Get(ctitle);    //

  sprintf(ctitleRef,"%s/MET/%s/Cleanup/METTask_%sEmEtFraction020",cprefixRef,METName.c_str(),METClass.c_str());
  TH1F *EmEtFraction020_Ref = _fileRef->Get(ctitleRef);
  EmEtFraction020_Ref->Scale(SF);

  //-------------------------------

  sprintf(title,"%s_EmEtFraction_run=%d (MET>20GeV)",METName.c_str(),run);
  sprintf(name,"%s_EmEtFraction020_%d",METName.c_str(),run);
  TCanvas *c_EmEtFraction020 = new TCanvas(title,name,600.,400.);

  gPad->SetBottomMargin(0.15);
  gPad->SetLeftMargin(0.15);
  //gPad->SetLogy(1);
  gPad->SetGrid(1);

  EmEtFraction020->SetLineColor(1);
  EmEtFraction020_JetID->SetLineColor(2);
  EmEtFraction020_JetID->SetLineWidth(3);
  EmEtFraction020_JetIDTight->SetLineColor(9);
  EmEtFraction020_HcalNoiseFilter->SetLineColor(4);
  EmEtFraction020_HcalNoiseFilterTight->SetLineColor(6);
  EmEtFraction020_Cleanup->SetLineColor(8);
  if (METClass=="Calo") EmEtFraction020_CleanupPV->SetLineColor(4);
  EmEtFraction020_BeamHaloIDLoosePass->SetLineColor(2);
  EmEtFraction020_BeamHaloIDTightPass->SetLineColor(9);
  EmEtFraction020_BeamHaloIDLoosePass->SetLineWidth(2);
  EmEtFraction020_BeamHaloIDTightPass->SetLineWidth(2);

  EmEtFraction020->SetStats(kFALSE);
  EmEtFraction020->SetTitle(title);
  EmEtFraction020->GetXaxis()->SetRange(221,280);
  //EmEtFraction020->SetMaximum(1000.);
  EmEtFraction020->SetMinimum(0.0);
  EmEtFraction020->DrawCopy();
  //EmEtFraction020_JetID->Draw("same");
  //EmEtFraction020_JetIDTight->Draw("same");
  //EmEtFraction020_HcalNoiseFilter->Draw("same");
  //EmEtFraction020_HcalNoiseFilterTight->Draw("same");
  //EmEtFraction020_BeamHaloIDTightPass->Draw("same");
  //EmEtFraction020_BeamHaloIDLoosePass->Draw("same");
  EmEtFraction020_Cleanup->DrawCopy("same");

  EmEtFraction020_Ref->SetLineColor(2);
  EmEtFraction020_Ref->SetFillColor(2);
  EmEtFraction020_Ref->SetLineWidth(3);
  EmEtFraction020_Ref->DrawCopy("same");

  EmEtFraction020->DrawCopy("same,s");
  EmEtFraction020->DrawCopy("same,e");
  EmEtFraction020_Cleanup->DrawCopy("same,s");
  EmEtFraction020_Cleanup->DrawCopy("same,e");
  if (METClass=="Calo") EmEtFraction020_CleanupPV->DrawCopy("same,s");
  if (METClass=="Calo") EmEtFraction020_CleanupPV->DrawCopy("same,e");

  TLegend *tl_EmEtFraction020 = new TLegend(0.50,0.65,0.85,0.85);
  tl_EmEtFraction020->AddEntry(EmEtFraction020,"All triggers (data)","l");
  //tl_EmEtFraction020->AddEntry(EmEtFraction020_JetID,"JetID cuts Minimal","l");
  //tl_EmEtFraction020->AddEntry(EmEtFraction020_JetIDTight,"JetID cuts Tight","l");
  //tl_EmEtFraction020->AddEntry(EmEtFraction020_HcalNoiseFilter,"Hcal Noise Filter","l");
  //tl_EmEtFraction020->AddEntry(EmEtFraction020_HcalNoiseFilterTight,"Hcal Noise Filter Tight","l");
  //tl_EmEtFraction020->AddEntry(EmEtFraction020_BeamHaloIDLoosePass,"BeamHalo ID Loose Pass","l");
  //tl_EmEtFraction020->AddEntry(EmEtFraction020_BeamHaloIDTightPass,"BeamHalo ID Tight Pass","l");
  tl_EmEtFraction020->AddEntry(EmEtFraction020_Cleanup,"JetID + Hcal Noise + Halo Filter","l");
  if (METClass=="Calo") tl_EmEtFraction020->AddEntry(EmEtFraction020_CleanupPV,"+ Primary Vertex","l");
  tl_EmEtFraction020->AddEntry(EmEtFraction020_Ref,"MinBias MC","l");
  tl_EmEtFraction020->SetFillColor(0);
  tl_EmEtFraction020->Draw();

  sprintf(cjpgname,"%s/EmEtFraction020.gif",METName.c_str());
  c_EmEtFraction020->SaveAs(cjpgname);

  }

  //-------------------------------

  return;

  //-------------------------------

}
