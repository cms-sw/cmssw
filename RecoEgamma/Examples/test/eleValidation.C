{
  // style

  gROOT->Reset();

  // style:
  TStyle *tdrStyle = new TStyle("tdrStyle", "Style for P-TDR");

  //For the canvas:
  tdrStyle->SetCanvasBorderMode(0);
  tdrStyle->SetCanvasColor(kWhite);
  tdrStyle->SetCanvasDefH(600);
  //Height of canvas
  tdrStyle->SetCanvasDefW(800);  //Width of canvas
  tdrStyle->SetCanvasDefX(0);    //POsition on screen
  tdrStyle->SetCanvasDefY(0);

  // For the Pad:
  tdrStyle->SetPadBorderMode(0);
  // tdrStyle->SetPadBorderSize(Width_t size = 1);
  tdrStyle->SetPadColor(kWhite);
  tdrStyle->SetPadGridX(false);
  tdrStyle->SetPadGridY(false);
  tdrStyle->SetGridColor(0);
  tdrStyle->SetGridStyle(3);
  tdrStyle->SetGridWidth(1);

  //For the frame:
  tdrStyle->SetFrameBorderMode(0);
  tdrStyle->SetFrameBorderSize(1);
  tdrStyle->SetFrameFillColor(0);
  tdrStyle->SetFrameFillStyle(0);
  tdrStyle->SetFrameLineColor(1);
  tdrStyle->SetFrameLineStyle(1);
  tdrStyle->SetFrameLineWidth(1);

  // For the histo:
  tdrStyle->SetHistLineColor(1);
  tdrStyle->SetHistLineStyle(0);
  tdrStyle->SetHistLineWidth(2);
  tdrStyle->SetEndErrorSize(2);
  //tdrStyle->SetErrorMarker(20);
  tdrStyle->SetErrorX(0.);
  tdrStyle->SetMarkerStyle(8);

  // For the statistics box:
  tdrStyle->SetOptFile(0);
  //tdrStyle->SetOptStat(1);
  tdrStyle->SetOptStat(0);
  tdrStyle->SetStatColor(kWhite);
  //tdrStyle->SetStatFont(42);
  //tdrStyle->SetStatFontSize(0.025);
  //tdrStyle->SetStatTextColor(1);
  //tdrStyle->SetStatFormat("6.4g");
  //tdrStyle->SetStatBorderSize(1);
  //tdrStyle->SetStatH(.1);
  //tdrStyle->SetStatW(.15);

  //  tdrStyle->SetStatX(.9);
  // tdrStyle->SetStatY(.9);

  // For the Global title:
  tdrStyle->SetOptTitle(0);
  tdrStyle->SetTitleFont(42);
  tdrStyle->SetTitleColor(1);
  tdrStyle->SetTitleTextColor(1);
  tdrStyle->SetTitleFillColor(10);
  tdrStyle->SetTitleFontSize(0.05);

  // For the axis titles:
  tdrStyle->SetTitleColor(1, "XYZ");
  tdrStyle->SetTitleFont(42, "XYZ");
  //  tdrStyle->SetTitleSize(0.06, "XYZ");
  tdrStyle->SetTitleSize(0.05, "XYZ");
  tdrStyle->SetTitleXOffset(0.9);
  // tdrStyle->SetTitleYOffset(1.25);
  //tdrStyle->SetTitleXOffset(0.5);
  tdrStyle->SetTitleYOffset(1.0);

  // For the axis labels:
  tdrStyle->SetLabelColor(1, "XYZ");
  tdrStyle->SetLabelFont(42, "XYZ");
  tdrStyle->SetLabelOffset(0.007, "XYZ");

  //tdrStyle->SetLabelSize(0.05, "XYZ");
  tdrStyle->SetLabelSize(0.04, "XYZ");

  // For the axis:
  tdrStyle->SetAxisColor(1, "XYZ");
  tdrStyle->SetStripDecimals(kTRUE);
  tdrStyle->SetTickLength(0.03, "XYZ");
  tdrStyle->SetNdivisions(510, "XYZ");
  tdrStyle->SetPadTickX(1);  // To get tick marks on the opposite side of the frame
  tdrStyle->SetPadTickY(1);
  // // Postscript options:
  //   //tdrStyle->SetPaperSize(20.,20.);

  // CC style
  tdrStyle->SetTitleXOffset(0.8);
  tdrStyle->SetTitleYOffset(0.8);
  tdrStyle->SetLabelOffset(0.005, "XYZ");
  tdrStyle->SetTitleSize(0.07, "XYZ");
  tdrStyle->SetTitleFont(22, "X");
  tdrStyle->SetTitleFont(22, "Y");
  //  tdrStyle->SetPadBottomMargin(0.2);
  //  tdrStyle->SetPadLeftMargin(0.2);
  tdrStyle->SetPadBottomMargin(0.13);
  tdrStyle->SetPadLeftMargin(0.15);
  //  tdrStyle->SetHistLineWidth(3);
  tdrStyle->SetHistLineWidth(2);

  tdrStyle->cd();

  gROOT->ForceStyle();

  // output figures: file type suffix
  char suffix[] = "gif";
  // output figures: directory
  char outDir[] = "./analysisPreselLowLumiPUBugfix";
  // temp variables
  char str[128];

  bool out = true;
  //  bool out = false;

  //  bool pause = true;
  bool pause = false;

  TFile hist("gsfElectronHistos_DiEle-Pt5To100_rereco131_LowLumiPU_bugfix.root");

  // electron quantities
  TH1F *h_ele_PoPtrue = (TH1F *)hist.Get("h_ele_PoPtrue");
  TH1F *h_ele_EtaMnEtaTrue = (TH1F *)hist.Get("h_ele_EtaMnEtaTrue");
  TH1F *h_ele_PhiMnPhiTrue = (TH1F *)hist.Get("h_ele_PhiMnPhiTrue");
  TH1F *h_ele_vertexP = (TH1F *)hist.Get("h_ele_vertexP");
  TH1F *h_ele_vertexPt = (TH1F *)hist.Get("h_ele_vertexPt");
  TH1F *h_ele_outerP_mode = (TH1F *)hist.Get("h_ele_outerP_mode");
  TH1F *h_ele_outerPt_mode = (TH1F *)hist.Get("h_ele_outerPt_mode");
  TH1F *h_ele_vertexZ = (TH1F *)hist.Get("h_ele_vertexZ");

  TCanvas *c_PoPtrue = new TCanvas("PoPtrue", "PoPtrue");
  c_PoPtrue->cd();
  h_ele_PoPtrue->Draw();
  h_ele_PoPtrue->GetXaxis()->SetTitle("p_{rec}/p_{true}");
  h_ele_PoPtrue->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/PoPtrue.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_EtaMnEtaTrue = new TCanvas("EtaMnEtaTrue", "EtaMnEtaTrue");
  c_EtaMnEtaTrue->cd();
  h_ele_EtaMnEtaTrue->Draw();
  h_ele_EtaMnEtaTrue->GetXaxis()->SetTitle("#eta_{rec}-#eta_{true}");
  h_ele_EtaMnEtaTrue->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/EtaMnEtaTrue.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_PhiMnPhiTrue = new TCanvas("PhiMnPhiTrue", "PhiMnPhiTrue");
  c_PhiMnPhiTrue->cd();
  h_ele_PhiMnPhiTrue->Draw();
  h_ele_PhiMnPhiTrue->GetXaxis()->SetTitle("#phi_{rec}-#phi_{true}");
  h_ele_PhiMnPhiTrue->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/PhiMnPhiTrue.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_vertexP = new TCanvas("vertexP", "vertexP");
  c_vertexP->cd();
  h_ele_vertexP->Draw();
  h_ele_vertexP->GetXaxis()->SetTitle("p_{vertex}");
  h_ele_vertexP->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/vertexP.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_vertexPt = new TCanvas("vertexPt", "vertexPt");
  c_vertexPt->cd();
  h_ele_vertexPt->Draw();
  h_ele_vertexPt->GetXaxis()->SetTitle("p_{T} from vertex");
  h_ele_vertexPt->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/vertexPt.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_outerP_mode = new TCanvas("outerP_mode", "outerP_mode");
  c_outerP_mode->cd();
  h_ele_outerP_mode->Draw();
  h_ele_outerP_mode->GetXaxis()->SetTitle("p from out, mode");
  h_ele_outerP_mode->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/outerP_mode.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_outerPt_mode = new TCanvas("outerPt_mode", "outerPt_mode");
  c_outerPt_mode->cd();
  h_ele_outerPt_mode->Draw();
  h_ele_outerPt_mode->GetXaxis()->SetTitle("p_{T} from out, mode");
  h_ele_outerPt_mode->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/outerPt_mode.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_vertexZ = new TCanvas("vertexZ", "vertexZ");
  c_vertexZ->cd();
  h_ele_vertexZ->Draw();
  h_ele_vertexZ->GetXaxis()->SetTitle("z_{rec}");
  h_ele_vertexZ->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/vertexZ.%s", outDir, suffix);
    gPad->Print(str);
  }
  if (pause)
    c_vertexZ->WaitPrimitive();

  // efficiency
  TH1F *h_ele_absetaEff = (TH1F *)hist.Get("h_ele_absetaEff");
  TH1F *h_simAbsEta = (TH1F *)hist.Get("h_mc_abseta");
  TH1F *h_ele_etaEff = (TH1F *)hist.Get("h_ele_etaEff");
  TH1F *h_simEta = (TH1F *)hist.Get("h_mc_eta");
  TH1F *h_ele_ptEff = (TH1F *)hist.Get("h_ele_ptEff");
  TH1F *h_simPt = (TH1F *)hist.Get("h_smc_Pt");

  TCanvas *c_absetaEff = new TCanvas("absetaEff", "absetaEff");
  c_absetaEff->cd();
  for (int ibin = 1; ibin < h_ele_absetaEff->GetNbinsX(); ibin++) {
    double binContent = h_ele_absetaEff->GetBinContent(ibin);
    double error2 = binContent * (1. - binContent) / h_simAbsEta->GetBinContent(ibin);
    //h_ele_absetaEff->SetBinError(ibin,sqrt(error2));
    std::cout << "ibin " << ibin << " efficiency " << binContent << " error " << sqrt(error2) << std::endl;
  }
  h_ele_absetaEff->SetMarkerStyle(21);
  h_ele_absetaEff->SetMinimum(0.0);
  h_ele_absetaEff->SetMaximum(1.0);
  h_ele_absetaEff->Draw("");
  //h_ele_absetaEff->Draw("SEP");
  h_ele_absetaEff->GetXaxis()->SetTitle("|#eta|");
  h_ele_absetaEff->GetYaxis()->SetTitle("Efficiency");
  if (out) {
    snprintf(str, 128, "%s/absetaEff.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_etaEff = new TCanvas("etaEff", "etaEff");
  c_etaEff->cd();
  h_ele_etaEff->SetMarkerStyle(21);
  h_ele_etaEff->SetMinimum(0.0);
  h_ele_etaEff->SetMaximum(1.0);
  h_ele_etaEff->Draw("");
  h_ele_etaEff->GetXaxis()->SetTitle("#eta");
  h_ele_etaEff->GetYaxis()->SetTitle("Efficiency");
  if (out) {
    snprintf(str, 128, "%s/etaEff.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_ptEff = new TCanvas("ptEff", "ptEff");
  c_ptEff->cd();
  h_ele_ptEff->SetMarkerStyle(21);
  h_ele_ptEff->SetMinimum(0.0);
  h_ele_ptEff->SetMaximum(1.0);
  h_ele_ptEff->Draw("");
  h_ele_ptEff->GetXaxis()->SetTitle("p_{T} (GeV/c)");
  h_ele_ptEff->GetYaxis()->SetTitle("Efficiency");
  if (out) {
    snprintf(str, 128, "%s/ptEff.%s", outDir, suffix);
    gPad->Print(str);
  }
  if (pause)
    c_ptEff->WaitPrimitive();

  // match
  TH1F *h_ele_EoP = (TH1F *)hist.Get("h_ele_EoP");
  TH1F *h_ele_EoPout = (TH1F *)hist.Get("h_ele_EoPout");
  TH1F *h_ele_EseedOP = (TH1F *)hist.Get("h_ele_EseedOP");
  TH1F *h_ele_EeleOPout = (TH1F *)hist.Get("h_ele_EeleOPout");
  TH1F *h_ele_dEtaCl_propOut = (TH1F *)hist.Get("h_ele_dEtaCl_propOut");
  TH1F *h_ele_dEtaSc_propVtx = (TH1F *)hist.Get("h_ele_dEtaSc_propVtx");
  TH1F *h_ele_dPhiCl_propOut = (TH1F *)hist.Get("h_ele_dPhiCl_propOut");
  TH1F *h_ele_dPhiSc_propVtx = (TH1F *)hist.Get("h_ele_dPhiSc_propVtx");
  TH1F *h_ele_dEtaEleCl_propOut = (TH1F *)hist.Get("h_ele_dEtaEleCl_propOut");
  TH1F *h_ele_dPhiEleCl_propOut = (TH1F *)hist.Get("h_ele_dPhiEleCl_propOut");
  TH1F *h_ele_HoE = (TH1F *)hist.Get("h_ele_HoE");

  TCanvas *c_EoP = new TCanvas("EoP", "EoP");
  c_EoP->cd();
  h_ele_EoP->Draw();
  h_ele_EoP->GetXaxis()->SetTitle("E/p");
  h_ele_EoP->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/EoP.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_EoPout = new TCanvas("EoPout", "EoPout");
  c_EoPout->cd();
  h_ele_EoPout->Draw();
  h_ele_EoP->GetXaxis()->SetTitle("E_{seed}/p_{pout}");
  h_ele_EoP->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/EoPout.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_EeleOPout = new TCanvas("EeleOPout", "EeleOPout");
  c_EeleOPout->cd();
  h_ele_EeleOPout->Draw();
  h_ele_EeleOPout->GetXaxis()->SetTitle("E_{ele}/p_{pout}");
  h_ele_EeleoPout->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/EeleOPout.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_EseedOP = new TCanvas("EseedOP", "EseedOP");
  c_EseedOP->cd();
  h_ele_EseedOP->Draw();
  h_ele_EseedoP->GetXaxis()->SetTitle("E_{seed}/p");
  h_ele_EseedOP->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/EseedOP.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_dEtaCl_propOut = new TCanvas("dEtaCl_propOut", "dEtaCl_propOut");
  c_dEtaCl_propOut->cd();
  h_ele_dEtaCl_propOut->Draw();
  h_ele_dEtaCl_propOut->GetXaxis()->SetTitle("#eta_{seed}-#eta_{tk, extrp. from out}");
  h_ele_dEtaCl_propOut->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/dEtaCl_propOut.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_dEtaEleCl_propOut = new TCanvas("dEtaEleCl_propOut", "dEtaEleCl_propOut");
  c_dEtaEleCl_propOut->cd();
  h_ele_dEtaEleCl_propOut->Draw();
  h_ele_dEtaEleCl_propOut->GetXaxis()->SetTitle("#eta_{ele}-#eta_{tk, extrp. from out}");
  h_ele_dEtaEleCl_propOut->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/dEtaEleCl_propOut.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_dEtaSc_propVtx = new TCanvas("dEtaSc_propVtx", "dEtaSc_propVtx");
  c_dEtaSc_propVtx->cd();
  h_ele_dEtaSc_propVtx->Draw();
  h_ele_dEtaSc_propVtx->GetXaxis()->SetTitle("#eta_{sc}-#eta_{tk, extrp. from vtx}");
  h_ele_dEtaSc_propVtx->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/dEtaSc_propVtx.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_dPhiCl_propOut = new TCanvas("dPhiCl_propOut", "dPhiCl_propOut");
  c_dPhiCl_propOut->cd();
  h_ele_dPhiCl_propOut->Draw();
  h_ele_dPhiCl_propOut->GetXaxis()->SetTitle("#phi_{seed}-#phi_{tk, extrp. from out}");
  h_ele_dPhiCl_propOut->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/dPhiCl_propOut.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_dPhiEleCl_propOut = new TCanvas("dPhiEleCl_propOut", "dPhiEleCl_propOut");
  c_dPhiEleCl_propOut->cd();
  h_ele_dPhiEleCl_propOut->Draw();
  h_ele_dPhiEleCl_propOut->GetXaxis()->SetTitle("#phi_{ele}-#phi_{tk, extrp. from out}");
  h_ele_dPhiEleCl_propOut->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/dPhiEleCl_propOut.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_dPhiSc_propVtx = new TCanvas("dPhiSc_propVtx", "dPhiSc_propVtx");
  c_dPhiSc_propVtx->cd();
  h_ele_dPhiSc_propVtx->Draw();
  h_ele_dPhiSc_propVtx->GetXaxis()->SetTitle("#phi_{sc}-#phi_{tk, extrp. from vtx}");
  h_ele_dPhiSc_propVtx->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/dPhiSc_propVtx.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_HoE = new TCanvas("HoE", "HoE");
  c_HoE->cd();
  h_ele_HoE->Draw();
  h_ele_HoE->GetXaxis()->SetTitle("H/E");
  h_ele_HoE->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/HoE.%s", outDir, suffix);
    gPad->Print(str);
  }
  if (pause)
    c_HoE->WaitPrimitive();

  // track
  TH1F *h_ele_chi2 = (TH1F *)hist.Get("h_ele_chi2");
  TH1F *h_ele_foundHits = (TH1F *)hist.Get("h_ele_foundHits");
  TH1F *h_ele_lostHits = (TH1F *)hist.Get("h_ele_lostHits");
  TH1F *h_ele_ambiguousTracks = (TH1F *)hist.Get("h_ele_ambiguousTracks");

  TCanvas *c_chi2 = new TCanvas("chi2", "chi2");
  c_chi2->cd();
  h_ele_chi2->Draw();
  h_ele_chi2->GetXaxis()->SetTitle("track #Chi^{2}");
  h_ele_chi2->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/chi2.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_foundHits = new TCanvas("foundHits", "foundHits");
  c_foundHits->cd();
  h_ele_foundHits->Draw();
  h_ele_foundHits->GetXaxis()->SetTitle("# hits");
  h_ele_foundHits->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/foundHits.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_lostHits = new TCanvas("lostHits", "lostHits");
  c_lostHits->cd();
  h_ele_lostHits->Draw();
  h_ele_lostHits->GetXaxis()->SetTitle("# lost hits");
  h_ele_lostHits->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/lostHits.%s", outDir, suffix);
    gPad->Print(str);
  }
  TCanvas *c_ambiguousTracks = new TCanvas("ambiguousTracks", "ambiguousTracks");
  c_ambiguousTracks->cd();
  h_ele_ambiguousTracks->Draw();
  h_ele_ambiguousTracks->GetXaxis()->SetTitle("# ambiguous tracks");
  h_ele_ambiguousTracks->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/ambiguousTracks.%s", outDir, suffix);
    gPad->Print(str);
  }

  if (pause)
    c_lostHits->WaitPrimitive();

  // classes
  TH1F *h_ele_PinMnPout_mode = (TH1F *)hist.Get("h_ele_PinMnPout");
  TH1F *h_ele_classes = (TH1F *)hist.Get("h_ele_classes");
  TH1F *h_ele_eta_bbremFrac = (TH1F *)hist.Get("h_ele_eta_bbremFrac");
  TH1F *h_ele_eta_goldenFrac = (TH1F *)hist.Get("h_ele_eta_goldenFrac");
  TH1F *h_ele_eta_narrowFrac = (TH1F *)hist.Get("h_ele_eta_narrowFrac");
  TH1F *h_ele_eta_showerFrac = (TH1F *)hist.Get("h_ele_eta_showerFrac");

  TCanvas *c_PinMnPout_mode = new TCanvas("PinMnPout_mode", "PinMnPout_mode");
  c_PinMnPout_mode->cd();
  h_ele_PinMnPout_mode->Draw();
  h_ele_PinMnPout_mode->GetXaxis()->SetTitle("P_{in} - p_{out} (GeV/c)");
  h_ele_PinMnPout_mode->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/PinMnPout_mode.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_classes = new TCanvas("classes", "classes");
  c_classes->cd();
  h_ele_classes->Draw();
  h_ele_classes->GetXaxis()->SetTitle("Class id");
  h_ele_classes->GetYaxis()->SetTitle("Events");
  if (out) {
    snprintf(str, 128, "%s/classes.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_eta_bbremFrac = new TCanvas("eta_bbremFrac", "eta_bbremFrac");
  c_eta_bbremFrac->cd();
  h_ele_eta_bbremFrac->Draw();
  h_ele_eta_bbremFrac->GetXaxis()->SetTitle("|#eta|");
  h_ele_eta_bbremFrac->GetYaxis()->SetTitle("Fraction of bbrem");
  if (out) {
    snprintf(str, 128, "%s/eta_bbremFrac.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_eta_goldenFrac = new TCanvas("eta_goldenFrac", "eta_goldenFrac");
  c_eta_goldenFrac->cd();
  h_ele_eta_goldenFrac->Draw();
  h_ele_eta_goldenFrac->GetXaxis()->SetTitle("|#eta|");
  h_ele_eta_goldenFrac->GetYaxis()->SetTitle("Fraction of golden");
  if (out) {
    snprintf(str, 128, "%s/eta_goldenFrac.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_eta_narrowFrac = new TCanvas("eta_narrowFrac", "eta_narrowFrac");
  c_eta_narrowFrac->cd();
  h_ele_eta_narrowFrac->Draw();
  h_ele_eta_narrowFrac->GetXaxis()->SetTitle("|#eta|");
  h_ele_eta_narrowFrac->GetYaxis()->SetTitle("Fraction of narrow");
  if (out) {
    snprintf(str, 128, "%s/eta_narrowFrac.%s", outDir, suffix);
    gPad->Print(str);
  }

  TCanvas *c_eta_showerFrac = new TCanvas("eta_showerFrac", "eta_showerFrac");
  c_eta_showerFrac->cd();
  h_ele_eta_showerFrac->Draw();
  h_ele_eta_showerFrac->GetXaxis()->SetTitle("|#eta|");
  h_ele_eta_showerFrac->GetYaxis()->SetTitle("Fraction of showering");
  if (out) {
    snprintf(str, 128, "%s/eta_showerFrac.%s", outDir, suffix);
    gPad->Print(str);
  }
}
