void setTDRStyle(Int_t xlog, Int_t ylog) {

  TStyle *tdrStyle = new TStyle("tdrStyle","Style for P-TDR");

// For the canvas:
  tdrStyle->SetCanvasBorderMode(0);
  tdrStyle->SetCanvasColor(kWhite);
  tdrStyle->SetCanvasDefH(600); //Height of canvas
  tdrStyle->SetCanvasDefW(600); //Width of canvas
  tdrStyle->SetCanvasDefX(0);   //POsition on screen
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

// For the frame:
  tdrStyle->SetFrameBorderMode(0);
  tdrStyle->SetFrameBorderSize(1);
  tdrStyle->SetFrameFillColor(0);
  tdrStyle->SetFrameFillStyle(0);
  tdrStyle->SetFrameLineColor(1);
  tdrStyle->SetFrameLineStyle(1);
  tdrStyle->SetFrameLineWidth(1);

// For the histo:
  // tdrStyle->SetHistFillColor(1);
  // tdrStyle->SetHistFillStyle(0);
  tdrStyle->SetHistLineColor(1);
  tdrStyle->SetHistLineStyle(0);
  tdrStyle->SetHistLineWidth(2);
  // tdrStyle->SetLegoInnerR(Float_t rad = 0.5);
  // tdrStyle->SetNumberContours(Int_t number = 20);

  tdrStyle->SetEndErrorSize(4);
  //  tdrStyle->SetErrorMarker(20);
  //  tdrStyle->SetErrorX(0.);
  
  tdrStyle->SetMarkerStyle(20);

//For the fit/function:
  tdrStyle->SetOptFit(1);
  tdrStyle->SetFitFormat("5.4g");
  tdrStyle->SetFuncColor(1);
  tdrStyle->SetFuncStyle(1);
  tdrStyle->SetFuncWidth(1);

//For the date:
  tdrStyle->SetOptDate(0);
  // tdrStyle->SetDateX(Float_t x = 0.01);
  // tdrStyle->SetDateY(Float_t y = 0.01);

// For the statistics box:
  tdrStyle->SetOptFile(0);
  tdrStyle->SetOptStat(0); // To display the mean and RMS:   SetOptStat("mr");
  tdrStyle->SetStatColor(kWhite);
  tdrStyle->SetStatFont(42);
  tdrStyle->SetStatFontSize(0.025);
  tdrStyle->SetStatTextColor(1);
  tdrStyle->SetStatFormat("6.4g");
  tdrStyle->SetStatBorderSize(1);
  tdrStyle->SetStatH(0.1);
  tdrStyle->SetStatW(0.15);
  // tdrStyle->SetStatStyle(Style_t style = 1001);
  // tdrStyle->SetStatX(Float_t x = 0);
  // tdrStyle->SetStatY(Float_t y = 0);

// Margins:
  tdrStyle->SetPadTopMargin(0.05);
  tdrStyle->SetPadBottomMargin(0.13);
  tdrStyle->SetPadLeftMargin(0.13);
  tdrStyle->SetPadRightMargin(0.05);

// For the Global title:

  tdrStyle->SetOptTitle(0);
  tdrStyle->SetTitleFont(42);
  tdrStyle->SetTitleColor(1);
  tdrStyle->SetTitleTextColor(1);
  tdrStyle->SetTitleFillColor(10);
  tdrStyle->SetTitleFontSize(0.05);
  // tdrStyle->SetTitleH(0); // Set the height of the title box
  // tdrStyle->SetTitleW(0); // Set the width of the title box
  // tdrStyle->SetTitleX(0); // Set the position of the title box
  // tdrStyle->SetTitleY(0.985); // Set the position of the title box
  // tdrStyle->SetTitleStyle(Style_t style = 1001);
  // tdrStyle->SetTitleBorderSize(2);

// For the axis titles:

  tdrStyle->SetTitleColor(1, "XYZ");
  tdrStyle->SetTitleFont(42, "XYZ");
  tdrStyle->SetTitleSize(0.06, "XYZ");
  // tdrStyle->SetTitleXSize(Float_t size = 0.02); // Another way to set the size?
  // tdrStyle->SetTitleYSize(Float_t size = 0.02);
  tdrStyle->SetTitleXOffset(0.9);
  tdrStyle->SetTitleYOffset(1.05);
  // tdrStyle->SetTitleOffset(1.1, "Y"); // Another way to set the Offset

// For the axis labels:

  tdrStyle->SetLabelColor(1, "XYZ");
  tdrStyle->SetLabelFont(42, "XYZ");
  tdrStyle->SetLabelOffset(0.007, "XYZ");
  tdrStyle->SetLabelSize(0.05, "XYZ");

// For the axis:

  tdrStyle->SetAxisColor(1, "XYZ");
  tdrStyle->SetStripDecimals(kTRUE);
  tdrStyle->SetTickLength(0.03, "XYZ");
  tdrStyle->SetNdivisions(510, "XYZ");
  tdrStyle->SetPadTickX(1);  // To get tick marks on the opposite side of the frame
  tdrStyle->SetPadTickY(1);

// Change for log plots:
  tdrStyle->SetOptLogx(xlog);
  tdrStyle->SetOptLogy(ylog);
  tdrStyle->SetOptLogz(0);

// Postscript options:

//  tdrStyle->SetPaperSize(7.5,7.5);

  tdrStyle->SetPaperSize(15.,15.);

//  tdrStyle->SetPaperSize(20.,20.);

  // tdrStyle->SetLineScalePS(Float_t scale = 3);
  // tdrStyle->SetLineStyleString(Int_t i, const char* text);
  // tdrStyle->SetHeaderPS(const char* header);
  // tdrStyle->SetTitlePS(const char* pstitle);

  // tdrStyle->SetBarOffset(Float_t baroff = 0.5);
  // tdrStyle->SetBarWidth(Float_t barwidth = 0.5);
  // tdrStyle->SetPaintTextFormat(const char* format = "g");
  // tdrStyle->SetPalette(Int_t ncolors = 0, Int_t* colors = 0);
  // tdrStyle->SetTimeOffset(Double_t toffset);
  // tdrStyle->SetHistMinimumZero(kTRUE);

  tdrStyle->cd();
}

void Draw()
{
   setTDRStyle(1,0);

   TFile* file = new TFile("nosr_endcap.root");

   TCanvas* c1 = new TCanvas("X","Y",1);
   TAxis* xaxis = hprEH11x5->GetXaxis();
   hprEH11x5->GetXaxis()->SetTitle("p_{T} of #pi ^{+} and #pi ^{-}, GeV");
   hprEH11x5->GetYaxis()->SetTitleOffset(1.3);
   hprEH11x5->GetYaxis()->SetTitle("Energy / E^{true}");
   hprEH11x5->SetMarkerStyle(21);
   hprEH11x5->SetMaximum(1.0);
   hprEH11x5->SetMinimum(0.4);
   hprEH11x5->Draw();

   hprEH7x5->SetMarkerStyle(25);
   hprEH7x5->Draw("same");

   hprEH11x3->SetMarkerStyle(24);
   hprEH11x3->Draw("same");

   hprEH7x3->SetMarkerStyle(22);
   hprEH7x3->SetMarkerSize(1.2);
   hprEH7x3->Draw("same");

   TLegend *leg = new TLegend(0.5,0.2,0.9,0.4,NULL,"brNDC");
   leg->SetFillColor(10);
   leg->AddEntry(hprEH11x5,"ECAL 11x11, HCAL 5x5","P");
   leg->AddEntry(hprEH7x5,"ECAL 7x7, HCAL 5x5","P");
   leg->AddEntry(hprEH11x3,"ECAL 11x11, HCAL 3x3","P");
   leg->AddEntry(hprEH7x3,"ECAL 7x7, HCAL 3x3","P");
   leg->Draw();
   TLatex *t = new TLatex();
   t->SetTextSize(0.042);
   t->DrawLatex(1.2,0.95,"CMSSW_1_6_12, |#eta| > 1.6");
   t->DrawLatex(1.2,0.90,"no ZSP in HCAL");
   t->DrawLatex(1.2,0.85,"no SR in ECAL");
   c1->SaveAs("EBHBmatrix_endcap.gif");

   TCanvas* c2 = new TCanvas("X","Y",1);
   TAxis* xaxis = hprH5->GetXaxis();
   hprH5->GetXaxis()->SetTitle("p_{T} of #pi ^{+} and #pi ^{-}, GeV");
   hprH5->GetYaxis()->SetTitleOffset(1.3);
   hprH5->GetYaxis()->SetTitle("Energy / E^{true}");
   hprH5->SetMarkerStyle(21);
   hprH5->SetMaximum(1.0);
   hprH5->SetMinimum(0.0);
   hprH5->Draw();

   hprH3->SetMarkerStyle(25);
   hprH3->Draw("same");

   TLegend *leg = new TLegend(0.7,0.2,0.9,0.4,NULL,"brNDC");
   leg->SetFillColor(10);
   leg->AddEntry(hprH5,"HCAL 5x5","P");
   leg->AddEntry(hprH3,"HCAL 3x3","P");
   leg->Draw();
   TLatex *t = new TLatex();
   t->SetTextSize(0.042);
   t->DrawLatex(1.5,0.80,"CMSSW_1_6_12, |#eta| > 1.6");
   t->DrawLatex(1.5,0.70,"no ZSP in HCAL");

   c2->SaveAs("HBmatrix_endcap.gif");

   setTDRStyle(0,0);
   TCanvas* c3 = new TCanvas("X","Y",1);
   TAxis* xaxis = hH3_1_2GeV->GetXaxis();
   hH3_1_2GeV->GetXaxis()->SetTitle("Energy/E^{true}");
   hH3_1_2GeV->GetYaxis()->SetTitleOffset(1.3);
   hH3_1_2GeV->GetYaxis()->SetTitle("");
   Double_t scale = 1./hH3_1_2GeV->Integral();
   hH3_1_2GeV->Scale(scale);
   hH3_1_2GeV->SetLineWidth(3);
   hH3_1_2GeV->Draw("hist");

   hH5_1_2GeV->SetLineStyle(2);
   hH5_1_2GeV->SetLineWidth(3);
   scale = 1./hH5_1_2GeV->Integral();
   hH5_1_2GeV->Scale(scale);
   hH5_1_2GeV->Draw("same");
   TLegend *leg = new TLegend(0.7,0.5,0.9,0.7,NULL,"brNDC");
   leg->SetFillColor(10);
   leg->AddEntry(hH5_1_2GeV,"HCAL 5x5","L");
   leg->AddEntry(hH3_1_2GeV,"HCAL 3x3","L");
   leg->Draw();
   TLatex *t = new TLatex();
   t->SetTextSize(0.042);
   t->DrawLatex(-1.8,0.12,"CMSSW_1_6_12");
   t->DrawLatex(-1.8,0.10,"no ZSP in HCAL");
   t->DrawLatex(-1.8,0.08,"|#eta| > 1.6");
   t->DrawLatex(-1.8,0.06,"1 < p_{T}^{trk} < 2 GeV");
   c3->SaveAs("hHB_1_2GeV_endcap.gif");

   /*
   TCanvas* c4 = new TCanvas("X","Y",1);
   TAxis* xaxis = hE11_1_2GeV->GetXaxis();
   hE11_1_2GeV->GetXaxis()->SetTitle("E in ECAL 11x11 / E^{true}");
   hE11_1_2GeV->GetYaxis()->SetTitleOffset(1.3);
   hE11_1_2GeV->GetYaxis()->SetTitle("");
   Double_t scale = 1./hE11_1_2GeV->Integral();
   hE11_1_2GeV->Scale(scale);
   hE11_1_2GeV->SetMaximum(0.15);
   hE11_1_2GeV->SetLineWidth(3);
   hE11_1_2GeV->Draw("hist");

   hE11_3_4GeV->SetLineStyle(2);
   hE11_3_4GeV->SetLineWidth(3);
   scale = 1./hE11_3_4GeV->Integral();
   hE11_3_4GeV->Scale(scale);
   hE11_3_4GeV->Draw("same");

   hE11_5_10GeV->SetLineStyle(3);
   hE11_5_10GeV->SetLineWidth(3);
   scale = 1./hE11_5_10GeV->Integral();
   hE11_5_10GeV->Scale(scale);
   hE11_5_10GeV->Draw("same");

   c4->SaveAs("hEB.gif");
   */
}
