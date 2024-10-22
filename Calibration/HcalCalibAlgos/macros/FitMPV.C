//////////////////////////////////////////////////////////////////////////////
// Usage:
// .L FitPV.C+g
//             For carrying out Gaussian convoluted Landau function to
//             the charge
//     void doFit(infile, text, iPeriod, etaMin, etamax, dmin, dmax, vxmin,
//                vxmax, phimin, phimax, phiInc, ifHPD, writeExtraText)
//
//
//  where
//   infile  (std::string) Nme of the input file
//   text    (std::string)
//   iPeriod (int)         ieta value of the tower
//   etaMin  (int)       | range of ieta's to
//   etaMax  (int)       | be considered
//   dmin, dmax (int)      range of depths to be done
//   vxmin,vxmax(int)      range of number of vertex to be considered
//   phiMin  (int)       | range of iphi's to
//   phiMax  (int)       | be considered
//   phiInc  (int)         incerement in phi (1 or 2 depending on iphi value)
//   ifHPD   (bool)        if the ditribution is from HPD or SiPM
//   writeExtraText(bool)  if extra text to be written
//
///////////////////////////////////////////////////////////////////////////////

#include <TArrow.h>
#include <TASImage.h>
#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TGraphAsymmErrors.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <TProfile.h>
#include <TROOT.h>
#include <TStyle.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace RooFit;

void setTDRStyle() {
  TStyle* tdrStyle = new TStyle("tdrStyle", "Style for P-TDR");

  // For the canvas:
  tdrStyle->SetCanvasBorderMode(0);
  tdrStyle->SetCanvasColor(kWhite);
  tdrStyle->SetCanvasDefH(600);  //Height of canvas
  tdrStyle->SetCanvasDefW(600);  //Width of canvas
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
  tdrStyle->SetHistLineWidth(1);
  // tdrStyle->SetLegoInnerR(Float_t rad = 0.5);
  tdrStyle->SetNumberContours(20);

  tdrStyle->SetEndErrorSize(2);
  // tdrStyle->SetErrorMarker(20);
  // tdrStyle->SetErrorX(0.);

  tdrStyle->SetMarkerStyle(20);

  //For the fit/function:
  tdrStyle->SetOptFit(1);
  tdrStyle->SetFitFormat("5.4g");
  tdrStyle->SetFuncColor(2);
  tdrStyle->SetFuncStyle(1);
  tdrStyle->SetFuncWidth(1);

  //For the date:
  tdrStyle->SetOptDate(0);
  // tdrStyle->SetDateX(Float_t x = 0.01);
  // tdrStyle->SetDateY(Float_t y = 0.01);

  // For the statistics box:
  tdrStyle->SetOptFile(0);
  tdrStyle->SetOptStat(0);  // To display the mean and RMS:   SetOptStat("mr");
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
  tdrStyle->SetPadLeftMargin(0.16);
  tdrStyle->SetPadRightMargin(0.02);

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
  tdrStyle->SetTitleYOffset(1.25);
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
  tdrStyle->SetOptLogx(0);
  tdrStyle->SetOptLogy(0);
  tdrStyle->SetOptLogz(0);

  // Postscript options:
  tdrStyle->SetPaperSize(20., 20.);
  tdrStyle->SetHatchesLineWidth(5);
  tdrStyle->SetHatchesSpacing(0.05);

  tdrStyle->cd();
}

void CMS_lumi(TPad* pad, int iPeriod, int iPosX) {
  TString cmsText = "CMS";
  const float cmsTextFont = 61;
  const bool writeExtraText = false;
  TString extraText = "Preliminary";
  const float extraTextFont = 52;
  const float lumiTextSize = 0.6;
  const float lumiTextOffset = 0.2;
  const float cmsTextSize = 0.75;
  const float relPosX = 0.045;
  const float relPosY = 0.035;
  const float relExtraDY = 1.2;
  const float extraOverCmsTextSize = 0.76;
  TString lumi_13TeV = "20.1 fb^{-1}";
  TString lumi_8TeV = "19.7 fb^{-1}";
  TString lumi_7TeV = "5.1 fb^{-1}";
  TString lumi_sqrtS = "";
  const bool drawLogo = false;

  bool outOfFrame = (iPosX / 10 == 0) ? true : false;
  int alignY_ = 3;
  int alignX_ = 2;
  if (iPosX / 10 == 0)
    alignX_ = 1;
  if (iPosX == 0)
    alignX_ = 1;
  if (iPosX == 0)
    alignY_ = 1;
  if (iPosX / 10 == 1)
    alignX_ = 1;
  if (iPosX / 10 == 2)
    alignX_ = 2;
  if (iPosX / 10 == 3)
    alignX_ = 3;
  //if( iPosX == 0  ) relPosX = 0.12;
  int align_ = 10 * alignX_ + alignY_;

  float H = pad->GetWh();
  float W = pad->GetWw();
  float l = pad->GetLeftMargin();
  float t = pad->GetTopMargin();
  float r = pad->GetRightMargin();
  float b = pad->GetBottomMargin();
  //  float e = 0.025;

  pad->cd();

  TString lumiText;
  if (iPeriod == 1) {
    lumiText += lumi_7TeV;
    lumiText += " (7 TeV)";
  } else if (iPeriod == 2) {
    lumiText += lumi_8TeV;
    lumiText += " (8 TeV)";
  } else if (iPeriod == 3) {
    lumiText = lumi_8TeV;
    lumiText += " (8 TeV)";
    lumiText += " + ";
    lumiText += lumi_7TeV;
    lumiText += " (7 TeV)";
  } else if (iPeriod == 4) {
    lumiText += lumi_13TeV;
    lumiText += " (13 TeV)";
  } else if (iPeriod == 7) {
    if (outOfFrame)
      lumiText += "#scale[0.85]{";
    lumiText += lumi_13TeV;
    lumiText += " (13 TeV)";
    lumiText += " + ";
    lumiText += lumi_8TeV;
    lumiText += " (8 TeV)";
    lumiText += " + ";
    lumiText += lumi_7TeV;
    lumiText += " (7 TeV)";
    if (outOfFrame)
      lumiText += "}";
  } else if (iPeriod == 12) {
    lumiText += "8 TeV";
  } else if (iPeriod == 0) {
    lumiText += lumi_sqrtS;
  }

  std::cout << lumiText << endl;

  TLatex latex;
  latex.SetNDC();
  latex.SetTextAngle(0);
  latex.SetTextColor(kBlack);

  float extraTextSize = extraOverCmsTextSize * cmsTextSize;

  latex.SetTextFont(42);
  latex.SetTextAlign(31);
  latex.SetTextSize(lumiTextSize * t);
  latex.DrawLatex(1 - r, 1 - t + lumiTextOffset * t, lumiText);

  if (outOfFrame) {
    latex.SetTextFont(cmsTextFont);
    latex.SetTextAlign(11);
    latex.SetTextSize(cmsTextSize * t);
    latex.DrawLatex(l, 1 - t + lumiTextOffset * t, cmsText);
  }

  pad->cd();

  float posX_ = 0;
  if (iPosX % 10 <= 1) {
    posX_ = l + relPosX * (1 - l - r);
  } else if (iPosX % 10 == 2) {
    posX_ = l + 0.5 * (1 - l - r);
  } else if (iPosX % 10 == 3) {
    posX_ = 1 - r - relPosX * (1 - l - r);
  }
  float posY_ = 1 - t - relPosY * (1 - t - b);
  if (!outOfFrame) {
    if (drawLogo) {
      posX_ = l + 0.045 * (1 - l - r) * W / H;
      posY_ = 1 - t - 0.045 * (1 - t - b);
      float xl_0 = posX_;
      float yl_0 = posY_ - 0.15;
      float xl_1 = posX_ + 0.15 * H / W;
      float yl_1 = posY_;
      TASImage* CMS_logo = new TASImage("CMS-BW-label.png");
      TPad* pad_logo = new TPad("logo", "logo", xl_0, yl_0, xl_1, yl_1);
      pad_logo->Draw();
      pad_logo->cd();
      CMS_logo->Draw("X");
      pad_logo->Modified();
      pad->cd();
    } else {
      latex.SetTextFont(cmsTextFont);
      latex.SetTextSize(cmsTextSize * t);
      latex.SetTextAlign(align_);
      latex.DrawLatex(posX_, posY_, cmsText);
      if (writeExtraText) {
        latex.SetTextFont(extraTextFont);
        latex.SetTextAlign(align_);
        latex.SetTextSize(extraTextSize * t);
        latex.DrawLatex(posX_, posY_ - relExtraDY * cmsTextSize * t, extraText);
      }
    }
  } else if (writeExtraText) {
    if (iPosX == 0) {
      posX_ = l + relPosX * (1 - l - r);
      posY_ = 1 - t + lumiTextOffset * t;
    }
    latex.SetTextFont(extraTextFont);
    latex.SetTextSize(extraTextSize * t);
    latex.SetTextAlign(align_);
    latex.DrawLatex(posX_, posY_, extraText);
  }
  return;
}

TCanvas* example_plot(int iPeriod,
                      int iPos,
                      int ieta,
                      int phi,
                      int idep,
                      int inv,
                      TFile* f,
                      std::string sample1,
                      bool writeExtraText,
                      bool ifHPD,
                      std::string outfile) {
  int W = 800;
  int H = 600;
  int H_ref = 600;
  int W_ref = 800;

  // references for T, B, L, R
  float T = 0.08 * H_ref;
  float B = 0.12 * H_ref;
  float L = 0.12 * W_ref;
  float R = 0.04 * W_ref;
  TString canvName = "FigExample_";
  canvName += W;
  canvName += "-";
  canvName += H;
  canvName += "_";
  canvName += iPeriod;
  if (writeExtraText)
    canvName += "-prelim";
  if (iPos % 10 == 0)
    canvName += "-out";
  else if (iPos % 10 == 1)
    canvName += "-left";
  else if (iPos % 10 == 2)
    canvName += "-center";
  else if (iPos % 10 == 3)
    canvName += "-right";

  TCanvas* canv = new TCanvas(canvName, canvName, 50, 50, W, H);
  canv->SetFillColor(0);
  canv->SetBorderMode(0);
  canv->SetFrameFillStyle(0);
  canv->SetFrameBorderMode(0);
  canv->SetLeftMargin(L / W);
  canv->SetRightMargin(R / W);
  canv->SetTopMargin(T / H);
  canv->SetBottomMargin(B / H);
  canv->SetTickx(0);
  canv->SetTicky(0);

  TH1* h = new TH1F("h", "h", 40, 70, 110);
  // h->GetXaxis()->SetNdivisions(6,5,0);
  // h->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV)");
  h->GetYaxis()->SetNdivisions(6, 5, 0);
  h->GetYaxis()->SetTitleOffset(1);
  h->GetYaxis()->SetTitle("Tracks");
  h->SetMaximum(260);
  if (iPos == 1)
    h->SetMaximum(300);
  h->Draw();

  int histLineColor = kOrange + 7;
  int histFillColor = kOrange - 2;
  float markerSize = 1.0;

  {
    TLatex latex;
    int n_ = 2;

    float x1_l = 0.92;
    float y1_l = 0.60;

    float dx_l = 0.30;
    float dy_l = 0.18;
    float x0_l = x1_l - dx_l;
    float y0_l = y1_l - dy_l;

    TPad* legend = new TPad("legend_0", "legend_0", x0_l, y0_l, x1_l, y1_l);
    //    legend->SetFillColor( kGray );
    legend->Draw();
    legend->cd();

    float ar_l = dy_l / dx_l;

    float x_l[1];
    float ex_l[1];
    float y_l[1];
    float ey_l[1];

    //    float gap_ = 0.09/ar_l;
    float gap_ = 1. / (n_ + 1);

    float bwx_ = 0.12;
    float bwy_ = gap_ / 1.5;

    x_l[0] = 1.2 * bwx_;
    //    y_l[0] = 1-(1-0.10)/ar_l;
    y_l[0] = 1 - gap_;
    ex_l[0] = 0;
    ey_l[0] = 0.04 / ar_l;

    TGraph* gr_l = new TGraphErrors(1, x_l, y_l, ex_l, ey_l);

    gStyle->SetEndErrorSize(0);
    gr_l->SetMarkerSize(0.9);
    gr_l->Draw("0P");

    latex.SetTextFont(42);
    latex.SetTextAngle(0);
    latex.SetTextColor(kBlack);
    latex.SetTextSize(0.25);
    latex.SetTextAlign(12);

    TLine line_;
    TBox box_;
    float xx_ = x_l[0];
    float yy_ = y_l[0];
    latex.DrawLatex(xx_ + 1. * bwx_, yy_, "Data");

    yy_ -= gap_;
    box_.SetLineStyle(kSolid);
    box_.SetLineWidth(1);
    // box_.SetLineColor( kBlack );
    box_.SetLineColor(histLineColor);
    box_.SetFillColor(histFillColor);
    box_.DrawBox(xx_ - bwx_ / 2, yy_ - bwy_ / 2, xx_ + bwx_ / 2, yy_ + bwy_ / 2);
    box_.SetFillStyle(0);
    box_.DrawBox(xx_ - bwx_ / 2, yy_ - bwy_ / 2, xx_ + bwx_ / 2, yy_ + bwy_ / 2);
    latex.DrawLatex(xx_ + 1. * bwx_, yy_, "Z #rightarrow e^{+}e^{-} (MC)");

    canv->cd();
  }

  {
    char name[256], name1[256], name1a[256], name1b[256], name1c[256];

    //  if(idep  == 0) vari=10000;
    int hmax = ifHPD ? 25 : 5000;
    RooRealVar t("t", "Charge [fC]", 0, hmax);

    int rbo = 0;
    std::cout << "problem:"
              << "ieta"
              << ":" << ieta << std::endl;
    std::cout << "problem:"
              << "idep"
              << ":" << idep << std::endl;
    std::cout << "problem:"
              << "inv"
              << ":" << inv << std::endl;

    //sprintf(name1,"DieMuonEta%d/ChrgE%dF1D%dV%dP0",ieta,ieta, idep, inv); ChrgE19F72D5V1P0
    sprintf(name1, "DieMuonEta%d/ChrgE%dF%dD%dV%dP0", ieta, ieta, phi, idep, inv);

    TH1F* h1 = (TH1F*)f->Get(name1);
    // if (!h1) continue;
    if (h1->GetBinContent(1) > h1->GetBinContent(2))
      h1->SetBinContent(1, 0);
    std::cout << "did you pass" << std::endl;
    std::cout << h1->Integral() << std::endl;
    //		if(h->Integral() >= 1) continue;
    std::cout << "you didnt" << std::endl;

    // h1->Scale(1.0/h1->Integral());
    // h1->Sumw2();
    int nrebin = ifHPD ? 100 : 200;
    h1->Rebin(nrebin);
    h1->Sumw2();
    // h ->SetBinContent(1,0);
    // h->GetXaxis()->SetRangeUser(0,5.);
    // if ((i  == 2  &&  hi ==3) || (i  == 2  &&  hi == 4 ) ) { h->Rebin(40) ;} else if( i == 5) { h->Rebin(50);} else {h->Rebin(80);}
    std::cout << name1 << std::endl;
    // h->SetBinContent(1,0);
    // RooDataHist data("data", "binned version of data",t,Import(*h));
    RooDataHist data("data", "binned version of data", t, Import(*h1));

    // Construct landau(t,ml,sl) ;
    std::cout << "==============================================================" << std::endl;
    std::cout << "mean of landau:" << h1->GetMean() << ":" << h1->GetXaxis()->GetXmax() << std::endl;
    std::cout << "==============================================================" << std::endl;
    // double ijk=3; double klm= 0.9;
    // if (i >= 10 && i <= 20) ijk=1.5;
    // if (i > 20) { ijk=7; klm=9;}
    double mlstrt = (ifHPD) ? 0.2 : 200.0;
    double mlmaxi = (ifHPD) ? 20. : 4000.0;
    double slstrt = (ifHPD) ? 0.019 : 150.019;
    double slmaxi = (ifHPD) ? 20. : 5000.0;
    RooRealVar ml("ml", "mean landau", mlstrt, 0, mlmaxi);
    RooRealVar sl("sl", "sigma landau", slstrt, 0, slmaxi);
    RooLandau landau("lx", "lx", t, ml, sl);
    // RooRealVar mg("mg","mg",1100);//,0,1300);//,0,1.3);
    // RooRealVar mg("mg","mg",600, 500,800);
    std::cout << "mean before gauss" << h->GetMean() << std::endl;
    double sgstrt = (ifHPD) ? 0.003 : 100.003;
    double sgmini = (ifHPD) ? 0.001 : 0.0;
    double sgmaxi = (ifHPD) ? 20. : 2000.0;
    RooRealVar mg("mg", "mg", 0);
    RooRealVar sg("sg", "sg", sgstrt, sgmini, sgmaxi);
    RooGaussian gauss("gauss", "gauss", t, mg, sg);

    //	RooRealVar sgl("sgl","sgl",1300,1000,1400);
    //	RooRealVar sgr("sgr","sgr",2000,1700,3000);
    //	RooBifurGauss lxg("lxg", "lxg", t,mg, sgl,sgr);

    // C o n s t r u c t   c o n v o l u t i o n   p d f
    // ---------------------------------------

    // Construct landau (x) gauss
    char nameo[256];
    sprintf(nameo, "charge for ieta%d_iphi%d_depth%d", ieta, phi, idep + 1);
    RooNumConvPdf lxg("lxg", "gauss (x) landau convolution", t, gauss, landau);  //landau,gauss) ;
    // S a m p l e ,   f i t   a n d   p l o t   c o n v o l u t e d   p d f
    // ----------------------------------------------------------------------

    lxg.fitTo(data, Minos(kTRUE));

    RooPlot* frame = t.frame(Title(nameo));
    data.plotOn(frame);

    lxg.plotOn(frame);
    lxg.paramOn(frame, Layout(0.7, 0.95, 0.9));  //LineColor(kRed),LineStyle(kDashed));

    sprintf(name1a, "i#eta=%d", ieta);
    //sprintf(name1c, "depth=%d, NVtx=%d", idep+1,inv+1);
    sprintf(name1b, "i#phi=%d", phi);
    sprintf(name1c, "depth=%d", idep + 1);
    frame->GetYaxis()->SetTitle("Events");
    frame->GetYaxis()->SetTitleOffset(1.0);
    TLatex* txt = new TLatex(0.7, 0.65, name1a);
    TLatex* txt1 = new TLatex(0.7, 0.6, name1b);
    TLatex* txt2 = new TLatex(0.7, 0.55, name1c);

    txt->SetNDC();
    txt->SetTextSize(0.04);
    txt1->SetNDC();
    txt1->SetTextSize(0.04);
    txt2->SetNDC();
    txt2->SetTextSize(0.04);

    // txt->SetTextColor(kRed) ;
    frame->addObject(txt);
    frame->addObject(txt1);
    frame->addObject(txt2);
    frame->SetTitle("");

    frame->Draw();

    TF1* f1 = lxg.asTF(RooArgList(t));
    Double_t xmax = f1->GetMaximumX();
    Double_t peak = ml.getVal();

    Double_t chiSquare = frame->chiSquare();

    std::ofstream log1(outfile.c_str(), std::ios_base::app | std::ios_base::out);
    //log1<<"depth"<<i+1<<"\t"<<"vtx bin"<< hi << "\t"<< xmax<<"\t"<<h->GetStdDevError()<<std::endl; //Standard deviation error

    //double gError = mg->getError();
    double lError = ml.getError();
    //double error = sqrt(gError*gError + lError*lError);
    log1 << sample1.c_str() << "\tieta " << ieta << "\tiphi " << phi << "\t\tDEP " << (idep + 1) << "\tNvtx " << inv
         << "\t" << peak << "\t" << lError << "\t" << chiSquare << std::endl;
    log1.close();
  }

  // writing the lumi information and the CMS "logo"
  CMS_lumi(canv, iPeriod, iPos);
  canv->Update();
  canv->RedrawAxis();
  canv->GetFrame()->Draw();

  // h->Draw();
  // char name1[256];
  char name_out[256], name_out1[256];

  char name1[256];
  sprintf(name1, "ieta%d_iphi%d_depth%d", ieta, phi, (idep + 1));
  //	canv->SaveAs("check.png");

  canv->SaveAs(("out_" + sample1).c_str() + TString(name1) + ".pdf", ".pdf");
  canv->SaveAs(("out_" + sample1).c_str() + TString(name1) + ".png", ".png");
  canv->SaveAs(("out_" + sample1).c_str() + TString(name1) + ".root", ".root");

  return canv;
}

void doFit(std::string infile,
           std::string text,
           int iPeriod = 0,
           int etaMin = 2,
           int etamax = 3,
           int dmin = 1,
           int dmax = 1,
           int vxmin = 2,
           int vxmax = 2,
           int phimin = 1,
           int phimax = 72,
           int phiInc = 1,
           bool ifHPD = true,
           bool writeExtraText = false) {
  TFile* f = new TFile(infile.c_str());
  if (f != 0) {
    setTDRStyle();
    for (unsigned int ieta = etaMin; ieta <= etamax; ieta++) {
      for (unsigned int idep = dmin; idep <= dmax; idep++) {
        // NVtx
        for (unsigned int inv = vxmin; inv <= vxmax; inv++) {
          for (unsigned int phi = phimin; phi <= phimax; phi += phiInc) {
            example_plot(iPeriod, 0, ieta, phi, idep, inv, f, text, writeExtraText, ifHPD, text);
          }
        }
      }
    }
    f->Close();
  }
}
