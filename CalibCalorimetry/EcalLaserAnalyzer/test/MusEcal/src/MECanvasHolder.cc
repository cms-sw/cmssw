#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <assert.h>
using namespace std;

#include <math.h>
#include <TStyle.h>

#include "TImage.h"
#include "TAttImage.h"

#include "MECanvasHolder.hh"

ClassImp(MECanvasHolder);

MECanvasHolder::MECanvasHolder()
    : fCanvas(0), fWelcomePave(0), fWelcomeState(true), fWelcomeTitle(0), fWelcomeL0(0), _h(0), _scale(1), _refw(1092) {}

MECanvasHolder::~MECanvasHolder() { ClearWelcome(); }

void MECanvasHolder::CanvasModified() {
  //Something has been modified in fCanvas
  TPad* pad = (TPad*)gPad;
  fCanvas->cd();
  fCanvas->Modified();
  fCanvas->Update();
  pad->cd();
}

void MECanvasHolder::ClearWelcome() {
  if (fWelcomePave) {
    delete fWelcomePave;
    fWelcomePave = 0;
    gPad->Clear();
  }
}

void MECanvasHolder::SetCanvas(
    TCanvas* canvas, const char* textTL, const char* textBL, const char* textTR, const char* textBR) {
  assert(canvas != 0);
  //  fCanvas = fEcanvas->GetCanvas();
  fCanvas = canvas;
  fCanvas->Clear();
  fTopXGen = fCanvas->GetWindowTopX();
  fTopYGen = fCanvas->GetWindowTopY();
  //  fWidthGen  = fCanvas->GetWindowWidth();
  //  fHeigthGen = fCanvas->GetWindowHeight();
  fWidthGen = fCanvas->GetWw();
  fHeigthGen = fCanvas->GetWh();

  //  cout << "fTopXGen/fTopYGen/fWidthGen/fHeigthGen " << fTopXGen << "/" << fTopYGen << "/" << fWidthGen << "/" << fHeigthGen << endl;

  if (fWidthGen != 0) {
    _scale = fWidthGen / _refw;
  }
  _scale = 1;  // fixme !

  // Various settings
  Color_t fColCan = 45;
  Style_t fStyleCan = 1001;
  Color_t fStyleHistColor = 45;
  Color_t fStyleTitleColor = 1;
  Color_t fStyleTFColor = 191;
  Color_t fStyleTTextColor = 1;
  Style_t fStylePad1 = 1001;
  Int_t fPadLogX = 0;
  Int_t fPadLogY = 0;
  Color_t fColFrame = 18;
  Color_t fColPad1 = 30;

  TString fStringTL(textTL);  //        = "Welcome to MusEcal - Monitoring and Useful Survey of Ecal";
  TString fStringBL(
      textBL);  //        = "Visit our Twiki page at https://twiki.cern.ch/twiki/bin/view/CMS/EcalLaserMonitoring";
  TString fStringTR(textTR);  //        = "MusEcal Version 2.0";
  TString fStringBR(textBR);  //        = "J. Malcles & G. Hamel de Monchenault, CEA-Saclay";

  double fXTexTL = 0.02;
  double fYTexTL = 0.975;
  double fXTexBL = 0.02;
  double fYTexBL = 0.01;
  double fXTexBR = 0.65;
  double fYTexBR = 0.01;
  double fXTexTR = 0.65;
  double fYTexTR = 0.975;

  Font_t fFontTex = 132;
  Float_t fSizeTex = 0.02;

  Short_t fBszCan = 4;
  Short_t fBszPad = 6;
  Width_t fWidthTex = 2;

  double fEmptyX = _scale * 0.005;
  double fEmptyY = _scale * 0.03;
  double wpad = _scale * 0.5 * (1.0 - 4 * fEmptyX);
  double fXLowPad1 = fEmptyX;
  double fXUpPad1 = fXLowPad1 + wpad;
  double fXLowPad2 = fXUpPad1 + 2 * fEmptyX;
  double fXUpPad2 = fXLowPad2 + wpad;
  double fYLowPad = fEmptyY;
  double fYUpPad = 1.0 - fEmptyY;

  fCanvas->SetEditable();
  fCanvas->Range(0, 0, 1, 1);
  fCanvas->SetFillColor(fColCan);
  fCanvas->SetFillStyle(fStyleCan);
  fCanvas->SetBorderSize(fBszCan);
  gStyle->SetOptStat(1111);
  gStyle->SetStatFont(22);
  gStyle->SetStatColor(18);
  if (fHeigthGen >= 700) {
    gStyle->SetStatH(0.1);
    gStyle->SetTitleSize(0.01);
  } else {
    gStyle->SetStatH(_scale * 0.15);
    gStyle->SetTitleSize(_scale * 0.015);
  }
  gStyle->SetTitleFillColor(fStyleTFColor);
  gStyle->SetTitleTextColor(fStyleTTextColor);
  gStyle->SetTitleW(_scale * 0.76);
  gStyle->SetHistFillColor(fStyleHistColor);
  gStyle->SetTitleFont(22);
  gStyle->SetTitleColor(173);
  gStyle->SetTitleFillColor(18);
  gStyle->SetTitleColor(fStyleTitleColor);
  gStyle->SetTitleTextColor(46);
  gStyle->SetLabelSize(_scale * 0.02, "XYZ");

  fTexTL = new TLatex(fXTexTL, fYTexTL, fStringTL.Data());
  fTexTR = new TLatex(fXTexTR, fYTexTR, fStringTR.Data());
  fTexBL = new TLatex(fXTexBL, fYTexBL, fStringBL.Data());
  fTexBR = new TLatex(fXTexBR, fYTexBR, fStringBR.Data());
  // from DrawLabels()
  //Draws the 4 labels on the Canvas
  fTexTL->SetTextFont(fFontTex);
  fTexTL->SetTextSize(fSizeTex);
  fTexTL->SetLineWidth(fWidthTex);
  fTexBL->SetTextFont(fFontTex);
  fTexBL->SetTextSize(fSizeTex);
  fTexBL->SetLineWidth(fWidthTex);
  fTexBR->SetTextFont(fFontTex);
  fTexBR->SetTextSize(fSizeTex);
  fTexBR->SetLineWidth(fWidthTex);
  fTexTR->SetTextFont(fFontTex);
  fTexTR->SetTextSize(fSizeTex);
  fTexTR->SetLineWidth(fWidthTex);
  fCanvas->cd();
  fTexTL->Draw();
  fTexBL->Draw();
  fTexBR->Draw();
  fTexTR->Draw();

  // from BookPad1()
  fCanvas->cd();
  fPad = new TPad("LeftPad", "Left pad", fXLowPad1, fYLowPad, fXUpPad2, fYUpPad);
  fPad->SetNumber(1);
  fPad->SetFillColor(fColPad1);
  fPad->SetFillStyle(fStylePad1);
  fPad->SetBorderSize(fBszPad);
  fPad->SetGridx();
  fPad->SetGridy();
  fPad->SetLogx(fPadLogX);
  fPad->SetLogy(fPadLogY);
  fPad->SetFrameFillColor(fColFrame);
  fPad->SetFillStyle(fStylePad1);
  fPad->Draw();
  fPad->cd();

  setSessionStyle();
  ShowWelcome(false);
}

void MECanvasHolder::ShowWelcome(bool image) {
  gPad->Clear();
  if (image) {
    TString imgpath = TString(std::getenv("MECONFIG"));
    TImage* img = TImage::Open(imgpath + "/LVB.jpg");
    assert(img != 0);
    //  img->SetConstRatio(0);
    TText* ttext1 = new TText(450, 500, "MusEcal");
    ttext1->SetTextSize(0.5);
    ttext1->SetTextColor(kRed);
    ttext1->SetTextFont(62);
    img->SetImageQuality(TAttImage::kImgBest);
    img->DrawText(ttext1, 450, 500);
    TText* ttext2 = new TText(450, 800, "ECAL Laser Monitoring");
    ttext2->SetTextSize(0.3);
    ttext2->SetTextColor(kRed);
    ttext2->SetTextFont(62);
    img->DrawText(ttext2, 450, 800);
    img->Draw("xxx");
    img->SetEditable(kTRUE);
  }
  gPad->Modified();
  gPad->Update();
}

void MECanvasHolder::setSessionStyle() {
  // use plain black on white colors
  gStyle->SetFrameBorderMode(0);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetPadColor(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetTitleBorderSize(0);
  //  gStyle->SetTitleColor(0);
  //  gStyle->SetStatColor(0);
  //  gStyle->SetFillColor(0);

  // set the paper & margin sizes
  gStyle->SetPaperSize(20, 26);
  //  gStyle->SetPadTopMargin(_scale*0.05);
  gStyle->SetPadTopMargin(_scale * 0.10);
  gStyle->SetPadRightMargin(_scale * 0.165);
  gStyle->SetPadBottomMargin(_scale * 0.15);
  gStyle->SetPadLeftMargin(_scale * 0.135);

  // use large Times-Roman fonts
  gStyle->SetTextFont(132);
  gStyle->SetTextSize(0.08);
  gStyle->SetLabelFont(132, "x");
  gStyle->SetLabelFont(132, "y");
  gStyle->SetLabelFont(132, "z");
  gStyle->SetTitleFont(132, "x");
  gStyle->SetTitleFont(132, "y");
  gStyle->SetTitleFont(132, "z");
  gStyle->SetTitleFont(132);
  gStyle->SetLabelSize(_scale * 0.05, "x");
  gStyle->SetLabelOffset(_scale * 0.01, "x");
  gStyle->SetTitleOffset(_scale * 1, "x");
  gStyle->SetTitleSize(_scale * 0.06, "x");
  gStyle->SetLabelSize(_scale * 0.05, "y");
  gStyle->SetLabelOffset(_scale * 0.01, "y");
  gStyle->SetTitleSize(_scale * 0.06, "y");
  gStyle->SetTitleOffset(_scale * 1, "y");
  gStyle->SetLabelSize(_scale * 0.05, "z");
  gStyle->SetLabelOffset(_scale * 0.01, "z");
  gStyle->SetTitleSize(_scale * 0.06, "z");
  gStyle->SetTitleOffset(_scale * 1, "z");

  // use bold lines and markers
  //  gStyle->SetMarkerStyle(8);
  gStyle->SetHistLineWidth(2);
  gStyle->SetLineStyleString(2, "[12 12]");  // postscript dashes

  // do not display any of the standard histogram decorations
  //  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(111111);
  gStyle->SetOptFit(0);

  // put tick marks on top and RHS of plots
  //  gStyle->SetPadTickX(1);
  //  gStyle->SetPadTickY(1);

  //  TColor::CreateColorWheel();

  gStyle->SetPalette(1);
  // setHessPalette();

  const int ncol = 60;
  gStyle->SetNumberContours(ncol);

  fCanvas->SetFillColor(kYellow);
  fCanvas->SetGridx(0);
  fCanvas->SetGridx(1);

  fPad->SetFillColor(kWhite);
  fPad->SetGridx(0);
  fPad->SetGridy(0);
  fPad->SetCrosshair(0);
  fPad->SetFrameFillColor(kWhite);
}

void MECanvasHolder::setHistoStyle(TH1* h) {
  if (h == 0)
    return;

  h->SetLineColor(4);
  h->SetLineWidth(1);
  h->SetFillColor(38);
  TAxis* axis[3];
  axis[0] = h->GetXaxis();
  axis[1] = h->GetYaxis();
  axis[2] = h->GetZaxis();
  for (int ii = 0; ii < 3; ii++) {
    TAxis* a = axis[ii];
    if (!a)
      continue;
    a->SetLabelFont(132);
    a->SetLabelOffset(0.005);
    a->SetLabelSize(0.04);
    a->SetTitleFont(132);
    a->SetTitleOffset(1);
    a->SetTitleSize(0.04);
  }
  h->SetStats(kTRUE);
}

void MECanvasHolder::SetDate() {
  // from F-X Gentit
  const Int_t cent = 100;
  Int_t date, time;
  Int_t day, month, year;
  Int_t hour, minute, second;
  TDatime td;
  date = td.GetDate();
  time = td.GetTime();
  //
  day = date % cent;
  date /= cent;
  month = date % cent;
  date /= cent;
  year = date;
  second = time % cent;
  time /= cent;
  minute = time % cent;
  time /= cent;
  hour = time;
  //
  fDate = "  ";
  fDate += day;
  fDate.Append(" / ");
  fDate += month;
  fDate.Append(" / ");
  fDate += year;
  fDate.Append("  ");
  //
  fTime = "";
  fTime += hour;
  fTime.Append('_');
  fTime += minute;
}

void MECanvasHolder::setPxAndPy(int px, int py) {
  _px = px;
  _py = py;
  _x = 0;
  _y = 0;
  if (_h != 0) {
    TString objectInfo;
    objectInfo = _h->GetObjectInfo(_px, _py);
    //      cout << "_px/_py/_h " << _px << "/" << _py  << endl;
    //      _h->Print();
    //      cout << objectInfo << endl;
    //
    // Yuk !!!
    //
    int istart1 = objectInfo.Index("(x=");
    int iend1 = istart1 + 3;
    int istart2 = objectInfo.Index(", y=");
    int iend2 = istart2 + 4;
    int istart3 = objectInfo.Index(", binx=");  // int iend3 = istart3+7;

    _x = TString(objectInfo(iend1, istart2 - iend1)).Atof();
    _y = TString(objectInfo(iend2, istart3 - iend2)).Atof();

    //      cout << "x/y " << _x << "/" << _y << endl;
  }
}

void MECanvasHolder::setPad() {
  if (fPad == 0)
    return;
  fPad->cd();
}

void MECanvasHolder::setHessPalette() {
  const int nfix = 5;

  const float Pi = acos(-1.);

  const int ninter = 10;
  int nstep = ninter + 1;
  double step = Pi / nstep;

  const int ncoltot = (nfix - 1) * ninter + nfix;

  TColor* myCol;
  Int_t palette[ncoltot];
  for (int i = 0; i < ncoltot; i++)
    palette[i] = 1;

  // 1:black, 4:blue, 2:red, 5:yellow, 10:white
  int colfix[nfix] = {1, 4, 2, 5, 10};

  int colOff7 = 4300;
  int icol = colOff7;  // new color number

  float red, green, blue;

  int ifix = 0;
  for (int ii = 0; ii < nfix; ii++) {
    TString myColName("myHessCol_");
    myColName += icol;
    TColor* theCol = (TColor*)gROOT->GetColor(colfix[ii]);
    theCol->GetRGB(red, green, blue);
    myCol = new TColor(icol, red, green, blue, myColName);
    cout << "ifix " << ifix << " r/g/b  " << red << "/" << green << "/" << blue << endl;
    palette[ifix] = icol++;
    ifix += nstep;
  }

  float r1, g1, b1;
  float r2, g2, b2;
  int ifix1 = 0;
  int ifix2 = 0;
  for (int ii = 0; ii < nfix - 1; ii++) {
    ifix2 = ifix1 + nstep;

    int icol1 = palette[ifix1];
    int icol2 = palette[ifix2];
    TColor* col1 = gROOT->GetColor(icol1);
    col1->Print();
    col1->GetRGB(r1, g1, b1);
    TColor* col2 = gROOT->GetColor(icol2);
    col2->Print();
    col2->GetRGB(r2, g2, b2);

    ifix = ifix1;
    double x = -Pi / 2.;
    for (int istep = 0; istep < ninter; istep++) {
      x += step;
      ifix++;

      double sinx = sin(x);
      red = 0.5 * ((r2 - r1) * sinx + (r1 + r2));
      green = 0.5 * ((g2 - g1) * sinx + (g1 + g2));
      blue = 0.5 * ((b2 - b1) * sinx + (b1 + b2));

      TString myColName("myHessCol_");
      myColName += icol;
      myCol = new TColor(icol, red, green, blue, myColName);
      cout << "ifix " << ifix << " r/g/b  " << red << "/" << green << "/" << blue << endl;
      palette[ifix] = icol++;
    }

    ifix1 = ifix2;
  }

  gStyle->SetPalette(ncoltot, palette);
}
