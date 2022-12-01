#ifndef _ALIGNMENT_OFFLINEVALIDATION_CMS_LUMI_H_
#define _ALIGNMENT_OFFLINEVALIDATION_CMS_LUMI_H_

#include "TPad.h"
#include "TLatex.h"
#include "TLine.h"
#include "TBox.h"
#include <iostream>

//
// Global variables
//

TString cmsText = "CMS";
float cmsTextFont = 61;  // default is helvetic-bold

bool writeExtraText = false;
TString extraText = "Preliminary";
float extraTextFont = 52;  // default is helvetica-italics

// text sizes and text offsets with respect to the top frame
// in unit of the top margin size
float lumiTextSize = 0.6;
float lumiTextOffset = 0.2;
float cmsTextSize = 0.75;
float cmsTextOffset = 0.1;  // only used in outOfFrame version

float relPosX = 0.045;
float relPosY = 0.035;
float relExtraDY = 1.2;

// ratio of "CMS" and extra text size
float extraOverCmsTextSize = 0.76;

TString lumi_13TeV = "20.1 fb^{-1}";
TString lumi_8TeV = "19.7 fb^{-1}";
TString lumi_7TeV = "5.1 fb^{-1}";
TString lumi_0p9TeV = "";
TString lumi_13p6TeV = "";
TString lumi_sqrtS = "";
bool writeExraLumi = false;
bool drawLogo = false;

void CMS_lumi(TPad* pad, int iPeriod = 3, int iPosX = 10);

inline void CMS_lumi(TPad* pad, int iPeriod, int iPosX) {
  bool outOfFrame = false;
  if (iPosX / 10 == 0) {
    outOfFrame = true;
  }
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
    if (writeExraLumi)
      lumiText += " (7 TeV)";
  } else if (iPeriod == 2) {
    lumiText += lumi_8TeV;
    if (writeExraLumi)
      lumiText += " (8 TeV)";
  } else if (iPeriod == 3) {
    lumiText = lumi_8TeV;
    if (writeExraLumi) {
      lumiText += " (8 TeV)";
      lumiText += " + ";
      lumiText += lumi_7TeV;
      lumiText += " (7 TeV)";
    }
  } else if (iPeriod == 4) {
    lumiText += lumi_13TeV;
    if (writeExraLumi)
      lumiText += " (#sqrt{s} = 13 TeV)";
  } else if (iPeriod == 5) {
    lumiText += lumi_0p9TeV;
    if (writeExraLumi)
      lumiText += " (#sqrt{s} = 0.9 TeV)";
  } else if (iPeriod == 6) {
    lumiText += lumi_13p6TeV;
    if (writeExraLumi)
      lumiText += " (#sqrt{s} = 13.6 TeV)";
  } else if (iPeriod == 7) {
    if (outOfFrame)
      lumiText += "#scale[0.85]{";
    lumiText += lumi_13TeV;
    if (writeExraLumi) {
      lumiText += " (13 TeV)";
      lumiText += " + ";
      lumiText += lumi_8TeV;
      lumiText += " (8 TeV)";
      lumiText += " + ";
      lumiText += lumi_7TeV;
      lumiText += " (7 TeV)";
    }
    if (outOfFrame)
      lumiText += "}";
  } else if (iPeriod == 12) {
    if (writeExraLumi)
      lumiText += "8 TeV";
  } else if (iPeriod == 0) {
    if (writeExraLumi)
      lumiText += lumi_sqrtS;
  }

  std::cout << lumiText << std::endl;

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
      TPad* pad_logo = new TPad("logo", "logo", xl_0, yl_0, xl_1, yl_1);
      pad_logo->Draw();
      pad_logo->cd();
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

#endif
