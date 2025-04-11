#ifndef PLOT_AUXILIARY_H
#define PLOT_AUXILIARY_H

#include "TCanvas.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TLine.h"

TCanvas * create_canvas(bool setOptStat=0)
{
 TCanvas* canv = new TCanvas("", "", 700, 600);
 gStyle->SetOptTitle(0);
 //gROOT->SetBatch( 1 );
 gStyle->SetOptStat(setOptStat);
 //gErrorIgnoreLevel = kWarning;
 return canv;
}

void formatLegend(TLegend* leg, double textsize=27)
{
   leg->SetBorderSize(0);
   leg->SetTextFont(43);
   leg->SetTextSize(textsize);
   leg->SetFillStyle(0);
   leg->SetFillColor(0);
   leg->SetLineColor(0);
}

TPad* create_Pad(const float& y1=0, const float& y2=1, const float& topmargin=0.055, const float& leftmargin=0.12, const float& bottommargin=0.03, const float& rightmargin=0.12, const bool& logy=true) {

  TPad * topPad = new TPad("topPad", "topPad", 0.00, y1, 1.00, y2);
  topPad->SetFillColor(10);
  topPad->SetLogy(logy);
  return topPad;
}

#endif //PLOT_AUXILIARY
