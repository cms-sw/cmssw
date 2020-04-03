/*
  \File BeamPixelRenderPlugin
  \Display Plugin for BeamSpot from Pixel-Vertices
  \author Mauro Dinardo
  \version $ Revision: 4.0 $
  \date $ Date: 2015/17/08 23:00:00 $
*/

#include "DQM/DQMRenderPlugin.h"
#include "utils.h"

#include "TROOT.h"
#include "TProfile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TText.h"
#include "TPaveStats.h"
#include "TGaxis.h"
#include <cassert>
#include <math.h>

class BeamPixelRenderPlugin : public DQMRenderPlugin {

  int hcalRainbowColors[100];
  int NCont_rainbow;

public:

 virtual void initialise (int, char **)
  {
    // Make rainbow colors. Assign colors positions 1101-1200
    NCont_rainbow = 100; // Specify number of contours for rainbow

    double stops_rainbow[] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
    double red_rainbow[]   = { 0.00, 0.00, 0.87, 1.00, 0.91 };
    double green_rainbow[] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
    double blue_rainbow[]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
    int NRGBs_rainbow      = 5; // Specify number of RGB boundaries for rainbow
    int nColorsGradient    = 0;
    int highestIndex       = 0;

    for (int g = 1; g < NRGBs_rainbow; g++)
      {
        nColorsGradient = (int) (floor(NCont_rainbow * stops_rainbow[g]) - floor(NCont_rainbow * stops_rainbow[g-1])); // Specify number of gradients between stops (g-1) and (g)
        for (int c = 0; c < nColorsGradient; c++)
	  {
	    hcalRainbowColors[highestIndex] = 1101 + highestIndex;
	    TColor* color = gROOT->GetColor(1101 + highestIndex);
	    // Make new color only if old color does not exist
	    if (!color)
	      color = new TColor(1101 + highestIndex,
				 red_rainbow[g-1]   + c * (red_rainbow[g]   - red_rainbow[g-1])   / nColorsGradient,
				 green_rainbow[g-1] + c * (green_rainbow[g] - green_rainbow[g-1]) / nColorsGradient,
				 blue_rainbow[g-1]  + c * (blue_rainbow[g]  - blue_rainbow[g-1])  / nColorsGradient,
				 "  ");
	    highestIndex++;
	  }
      }
  }
  
  virtual bool applies (const VisDQMObject& o, const VisDQMImgInfo& )
  {
    if (o.name.find("BeamPixel/") != std::string::npos) return true;

    return false;
  }

  virtual void preDraw (TCanvas* c, const VisDQMObject& o, const VisDQMImgInfo& , VisDQMRenderInfo& )
  {
    c->cd();

    if (dynamic_cast<TH2F*>(o.object))     preDrawTH2F(c, o);
    if (dynamic_cast<TH1F*>(o.object))     preDrawTH1F(c, o);
    if (dynamic_cast<TProfile*>(o.object)) preDrawTProfile(c, o);
  }

  virtual void postDraw (TCanvas* c, const VisDQMObject& o, const VisDQMImgInfo& )
  {
    c->cd();
    
    if (dynamic_cast<TH2F*>(o.object))     postDrawTH2F(c, o);
    if (dynamic_cast<TH1F*>(o.object))     postDrawTH1F(c, o);
    if (dynamic_cast<TProfile*>(o.object)) postDrawTProfile(c, o);
  }

private:

  void preDrawTH2F (TCanvas* c, const VisDQMObject& o)
  {
    TH2F* obj = dynamic_cast<TH2F*>(o.object);
    assert(obj);

    if ((o.name.find("E - vertex zx") != std::string::npos) || (o.name.find("E - vertex zy") != std::string::npos) || (o.name.find("E - vertex xy") != std::string::npos) ||
	(o.name.find("H - vertex zx cum") != std::string::npos) || (o.name.find("H - vertex zy cum") != std::string::npos) || (o.name.find("H - vertex xy cum") != std::string::npos))
      {
	c->SetGrid();
	c->SetRightMargin(0.12);

	obj->SetContour(NCont_rainbow);
	obj->SetStats(false);
	obj->SetOption("colz");

	return;
      }

    if (o.name.find("A - fit results") != std::string::npos)
      {
	c->SetGrid();

	obj->SetStats(false);
	obj->SetMarkerSize(2.);

	return;
      }

    if (o.name.find("reportSummaryMap") != std::string::npos)
      {
	c->SetGrid(false, false);

	obj->GetXaxis()->SetLabelSize(0);
	obj->GetXaxis()->SetTickLength(0);

	obj->GetYaxis()->SetLabelSize(0);
	obj->GetYaxis()->SetTickLength(0);

	obj->SetStats(false);

	return;
      }
  }

  void preDrawTH1F (TCanvas* c, const VisDQMObject& o)
  {
    TH1F* obj = dynamic_cast<TH1F*>(o.object);
    assert(obj);

    c->SetGrid(false, true);
    c->SetTopMargin(0.12);
    
    obj->SetMarkerStyle(20);
    obj->SetMarkerColor(4);
    
    if ((o.name.find("F - vertex x") != std::string::npos) || (o.name.find("F - vertex y") != std::string::npos) || (o.name.find("F - vertex z") != std::string::npos) ||
	(o.name.find("G - vertex x fit") != std::string::npos) || (o.name.find("G - vertex y fit") != std::string::npos) || (o.name.find("G - vertex z fit") != std::string::npos) ||
	(o.name.find("I - vertex x cum") != std::string::npos) || (o.name.find("I - vertex y cum") != std::string::npos) || (o.name.find("I - vertex z cum") != std::string::npos) ||
	(o.name.find("L - status vs lumi") != std::string::npos))
      c->SetGrid(false, false);

    if ((o.name.find("B - muX vs lumi") != std::string::npos) || (o.name.find("B - muY vs lumi") != std::string::npos) || (o.name.find("B - muZ vs lumi") != std::string::npos) ||
	(o.name.find("C - sigmaX vs lumi") != std::string::npos) || (o.name.find("C - sigmaY vs lumi") != std::string::npos) || (o.name.find("C - sigmaZ vs lumi") != std::string::npos) ||
	(o.name.find("D - dxdz vs lumi") != std::string::npos) || (o.name.find("D - dydz vs lumi") != std::string::npos) ||
	(o.name.find("G - vertex x fit") != std::string::npos) || (o.name.find("G - vertex y fit") != std::string::npos) || (o.name.find("G - vertex z fit") != std::string::npos) ||
	(o.name.find("J - pixelHits vs lumi") != std::string::npos) || (o.name.find("K - good vertices vs lumi") != std::string::npos) ||
	(o.name.find("L - status vs lumi") != std::string::npos))
      {
	obj->SetMarkerSize(0.5);

	return;
      }
  }

  void preDrawTProfile (TCanvas* c, const VisDQMObject& o)
  {
    TProfile* obj = dynamic_cast<TProfile*>(o.object);
    assert(obj);

    c->SetGrid();
  }

  void postDrawTH2F (TCanvas* c, const VisDQMObject& o)
  {
    TH2F* obj = dynamic_cast<TH2F*>(o.object);
    assert(obj);

    TGaxis::SetMaxDigits(3);

    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetPadBorderSize(0);

    TAxis* xa = obj->GetXaxis();
    TAxis* ya = obj->GetYaxis();

    if ((o.name.find("E - vertex zx") != std::string::npos) || (o.name.find("E - vertex zy") != std::string::npos) || (o.name.find("E - vertex xy") != std::string::npos) ||
	(o.name.find("H - vertex zx cum") != std::string::npos) || (o.name.find("H - vertex zy cum") != std::string::npos) || (o.name.find("H - vertex xy cum") != std::string::npos))
      {
	xa->SetTitleOffset(1.15);
	ya->SetTitleOffset(1.15);
	
	xa->SetTitleSize(0.04);
	ya->SetTitleSize(0.04);
	
	xa->SetLabelSize(0.04);
	ya->SetLabelSize(0.04);
	
	c->SetGrid();

	gStyle->SetPalette(NCont_rainbow, hcalRainbowColors);
	
	return;
      }

    if (o.name.find("A - fit results") != std::string::npos)
      {
	xa->SetTitleOffset(0.85);
	ya->SetTitleOffset(1.15);
	
	xa->SetTitleSize(0.055);
	ya->SetTitleSize(0.04);

	xa->SetLabelSize(0.06);
	ya->SetLabelSize(0.06);

        c->SetGrid();

        return;
      }

    if (o.name.find("reportSummaryMap") != std::string::npos)
      {
        c->SetGrid(false, false);
        TH2F* obj2 = dynamic_cast<TH2F*>(o.object);
        dqm::utils::reportSummaryMapPalette(obj2);
        return;
      }
  }

  void postDrawTH1F (TCanvas* c, const VisDQMObject& o)
  {
    TH1F* obj = dynamic_cast<TH1F*>(o.object);
    assert(obj);

    TGaxis::SetMaxDigits(3);

    c->SetGrid(false, true);

    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetPadBorderSize(0);

    gStyle->SetOptStat(111110);

    TAxis* xa = obj->GetXaxis();
    TAxis* ya = obj->GetYaxis();

    xa->SetTitleOffset(1.15);
    ya->SetTitleOffset(1.15);

    xa->SetTitleSize(0.04);
    ya->SetTitleSize(0.04);

    xa->SetLabelSize(0.04);
    ya->SetLabelSize(0.04);

    if (o.name.find("L - status vs lumi") != std::string::npos)
      {
	c->SetGrid(false, false);

	ya->SetRangeUser(-5.5,5.5);

	gStyle->SetOptStat(10);
	gStyle->SetEndErrorSize(0.);

	TPaveStats* st = (TPaveStats*)obj->FindObject("stats");
	st->SetY1NDC(0.9);

	return;
      }
    
    if ((o.name.find("F - vertex x") != std::string::npos) || (o.name.find("F - vertex y") != std::string::npos) || (o.name.find("F - vertex z") != std::string::npos) ||
	(o.name.find("G - vertex x fit") != std::string::npos) || (o.name.find("G - vertex y fit") != std::string::npos) || (o.name.find("G - vertex z fit") != std::string::npos) ||
	(o.name.find("I - vertex x cum") != std::string::npos) || (o.name.find("I - vertex y cum") != std::string::npos) || (o.name.find("I - vertex z cum") != std::string::npos))
      c->SetGrid(false, false);

    if ((o.name.find("B - muX vs lumi") != std::string::npos) || (o.name.find("B - muY vs lumi") != std::string::npos) || (o.name.find("B - muZ vs lumi") != std::string::npos) ||
	(o.name.find("C - sigmaX vs lumi") != std::string::npos) || (o.name.find("C - sigmaY vs lumi") != std::string::npos) || (o.name.find("C - sigmaZ vs lumi") != std::string::npos) ||
	(o.name.find("D - dxdz vs lumi") != std::string::npos) || (o.name.find("D - dydz vs lumi") != std::string::npos) ||
	(o.name.find("G - vertex x fit") != std::string::npos) || (o.name.find("G - vertex y fit") != std::string::npos) || (o.name.find("G - vertex z fit") != std::string::npos) ||
	(o.name.find("J - pixelHits vs lumi") != std::string::npos) || (o.name.find("K - good vertices vs lumi") != std::string::npos))
      {
	gStyle->SetOptFit(1110);
	gStyle->SetOptStat(10);
	
 	gStyle->SetErrorX(0.);
	gStyle->SetEndErrorSize(0.);
	
	return;
      }
  }

  void postDrawTProfile (TCanvas* c, const VisDQMObject& o)
  {
    TProfile* obj = dynamic_cast<TProfile*>(o.object);
    assert(obj);

    TGaxis::SetMaxDigits(3);

    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetPadBorderSize(0);

    gStyle->SetOptStat(1110);

    TAxis* xa = obj->GetXaxis();
    TAxis* ya = obj->GetYaxis();

    xa->SetTitleOffset(1.15);
    ya->SetTitleOffset(1.15);

    xa->SetTitleSize(0.04);
    ya->SetTitleSize(0.04);

    xa->SetLabelSize(0.04);
    ya->SetLabelSize(0.04);

    c->SetGrid();
  }

};

static BeamPixelRenderPlugin instance;
