// Created 03-2016 by N. Smith <nick.smith@cern.ch>
#include "DQM/DQMRenderPlugin.h"
#include "TBox.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TFrame.h"
#include "TGaxis.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TText.h"
#include <cassert>
#include <iostream>

class L1T2016Layer1RenderPlugin : public DQMRenderPlugin
{
  TBox *exclusionBox_;
  TLine * correlation_;
  TText tlabels_;
  int palette_kry[256];
  int palette_yrk[256];

public:
  virtual void initialise(int, char **)
    {
      // For masking areas
      exclusionBox_ = new TBox();
      exclusionBox_->SetFillColor(kGray+2);
      exclusionBox_->SetFillStyle(3002);

      // For correlation eye guide
      correlation_ = new TLine();
      correlation_->SetLineStyle(kDashed);

      // For tower ieta,iphi labels
      tlabels_.SetTextFont(40);
      tlabels_.SetTextAlign(22);
      tlabels_.SetTextSize(0.03);

      // Laugh all you want, but they do look pretty
      // http://arxiv.org/pdf/1509.03700v1.pdf
      // linear_kry_5-98_c75_n256 reversed
      const float rValues[256] = {0.066001,0.080538,0.093113,0.10433,0.11447,0.12392,0.13324,0.1425,0.1517,0.16087,0.16995,0.17892,0.18783,0.19664,0.20537,0.21407,0.2227,0.23128,0.23978,0.24826,0.25665,0.265,0.2733,0.28153,0.28972,0.29783,0.3059,0.31386,0.32176,0.32961,0.33736,0.34502,0.35258,0.36003,0.36736,0.37459,0.38167,0.38862,0.39539,0.40197,0.40837,0.41456,0.42062,0.42666,0.4327,0.43874,0.44481,0.45086,0.45693,0.463,0.4691,0.47519,0.48128,0.48739,0.49349,0.49961,0.50573,0.51187,0.51802,0.52416,0.53031,0.53647,0.54263,0.54881,0.55498,0.56116,0.56735,0.57355,0.57975,0.58596,0.59219,0.59842,0.60468,0.61096,0.61724,0.62356,0.62988,0.63622,0.64258,0.64896,0.65535,0.66176,0.66819,0.67463,0.68108,0.68756,0.69405,0.70057,0.7071,0.71364,0.7202,0.72678,0.73337,0.73999,0.74661,0.75325,0.75991,0.76657,0.77322,0.77985,0.78648,0.79309,0.79969,0.80627,0.81283,0.81939,0.82593,0.83246,0.83897,0.84547,0.85195,0.85842,0.86487,0.87132,0.87774,0.88415,0.89054,0.89692,0.90328,0.90963,0.91595,0.92227,0.92856,0.93484,0.94104,0.9471,0.95296,0.95862,0.96409,0.96937,0.97446,0.97936,0.98405,0.98853,0.99282,0.9969,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
      const float gValues[256] = {0.066024,0.066611,0.067208,0.067823,0.068466,0.069079,0.069387,0.069367,0.069013,0.068361,0.067575,0.066658,0.065607,0.064412,0.063105,0.061489,0.059924,0.057935,0.055984,0.053694,0.051268,0.048696,0.045729,0.042641,0.039309,0.035771,0.031918,0.028215,0.024452,0.020653,0.016844,0.013048,0.0091504,0.005529,0.0020418,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0017832,0.0061612,0.010887,0.016478,0.023267,0.031841,0.042415,0.053281,0.064278,0.075188,0.086066,0.096997,0.10794,0.11888,0.12973,0.14054,0.1513,0.1621,0.17286,0.18361,0.19437,0.20505,0.21574,0.22642,0.23711,0.24776,0.25838,0.26903,0.27963,0.2902,0.30076,0.31115,0.32126,0.33113,0.34081,0.35027,0.35954,0.36866,0.3776,0.3864,0.39508,0.40364,0.41207,0.42039,0.42862,0.43675,0.44479,0.45274,0.4606,0.4684,0.47614,0.48379,0.4914,0.49892,0.5064,0.51383,0.52119,0.52852,0.53577,0.54298,0.55012,0.55721,0.56423,0.57119,0.57811,0.58499,0.59181,0.59857,0.6053,0.61199,0.61864,0.62524,0.6318,0.63833,0.64482,0.65128,0.6577,0.66409,0.67045,0.67678,0.68308,0.68935,0.69558,0.70178,0.70798,0.71415,0.72028,0.72637,0.73244,0.73847,0.74448,0.75047,0.75643,0.76237,0.7683,0.77421,0.78011,0.786,0.79186,0.79771,0.80356,0.80939,0.81521,0.82102,0.82682,0.83261,0.83839,0.84416,0.84993,0.85569,0.86143,0.86717,0.87291,0.87864,0.88437,0.89009,0.8958,0.90151,0.90721,0.91291,0.9186,0.92429,0.92997,0.93565,0.94133,0.947,0.95267,0.95834,0.964,0.96966,0.97532,0.98098,0.98663,0.99228,0.99792};
      const float bValues[256] = {0.066019,0.064712,0.063412,0.062042,0.060693,0.059406,0.057942,0.056714,0.055346,0.05402,0.052699,0.051395,0.050099,0.04882,0.047405,0.046005,0.044746,0.043254,0.042035,0.040618,0.039228,0.03786,0.036506,0.03519,0.033667,0.032419,0.031169,0.029947,0.028752,0.027583,0.026442,0.025331,0.024255,0.023213,0.022204,0.021232,0.020297,0.019403,0.018552,0.017747,0.016994,0.016295,0.015634,0.014992,0.014366,0.013752,0.013144,0.012574,0.012021,0.011365,0.010683,0.010135,0.0096387,0.0091289,0.0086244,0.0081373,0.0076619,0.0071948,0.0067375,0.0062922,0.0058577,0.0054319,0.0050145,0.0046053,0.0042044,0.0038134,0.0034314,0.0030565,0.0026882,0.0023271,0.0019742,0.0016322,0.001301,0.00097818,0.00066352,0.00035687,5.8077e-05,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.00010807,0.0016338,0.0031726,0.0047225,0.0062803,0.0078416,0.0094129,0.011021,0.012734,0.014303,0.015864,0.017417,0.018959,0.020491,0.022011,0.023517,0.025006,0.026472,0.027913,0.02933,0.030724,0.032109,0.033495,0.035116,0.036562,0.038032,0.039515,0.041015,0.042457,0.043871,0.045349,0.046804,0.048174,0.049638,0.050967,0.052314,0.053669,0.05503,0.056416,0.057622,0.05901,0.060291,0.061456,0.062815,0.063988,0.065188,0.06638,0.067574,0.068771,0.069853,0.07091,0.071928,0.072954,0.073945,0.074848,0.075762,0.076661,0.077556,0.078448,0.079336,0.080219,0.081101,0.081992,0.082776,0.083622,0.084477,0.085237,0.086035,0.086929,0.087733,0.088519,0.089324,0.090123,0.090933,0.091749,0.092489,0.093256,0.094098,0.094916,0.095666,0.096403,0.097221,0.098051,0.098807,0.099584,0.10036,0.10115,0.10194,0.10274,0.10354,0.10428,0.10506,0.10589,0.1067,0.10746,0.10821,0.10903,0.10986};
      for(size_t i=0; i<256; ++i) {
        auto c = new TColor(rValues[i], gValues[i], bValues[i]);
        palette_kry[i] = c->GetNumber();
        palette_yrk[255-i] = c->GetNumber();
      }

    }

  virtual bool applies( const VisDQMObject &o, const VisDQMImgInfo & )
    {
      if( o.name.find( "L1TStage2CaloLayer1" ) != std::string::npos )
        return true;
      // For a brief time (March-April 2016) this module was in different location
      if( o.name.find( "L1T/L1TLayer1" ) != std::string::npos )
        return true;
      // Also process data/emulator
      if( o.name.find( "L1TdeStage2CaloLayer1" ) != std::string::npos )
        return true;

      return false;
    }

  virtual void preDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &i, VisDQMRenderInfo &r )
    {
      c->cd();

      gStyle->Reset("Default");

      gStyle->SetCanvasColor(kWhite);
      gStyle->SetPadColor(kWhite);
      gStyle->SetFillColor(kWhite);
      gStyle->SetStatColor(kWhite);
      gStyle->SetTitleFillColor(kWhite);

      gStyle->SetOptTitle(kTRUE);
      gStyle->SetTitleBorderSize(0);

      // This tends to need reaffirming in postDraw
      gStyle->SetOptStat(11);
      gStyle->SetStatBorderSize(1);

      // Stash away our drawOptions for postDraw
      c->SetTitle(r.drawOptions.c_str());

      if( dynamic_cast<TH2F*>( o.object ) )
      {
        preDrawTH2F( c, o, r );
      }
      else if( dynamic_cast<TH1F*>( o.object ) )
      {
        preDrawTH1F( c, o, i, r );
      }
    }

  virtual void postDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &i )
    {
      c->cd();

      // Unstash the draw option
      std::string drawOptionString = c->GetTitle();
      c->SetTitle("");

      if( dynamic_cast<TH2F*>( o.object ) )
      {
        postDrawTH2F( c, o, drawOptionString );
      }
      else if( dynamic_cast<TH1F*>( o.object ) )
      {
        postDrawTH1F( c, o, i);
      }
    }

private:
  bool checkAndRemove(std::string &s, const char * key)
  {
    if( s.find(key) != std::string::npos )
    {
      s.replace(s.find(key), strlen(key), "");
      return true;
    }
    return false;
  }

  void drawExclusionBox(double x1, double y1, double x2, double y2)
  {
    // Clamp in case we are zoomed
    if ( x1 < gPad->GetFrame()->GetX1() ) x1 = gPad->GetFrame()->GetX1();
    if ( x2 > gPad->GetFrame()->GetX2() ) x2 = gPad->GetFrame()->GetX2();
    if ( y1 < gPad->GetFrame()->GetY1() ) y1 = gPad->GetFrame()->GetY1();
    if ( y2 > gPad->GetFrame()->GetY2() ) y2 = gPad->GetFrame()->GetY2();
    if ( y1 < y2 && x1 < x2 )
      exclusionBox_->DrawBox(x1, y1, x2, y2);
  }

  void preDrawTH2F( TCanvas *c, const VisDQMObject &o, VisDQMRenderInfo &r )
    {
      TH2F* obj = dynamic_cast<TH2F*>( o.object );
      assert( obj );

      // Remove any postDraw options we might use
      // so that ROOT does not misinterpret them somehow
      checkAndRemove(r.drawOptions, "tlabels");

      // Colormap setup
      if( checkAndRemove(r.drawOptions, "hottower") || checkAndRemove(r.drawOptions, "kry"))
      {
        // Choose hot tower palette
        obj->SetContour(256);
        gStyle->SetPalette(256, palette_kry);
        c->SetFrameFillColor(kBlack);
        obj->GetXaxis()->SetAxisColor(kWhite);
        obj->GetYaxis()->SetAxisColor(kWhite);
        tlabels_.SetTextColor(kWhite);
      }
      else if( checkAndRemove(r.drawOptions, "yrk") )
      {
        // Choose the Inverted Dark Body Radiator palette
        obj->SetContour(256);
        gStyle->SetPalette(256, palette_yrk);
      }
      else
      {
        // Set up default palette
        obj->SetContour(100);
        gStyle->SetPalette(1);
        tlabels_.SetTextColor(kBlack);
      }

      std::string name = o.name.substr(o.name.rfind("/")+1);
      if( name.find( "Occ" ) != std::string::npos
          || name.find( "calDiscrepancy" ) != std::string::npos
          || name.find( "calLinkError" ) != std::string::npos
        )
      {
        // Set one level of divisions for axes
        obj->GetXaxis()->SetNdivisions(10);
        obj->GetYaxis()->SetNdivisions(10);
        if ( r.drawOptions.length() == 0 ) {
          obj->SetOption("colz");
        }
        else if ( checkAndRemove(r.drawOptions, "box") )
        {
          obj->SetOption("box");
        }
      }
      else if ( name.find( "Mismatches" ) != std::string::npos )
      {
        c->SetLeftMargin(0.2);
        c->SetBottomMargin(0.15);
        if ( r.drawOptions.length() == 0 ) {
          obj->SetOption("col");
        }
      }
      else if ( name.find( "Correlation" ) != std::string::npos )
      {
        if ( r.drawOptions.length() == 0 ) {
          obj->SetOption("colz");
        }
      }
    }

  void preDrawTH1F( TCanvas *, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo &r )
    {
      TH1F* obj = dynamic_cast<TH1F*>( o.object );
      assert( obj );

      std::string name = o.name.substr(o.name.rfind("/")+1);

      if( name.find( "ByLumi" ) != std::string::npos )
      {
        obj->SetFillColor(kRed);
      }
      else if ( name.find( "PerBx" ) != std::string::npos )
      {
        obj->SetFillColor(kRed);
      }
      if ( r.drawOptions.length() == 0 ) {
        obj->SetOption("hist");
      }
      if( name.find( "dataEmulSummary" ) != std::string::npos )
      {
        obj->SetFillColor(kRed);
        obj->SetOption("texthist");
      }
    }

  void postDrawTH2F( TCanvas * c, const VisDQMObject &o, std::string &drawOptions )
    {
      TH2F* obj = dynamic_cast<TH2F*>( o.object );
      assert( obj );

      gStyle->SetOptStat(11);

      std::string name = o.name.substr(o.name.rfind("/")+1);

      if( name.find( "ecalOcc" ) != std::string::npos )
      {
        drawExclusionBox(-.5, 0.5, .5, 72.5);
      }
      else if( name.find( "hcalOcc" ) != std::string::npos )
      {
        drawExclusionBox(-.5, 0.5, .5, 72.5);
        drawExclusionBox(-29.5, 0.5, -28.5, 72.5);
        drawExclusionBox(28.5, 0.5, 29.5, 72.5);
        // HF granularity reduction
        for (int i=2; i<=72; i+=2) {
          drawExclusionBox(28.5, i-0.5, 41.5, i+0.5);
          drawExclusionBox(-41.5, i-0.5, -28.5, i+0.5);
        }
        for (int i=1; i<=72; i+=4) {
          drawExclusionBox(39.5, i-0.5, 41.5, i+0.5);
          drawExclusionBox(-41.5, i-0.5, -39.5, i+0.5);
        }
      }

      if ( name.find( "ecal" ) != std::string::npos && checkAndRemove(drawOptions, "tlabels") )
      {
        char text[20];
        for (int ieta=-28; ieta<29; ++ieta) {
          if ( ieta == 0 ) continue;
          for (int iphi=1; iphi<=72; ++iphi) {
            if ( ieta < c->GetFrame()->GetX1() || ieta > c->GetFrame()->GetX2() || iphi < c->GetFrame()->GetY1() || iphi > c->GetFrame()->GetY2() ) continue;
            sprintf(text, "% d,%d", ieta, iphi);
            tlabels_.DrawText(ieta, iphi, text);
          }
        }
      }
      else if ( (name.find( "hcal" ) != std::string::npos or o.name.find( "L1TdeStage2CaloLayer1/Occupancies" ) != std::string::npos) && checkAndRemove(drawOptions, "tlabels") )
      {
        char text[20];
        for (int ieta=-42; ieta<42; ++ieta) {
          if ( ieta == 0 || abs(ieta) == 29 ) continue;
          for (int iphi=1; iphi<=72; ++iphi) {
            if ( abs(ieta) > 29 && iphi%2==0 ) continue;
            if ( abs(ieta) > 39 && iphi%4!=3 ) continue;
            if ( ieta < c->GetFrame()->GetX1() || ieta > c->GetFrame()->GetX2() || iphi < c->GetFrame()->GetY1() || iphi > c->GetFrame()->GetY2() ) continue;
            sprintf(text, "% d,%d", ieta, iphi);
            tlabels_.DrawText(ieta, iphi, text);
          }
        }
      }
      else if ( name.find( "Correlation" ) != std::string::npos )
      {
        correlation_->DrawLine(0.,0.,512.,512.);
      }
      else if ( name.find( "Mismatches" ) != std::string::npos )
      {
        gStyle->SetOptStat(0);
      }

    }

  void postDrawTH1F( TCanvas *, const VisDQMObject &o, const VisDQMImgInfo &i )
    {
      TH1F* obj = dynamic_cast<TH1F*>( o.object );
      assert( obj );

      std::string name = o.name.substr(o.name.rfind("/")+1);

      if( name.find( "ByLumi" ) != std::string::npos )
      {
        if ( isnan(i.xaxis.min) && isnan(i.xaxis.max) )
        {
          double currentLumi = obj->GetBinContent(0);
          if ( currentLumi != 0. ) obj->GetXaxis()->SetRangeUser(1, currentLumi+2);
        }
        // (if NaN, compares false anyway)
        else if ( i.xaxis.min < 1. ) {
          obj->GetXaxis()->SetRangeUser(1., i.xaxis.max);
        }
        // To account for our underflow bin hack
        obj->SetEntries(obj->Integral(1, -1));
        gStyle->SetOptStat(11);
      }
      if( o.name.find( "maxEvt" ) != std::string::npos && o.name.find("MismatchDetail") == std::string::npos )
      {
        if ( isnan(i.yaxis.max) )
        {
          // Setting min to 0 breaks log scale unless user sets min>0
          // Not sure how to let ROOT choose min
          auto miny = (isnan(i.yaxis.min)) ? 0. : i.yaxis.min;
          obj->GetYaxis()->SetRangeUser(miny, 8856);
        }
      }
      if( name.find( "dataEmulSummary" ) != std::string::npos )
      {
        gStyle->SetOptStat(11);
        if ( isnan(i.yaxis.max) )
        {
          auto miny = (isnan(i.yaxis.min)) ? 0. : i.yaxis.min;
          obj->GetYaxis()->SetRangeUser(miny, 0.1);
        }
      }
    }
};

static L1T2016Layer1RenderPlugin instance;
