/*!
  \file DAQRenderPlugin.cc

*/

#include "DQMServices/DQMGUI/interface/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile2D.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TText.h"
#include "TPRegexp.h"
#include <cassert>

#define REMATCH(pat, str) (TPRegexp(pat).MatchB(str))

class DAQRenderPlugin : public DQMRenderPlugin
{
  Double_t contour_level[25];
public:
  virtual void initialise (int, char **)
  {
    Double_t contour_level_t[25]  = {0.02,0.025,0.03,0.032,0.036,0.040,0.044,0.048,0.052,0.056,0.060,
	                           0.065,0.070,0.075,0.085,0.090,0.095,0.12,0.25,0.4,0.5,0.6,0.7,0.8,0.9};
    for (int i=0;i<20;i++) contour_level[i]=contour_level_t[i];
  }

  virtual bool applies(const VisDQMObject &o, const VisDQMImgInfo &)
    {
      // determine whether core object is an DAQ object. DAQ objects start with "DAQ/", "DAQval/" or "DAQdev/"
      const size_t beginning = 0;
      if (o.name.find( "DAQ/" ) == beginning
          || o.name.find("DAQval/") == beginning
          || o.name.find("DAQdev/") == beginning
      )
        return true;

      return false;
    }

  virtual void preDraw (TCanvas * c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo &r)
    {
      c->cd();

      // object is TH1 histogram
      if( dynamic_cast<TH1F*>( o.object ) )
      {
        preDrawTH1F( c, o );
	//assert(r);
	r.drawOptions="HIST";
      }
      if( dynamic_cast<TH2F*>( o.object ) )
      {
        preDrawTH2F( c, o );
      }
    }

  virtual void postDraw (TCanvas * c, const VisDQMObject &o, const VisDQMImgInfo &)
    {
      // object is TH2 histogram
      if( dynamic_cast<TH2F*>( o.object ) )
      {
        postDrawTH2F( c, o );
      }
      // object is TH1 histogram
      else if( dynamic_cast<TH1F*>( o.object ) )
      {
        postDrawTH1F( c, o );
      }
    }

private:
  void preDrawTH1F ( TCanvas *, const VisDQMObject &o )
    {
      // Do we want to do anything special yet with TH1F histograms?
      TH1F* obj = dynamic_cast<TH1F*>( o.object );
      assert (obj); // checks that object indeed exists

      // rate histograms
      if ( 1 )
      {
        gStyle->SetOptStat(11);
        obj->GetXaxis()->SetTitle("Luminosity Section");
        if ( o.name.find("RATE") != std::string::npos) {
          obj->GetYaxis()->SetTitle("Rate (Hz)");
	  obj->SetMinimum(0.);
	}
	else if (o.name.find("reportSummaryMap")!=std::string::npos) {
	  obj->SetMaximum(100.);
	  obj->SetMinimum(0.);
          obj->GetYaxis()->SetTitle("Busy (%)");

	}
	else {
	  obj->GetYaxis()->SetTitle("Event processing time (ms)");
	  obj->SetMinimum(0.);
	}

        int nbins = obj->GetNbinsX();

        int maxRange = nbins;
        for ( int i = nbins; i > 0; --i )
        {
          if ( obj->GetBinContent(i) != 0 )
          {
            maxRange = i;
            break;
          }
        }

        int minRange = 0;
        for ( int i = 0; i <= nbins; ++i )
        {
          if ( obj->GetBinContent(i) != 0 )
          {
            minRange = i-1;
            break;
          }
        }

        obj->GetXaxis()->SetRange(minRange, maxRange);
      }

    }

  void preDrawTH2F ( TCanvas *, const VisDQMObject &o )
  {
      TH2F* obj = dynamic_cast<TH2F*>( o.object );
      assert( obj );

      if (o.name.find("MODULE_FRACTION" ) != std::string::npos) {
        gPad->SetGridx();
        gPad->SetGridy();
        gStyle->SetPalette(1);
        gStyle->SetCanvasBorderMode( 0 );
        gStyle->SetPadBorderMode( 0 );
        gStyle->SetPadBorderSize( 0 );

        gStyle->SetOptStat( kFALSE );
        obj->SetStats( kFALSE );

        // Default coloring scheme
        //setRainbowColor(obj); // sets to rainbow color with finer gradations than setPalette(1)
	obj->SetContour(25,contour_level);
        obj->SetOption("textcol");
      }
      if (o.name.find("_SUMMARY" ) != std::string::npos) {
        gPad->SetGridx();
        gPad->SetGridy();
        gStyle->SetCanvasBorderMode( 0 );
        gStyle->SetPadBorderMode( 0 );
        gStyle->SetPadBorderSize( 0 );
        gStyle->SetOptStat( 0 );
        obj->SetStats( kFALSE );
        obj->SetOption("text");
	obj->GetYaxis()->SetTitle("N_{EP}");
	obj->GetXaxis()->SetTitle("Luminosity section");
      }
  }

  void postDrawTH1F ( TCanvas *, const VisDQMObject &o )
  {
    TH1F* obj = dynamic_cast<TH1F*>( o.object );
    assert (obj); // checks that object indeed exists
    obj->SetLineColor(kRed+1);
    obj->SetFillColor(kGreen+1);
  }

  void postDrawTH2F ( TCanvas *, const VisDQMObject &o )
  {
    TH2F* obj = dynamic_cast<TH2F*>( o.object );
    assert (obj); // checks that object indeed exists
  }

  //void setRainbowColor(TH2* obj)
  //{
  //  obj->SetContour(NCont_rainbow);
  //  gStyle->SetPalette(NCont_rainbow,hcalRainbowColors);
  //}

};
static DAQRenderPlugin instance;
