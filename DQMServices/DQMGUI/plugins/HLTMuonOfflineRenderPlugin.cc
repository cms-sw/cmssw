/*!
  \file HLTRenderPlugin.cc
  \\ Current maintainer: slaunwhj
  \\ Based on HLTRenderPlugin
  \\

  $Id: HLTMuonOfflineRenderPlugin.cc,v 1.7 2011/09/09 11:53:42 lilopera Exp $
  $Log: HLTMuonOfflineRenderPlugin.cc,v $
  Revision 1.7  2011/09/09 11:53:42  lilopera
  updated render plugins to v6 of the GUI

  Revision 1.6  2011/03/24 20:13:38  klukas
  Updated to fit new offline DQM plots; changed efficiency plot styles

  Revision 1.5  2009/10/31 23:18:54  lat
  Update for DQM GUI 5.1.0

  Revision 1.4  2009/10/28 20:12:16  klukas
  Added TPRegexp to distinguish between RelVal and Offline; added some RelVal-specific content

  Revision 1.3  2009/10/27 17:50:52  slaunwhj
  Changed class name to resolve conflict with HLTRenderPlugin

  Revision 1.1  2009/10/14 11:17:53  slaunwhj
  RenderPlugin for HLT Muon offline DQM plots

  Revision 1.8  2009/09/21 14:34:02  rekovic
  Fix FourVector renders

  Revision 1.7  2009/05/22 19:09:33  lat
  Untabify.

  Revision 1.6  2009/05/22 19:05:23  lat
  Adapt to keeping render plug-ins outside server RPM. Clean up and harmonise code.

  Revision 1.5  2008/10/01 17:57:39  lorenzo
  changed EventInfo format

  Revision 1.4  2008/08/28 21:50:39  wittich
  Rate histos: Also put in low range minimums in case we start in the
  middle of a run
*/

#include "DQM/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile.h"
#include "TProfile2D.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TText.h"
#include "TPRegexp.h"
#include "TLine.h"
#include <cassert>

// Define constants
TPRegexp efficiencyRegexp("(efficiency[^_]*$|genEff|recEff|TurnOn|MaxPt)");

class HLTMuonOfflineRenderPlugin : public DQMRenderPlugin
{
public:
  virtual bool applies(const VisDQMObject &o, const VisDQMImgInfo &)
    {
      // determine whether core object is an HLT object
      if (o.name.find( "HLT/Muon" ) != std::string::npos  )
        return true;

      return false;
    }

  virtual void preDraw (TCanvas * c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo &)
    {
      c->cd();

      // object is TProfile histogram
      if( dynamic_cast<TProfile*>( o.object ) )
      {
        preDrawTProfile( c, o );
      }

      // object is TH2 histogram
      if( dynamic_cast<TH2F*>( o.object ) )
      {
        preDrawTH2F( c, o );
      }

      // object is TH1 histogram
      else if( dynamic_cast<TH1F*>( o.object ) )
      {
        preDrawTH1F( c, o );
      }
    }

  virtual void postDraw (TCanvas * c, const VisDQMObject &o, const VisDQMImgInfo &)
    {

      // object is TProfile histogram
      if( dynamic_cast<TProfile*>( o.object ) )
      {
        postDrawTProfile( c, o );
      }

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
  void preDrawTProfile ( TCanvas *, const VisDQMObject &o )
    {

      // Do we want to do anything special yet with TProfile histograms?
      TProfile* obj = dynamic_cast<TProfile*>( o.object );
      assert (obj); // checks that object indeed exists

      // if this isn't a muon hlt plot, skip it
      if (o.name.find("HLT/Muon") == std::string::npos)
        return;

      if (TString(o.name).Contains(efficiencyRegexp))
      {

          gStyle->SetOptStat(10);

            obj->SetMinimum(0);
          obj->SetMaximum(1.1);
          obj->SetMarkerStyle(4);

          }

      if (o.name.find("TurnOn") != std::string::npos)
          {

            gPad->SetLogx();
            obj->GetXaxis()->SetRangeUser(2., 300.);

          }

      }

  void preDrawTH1F ( TCanvas *, const VisDQMObject &o )
      {
      // Do we want to do anything special yet with TH1F histograms?
      TH1F* obj = dynamic_cast<TH1F*>( o.object );
      assert (obj); // checks that object indeed exists

      // if this isn't a muon hlt plot, skip it
      if (o.name.find("HLT/Muon") == std::string::npos)
        return;

      // Do these for all your histos
      gStyle->SetOptStat(10);

      // FourVector eff histograms

      if (TString(o.name).Contains(efficiencyRegexp))
          {

            obj->SetMinimum(0);
            obj->SetMaximum(1.0);

          }

      if (o.name.find("TurnOn") != std::string::npos)
          {

          gPad->SetLogx();
          obj->GetXaxis()->SetRangeUser(2., 300.);

          }

    }

  void preDrawTH2F ( TCanvas *, const VisDQMObject &o )
    {
      TH2F* obj = dynamic_cast<TH2F*>( o.object );
      assert( obj );

      // if this isn't a muon hlt plot, skip it
      if (o.name.find("HLT/Muon") == std::string::npos)
        return;

      // You'll want to do all of these things
      // Regardless of the histo

      gStyle->SetOptStat(10);
      gStyle->SetPalette(1,0);
      obj->SetOption("colz");
      // gPad->SetGrid(1,1);

      //Handle the 2D eff histograms
      if (TString(o.name).Contains(efficiencyRegexp))
        {

          obj->SetOption("text colz");
          gStyle->SetPaintTextFormat(".2f");

          const Int_t NRGBs = 3;
          const Int_t NCont = 255;
          Double_t stops[NRGBs] = { 0.00, 0.88, 1.00 };
          Double_t red[NRGBs]   = { 1.00, 1.00, 0.00 };
          Double_t green[NRGBs] = { 0.00, 1.00, 1.00 };
          Double_t blue[NRGBs]  = { 0.00, 0.00, 0.00 };
          TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
          gStyle->SetNumberContours(NCont);

          obj->SetMinimum(0.0);
          obj->SetMaximum(1.0);

          return;
        }

      // if not a 2D eff, do nothing special
      return;

    }

  void postDrawTProfile( TCanvas *, const VisDQMObject &o)
    {

      TProfile* obj = dynamic_cast<TProfile*>( o.object );

      if (TString(o.name).Contains(efficiencyRegexp))
        {

          TLine l;
          l.SetLineStyle(2); // dashed
          float xmin = obj->GetBinLowEdge(1);
          float xmax = obj->GetBinLowEdge(obj->GetNbinsX() + 1);
          l.DrawLine(xmin, 1.0, xmax, 1.0);

        }

      return;

    }

  void postDrawTH1F( TCanvas *, const VisDQMObject & )
    {

      // No special actions necessary right now

      return;

    }

  void postDrawTH2F( TCanvas *, const VisDQMObject & )
    {
      // nothing to put here just yet
      // in the future, we can add text output based on error status,
      // or set bin range based on filled histograms, etc.
      // Maybe add a big "OK" sign to histograms with no entries (i.e., no errors)?
    }
};

static HLTMuonOfflineRenderPlugin instance;
