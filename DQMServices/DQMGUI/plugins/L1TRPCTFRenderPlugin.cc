/*!
  \file L1TRPCTFRenderPlugin.cc
  \\
  \\ Code shamelessly taken from, which was shamelessly borrowed
  \\ from J. Temple's HcalRenderPlugin.cc code, which was shamelessly
  \\ borrowed from S. Dutta's SiStripRenderPlugin.cc  code, G. Della Ricca
  \\ and B. Gobbo's EBRenderPlugin.cc, and other existing subdetector plugins
*/

#include "DQM/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile2D.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TText.h"
#include "TLine.h"
#include <cassert>

class L1TRPCTFRenderPlugin : public DQMRenderPlugin
{
public:
  virtual bool applies(const VisDQMObject &o, const VisDQMImgInfo &)
    {
      if (o.name.find( "L1TRPCTF/" ) != std::string::npos )
        return true;

      return false;
    }

  virtual void preDraw (TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo &)
    {
      c->cd();

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

  virtual void postDraw (TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &)
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
    }

  void preDrawTH2F ( TCanvas *, const VisDQMObject &o )
    {
      TH2F* obj = dynamic_cast<TH2F*>( o.object );
      assert( obj );

      //put in preDrawTH2F
      if( o.name.find( "muons_eta_phi" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
      }
    }

  void postDrawTH1F( TCanvas *, const VisDQMObject & )
    {
      // Add error/warning text to 1-D histograms.  Do we want this at this time?
    }

  void postDrawTH2F( TCanvas *, const VisDQMObject &o )
    {
      TH2F* obj = dynamic_cast<TH2F*>( o.object );
      assert( obj );

      if( o.name.find( "muons_eta_phi" )  != std::string::npos)
      {
        TLine line;
        line.SetLineWidth(1);
        for (int tc=0;tc<12;++tc)
        {
          int phi = tc*12+2;
          //line.DrawLine(-3.5, 0.5, -3.5, 6.5);
          line.DrawLine(-16.5, phi, 16.5, phi);
        }

        for (int tb=0;tb<8;++tb)
        {
          float eta = -12.5;
          if (tb==1) eta = -8.5;
          else if (tb==2) eta = -4.5;
          else if (tb==3) eta = -1.5;
          else if (tb==4) eta = 1.5;
          else if (tb==5) eta = 4.5;
          else if (tb==6) eta = 8.5;
          else if (tb==7) eta = 12.5;

          line.DrawLine(eta, 0, eta, 143);
        }
      }
    }
};

static L1TRPCTFRenderPlugin instance;
