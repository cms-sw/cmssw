/*!
  \file HLTRenderPlugin.cc
  \\
  \\ Code shamelessly borrowed from J. Temple's HcalRenderPlugin.cc code,
  \\ which was shamelessly borrowed from S. Dutta's SiStripRenderPlugin.cc
  \\ code, G. Della Ricca and B. Gobbo's EBRenderPlugin.cc, and other existing
  \\ subdetector plugins
  \\ preDraw and postDraw methods now check whether histogram was a TH1
  \\ or TH2, and call a private method appropriate for the histogram type
  $Id: HLTRenderPlugin.cc,v 1.22 2011/09/09 11:53:42 lilopera Exp $
  $Log: HLTRenderPlugin.cc,v $
  Revision 1.22  2011/09/09 11:53:42  lilopera
  updated render plugins to v6 of the GUI

  Revision 1.21  2011/02/15 18:16:43  rekovic
  Move up the bottom margin from 0.16 to 0.3 for FourVector Filters histograms so that the names of filters can be seen

  Revision 1.20  2010/07/26 10:45:32  rekovic
  Extend the Renders from folder FourVector/PathsSummary to folder PathsSummary

  Revision 1.19  2010/06/10 14:59:48  rekovic
  Fix a bug of missing bracket

  Revision 1.18  2010/06/09 09:47:36  rekovic
  change the check for paths to source and client histograms.  Added change of X-axis range for 1D LS histogrsms.

  Revision 1.17  2010/03/29 09:16:26  rekovic
  Extend Renders from counts_LS to _LS plots

  Revision 1.16  2010/02/22 14:16:10  wittich
  updates for HLT Scalers

  Revision 1.15  2010/01/15 23:57:26  wteo
  show only non-empty bins for HLTMonBitSummary plots

  Revision 1.14  2009/12/09 16:47:53  lorenzo
  added log scale

  Revision 1.13  2009/12/06 17:30:11  rekovic
  Render for HLT_bx plot

  Revision 1.11  2009/12/04 15:11:13  lorenzo
  commented log

  Revision 1.10  2009/12/03 19:13:11  puigh
  use log scale and make axes readable

  Revision 1.9  2009/10/31 23:18:54  lat
  Update for DQM GUI 5.1.0

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

class HLTRenderPlugin : public DQMRenderPlugin
{
public:
  virtual bool applies(const VisDQMObject &o, const VisDQMImgInfo &)
    {
      // determine whether core object is an HLT object
      if (o.name.find( "HLT/" ) != std::string::npos  )
        return true;

      return false;
    }

  virtual void preDraw (TCanvas * c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo &)
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
      if ( o.name.find("rate_p") != std::string::npos ||
           o.name.find("counts_p") != std::string::npos ||
           o.name.find("hltCount") != std::string::npos ||
           o.name.find("hltRate") != std::string::npos ||
           o.name.find("hltRate") != std::string::npos ||
           o.name.find("mergeCount") != std::string::npos)
      {
        gStyle->SetOptStat(11);
        obj->GetXaxis()->SetTitle("Luminosity Segment Number");
        if ( o.name.find("counts_p") == std::string::npos)
          obj->GetYaxis()->SetTitle("Rate (Hz)");
        else
          obj->GetYaxis()->SetTitle("Counts (no units)");

        int nbins = obj->GetNbinsX();

        int maxRange = nbins;
        for ( int i = nbins; i > 0; --i )
        {
          if ( obj->GetBinContent(i) != 0 )
          {
            maxRange = i+1;
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

      // BitSummary histograms
      if(REMATCH("Efficiency_Summary_*", o.name) || REMATCH("HLTRate_*", o.name)
	 || REMATCH("PassingBits_Summary_*", o.name)){

				//gStyle->SetOptStat(11);
				//obj->GetXaxis()->SetTitle("");
				//obj->GetYaxis()->SetTitle("");
				int nbins = obj->GetNbinsX();
				int maxRange = nbins;
				for ( int i = nbins; i > 0; --i ) {
				  if ( strlen(obj->GetXaxis()->GetBinLabel(i)) != 0 ) {
				    maxRange = i;
				    break;
				  }
				}
				obj->GetXaxis()->SetRange(0, maxRange);
			}

      // FourVector histograms
      if ( o.name.find("FourVector/paths/") != std::string::npos){

	      if( o.name.find("custom-eff") != std::string::npos)
	      {
	        gStyle->SetOptStat(10);
	        obj->SetMinimum(-0.05);
	        obj->SetMaximum(1.2);
	        obj->GetYaxis()->SetTitle("_Eff_");

	        if ( o.name.find("l1Et_Eff") != std::string::npos) obj->GetXaxis()->SetTitle("L1 P_{T}");
	        if ( o.name.find("onEt_Eff") != std::string::npos) obj->GetXaxis()->SetTitle("HLT P_{T}");
	        if ( o.name.find("offEt_Eff") != std::string::npos) obj->GetXaxis()->SetTitle("RECO P_{T}");
	      }
	      if (  o.name.find("custom-eff") == std::string::npos)
	      {

	        if ( o.name.find("l1Et") != std::string::npos) obj->GetXaxis()->SetTitle("L1 P_{T}");
	        if ( o.name.find("onEt") != std::string::npos) obj->GetXaxis()->SetTitle("HLT P_{T}");
	        if ( o.name.find("offEt") != std::string::npos) obj->GetXaxis()->SetTitle("RECO P_{T}");
	        if ( o.name.find("l1DRL1On") != std::string::npos) obj->GetXaxis()->SetTitle("L1-HLT #Delta R [rad]");
	        if ( o.name.find("offDRL1Off") != std::string::npos) obj->GetXaxis()->SetTitle("L1-RECO #Delta R [rad]");
	        if ( o.name.find("offDROnOff") != std::string::npos) obj->GetXaxis()->SetTitle("HLT-RECO #Delta R [rad]");
	      }

      }

      if ( o.name.find("PathsSummary") != std::string::npos) {

        if ( o.name.find("Pass_Any") != std::string::npos  ||  o.name.find("Normalized_Any") != std::string::npos  )
        {
          gPad->SetBottomMargin(0.16);
          gPad->SetLogy(1);
        }
	      // rate histograms
	      if( o.name.find("_LS") != std::string::npos)
	      {
	        gStyle->SetOptStat(11);
	        obj->GetXaxis()->SetTitle("Luminosity Segment");
	        int nbins = obj->GetNbinsX();

	        int maxRange = nbins;
	        for ( int i = nbins; i > 0; --i )
	        {
	          if ( obj->GetBinContent(i) != 0 )
	          {
	            maxRange = i+1;
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
      if ( o.name.find("PathsSummary/Filters Counts") != std::string::npos)
      {
        if ( o.name.find("Filters") != std::string::npos    )
        {
          gPad->SetBottomMargin(0.3);
          gPad->SetLogy(1);
          obj->SetMaximum(obj->GetBinContent(obj->GetMaximumBin())*1.1);
        }
      }
      if ( o.name.find("PathsSummary/Filters Efficiencies") != std::string::npos)
      {
        if ( o.name.find("Filters") != std::string::npos    )
        {
          gPad->SetBottomMargin(0.3);
          gPad->SetLogy(1);
          obj->SetMaximum(obj->GetBinContent(obj->GetMaximumBin())*1.1);
        }
      }

  }

  void preDrawTH2F ( TCanvas *, const VisDQMObject &o )
  {
      TH2F* obj = dynamic_cast<TH2F*>( o.object );
      assert( obj );

      //put in preDrawTH2F
      if( o.name.find( "reportSummaryMap" )  != std::string::npos)
      {
        obj->SetStats( kFALSE );
        dqm::utils::reportSummaryMapPalette(obj);
        obj->SetOption("colz");
        obj->SetTitle("HLT Report Summary Map");
        obj->GetXaxis()->SetNdivisions(1,true);
        obj->GetYaxis()->SetNdivisions(5,true);
        obj->GetXaxis()->CenterLabels();
        obj->GetYaxis()->CenterLabels();
        gPad->SetGrid(1,1);
        return;
      }
      gStyle->SetCanvasBorderMode( 0 );
      gStyle->SetPadBorderMode( 0 );
      gStyle->SetPadBorderSize( 0 );

      // I don't think we want to set stats to 0 for Hcal
      //gStyle->SetOptStat( 0 );
      //obj->SetStats( kFALSE );

      // Use same labeling format as SiStripRenderPlugin.cc
      TAxis* xa = obj->GetXaxis();
      TAxis* ya = obj->GetYaxis();

      xa->SetTitleOffset(0.7);
      xa->SetTitleSize(0.05);
      xa->SetLabelSize(0.04);

      ya->SetTitleOffset(0.7);
      ya->SetTitleSize(0.05);
      ya->SetLabelSize(0.04);

      // Now the important stuff -- set 2D hist drawing option to "colz"
      gStyle->SetPalette(1);
      obj->SetOption("colz");

      //put in preDrawTH2F
      if( o.name.find( "FourVector" )  != std::string::npos)
      {
        gStyle->SetOptStat(10);
        obj->SetOption("colz");
        gStyle->SetPalette(1,0);
        //obj->GetXaxis()->CenterLabels();
        //obj->GetYaxis()->CenterLabels();

        if ( o.name.find("l1Eta") != std::string::npos) {obj->GetXaxis()->SetTitle("L1 #eta");obj->GetYaxis()->SetTitle("L1 #phi");}
        if ( o.name.find("onEta") != std::string::npos) {obj->GetXaxis()->SetTitle("HLT #eta");obj->GetYaxis()->SetTitle("HLT #phi");}
        if ( o.name.find("offEta") != std::string::npos) {obj->GetXaxis()->SetTitle("RECO #eta");obj->GetYaxis()->SetTitle("RECO #phi");}
      }

      if( o.name.find( "FourVector/paths/" )  != std::string::npos
       && o.name.find("custom-eff") != std::string::npos) {
       obj->SetMinimum(0);
       obj->SetMaximum(1.2);
      }

      if ( o.name.find("PathsSummary") != std::string::npos)
      {
        if (o.name.find("PassPass") != std::string::npos )
        {
          gPad->SetBottomMargin(0.16);
          gPad->SetRightMargin(0.14);
          gPad->SetLeftMargin(0.24);
        }

	      // HLT couts vs Bunch Crossing
	      if( o.name.find("HLT_bx") != std::string::npos)
	      {
	        gPad->SetRightMargin(0.14);
	        gPad->SetLeftMargin(0.24);
	        gStyle->SetOptStat(0);
	        obj->SetTitle("HLT count");
	        obj->GetXaxis()->SetTitle("Event Bunch Crossing");
        }

	      // rate histograms
	      if( o.name.find("_LS") != std::string::npos)
	      {
	        gPad->SetRightMargin(0.14);
	        gPad->SetLeftMargin(0.24);
	        gStyle->SetOptStat(11);
	        obj->GetXaxis()->SetTitle("Luminosity Segment");
	        int topBin = obj->GetNbinsY();
	        int nbins = obj->GetNbinsX();

	        int maxRange = nbins;
	        for ( int i = nbins; i > 0; --i )
	        {
	          if ( obj->GetBinContent(i,topBin) != 0 )
	          {
	            maxRange = i+1;
	            break;
	          }
	        }

	        int minRange = 0;
	        for ( int i = 0; i <= nbins; ++i )
	        {
	          if ( obj->GetBinContent(i,topBin) != 0 )
	          {
	            minRange = i-1;
	            break;
	          }
	        }

	        obj->GetXaxis()->SetRange(minRange, maxRange);
	      }
      }

      //HLTMonBitSummary
      if(REMATCH("PassingBits_Correlation_*", o.name)){

	//gStyle->SetOptStat(11);
	//obj->GetXaxis()->SetTitle("");
	//obj->GetYaxis()->SetTitle("");
	int nbinsx = obj->GetNbinsX();
	int maxRangex = nbinsx;
	for ( int i = nbinsx; i > 0; --i ) {
	  if ( strlen(obj->GetXaxis()->GetBinLabel(i)) != 0 ) {
	    maxRangex = i;
	    break;
	  }
	}
	int nbinsy = obj->GetNbinsY();
	int maxRangey = nbinsy;
	for ( int i = nbinsy; i > 0; --i ) {
	  if ( strlen(obj->GetYaxis()->GetBinLabel(i)) != 0 ) {
	    maxRangey = i;
	    break;
	  }
	}
	obj->GetXaxis()->SetRange(0, maxRangex);
	obj->GetYaxis()->SetRange(0, maxRangey);
      }

  }

  void postDrawTH1F( TCanvas *, const VisDQMObject & )
    {
      /*
        // Add error/warning text to 1-D histograms.  Do we want this at this time?
        TText tt;
        tt.SetTextSize(0.12);

        if (o.flags == 0)
                return;

        else
        {
          if (o.flags & DQMNet::DQM_PROP_REPORT_ERROR)
          {
                  tt.SetTextColor(2); // error color = RED
                  tt.DrawTextNDC(0.5, 0.5, "Error");
          } // DQM_PROP_REPORT_ERROR
          else if (o.flags & DQMNet::DQM_PROP_REPORT_WARN)
          {
                  tt.SetTextColor(5);
                  tt.DrawTextNDC(0.5, 0.5, "Warning"); // warning color = YELLOW
          } // DQM_PROP_REPORT_WARN
          else if (o.flags & DQMNet::DQM_PROP_REPORT_OTHER)
          {
                  tt.SetTextColor(1); // other color = BLACK
                  tt.DrawTextNDC(0.5, 0.5, "Other ");
          } // DQM_PROP_REPORT_OTHER
          else
          {
                  tt.SetTextColor(3);
                  tt.DrawTextNDC(0.5, 0.5, "Ok ");
          } //else
        } // else (  o.flags != 0  )
      */
    }

  void postDrawTH2F( TCanvas *, const VisDQMObject & )
    {
      // nothing to put here just yet
      // in the future, we can add text output based on error status,
      // or set bin range based on filled histograms, etc.
      // Maybe add a big "OK" sign to histograms with no entries (i.e., no errors)?
    }
};

static HLTRenderPlugin instance;
