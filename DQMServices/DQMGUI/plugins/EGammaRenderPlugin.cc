#include "DQM/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile2D.h"
#include "TProfile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLine.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TText.h"
#include "TClass.h"
#include <cassert>

class EGammaRenderPlugin : public DQMRenderPlugin
{

public:
  virtual bool applies( const VisDQMObject &o, const VisDQMImgInfo & )
    {
      // egamma filter
      if ( o.name.find( "Egamma/") == std::string::npos )
        return false;

      // electrons
      if ( o.name.find( "/Electrons/Ele") != std::string::npos )
        return true;

      // photons

      if (o.name.find( "PhotonAnalyzer/" ) == std::string::npos)
        return false;

      if( o.name.find( "/General/" ) != std::string::npos )
        return true;

      if( o.name.find( "/Efficiencies/" ) != std::string::npos )
        return true;

      if( o.name.find( "/AllPhotons/" ) != std::string::npos )
        return true;

      if( o.name.find( "/GoodCandidatePhotons/" ) != std::string::npos )
        return true;

      if( o.name.find( "/BackgroundPhotons/" ) != std::string::npos )
        return true;

      return false;
    }

  virtual void preDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo & )
    {
      c->cd();

      // electrons
      if ( o.name.find( "/Electrons/Ele") != std::string::npos )
       {
        TH1 * histo = dynamic_cast<TH1*>(o.object) ;
        if (!histo) return ;

        TString histo_option = histo->GetOption() ;
        if ((histo_option.Contains("ELE_LOGY")==kTRUE)&&(histo->GetMaximum()>0))
        { c->SetLogy(1) ; }

        if ( dynamic_cast<TH2*>(o.object) )
         {
          gStyle->SetPalette(1) ;
          gStyle->SetOptStat(110) ;
         }
        else if ( dynamic_cast<TProfile*>(o.object) )
         { gStyle->SetOptStat(110) ; }
        else // TH1
         { gStyle->SetOptStat(111110) ; }

        return ;
       }

      // photons
      if( dynamic_cast<TH2F*>( o.object ) )
      {
        preDrawTH2F( c, o );
      }
      else if( dynamic_cast<TH1F*>( o.object ) )
      {
        preDrawTH1F( c, o );
      }
      else if( dynamic_cast<TProfile*>( o.object ) )
      {
	      preDrawTProfile( c, o );
      }

    }

  virtual void postDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo & )
    {
      c->cd();

      // electrons : do nothing
      if ( o.name.find( "/Electrons/Ele") != std::string::npos )
       { return ; }

      // photons
      if( dynamic_cast<TH1F*>( o.object ) )
      {
        postDrawTH1F( c, o );
      }
      if( dynamic_cast<TH2F*>( o.object ) )
      {
        postDrawTH2F( c, o );
      }
    }

private:

  void preDrawTH2F( TCanvas *, const VisDQMObject &o )
    {
      TH2F* obj = dynamic_cast<TH2F*>( o.object );
      assert( obj );

      gStyle->SetPalette(1);
      gStyle->SetOptStat("e");
      obj->SetOption( "colz" );
    }

  void preDrawTH1F( TCanvas *, const VisDQMObject &o )
    {
      TH1F* obj = dynamic_cast<TH1F*>( o.object );
      assert( obj );

      gStyle->SetOptStat("emr");

      if( o.name.find( "nPho" )  != std::string::npos)
        gStyle->SetOptStat("em");

      if( o.name.find( "nConv" )  != std::string::npos)
        gStyle->SetOptStat("em");

      if( o.name.find( "nIsoTracks" )  != std::string::npos)
        gStyle->SetOptStat("em");

      if( o.name.find( "phoEta" )  != std::string::npos)
        gStyle->SetOptStat("e");

      if( o.name.find( "phoConvEta" )  != std::string::npos)
        gStyle->SetOptStat("e");

      if( o.name.find( "phoPhi" )  != std::string::npos)
        gStyle->SetOptStat("e");

      if( o.name.find( "phoConvPhi" )  != std::string::npos)
        gStyle->SetOptStat("e");

      if( o.name.find( "VsEta" )  != std::string::npos)
        gStyle->SetOptStat("e");
    }

  void preDrawTProfile( TCanvas *c, const VisDQMObject &o )
    {

      c->cd();
      TProfile* obj = dynamic_cast<TProfile*>( o.object );
      assert( obj );

      gStyle->SetOptStat("em");

    }

  void postDrawTH1F( TCanvas *c, const VisDQMObject &o )
    {
      TH1F* obj = dynamic_cast<TH1F*>( o.object );
      assert( obj );

      //gStyle->SetOptStat(11);
      obj->SetMinimum(0);

      if( o.name.find( "Filters" ) != std::string::npos )
      {
        c->SetBottomMargin(0.25);
        c->SetRightMargin(0.35);
        obj->SetStats(kFALSE);
        obj->SetMaximum(1.05);
      }
      if( o.name.find( "hOverE" )  != std::string::npos)
      {
        c->SetLogy(1);
        obj->SetMinimum(0.5);
      }
      if( o.name.find( "h1OverE" )  != std::string::npos)
      {
        c->SetLogy(1);
        obj->SetMinimum(0.5);
      }
      if( o.name.find( "h2OverE" )  != std::string::npos)
      {
        c->SetLogy(1);
        obj->SetMinimum(0.5);
      }

    }

  void postDrawTH2F( TCanvas *, const VisDQMObject &o )
    {
      TH2F* obj = dynamic_cast<TH2F*>( o.object );
      assert( obj );
      //gStyle->SetOptStat("e");

    }

};

static EGammaRenderPlugin instance;
