/*!
  \file OfflinePVAlignmentRenderPlugin
  \Display Plugin for Alignment Primary Vertex Validation 
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2017/06/29 10:51:56 $
*/

#include "DQM/DQMRenderPlugin.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TStyle.h"
#include "TGaxis.h"
#include "TPaveStats.h"
#include "TList.h"
#include <cassert>

class OfflinePVAlignmentRenderPlugin : public DQMRenderPlugin
{
public:

  virtual bool applies(const VisDQMObject &o, const VisDQMImgInfo &)
  {
    if(o.name.find( "OfflinePV/Alignment/" ) != std::string::npos)return true;
    else return false;
  }

  virtual void preDraw(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo &)
  {
    c->cd();

    if( dynamic_cast<TH1F*>( o.object ) )
    {
      this->preDrawTH1F( c, o );
    }
    else if( dynamic_cast<TH2F*>( o.object ) )
    {
      this->preDrawTH2F( c, o );
    }
    else if( dynamic_cast<TProfile*>( o.object ) )
    {
      this->preDrawTProfile( c, o );
    }
    else if( dynamic_cast<TProfile2D*>( o.object ) )
    {
      this->preDrawTProfile2D( c, o );
    }
  }

  virtual void postDraw(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &)
  {
    c->cd();

    if( dynamic_cast<TH1F*>( o.object ) )
    {
      this->postDrawTH1F( c, o );
    }
    else if( dynamic_cast<TH2F*>( o.object ) )
    {
      this->postDrawTH2F( c, o );
    }
    else if( dynamic_cast<TProfile*>( o.object ) )
    {
      this->postDrawTProfile( c, o );
    }
    else if( dynamic_cast<TProfile2D*>( o.object ) )
    {
      this->postDrawTProfile2D( c, o );
    }
  }

private:

  void preDrawTH1F(TCanvas *, const VisDQMObject &o)
  {
    TH1F* obj = dynamic_cast<TH1F*>( o.object );
    assert( obj );

    obj->SetLineWidth(2);

  }

  void preDrawTH2F(TCanvas *, const VisDQMObject &o)
  {
    TH2F* obj = dynamic_cast<TH2F*>( o.object );
    assert( obj );

  }

  void preDrawTProfile(TCanvas *, const VisDQMObject &o)
  {
    TProfile* obj = dynamic_cast<TProfile*>( o.object );
    assert( obj );

    obj->SetMarkerStyle(20);
    obj->SetMarkerSize(1);

  }

  void preDrawTProfile2D(TCanvas *, const VisDQMObject &o)
  {
    TProfile2D* obj = dynamic_cast<TProfile2D*>( o.object );
    assert( obj );

  }

  void postDrawTH1F(TCanvas *, const VisDQMObject &o)
  {
    TH1F* obj = dynamic_cast<TH1F*>( o.object );
    assert( obj );
    
    TAxis* xa = obj->GetXaxis();
    TAxis* ya = obj->GetYaxis();

    xa->SetTitleOffset(0.9);
    xa->SetTitleSize(0.05);
    xa->SetLabelSize(0.04);
    xa->CenterTitle();

    ya->SetTitleOffset(0.9);
    ya->SetTitleSize(0.05);
    ya->SetLabelSize(0.04);
    ya->CenterTitle();

    // first layout

    if( o.name.find("chi2ndf")!=std::string::npos ){
     
      xa->SetTitle("#chi^{2}/ndf of track");
      ya->SetTitle("tracks");

    } else if(o.name.find("chi2prob")!=std::string::npos ){

      xa->SetTitle("Prob(#chi^{2},ndf)");
      ya->SetTitle("tracks");

    } else if(o.name.find("sumpt")!=std::string::npos ){

      xa->SetTitle("#sum p^{2}_{T} [GeV^{2}]");
      ya->SetTitle("tracks");

    } else if(o.name.find("weight")!=std::string::npos ){
      
      xa->SetTitle("track weight");
      ya->SetTitle("tracks");
      
    } else if(o.name.find("ntracks")!=std::string::npos ){

      xa->SetTitle("n. of tracks (p_{T}>1GeV)");
      ya->SetTitle("vertices");
      gPad->SetLogy(1); 

    } else if(o.name.find("dxyErr")!=std::string::npos ){

      xa->SetTitle("error on d_{xy}(trk-PV) [#mum]");
      ya->SetTitle("tracks");
      gPad->SetLogy(1); 

    } else if(o.name.find("dzErr")!=std::string::npos ){

      xa->SetTitle("error on d_{z}(trk-PV) [#mum]");
      ya->SetTitle("tracks");
      gPad->SetLogy(1); 

    } else if(o.name.find("dxy")!=std::string::npos ){

      xa->SetTitle("d_{xy}(trk-PV) [#mum]");
      ya->SetTitle("tracks");

      if(o.name.find("dxyzoom")==std::string::npos ){
	gPad->SetLogy(1); 
      }

    } else if(o.name.find("dz")!=std::string::npos ){

      xa->SetTitle("d_{z}(trk-PV) [#mum]");
      ya->SetTitle("tracks");

    }

    gPad->Update();
    TPaveStats* st = (TPaveStats*)obj->GetListOfFunctions()->FindObject("stats");
    if(st!=0)  {
      st->SetBorderSize(0);
      st->SetOptStat( 1110 );
      st->SetTextColor( obj->GetLineColor() );

      if(o.name.find("ntrack")!=std::string::npos ||
	 o.name.find("sumpt")!=std::string::npos ||
	 o.name.find("chi2ndf")!=std::string::npos ||
	 o.name.find("Err")!=std::string::npos
	 ) {
	
	st->SetX1NDC( .65 );
	st->SetX2NDC( .88 );
	st->SetY1NDC( .73 );
	st->SetY2NDC( .89 );

      } else {

	st->SetX1NDC( .12 );
	st->SetX2NDC( .35 );
	st->SetY1NDC( .73 );
	st->SetY2NDC( .89 );
      }
    }
  }

  void postDrawTH2F(TCanvas *, const VisDQMObject &o)
  {
    TH2F* obj = dynamic_cast<TH2F*>( o.object );
    assert( obj );

  }

  void postDrawTProfile(TCanvas *, const VisDQMObject &o)
  {
    TProfile* obj = dynamic_cast<TProfile*>( o.object );
    assert( obj );

    gStyle->SetOptStat(0);

    TAxis* xa = obj->GetXaxis();
    TAxis* ya = obj->GetYaxis();

    xa->SetTitleOffset(0.9);
    xa->SetTitleSize(0.05);
    xa->SetLabelSize(0.04);
    xa->CenterTitle();

    ya->SetTitleOffset(0.9);
    ya->SetTitleSize(0.05);
    ya->SetLabelSize(0.04);
    ya->CenterTitle();

    gPad->Update();

    TLine tl;
    tl.SetLineColor(kMagenta);
    tl.SetLineWidth(3);
    tl.SetLineStyle(7);
    tl.DrawLine(gPad->GetUxmin(),0.,gPad->GetUxmax(),0.);

   
  }

void postDrawTProfile2D(TCanvas *, const VisDQMObject &o)
  {
    TProfile2D* obj = dynamic_cast<TProfile2D*>( o.object );
    assert( obj );

  }
};

static OfflinePVAlignmentRenderPlugin instance;
