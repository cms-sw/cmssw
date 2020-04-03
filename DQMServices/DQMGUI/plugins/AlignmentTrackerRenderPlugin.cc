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

class AlignmentTrackerRenderPlugin : public DQMRenderPlugin
{
public:

  virtual bool applies(const VisDQMObject &o, const VisDQMImgInfo &)
  {
    if(o.name.find( "Alignment/Tracker/" ) != std::string::npos)return true;
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

  void preDrawTH1F(TCanvas *c, const VisDQMObject &o)
  {
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetPadBorderSize(0);

    gStyle->SetOptStat(111110);
    gStyle->SetTitleSize(0.06,"");
    gStyle->SetTitleX(0.18);
//    gStyle->SetTitleSize(0.02,"XY");
//    gStyle->SetLabelSize(0.05,"XY");
//    gStyle->SetTitleOffset(1.2,"X");
//    gStyle->SetTitleOffset(1.6,"Y");

    TH1F* obj = dynamic_cast<TH1F*>( o.object );
    assert( obj );

    c->SetLogy(0);
    c->SetTopMargin(0.08);
    c->SetBottomMargin(0.14);
    c->SetLeftMargin(0.11);
    c->SetRightMargin(0.09);

    if((o.name.find( "/GlobalTrackVariables/" ) != std::string::npos))return;
    if((o.name.find( "h_summary" ) != std::string::npos))return;

    obj->GetXaxis()->SetTitleSize(0.06);
    obj->GetXaxis()->SetLabelSize(0.06);
    obj->GetXaxis()->SetTitleOffset(1.05);

    obj->GetYaxis()->SetTitleSize(0.06);
    obj->GetYaxis()->SetLabelSize(0.06);
    if(!(o.name.find( "h_Dmr" ) != std::string::npos)){
      obj->GetYaxis()->SetTitle("# hits");
    }

    obj->SetStats(kTRUE);
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

    gStyle->SetOptStat(111110);

    obj->SetStats( kTRUE );

    TPaveStats* stats = (TPaveStats*)obj->GetListOfFunctions()->FindObject("stats");
    if(!stats)return;
    stats->SetTextSize(0.055);
    stats->SetX1NDC(0.68);
    stats->SetX2NDC(0.99);
    stats->SetY1NDC(0.575);
    stats->SetY2NDC(0.925);
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

  }

void postDrawTProfile2D(TCanvas *, const VisDQMObject &o)
  {
    TProfile2D* obj = dynamic_cast<TProfile2D*>( o.object );
    assert( obj );

  }
};

static AlignmentTrackerRenderPlugin instance;
