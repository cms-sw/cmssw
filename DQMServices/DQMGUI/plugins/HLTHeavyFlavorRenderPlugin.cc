// Render Plugin for Heavy Flavor HLT validation
// author: Zoltan Gecse

#include "DQM/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile2D.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TText.h"
#include "TLegend.h"
#include <cassert>

class HLTHeavyFlavorRenderPlugin : public DQMRenderPlugin{
public:
  virtual bool applies( const VisDQMObject &o, const VisDQMImgInfo & ){
    return o.name.find( "HLT/HeavyFlavor" ) != std::string::npos;
  }

  virtual void preDraw( TCanvas * c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo & ){
    if(o.name.find( "effPathGlob_recoPt" ) != std::string::npos)
      return;
    c->cd();
    TH2* h2 = dynamic_cast<TH2*>( o.object );
    if( h2 ){
      preDrawTH2( c, h2, o.name );
      return;
    }
    TH1* h1 = dynamic_cast<TH1*>( o.object );
    if( h1 ){
      preDrawTH1( c, h1, o.name );
      return;
    }
  }

  virtual void postDraw( TCanvas *c , const VisDQMObject &o, const VisDQMImgInfo & ){
    if(o.name.find( "effPathGlob_recoPt" ) == std::string::npos)
      return;
    c->cd();
    TH2* h2 = dynamic_cast<TH2*>( o.object );
    if( h2 ){
      postDrawTH2( c, h2, o.name );
      return;
    }
  }

private:
  void preDrawTH1( TCanvas *c, TH1* obj, const std::string &name ){
    if( name.find("eff") != std::string::npos ){
      gStyle->SetOptStat(0);
      obj->SetOption("PE");
      obj->SetTitle("");
//      obj->Scale(100);
      // "colz" doesn't draw zero bins so increase them a bit. If bin has 0/0 i.e. 50+-50 make it -1 to avoid drawing
      for(int i=1; i<=obj->GetNbinsX(); i++){
        if( obj->GetBinError(i) > 0.49 )
          obj->SetBinContent(i, -1);
      }
      obj->GetYaxis()->SetRangeUser(-0.001,1.001);
      c->SetGridy();
    }else{
      gStyle->SetOptStat("e");
    }
    // if axis label starts with space, set log scale
    if( TString(obj->GetXaxis()->GetTitle()).BeginsWith(' ') ){
      c->SetLogx();
    }
    obj->SetLineColor(2);
    obj->SetLineWidth(2);
    obj->SetMarkerStyle(20);
    obj->SetMarkerSize(0.8);
  }

  void preDrawTH2 ( TCanvas *c, TH2* obj, const std::string &name ){
    if( name.find("eff") != std::string::npos ){
      gStyle->SetOptStat(0);
      gStyle->SetPaintTextFormat(".0f");
      obj->SetOption("colztexte");
      obj->SetTitle("");
      //convert to percents, less digits to draw
      obj->Scale(100);
      //"colz" doesn't draw zero bins so increase them a bit. If bin has 0/0 i.e. 50+-50 make it -1 to avoid drawing
      for(int i=1; i<=obj->GetNbinsX(); i++){
        for(int j=1; j<=obj->GetNbinsY(); j++){
          if( obj->GetBinContent(i, j) == 0 )
            obj->SetBinContent(i, j, 0.0001);
          if( obj->GetBinError(i, j) > 49 )
            obj->SetBinContent(i, j, -1);
        }
      }
      obj->GetZaxis()->SetRangeUser(-0.001,100.001);
    }else if( name.find("deltaEtaDeltaPhi") != std::string::npos ){
      gStyle->SetOptStat("emr");
      obj->SetOption("colz");
      c->SetLogz();
    }else{
      gStyle->SetOptStat("e");
      obj->SetOption("colztext");
    }
    // if axis label starts with space, set log scale
    if( TString(obj->GetXaxis()->GetTitle()).BeginsWith(' ') ){
      c->SetLogx();
    }
    if( TString(obj->GetYaxis()->GetTitle()).BeginsWith(' ') ){
      c->SetLogy();
    }
    gStyle->SetPalette(1);
  }

  void postDrawTH2 ( TCanvas *c, TH2* h, const std::string & ){
    gStyle->SetOptStat(0);
    c->SetGridy();
    // if axis label starts with space, set log scale
    if( TString(h->GetXaxis()->GetTitle()).BeginsWith(' ') ){
      c->SetLogx();
    }
    TLegend* l = new TLegend(0.01,.92,0.99,0.99);
    l->SetBit(kCanDelete);
    l->SetFillColor(0);
    l->SetNColumns(h->GetNbinsY());
    for(int i=1; i<=h->GetNbinsY(); i++){
      TH1F * h1 = (h->GetXaxis()->GetXbins()->GetSize()==0) ?
        new TH1F(TString::Format("tmp%d",i),TString::Format("tmp%d",i),h->GetXaxis()->GetNbins(),h->GetXaxis()->GetXmin(),h->GetXaxis()->GetXmax()):
        new TH1F(TString::Format("tmp%d",i),TString::Format("tmp%d",i),h->GetXaxis()->GetNbins(),h->GetXaxis()->GetXbins()->GetArray());
      h1->SetBit(kCanDelete);
      for(int j=1; j<=h->GetNbinsX(); j++){
        h1->SetBinContent(j,h->GetBinContent(j,i));
        h1->SetBinError(j,h->GetBinError(j,i));
      }
      h1->SetLineColor(1+i);
      h1->SetLineWidth(2);
      h1->SetMarkerStyle(19+i);
      h1->SetMarkerSize(0.8);
      l->AddEntry(h1,h->GetYaxis()->GetBinLabel(i),"lp");
      if(i==1){
        h1->GetXaxis()->SetTitle(h->GetXaxis()->GetTitle());
        h1->GetYaxis()->SetRangeUser(-0.001,1.001);
        h1->GetYaxis()->SetTitle("Efficiency");
        h1->SetTitle("");
        h1->Draw("PE");
      }else{
        h1->Draw("PEsame");
      }
    }
    l->Draw();
  }

};

static HLTHeavyFlavorRenderPlugin instance;
