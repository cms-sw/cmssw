#include "DQM/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile2D.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include <cassert>
#include "TLatex.h"
#include "TLine.h"

class MuonRenderPlugin : public DQMRenderPlugin
{
  TLatex *label_resTk;
  TLatex *label_resSTA;
  TLatex *label_muonIdSTA;
  TLatex *label_glbHO;
  TLatex *label_tkHO;
  TLatex *label_staHO;

public:
  MuonRenderPlugin()
    {
      label_resTk = new TLatex(2.5,4.5,"NA");
      label_resSTA = new TLatex(3.5,4.5,"NA");
      label_muonIdSTA = new TLatex(3.5,5.5,"NA");
      label_glbHO = new TLatex(1.5,3.5,"NA");
      label_tkHO = new TLatex(2.5,3.5,"NA");
      label_staHO = new TLatex(3.5,3.5,"NA");
    }

  virtual bool applies( const VisDQMObject &o, const VisDQMImgInfo & )
    {
      if((o.name.find( "Muons/E" ) != std::string::npos) ||
         (o.name.find( "Muons/M" ) != std::string::npos) ||
         (o.name.find( "Muons/R" ) != std::string::npos) ||
         (o.name.find( "Muons/S" ) != std::string::npos) ||
         (o.name.find( "Muons/T" ) != std::string::npos) ||
         (o.name.find( "Muons/c" ) != std::string::npos) ||
         (o.name.find( "Muons/g" ) != std::string::npos) ||
         (o.name.find( "Muons/s" ) != std::string::npos))
        return true;

      return false;
    }

  virtual void preDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo & )
    {
      c->cd();

      if( dynamic_cast<TProfile2D*>( o.object ) )
      {
        preDrawTProfile2D( c, o );
      }
      else if( dynamic_cast<TProfile*>( o.object ) )
      {
        preDrawTProfile( c, o );
      }
      else if( dynamic_cast<TH2*>( o.object ) )
      {
        preDrawTH2( c, o );
      }
      else if( dynamic_cast<TH1*>( o.object ) )
      {
        preDrawTH1( c, o );
      }
    }

  virtual void postDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo & )
    {
      c->cd();

      if( dynamic_cast<TProfile2D*>( o.object ) )
      {
        postDrawTProfile2D( c, o );
      }
      else if( dynamic_cast<TProfile*>( o.object ) )
      {
        postDrawTProfile( c, o );
      }
      else if( dynamic_cast<TH2*>( o.object ) )
      {
        postDrawTH2( c, o );
      }
      else if( dynamic_cast<TH1*>( o.object ) )
      {
        postDrawTH1( c, o );
      }
    }

private:
  void preDrawTProfile2D( TCanvas *, const VisDQMObject & )
    {
    }

  void preDrawTProfile( TCanvas *, const VisDQMObject & )
    {
    }

  void preDrawTH2( TCanvas *c, const VisDQMObject &o )
    {
      TH2F* obj = dynamic_cast<TH2F*>( o.object );
      assert( obj );

      // This applies to all
      gStyle->SetCanvasBorderMode(0);
      gStyle->SetPadBorderMode(0);
      gStyle->SetPadBorderSize(0);

      gStyle->SetOptStat(0);
      gStyle->SetPalette(1);

      obj->SetStats(kFALSE);
      obj->SetOption("colz");

      if(obj->GetEntries() != 0)
        c->SetLogz(0);

      obj->GetXaxis()->SetLabelSize(0.06);
      obj->GetYaxis()->SetLabelSize(0.06);

      // Summary map
      if( o.name.find( "reportSummaryMap" ) != std::string::npos )
      {
        dqm::utils::reportSummaryMapPalette(obj);
        obj->GetXaxis()->SetNdivisions(4,true);
        obj->GetYaxis()->SetNdivisions(8,true);
        obj->GetXaxis()->CenterLabels();
        obj->GetYaxis()->CenterLabels();
        c->SetGrid(1,1);
        return;
      }

      // ------------------- summary plots -------------------
      if(o.name.find("energySummaryMap") != std::string::npos)
      {
        obj->GetXaxis()->SetNdivisions(4,true);
        obj->GetYaxis()->SetNdivisions(4,true);
        obj->GetXaxis()->CenterLabels();
        obj->GetYaxis()->CenterLabels();
        c->SetGrid(1,1);
        obj->GetXaxis()->SetTitleOffset(1.15);
        c->SetBottomMargin(0.1);
        c->SetLeftMargin(0.15);
        c->SetRightMargin(0.12);
        obj->SetMinimum(-0.00000001);
        obj->SetMaximum(2.0);

        int colorErrorDI[2];
        colorErrorDI[1] = 416;// kGreen
        colorErrorDI[0] = 632;// kRed
        gStyle->SetPalette(2, colorErrorDI);
        return;
      }

      if(o.name.find("kinematicsSummaryMap") != std::string::npos)
      {
        obj->GetXaxis()->SetNdivisions(6,true);
        obj->GetYaxis()->SetNdivisions(4,true);
        obj->GetXaxis()->CenterLabels();
        obj->GetYaxis()->CenterLabels();
        c->SetGrid(1,1);
        obj->GetXaxis()->SetTitleOffset(1.15);
        c->SetBottomMargin(0.1);
        c->SetLeftMargin(0.15);
        c->SetRightMargin(0.12);
        obj->SetMinimum(-0.00000001);
        obj->SetMaximum(2.0);

        int colorErrorDI[2];
        colorErrorDI[1] = 416;// kGreen
        colorErrorDI[0] = 632;// kRed
        gStyle->SetPalette(2, colorErrorDI);
        return;
      }

      if(o.name.find("muonIdSummaryMap") != std::string::npos)
      {
        obj->GetXaxis()->SetNdivisions(3,true);
        obj->GetYaxis()->SetNdivisions(4,true);
        obj->GetXaxis()->CenterLabels();
        obj->GetYaxis()->CenterLabels();
        c->SetGrid(1,1);
        obj->GetXaxis()->SetTitleOffset(1.15);
        c->SetBottomMargin(0.1);
        c->SetLeftMargin(0.15);
        c->SetRightMargin(0.12);
        obj->SetMinimum(-0.00000001);
        obj->SetMaximum(1.5);

        int colorErrorDI[3];
        colorErrorDI[2] = 416;// kGreen
        colorErrorDI[1] = 400;// kYellow
        colorErrorDI[0] = 632;// kRed
        gStyle->SetPalette(3, colorErrorDI);
        return;
      }

      if(o.name.find("residualsSummaryMap") != std::string::npos)
      {
        obj->GetXaxis()->SetNdivisions(4,true);
        obj->GetYaxis()->SetNdivisions(5,true);
        obj->GetXaxis()->CenterLabels();
        obj->GetYaxis()->CenterLabels();
        c->SetGrid(1,1);
        obj->GetXaxis()->SetTitleOffset(1.15);
        c->SetBottomMargin(0.1);
        c->SetLeftMargin(0.15);
        c->SetRightMargin(0.12);
        obj->SetMinimum(-0.00000001);
        obj->SetMaximum(2.0);

        int colorErrorDI[2];
        colorErrorDI[1] = 416;// kGreen
        colorErrorDI[0] = 632;// kRed
        gStyle->SetPalette(2, colorErrorDI);
        return;
      }
    }

  void preDrawTH1( TCanvas *, const VisDQMObject & )
    {
      return;
    }

  void postDrawTProfile2D( TCanvas *, const VisDQMObject & )
    {
      return;
    }

  void postDrawTProfile( TCanvas *, const VisDQMObject & )
    {
      return;
    }

  void postDrawTH2( TCanvas *, const VisDQMObject &o )
    {
      if( o.name.find( "reportSummaryMap" ) != std::string::npos )
      {
        label_resTk->Draw("same");
        label_resSTA->Draw("same");
        label_muonIdSTA->Draw("same");
        return;
      }
      if(o.name.find("energySummaryMap") != std::string::npos)
      {
        label_glbHO->Draw("same");
        label_tkHO->Draw("same");
        label_staHO->Draw("same");
        return;
      }
    }

  void postDrawTH1( TCanvas *, const VisDQMObject & )
    {
    }
};

static MuonRenderPlugin instance;
