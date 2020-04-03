/*!
  \file SiStripLASRenderPlugin
  \brief Display Plugin for Pixel DQM Histograms
  \author A. Perieanu
  \version $Revision: 1.3 $
  \date $Date: 2011/09/09 11:53:43 $
*/

#include "DQM/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile2D.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TGraphPolar.h"
#include "TColor.h"
#include "TText.h"
#include "TLine.h"
#include <cassert>

class SiStripLASRenderPlugin : public DQMRenderPlugin{
public:
  virtual bool applies( const VisDQMObject &o, const VisDQMImgInfo & ){
      if( o.name.find( "SiStripLAS/" ) != std::string::npos )
        return true;

      return false;
    }

  virtual void preDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo & )
    {
      c->cd();
      if( dynamic_cast<TH2*>( o.object ) )
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

      if( dynamic_cast<TH2*>( o.object ) )
      {
        postDrawTH2( c, o );
      }
      else if( dynamic_cast<TH1*>( o.object ) )
      {
        postDrawTH1( c, o );
      }
    }

private:

  void preDrawTH2( TCanvas* c, const VisDQMObject &o ){
    TH2* obj = dynamic_cast<TH2*>( o.object );
    assert( obj );

    gStyle->SetCanvasBorderMode( 0 );
    gStyle->SetPadBorderMode( 0 );
    gStyle->SetPadBorderSize( 0 );
    gStyle->SetOptStat( 0 );
    obj->SetStats( kFALSE );

    c->SetRightMargin(0.15);

    //if( o.name.find( "NumberOfSignals_AlignmentTubes" ) != std::string::npos){
    if( o.name.find( "NumberOfSignals_" ) != std::string::npos ||
        o.name.find( "reportSummaryMap" )  != std::string::npos){
      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);
      gStyle->SetOptTitle(0);
      gStyle->SetNumberContours(4);

      const int Number = 4;
      double Red[4]    = { 1.00, 1.00, 0.00, 0.00};
      double Yellow[Number] = { 0.00, 1.00, 0.80, 0.10};
      double Blue[Number]   = { 0.00, 0.00, 0.10, 0.10};
      double Length[Number] = { 0.00, 0.25, 0.50, 1.00};
      int nb=4;
      TColor::CreateGradientColorTable(Number,Length,Red,Yellow,Blue,nb);
      obj->SetContour(nb);

      TAxis* ya = obj->GetYaxis();
      ya->SetTitleOffset(0.75);
      ya->SetTitleSize(0.05);
      ya->SetLabelSize(0.05);

      obj->SetOption("colz");
      gPad->SetGridx();
      gPad->SetGridy();
    }
   }
  //_____________________________________________________
  void preDrawTH1( TCanvas *, const VisDQMObject &o ){
    TH1* obj = dynamic_cast<TH1*>( o.object );
    assert( obj );

    // This applies to all
    gStyle->SetOptStat(111);

    TAxis* xa = obj->GetXaxis();
    TAxis* ya = obj->GetYaxis();
    xa->SetTitleOffset(0.7);
    xa->SetTitleSize(0.065);
    xa->SetLabelSize(0.065);
    ya->SetTitleOffset(0.75);
    ya->SetTitleSize(0.065);
    ya->SetLabelSize(0.065);

  }

  void postDrawTH1( TCanvas *, const VisDQMObject &o ){
    TH1* obj = dynamic_cast<TH1*>( o.object );
    assert( obj );

  }

  void postDrawTH2( TCanvas *, const VisDQMObject &o ){

    TH2* obj = dynamic_cast<TH2*>( o.object );
    assert( obj );
  }

};

static SiStripLASRenderPlugin instance;
