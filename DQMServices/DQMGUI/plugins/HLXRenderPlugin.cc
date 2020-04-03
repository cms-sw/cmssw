// -*- C++ -*-
//
// Package:     RenderPlugins
// Class  :     HLXRenderPlugin
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sat Apr 19 20:02:57 CEST 2008
// $Id: HLXRenderPlugin.cc,v 1.21 2011/09/09 11:53:42 lilopera Exp $
//

// user include files
#include "DQM/DQMRenderPlugin.h"
#include "utils.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TGraph.h"
#include "TLine.h"
#include "TROOT.h"
#include <iostream>
#include <cassert>

class HLXRenderPlugin : public DQMRenderPlugin
{
public:
  virtual bool applies( const VisDQMObject &o, const VisDQMImgInfo &)
    {
      if( o.name.find( "HLX/Luminosity" )  != std::string::npos )
        return true;

      if( o.name.find( "HLX/HFMinus" ) != std::string::npos )
        return true;

      if( o.name.find( "HLX/HFPlus" ) != std::string::npos )
        return true;

      if( o.name.find( "HLX/HFCompare" ) != std::string::npos )
        return true;

      if( o.name.find( "HLX/Average" ) != std::string::npos )
        return true;

      if( o.name.find( "HLX/CheckSums" ) != std::string::npos )
        return true;

      if( o.name.find( "HLX/SigBkgLevels" ) != std::string::npos )
        return true;

      if( o.name.find( "HLX/History" ) != std::string::npos )
        return true;

      if( o.name.find( "HLX/EventInfo" ) != std::string::npos )
        return true;

      return false;
    }

  virtual void preDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo &r)
    {
      c->cd();

      gStyle->Reset("Plain");
      gStyle->SetErrorX(0);
      gStyle->SetEndErrorSize(10);
      gStyle->SetCanvasColor(10);
      gStyle->SetPadColor(10);
      gStyle->SetFillColor(10);
      gStyle->SetOptTitle(kTRUE);
      gStyle->SetTitleBorderSize(0);
      gStyle->SetOptStat(kTRUE);
      gStyle->SetStatBorderSize(1);
      gStyle->SetStatColor(kWhite);
      gStyle->SetTitleFillColor(kWhite);
      gStyle->SetPadTickX(1);
      gStyle->SetPadTickY(1);
      gStyle->SetPadLeftMargin(0.15);

      gROOT->ForceStyle();

      r.drawOptions = "";
      if( dynamic_cast<TProfile*>( o.object ) )
      {
        preDrawTProfile( c, o );
      }
      else if( dynamic_cast<TH1F*>( o.object ) )
      {
        preDrawTH1F( c, o );
      }
      else if( dynamic_cast<TH2F*>( o.object ) )
      {
        preDrawTH2F( c, o );
        //r.drawOptions = "colz";
      }
    }

private:
  void preDrawTProfile( TCanvas *, const VisDQMObject &o )
    {
      //std::cout << "NADIA: Here in predraw PROFILE!!!!" << std::endl;

      TProfile* obj = dynamic_cast<TProfile*>( o.object );

      assert( obj );

      // Average plots for Etsum/Tower Occupancy
      obj->SetStats(kTRUE);
      obj->GetYaxis()->SetTitleOffset(1.3);

      if( (o.name.find("AvgEtSum") != std::string::npos ||
	   o.name.find("AvgOcc")   != std::string::npos) &&
	  (o.name.find("Lumi") == std::string::npos &&
	   o.name.find("Hist") == std::string::npos) ){

	 // If there isn't a big range we need more room for the
	 // fractions ...
	 if( (obj->GetMaximum() - obj->GetMinimum()) < 0.001 &&
	     obj->GetMaximum() > 0.1 ){
	    gPad->SetLeftMargin(0.20);
	    obj->GetYaxis()->SetLabelSize(0.03);
	    obj->GetYaxis()->SetTitleOffset(2.2);
	 }

	 if( o.name.find("AvgEtSum") != std::string::npos ){
	    gPad->SetTopMargin(0.15);
	 }else{
	    obj->SetErrorOption("");
	 }

	 obj->SetMarkerStyle(21);
	 obj->SetMarkerSize(0.6);
	 obj->SetMarkerColor(kBlue);
      }

      if( o.name.find("LumiAvg") != std::string::npos ){
	 obj->SetMarkerStyle(kPlus);
	 obj->SetMarkerColor(kBlue);
      }

      // Sum All Occupancy plots (check sums)
      if( o.name.find("SumAllOcc") != std::string::npos ){
        obj->SetMaximum( 1.25*obj->GetMaximum() );
        obj->SetMinimum( 0.75*obj->GetMinimum() );
        obj->SetMarkerStyle(21);
        obj->SetMarkerSize(0.6);
	obj->SetMarkerColor(6);
      }

      // History histograms ...
      if( o.name.find("Hist") != std::string::npos ){
        obj->SetMarkerStyle(kFullCircle);
        obj->SetMarkerSize(0.8);
        // Loop over the bins and find the first empty one
        int firstEmptyBin = obj->GetNbinsX();
        for( int iBin = 1; iBin<=obj->GetNbinsX(); ++iBin ){
          if( obj->GetBinContent(iBin) == 0 ){
            firstEmptyBin = iBin;
            break;
          }
        }
        if( firstEmptyBin < 2 ) firstEmptyBin = 2;
        // Now set the new range
        obj->GetXaxis()->SetRange(0,firstEmptyBin-1);
      }
    }

  void preDrawTH1F( TCanvas *, const VisDQMObject &o )
    {
      TH1F* obj = dynamic_cast<TH1F*>( o.object );
      assert( obj );

      // Average plots for Etsum/Tower Occupancy
      obj->SetStats(kTRUE);
      obj->GetYaxis()->SetTitleOffset(1.3);

      if( o.name.find("Lumi") == std::string::npos )
      {
        obj->SetFillColor(kBlue);
        obj->SetLineColor(kBlue);
        obj->SetFillStyle(1001);
      }

      if( o.name.find("MaxInstLumiBXNum") != std::string::npos )
      {
        obj->SetFillColor(kBlue);
        obj->SetLineColor(kBlue);
        obj->SetFillStyle(1001);
      }

      // HF comparison histograms
      if( o.name.find("HFCompareOcc") != std::string::npos ){
	 if( (obj->GetMaximum() - obj->GetMinimum()) < 0.001 &&
	     obj->GetMaximum() > 0.1 ){
	    gPad->SetLeftMargin(0.20);
	    obj->GetYaxis()->SetLabelSize(0.03);
	    obj->GetYaxis()->SetTitleOffset(2.2);
	 }
      }

      // Wedge by wedge histograms
      if( o.name.find("ETSum") != std::string::npos ||
	  o.name.find("Set1_") != std::string::npos ||
	  o.name.find("Set2_") != std::string::npos ){
	 gPad->SetLeftMargin(0.15);
	 //obj->GetYaxis()->SetLabelSize(0.03);
	 obj->GetYaxis()->SetTitleOffset(1.7);

	 if( (obj->GetMaximum() - obj->GetMinimum()) < 0.001 &&
	     obj->GetMaximum() > 0.1 ){
	    gPad->SetLeftMargin(0.20);
	    obj->GetYaxis()->SetLabelSize(0.03);
	    obj->GetYaxis()->SetTitleOffset(2.2);
	 }
      }

      if( (o.name.find("EtSum") != std::string::npos && o.name.find("Lumi") == std::string::npos) ||
          o.name.find("ETSum") != std::string::npos )
      {
        int maxBin = obj->GetMaximumBin();
        int minBin = obj->GetMinimumBin();

        double objMax = obj->GetBinContent(maxBin);
        double objMin = obj->GetBinContent(minBin);
        if( objMax > 0 && objMin > 0 )
        {
          if( (objMax/objMin) > 1e4 ) gPad->SetLogy();
        }
        obj->SetMaximum(objMax*1.02);
        obj->SetMinimum(objMin*0.98);
      }

      if( o.name.find("Lumi") != std::string::npos &&
          o.name.find("HistInstantLumi") == std::string::npos &&
          o.name.find("HistIntegratedLumi") == std::string::npos )
      {
        obj->SetLineColor(kBlack);
        obj->SetMarkerStyle(kPlus);

	if( o.name.find("LumiInstant") != std::string::npos )
	   obj->SetMarkerColor(8);
	else if( o.name.find("LumiIntegrated") != std::string::npos )
	   obj->SetMarkerColor(kRed);

      }

      if( o.name.find("MaxInstLumiBX") != std::string::npos &&
	  o.name.find("MaxInstLumiBXNum") == std::string::npos ){
	 obj->SetLineColor(kBlack);
	 obj->SetMarkerStyle(21);
	 obj->SetMarkerSize(0.6);
	 obj->SetMarkerColor(kBlue);
	 obj->SetOption("e0");
      }

      if( o.name.find("HistInstantLumi") != std::string::npos || o.name.find("HistIntegratedLumi") != std::string::npos  )
      {
	 if( o.name.find("Error") == std::string::npos ){
	    obj->SetMarkerStyle(kFullCircle);
	    obj->SetMarkerSize(0.8);
	    obj->SetLineColor(kBlack);
	    obj->SetOption("e0");
	 } else {
	    obj->SetMarkerStyle(21);
	    obj->SetMarkerColor(kRed);
	    obj->SetMarkerSize(1.2);
	    obj->SetOption("p");
	 }
	 // Loop over the bins and find the first empty one
	 int firstEmptyBin = obj->GetNbinsX();
	 for( int iBin = 1; iBin<=obj->GetNbinsX(); ++iBin )
	 {
	    if( obj->GetBinContent(iBin) == 0 )
	    {
	       firstEmptyBin = iBin;
	       break;
	    }
	 }
	 if( firstEmptyBin < 2 ) firstEmptyBin = 2;
	 // Now set the new range
	 obj->GetXaxis()->SetRange(0,firstEmptyBin-1);
      }
    }

  void preDrawTH2F( TCanvas *c, const VisDQMObject &o )
    {
      TH2F* obj = dynamic_cast<TH2F*>( o.object );

      assert( obj );
      obj->SetStats(kFALSE);

      if( o.name.find("Time") != std::string::npos )
      {
        // Loop over the bins and find the first empty one
        int firstEmptyBin = obj->GetNbinsX();
        for( int iBin = 1; iBin<=obj->GetNbinsX(); ++iBin )
        {
          if( obj->GetBinContent(iBin,1) == 0 )
          {
            firstEmptyBin = iBin;
            break;
          }
        }
        if( firstEmptyBin <= 2 ) firstEmptyBin  = 1;
        else                     firstEmptyBin -= 2;
        // Now set the new range
        obj->GetXaxis()->SetRange(0,firstEmptyBin);

        unsigned int Number = 3;
        double       Red[]   = { 0.00, 0.00, 1.00};
        double       Green[] = { 0.00, 1.00, 0.00};
        double       Blue[]  = { 1.00, 0.00, 0.00};
        double       Stops[] = { 0.00, 0.50, 1.00 };
        int          nb = 100;
        TColor::CreateGradientColorTable(Number,Stops,Blue,Green,Red,nb);
        obj->SetOption("colz");

        c->SetRightMargin(2*c->GetRightMargin());
      }
      else
      {
        obj->SetMinimum(-.001);
        obj->SetMaximum(1.02);

        gStyle->SetPaintTextFormat("5.4g");
        dqm::utils::reportSummaryMapPalette(obj);
        obj->SetMarkerSize(3);
        obj->GetYaxis()->SetLabelSize(0.07);
        obj->GetXaxis()->SetTitleSize(0.05);
        obj->GetXaxis()->SetTitleOffset(0.8);
        obj->SetOption("colztext90");
      }
      //    //gStyle->SetPalette(1);
    }
};

static HLXRenderPlugin instance;
