/*
 *	file:			        HcalrelvalObjectCustomizer.h
 *	original author:		Viktor Khristenko
 *	modified by:			Shubham Pandey (shubham.pandey@cern.ch)
 */

#ifndef _HCAL_RELVAL_OBJECTCUSTOMIZER_H
#define _HCAL_RELVAL_OBJECTCUSTOMIZER_H

//	ROOT Includes
#include "TCanvas.h"
#include "TText.h"
#include "TColor.h"
#include "TROOT.h"
#include "TH1.h"
#include "TH1D.h"
#include "TH1F.h"
#include "TH2.h"
#include "TH2D.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TH3D.h"
#include "TStyle.h"
#include "TGaxis.h"
#include "TText.h"
#include "TLine.h"
#include "TPaletteAxis.h"

//	STD includes
#include <string>
#include <vector>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <map>
using namespace std;

//	hcaldqm includes

//	hcaldqm namespace for HcalObjectCustomization API
namespace hcaldqm {

  //	Render Types: 2 so far
  enum RenderType { kHcal = 0, kHcalCalib = 1, nRenderType = 2 };

  //	Specifies the ROOT Class Type
  enum ROOTType {
    kTH1D = 0,
    kTH1F = 1,
    kTH2D = 2,
    kTH2F = 3,
    kTProfile = 4,
    kTProfile2D = 5,
    kTH3D = 6,
    kInvalid = 7,
    nROOTTypes = 8
  };

  //	Specifies the TObject bits convention
  int const bitshift = 19;
  enum ObjectBits { kAxisLogx = 0, kAxisLogy = 1, kAxisLogz = 2, kAxisLS = 3, kAxisFlag = 4, nObjectBits = 5 };

  //	Summary Detector Status Constants
  //	to be used in the next iteration of cmssw
  enum State { fNONE = 0, fNCDAQ = 1, fNA = 2, fGOOD = 3, fPROBLEMATIC = 4, fBAD = 5, fRESERVED = 6, nState = 7 };

  //	Class HcalrelvalObjectCustomizer
  class HcalrelvalObjectCustomizer {
  public:
    HcalrelvalObjectCustomizer() : _type(kHcal), _verbosity(0) {}
    ~HcalrelvalObjectCustomizer() {}

    //	Apply standard customizations for Styles or Canvas or anything
    //	Apply standard customizations for all of the objects
    void preDraw_Standard(TCanvas* c) {
      if (_verbosity > 0)
        std::cout << "Calling preDraw_Standard" << std::endl;
      gStyle->Reset("Default");
      gStyle->SetCanvasColor(10);
      gStyle->SetPadColor(10);
      gStyle->SetFillColor(10);
      gStyle->SetStatColor(10);
      gStyle->SetTitleFillColor(10);
      gStyle->SetOptTitle(kTRUE);
      gStyle->SetTitleBorderSize(0);
      gStyle->SetOptStat(kFALSE);
      gStyle->SetStatBorderSize(1);
      gROOT->ForceStyle();

      TGaxis::SetMaxDigits(4);
      c->Draw();
    }

    void postDraw_Standard(TCanvas*, VisDQMObject const&, VisDQMImgInfo const&) {}

    //	Initialize all the Color Schemes
    void initialize_ColorSchemes() {
      //	summary
      _n_summary = nState - fNONE;
      _colors_summary[0] = kWhite;
      _colors_summary[1] = kGray;
      _colors_summary[2] = kWhite;
      _colors_summary[3] = kGreen;
      _colors_summary[4] = kYellow;
      _colors_summary[5] = kRed;
      _colors_summary[6] = kBlack;
      _contours_summary[0] = fNONE;
      _contours_summary[1] = fNCDAQ;
      _contours_summary[2] = fNA;
      _contours_summary[3] = fGOOD;
      _contours_summary[4] = fPROBLEMATIC;
      _contours_summary[5] = fBAD;
      _contours_summary[6] = fRESERVED;
      _contours_summary[7] = nState;
    }

    //	Initialize Filters - Names for Searching
    void initialize_Filters() {}

    //	Initialize Render Type
    void initialize_Type(RenderType type) { _type = type; }

    //	Customize 1D
    void pre_customize_1D(
        hcaldqm::ROOTType type, TCanvas* c, VisDQMObject const& o, VisDQMImgInfo const& ii, VisDQMRenderInfo& ri) {
      if (_verbosity > 0)
        std::cout << "Calling customize_1D" << std::endl;

      //spandey: Basic cosmetics
      c->cd();
      c->GetPad(0)->SetFillColor(kCyan - 10);
      c->GetPad(0)->SetGridx();
      c->GetPad(0)->SetGridy();

      //Get appropriate ranges according to the last loaded sample
      hist_range values = driver(o.object->GetName());

      if (values.rebin != -1) {
        if (values.xmax > 0 || values.xmin != 0)  //change xAxis range
          ((TH1F*)o.object)->GetXaxis()->SetRangeUser(values.xmin, values.xmax);

        //yAxis range, log_flag etc
        if (values.ymin != 0)
          ((TH1F*)o.object)->SetMinimum(values.ymin);
        if (values.ymax > 0)
          ((TH1F*)o.object)->SetMaximum(values.ymax);
        if (values.log_flag) {
          c->GetPad(0)->SetLogy();
          if (values.ymax < 0)
            ((TH1F*)o.object)->SetMaximum((((TH1F*)o.object)->GetMaximum()) * 5);
        }

        //cosmetics
        if ((type == kTH1F) || (type == kTH1D)) {
          ((TH1F*)o.object)->SetLineStyle(2);
          ((TH1F*)o.object)->SetLineWidth(2);
          if (values.chi2_flag) {
            ((TH1F*)o.object)->SetFillColor(42);
            ((TH1F*)o.object)->SetLineColor(kBlack);
          } else
            ((TH1F*)o.object)->SetLineColor(kPink);

          ri.drawOptions = "hist";
          c->Modified();
        }
      }
      //	by default

      //	further customization
      this->pre_customize_ByName(c, o, ii, ri);
      this->pre_customize_ByBits(type, c, o, ii, ri);

      //	for 1D Profiles
      //	NOTE: for profiles always hide non occupied xbins
      //	independent of the type of the axis (LS or not...)
      if (type == kTProfile) {
        TProfile* obj = dynamic_cast<TProfile*>(o.object);
        //cosmetics

        obj->SetErrorOption("");
        obj->SetMarkerStyle(20);
        obj->SetLineColor(kGreen + 2);
        obj->SetLineStyle(1);
        obj->SetLineWidth(1);
        obj->SetMarkerColor(kGreen + 2);
        obj->SetMarkerStyle(22);
        obj->SetMarkerSize(1.0);
        ri.drawOptions = "hist pl";
        c->Modified();
      }
    }

    //	Customize 2D
    void pre_customize_2D(
        hcaldqm::ROOTType type, TCanvas* c, VisDQMObject const& o, VisDQMImgInfo const& ii, VisDQMRenderInfo& ri) {
      if (_verbosity > 0)
        std::cout << "Calling customize_2D" << std::endl;

      //	By default
      //	1) colz
      //	2) No Statistics for 2D plots
      //	3) z plot range (min, max)
      //	4) Set Rainbow Palette(1 is for Rainbow Color Map)
      ri.drawOptions = "colz";
      ((TH2*)o.object)->SetStats(kFALSE);
      ((TH2*)o.object)->GetZaxis()->SetRangeUser(((TH2*)o.object)->GetMinimum(), ((TH2*)o.object)->GetMaximum());
      gStyle->SetPalette(1);

      //
      this->pre_customize_ByName(c, o, ii, ri);
      this->pre_customize_ByBits(type, c, o, ii, ri);
    }

    //	post customize 1D
    void post_customize_1D(hcaldqm::ROOTType, TCanvas*, VisDQMObject const&, VisDQMImgInfo const&) {}

    //	post customize 2D
    void post_customize_2D(hcaldqm::ROOTType, TCanvas*, VisDQMObject const& o, VisDQMImgInfo const&) {
      if (_verbosity > 0)
        std::cout << "Caliing post_customize_2D" << std::endl;

      TString fullpath(o.name.c_str());

      //	for summaries
      /*if (fullpath.Contains("Summary"))
	{
	gPad->Update();
	TBox *box_GOOD = new TBox(0.8, 0.8, 0.9, 0.9);
	box_GOOD->SetFillColor(kGreen);
	box_GOOD->Draw();
	}*/
    }

    //	Customize By Name
    void pre_customize_ByName(TCanvas* c, VisDQMObject const& o, VisDQMImgInfo const&, VisDQMRenderInfo& ri) {
      if (_verbosity > 0)
        std::cout << "Calling customize_ByName" << std::endl;

      TString fullpath(o.name.c_str());

      //	for summaries
      if (fullpath.Contains("Summary")) {
        ri.drawOptions = "col";
        if (!fullpath.Contains("runSummary"))
          c->SetGrid();
        gStyle->SetPalette(_n_summary, _colors_summary);
        ((TH2*)o.object)->SetContour(_n_summary + 1, _contours_summary);
      }
    }

    //	Customize by Bits
    void pre_customize_ByBits(
        hcaldqm::ROOTType type, TCanvas*, VisDQMObject const& o, VisDQMImgInfo const&, VisDQMRenderInfo&) {
      if (_verbosity > 0)
        std::cout << "Calling customize_ByBits" << std::endl;

      for (int i = 0; i < nObjectBits; i++) {
        if (!isBitSet(i, o.object))
          continue;

        //	do based on which bit it is
        switch (i) {
          case kAxisLogx:
            gPad->SetLogx(1);
            break;
          case kAxisLogy:
            gPad->SetLogy(1);
            break;
          case kAxisLogz:
            gPad->SetLogz(1);
            break;
          case kAxisLS:
            if (type == kTH2D || type == kTH2F) {
              //	for 2D with X axis as LS
              TH2* obj = dynamic_cast<TH2*>(o.object);
              bool foundfirst = false;
              int first = 1;
              int last = 1;
              for (int i = first; i <= obj->GetNbinsX(); i++) {
                if (!foundfirst && obj->GetBinContent(i, 1) != 0) {
                  first = i;
                  foundfirst = true;
                }
                if (obj->GetBinContent(i, 1) != 0)
                  last = i + 1;
              }
              if (last - first >= 1)
                obj->GetXaxis()->SetRange(first, last);
            } else if (type == kTH1F || type == kTH1D) {
              //	for 1D with X axis as LS
              TH1* obj = dynamic_cast<TH1*>(o.object);
              bool foundfirst = false;
              int first = 1;
              int last = 1;
              for (int i = first; i <= obj->GetNbinsX(); i++) {
                if (!foundfirst && obj->GetBinContent(i) != 0) {
                  first = i;
                  foundfirst = true;
                }
                if (obj->GetBinContent(i) != 0)
                  last = i + 1;
              }
              if (last - first >= 1)
                obj->GetXaxis()->SetRange(first, last);
              obj->SetMarkerStyle(20);
            }
            break;
          case kAxisFlag:
            break;
          default:
            break;
        }
      }
    }

    //This is the driver function
    // This function returns ranges according to histogram name
    hist_range driver(std::string hist_name) {
      std::vector<std::map<std::string, hist_range> > vec;

      //initializes all maps from HcalrelvalMaps.h file
      vec.push_back(map0());
      vec.push_back(map1());
      vec.push_back(map2());
      vec.push_back(map3());
      int ii = 2;  //Using standard value i.e. rangeMedium
      std::map<std::string, hist_range>::iterator search = vec[ii].find(hist_name);
      //hist_range foobar= NULL;
      hist_range foobar;
      if (search != vec[ii].end()) {
        foobar = search->second;
        //std::cout<<"Key found!!"<<std::endl;
        //std::cout<<"hist_name:"<<hist_name<<", xmin:"<<foobar.xmin<<", xmax:"<<foobar.xmax
        //<<", ymin:"<<foobar.ymin<<", ymax:"<<foobar.ymax<<", log_flag:"<<foobar.log_flag
        //<<", chi2:"<<foobar.chi2_flag<<", (search->second.chi2_flag):"<<(search->second.chi2_flag)<<std::endl;
      } else {
        foobar.xmin = -1.0;
        foobar.xmax = -1.0;
        foobar.ymin = -1.0;
        foobar.ymax = -1.0;
        foobar.rebin = -1;
        foobar.log_flag = false;
        foobar.chi2_flag = false;
        //std::cout<<"Key:"<<hist_name<<" Not Found!!\n";
      }

      return foobar;
    }

    /*
     *  End of function
     *
     */

  protected:
    bool isBitSet(int ibit, TObject* o) { return o->TestBit(BIT(ibit + bitshift)); }

    // Some members....
  protected:
    RenderType _type;
    int _verbosity;

    /*
     *	Declare all the colors/contours
     */
    //	summary
    unsigned int _n_summary;
    Int_t _colors_summary[10];
    Double_t _contours_summary[10];
  };
}  // namespace hcaldqm

#endif
