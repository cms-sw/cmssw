/*
 *	file:			HcalObjectCustomizer.h
 *	author:			Viktor Khristenko
 *	Revision
 *	Description:	A wrapper around any Render Plugin.
 *		Created specifically for HcalRenderPlugin and 
 *		HcalCalibRenderPlugin. Provides object customization based on the 
 *		following hierarchical selections. 
 *		1) TObject::Class() -> TH1, TH2, TProfile, etc....
 *		=>	2.1) Based on the Object's name. See naming conventions
 *		=>	2.2) Based on the Bits of TObject using SetBit/TestBit
 *		Naming Conventions(or Name Filters):
 */

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

  //	Class HcalObjectCustomizer
  class HcalObjectCustomizer {
  public:
    HcalObjectCustomizer() : _type(kHcal), _verbosity(0) {}
    ~HcalObjectCustomizer() {}

    //	Apply standard customizations for Styles or Canvas or anything
    //	Apply standard customizations for all of the objects
    void preDraw_Standard(TCanvas* c) {
      if (_verbosity > 0)
        std::cout << "Calling preDraw_Standard" << std::endl;

      c->cd();

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

      //	by default

      //	further customization
      this->pre_customize_ByName(c, o, ii, ri);
      this->pre_customize_ByBits(type, c, o, ii, ri);

      //	for 1D Profiles
      //	NOTE: for profiles always hide non occupied xbins
      //	independent of the type of the axis (LS or not...)
      if (type == kTProfile) {
        TProfile* obj = dynamic_cast<TProfile*>(o.object);
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
