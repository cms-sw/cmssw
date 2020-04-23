/*!
  \File BeamRenderPlugin
  \Display Plugin for BeamSpot DQM Histograms
  \author
  \version $Revision: 1.12 $
  \date $Date: 2011/09/09 11:53:42 $
*/

#include "DQMServices/DQMGUI/interface/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile2D.h"
#include "TProfile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TText.h"
#include <cassert>

class BeamRenderPlugin : public DQMRenderPlugin {
public:
  virtual bool applies(const VisDQMObject &o, const VisDQMImgInfo &) {
    if ((o.name.find("BeamMonitor/") == std::string::npos) &&
        (o.name.find("BeamMonitor_PixelLess/") == std::string::npos) &&
        (o.name.find("TrackingHLTBeamspotStream/") == std::string::npos))
      return false;

    if (o.name.find("/EventInfo/") != std::string::npos)
      return true;

    if (o.name.find("/Fit/") != std::string::npos)
      return true;

    if (o.name.find("/FitBx/") != std::string::npos)
      return true;

    if (o.name.find("/PrimaryVertex/") != std::string::npos)
      return true;

    return false;
  }

  virtual void preDraw(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo &) {
    c->cd();

    if (dynamic_cast<TH2F *>(o.object)) {
      preDrawTH2F(c, o);
    }

    if (dynamic_cast<TH1F *>(o.object)) {
      preDrawTH1F(c, o);
    }

    if (dynamic_cast<TProfile *>(o.object)) {
      preDrawTProfile(c, o);
    }
  }

  virtual void postDraw(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &) {
    c->cd();

    if (dynamic_cast<TH2F *>(o.object)) {
      postDrawTH2F(c, o);
    }

    if (dynamic_cast<TH1F *>(o.object)) {
      postDrawTH1F(c, o);
    }
  }

private:
  void preDrawTH2F(TCanvas *c, const VisDQMObject &o) {
    TH2F *obj = dynamic_cast<TH2F *>(o.object);
    assert(obj);

    // This applies to all
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetPadBorderSize(0);

    TAxis *xa = obj->GetXaxis();
    TAxis *ya = obj->GetYaxis();

    //xa->SetTitleOffset(0.7);
    xa->SetTitleSize(0.04);
    xa->SetLabelSize(0.03);

    //ya->SetTitleOffset(0.7);
    ya->SetTitleSize(0.04);
    ya->SetLabelSize(0.03);

    if (o.name.find("trk_vx_vy") != std::string::npos) {
      gStyle->SetOptStat(11);
      obj->SetOption("colz");
      return;
    }

    xa->SetTitleSize(0.04);
    xa->SetLabelSize(0.045);
    ya->SetLabelSize(0.045);

    if (o.name.find("fitResults") != std::string::npos || o.name.find("pvResults") != std::string::npos ||
        o.name.find("_bx") != std::string::npos) {
      c->SetGrid();
      obj->SetStats(kFALSE);
      obj->SetMarkerSize(2.);
      return;
    }

    if (o.name.find("reportSummaryMap") != std::string::npos) {
      obj->SetStats(kFALSE);
      dqm::utils::reportSummaryMapPalette(obj);
      obj->SetOption("colz");
      return;
    }
  }

  void preDrawTH1F(TCanvas *c, const VisDQMObject &o) {
    TH1F *obj = dynamic_cast<TH1F *>(o.object);
    assert(obj);

    TAxis *xa = obj->GetXaxis();
    TAxis *ya = obj->GetYaxis();
    ya->SetTitleOffset(1.15);
    ya->SetTitleSize(0.04);
    ya->SetLabelSize(0.03);

    if (o.name.find("_lumi") != std::string::npos) {
      gStyle->SetOptStat(11);
      return;
    }

    xa->SetTitleOffset(1.15);
    xa->SetTitleSize(0.04);
    xa->SetLabelSize(0.03);

    if (o.name.find("_time") != std::string::npos) {
      gStyle->SetOptStat(11);
      return;
    }

    if (o.name.find("trkPt") != std::string::npos || o.name.find("cutFlowTable") != std::string::npos) {
      c->SetLogy();
      return;
    }
  }

  void preDrawTProfile(TCanvas *, const VisDQMObject &o) {
    TProfile *obj = dynamic_cast<TProfile *>(o.object);
    assert(obj);

    // This applies to all
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetPadBorderSize(0);

    TAxis *xa = obj->GetXaxis();
    TAxis *ya = obj->GetYaxis();

    //xa->SetTitleOffset(0.9);
    xa->SetTitleSize(0.04);
    xa->SetLabelSize(0.03);

    //ya->SetTitleOffset(0.9);
    ya->SetTitleSize(0.04);
    ya->SetLabelSize(0.03);

    if (o.name.find("d0_phi0") != std::string::npos) {
      gStyle->SetOptStat(11);
      return;
    }
  }

  void postDrawTH2F(TCanvas *c, const VisDQMObject &o) {
    TH2F *obj = dynamic_cast<TH2F *>(o.object);
    assert(obj);

    std::string name = o.name.substr(o.name.rfind("/") + 1);

    if (name.find("reportSummaryMap") != std::string::npos) {
      c->SetGridx();
      c->SetGridy();
      return;
    }
  }

  void postDrawTH1F(TCanvas *c, const VisDQMObject &o) {
    TH1F *obj = dynamic_cast<TH1F *>(o.object);
    assert(obj);

    std::string name = o.name.substr(o.name.rfind("/") + 1);

    if ((o.name.find("_lumi") != std::string::npos || o.name.find("_time") != std::string::npos) &&
        o.name.find("_all") == std::string::npos && o.name.find("nTrk") == std::string::npos &&
        o.name.find("nVtx") == std::string::npos && o.name.find("bx") == std::string::npos) {
      gStyle->SetErrorX(0.);
      obj->SetLineColor(2);
      obj->SetMarkerStyle(20);
      obj->SetMarkerSize(0.8);
      obj->SetMarkerColor(4);
      return;
    }

    if (o.name.find("_all") != std::string::npos && o.name.find("nVtx") == std::string::npos) {
      c->SetGridy();
      gStyle->SetErrorX(0.);
      gStyle->SetEndErrorSize(0.);
      obj->SetLineColor(2);
      obj->SetMarkerStyle(20);
      obj->SetMarkerSize(0.6);
      obj->SetMarkerColor(4);
      return;
    }

    if (o.name.find("Trending") != std::string::npos) {
      c->SetGridy();
      gStyle->SetErrorX(0.);
      gStyle->SetEndErrorSize(0.);
      obj->SetLineColor(2);
      obj->SetMarkerStyle(20);
      obj->SetMarkerSize(0.8);
      obj->SetMarkerColor(4);
      return;
    }
  }
};

static BeamRenderPlugin instance;
