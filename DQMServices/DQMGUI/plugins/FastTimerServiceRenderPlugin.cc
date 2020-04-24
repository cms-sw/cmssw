#include <cmath>

#include <TProfile2D.h>
#include <TProfile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TText.h>

#include "DQMServices/DQMGUI/interface/DQMRenderPlugin.h"
#include "utils.h"

class FastTimerServiceRenderPlugin : public DQMRenderPlugin {
public:
  bool applies(const VisDQMObject &o, const VisDQMImgInfo &) override {
    if (o.name.find("HLT/TimerService") == 0)
      return true;

    return false;
  }

  void preDraw(TCanvas *, const VisDQMObject &, const VisDQMImgInfo &, VisDQMRenderInfo &) override {}

  void postDraw(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &i) override {
    if ((o.name == "HLT/TimerService/event_bylumi") or (o.name == "HLT/TimerService/source_bylumi") or
        (o.name == "HLT/TimerService/all_paths_bylumi") or (o.name == "HLT/TimerService/all_endpaths_bylumi")) {
      customiseLumisectionRange(c, o, i);
    }

    if ((o.name == "HLT/TimerService/paths_active_time") or (o.name == "HLT/TimerService/paths_total_time") or
        (o.name == "HLT/TimerService/paths_exclusive_time") or (o.name == "HLT/TimerService/event_bylumi") or
        (o.name == "HLT/TimerService/source_bylumi") or (o.name == "HLT/TimerService/all_paths_bylumi") or
        (o.name == "HLT/TimerService/all_endpaths_bylumi")) {
      customiseTProfile(c, o, i);
    }
  }

private:
  void customiseTProfile(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &) {
    // is this needed ?
    c->cd();

    TProfile *obj = dynamic_cast<TProfile *>(o.object);
    // this is supposed to be a TProfile
    if (not obj)
      return;

    // disable statistics panel, draw as a histogram, set black line color
    obj->SetStats(false);
    obj->SetDrawOption("H");
    obj->SetLineColor(kBlack);

    // make a copy and draw red Y error bars
    TProfile *copy = (TProfile *)obj->DrawCopy("SAME E1 X0");
    copy->SetLineColor(kRed);
  }

  void customiseLumisectionRange(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &i) {
    // is this needed ?
    c->cd();

    TH1 *obj = dynamic_cast<TH1 *>(o.object);
    // this is supposed to be a TH1 or derived object
    if (not obj)
      return;

    // if the maximum is not specified, find maximum filled bin
    if (std::isnan(i.xaxis.max)) {
      int xmax = 0;
      for (int i = 1; i <= obj->GetNbinsX(); ++i)
        if (obj->GetBinContent(i))
          xmax = i;
      TAxis *a = obj->GetXaxis();
      a->SetRange(a->GetFirst(), xmax);
    }
  }
};

static FastTimerServiceRenderPlugin instance;
