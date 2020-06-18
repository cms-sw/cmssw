/*!
  \file Ratio2DComparisonRenderPlugin
  \brief RenderPlugin producing a comparison for 2D histograms.

This render plugin uses the API for references to render a simple 2D ratio
Comparison for 2D histograms. This allows using the "overlay" comparison with
2D histograms. 

This is faster than PullValue for big occupancy maps etc.

  \author Marcel Schneider
*/

#include "DQMServices/DQMGUI/interface/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile2D.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TLegend.h"

#include <cassert>
#include <string>

using namespace std;

class Ratio2DComparisonRenderPlugin : public DQMRenderPlugin {
public:
  bool applies(const VisDQMObject& o, const VisDQMImgInfo& i) override {
    // TODO: If you don't like the Ratio mode on your histograms, you could
    // opt-out here and provide your own render plugin.
    // Only use this beyond 1000 bins -- PullVal is better but too slow there.
    auto object2d = dynamic_cast<TH2*>(o.object);
    if (i.objects.size() == 2 && object2d && object2d->GetNcells() >= 1000) {
      return true;
    } else {
      return false;
    }
  }

  // Here, we replace the main object with out own comparison.
  void preDraw(TCanvas* canvas, const VisDQMObject& o, const VisDQMImgInfo& ii, VisDQMRenderInfo& renderInfo) override {
    canvas->cd();
    // ii.objects[0] == o
    TH2* data_hist = dynamic_cast<TH2*>(ii.objects[0].object);
    TH2* ref_hist = dynamic_cast<TH2*>(ii.objects[1].object);

    if (!data_hist || data_hist->GetDimension() != 2 || !ref_hist || ref_hist->GetDimension() != 2 ||
        ref_hist->GetNbinsX() != data_hist->GetNbinsX() || ref_hist->GetNbinsY() != data_hist->GetNbinsY()) {
      return;
    }

    TH2* ratio_hist = dynamic_cast<TH2*>(data_hist->Clone());

    double normfact = data_hist->GetEntries() / ref_hist->GetEntries();
    const uint32_t SUMMARY_PROP_EFFICIENCY_PLOT = 0x00200000;
    if (dynamic_cast<TProfile2D*>(data_hist) || (o.flags & SUMMARY_PROP_EFFICIENCY_PLOT)) {
      normfact = 1;
    }

    ratio_hist->Divide(data_hist, ref_hist, 1, normfact);

    double min = ratio_hist->GetMinimum(1e-2);
    double max = ratio_hist->GetMaximum(1e2);

    // only mess with the axis if the user does not override it.
    // This will also override most other render plugins.
    if (ii.zaxis.type == "def") {
      // grow axis range to be symmetrical wrt log-scale.
      if (min < (1.0 / max))
        max = 1.0 / min;
      if (max > (1.0 / min))
        min = 1.0 / max;
      ratio_hist->SetMinimum(min);
      ratio_hist->SetMaximum(max);

      // use log scale, except when the range is very small
      // (lin and log are basically equal then, but lin has better ticks)
      if (min < 0.90) {
        auto axistype = const_cast<std::string*>(&ii.zaxis.type);
        *axistype = "log";
      }

      ratio_hist->GetZaxis()->SetMoreLogLabels(1);
    }

    if (ii.reflabels.size() >= 2) {
      std::string title =
          std::string(ratio_hist->GetTitle()) + " (Ratio " + ii.reflabels[0] + " / " + ii.reflabels[1] + ")";
      ratio_hist->SetTitle(title.c_str());
    }

    // Only set this if no other options are set, it will then override the
    // object settings (which tend to get set by other  Render Plugins)
    if (renderInfo.drawOptions.empty()) {
      renderInfo.drawOptions = "COLZ";
    }

    // Replace object. Now normal rendering takes over.
    VisDQMObject* mut_o = const_cast<VisDQMObject*>(&o);
    delete mut_o->object;
    mut_o->object = ratio_hist;
  }

  void postDraw(TCanvas* canvas, const VisDQMObject& o, const VisDQMImgInfo& ii) override {
    // TODO: this tends to be overridden by other render plugins.
    // But setting it only in postDraw helps, most render plugins set it in preDraw
    // The unusual color palette helps to remind the user that these are comparisons.
    gStyle->SetPalette(kLightTemperature);
    gStyle->SetNumberContours(255);
  }

private:
};

static Ratio2DComparisonRenderPlugin instance;
