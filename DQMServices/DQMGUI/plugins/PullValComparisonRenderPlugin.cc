/*!
  \file PullValComparisonRenderPlugin
  \brief RenderPlugin producing a comparison for 2D histograms.

This render plugin uses the API for references to render a Pull Value
Comparison for 2D histograms. This allows using the "overlay" comparison with
2D histograms. 

The Pull Value metric is also used by the AutoDQM tool, introduced by the Muon
groups around 2018.

  \author Marcel Schneider
*/

#include "DQMServices/DQMGUI/interface/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile2D.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TLegend.h"
#include "Math/Math.h"
#include "Math/QuantFuncMathCore.h"

#include <cassert>
#include <string>
#include <tuple>

using namespace std;

class PullValComparisonRenderPlugin : public DQMRenderPlugin {
public:
  bool applies(const VisDQMObject& o, const VisDQMImgInfo& i) override {
    // TODO: If you don't like the Pull-Value mode on your histograms, you could
    // opt-out here and provide your own render plugin.
    // Only use this up to 1000 bins -- beyond that things get too slow.
    auto object2d = dynamic_cast<TH2*>(o.object);
    if (i.objects.size() == 2 && object2d && object2d->GetNcells() < 1000) {
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

    auto ref_profile = dynamic_cast<TProfile2D*>(ref_hist);
    auto data_profile = dynamic_cast<TProfile2D*>(data_hist);
    // Get empty clone of reference histogram for pull hist
    TH2* pull_hist;
    if (ref_profile != nullptr) {
      pull_hist = ref_profile->ProjectionXY("PullVal");
    } else {
      pull_hist = dynamic_cast<TH2*>(ref_hist->Clone("PullVal"));
    }

    pull_hist->Reset();

    // Reject empty histograms
    //auto min_entries = 1000;
    //auto is_good = data_hist->GetEntries() != 0 && data_hist->GetEntries() >= min_entries;

    // Normalize data_hist
    if (data_hist->GetEntries() > 0) {
      data_hist->Scale(ref_hist->GetSumOfWeights() / data_hist->GetSumOfWeights());
    }

    double pull_cap = 25;
    double max_pull = 0.0;
    for (auto x = 1; x <= ref_hist->GetNbinsX(); x++) {
      for (auto y = 1; y <= ref_hist->GetNbinsY(); y++) {
        auto bin1 = data_hist->GetBinContent(x, y);
        auto bin2 = ref_hist->GetBinContent(x, y);
        bool bad = false;

        // Proper Poisson error
        double bin1err = 0, bin2err = 0;
        if (data_profile && ref_profile) {
          bin1err = data_profile->GetBinError(x, y);
          bin2err = ref_profile->GetBinError(x, y);
        } else {
          if (bin1 >= 0 && bin2 >= 0) {
            std::tie(bin1err, bin2err) = get_poisson_errors(bin1, bin2);
          } else {
            bad = true;
          }
        }

        double new_pull = 0;
        if (bin1err == 0 && bin2err == 0) {
          new_pull = 0;
          bad = true;
        } else {
          new_pull = pull(bin1, bin1err, bin2, bin2err);
        }

        // Clamp the displayed value
        auto fill_val = std::max(std::min(new_pull, pull_cap), -pull_cap);

        // Check if max_pull
        max_pull = std::max(max_pull, std::abs(fill_val));

        // If the input bins were explicitly empty, make this bin white by
        // setting it out of range
        if (bad || (bin1 == 0 && bin2 == 0)) {
          fill_val = -999;
        }

        pull_hist->SetBinContent(x, y, fill_val);
      }
    }

    pull_hist->SetMinimum(-max_pull);
    pull_hist->SetMaximum(max_pull);
    if (ii.reflabels.size() >= 2) {
      std::string title = std::string(pull_hist->GetTitle()) + " (" + ii.reflabels[0] + " vs. " + ii.reflabels[1] + ")";
      pull_hist->SetTitle(title.c_str());
    }
    // TODO: this tends to be overridden by other render plugins.
    //renderInfo.drawOptions += " COLZ";
    gStyle->SetPalette(kLightTemperature);
    gStyle->SetNumberContours(255);

    // Replace object. Now normal rendering takes over.
    VisDQMObject* mut_o = const_cast<VisDQMObject*>(&o);
    delete mut_o->object;
    mut_o->object = pull_hist;
  }

  void postDraw(TCanvas* canvas, const VisDQMObject& o, const VisDQMImgInfo& ii) override {}

  double pull(double bin1, double binerr1, double bin2, double binerr2) {
    return (bin1 - bin2) / (sqrt(binerr1 * binerr1 + binerr2 * binerr2));
  }

  std::tuple<double, double> get_poisson_errors(double bin1, double bin2) {
    auto alpha = 1 - 0.6827;
    double m_error1, m_error2, p_error1, p_error2;
    if (bin1 == 0) {
      m_error1 = 0;
      p_error1 = ROOT::Math::gamma_quantile_c(alpha / 2, bin1 + 1, 1);
    } else {
      m_error1 = ROOT::Math::gamma_quantile(alpha / 2, bin1, 1);
      p_error1 = ROOT::Math::gamma_quantile_c(alpha / 2, bin1 + 1, 1);
    }

    if (bin2 == 0) {
      m_error2 = 0;
      p_error2 = ROOT::Math::gamma_quantile_c(alpha / 2, bin2 + 1, 1);
    } else {
      m_error2 = ROOT::Math::gamma_quantile(alpha / 2, bin2, 1);
      p_error2 = ROOT::Math::gamma_quantile_c(alpha / 2, bin2 + 1, 1);
    }

    double bin1err, bin2err;
    if (bin1 > bin2) {
      bin1err = bin1 - m_error1;
      bin2err = p_error2 - bin2;
    } else {
      bin2err = bin2 - m_error2;
      bin1err = p_error1 - bin1;
    }

    return std::make_tuple(bin1err, bin2err);
  }

private:
};

static PullValComparisonRenderPlugin instance;
