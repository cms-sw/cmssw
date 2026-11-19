#include "DQMServices/Core/interface/DQMBookingHelpers.h"

namespace dqm {
  namespace booking {

    // =========================================================================
    // Internal helpers — not part of the public interface
    // =========================================================================

    namespace {
      // Resets the bin edges for a given axis to log10 scale.
      void rebinAxisToLog(TAxis* axis) {
        const int bins = axis->GetNbins();
        const double from = TMath::Log10(axis->GetXmin());
        const double to = TMath::Log10(axis->GetXmax());
        const double width = (to - from) / bins;
        std::vector<double> new_bins(bins + 1, 0.0);
        for (int i = 0; i <= bins; ++i)
          new_bins[i] = TMath::Power(10, from + i * width);
        axis->Set(bins, new_bins.data());
      }
    }  // namespace

    namespace detail {

      void rebinXToLog(TH1* h) {
        TAxis* axis = h->GetXaxis();
        rebinAxisToLog(axis);
      }

      void rebinYToLog(TH1* h) {
        TAxis* axis = h->GetYaxis();
        rebinAxisToLog(axis);
      }

    }  // namespace detail
  }  // namespace booking
}  // namespace dqm
