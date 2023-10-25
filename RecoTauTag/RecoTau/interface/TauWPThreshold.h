#ifndef RecoTauTag_RecoTau_TauWPThreshold_h
#define RecoTauTag_RecoTau_TauWPThreshold_h

#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include <TF1.h>

namespace tau {
  class TauWPThreshold {
  public:
    explicit TauWPThreshold(const std::string& cut_str) {
      bool simple_value = false;
      try {
        size_t pos = 0;
        value_ = std::stod(cut_str, &pos);
        simple_value = (pos == cut_str.size());
      } catch (std::invalid_argument&) {
      } catch (std::out_of_range&) {
      }
      if (!simple_value) {
        static const std::string prefix =
            "[&](double *x, double *p) { const int decayMode = p[0];"
            "const double pt = p[1]; const double eta = p[2];";
        static const int n_params = 3;
        static const auto handler = [](int, Bool_t, const char*, const char*) -> void {};

        std::string fn_str = prefix;
        if (cut_str.find("return") == std::string::npos)
          fn_str += " return " + cut_str + ";}";
        else
          fn_str += cut_str + "}";
        auto old_handler = SetErrorHandler(handler);
        fn_ = std::make_unique<TF1>("fn_", fn_str.c_str(), 0, 1, n_params);
        SetErrorHandler(old_handler);
        if (!fn_->IsValid())
          throw cms::Exception("TauWPThreshold: invalid formula") << "Invalid WP cut formula = '" << cut_str << "'.";
      }
    }

    double operator()(int dm, double pt, double eta) const {
      if (!fn_)
        return value_;

      fn_->SetParameter(0, dm);
      fn_->SetParameter(1, pt);
      fn_->SetParameter(2, eta);
      return fn_->Eval(0);
    }

    double operator()(const reco::BaseTau& tau, bool isPFTau) const {
      const int dm =
          isPFTau ? dynamic_cast<const reco::PFTau&>(tau).decayMode() : dynamic_cast<const pat::Tau&>(tau).decayMode();
      return (*this)(dm, tau.pt(), tau.eta());
    }

    double operator()(const reco::Candidate& tau) const { return (*this)(-1, tau.pt(), tau.eta()); }

  private:
    std::unique_ptr<TF1> fn_;
    double value_;
  };
}  // namespace tau

#endif
