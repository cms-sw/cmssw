#ifndef PhysicsTools_Utilities_HistoPoissonLikelihoodRatio_h
#define PhysicsTools_Utilities_HistoPoissonLikelihoodRatio_h
#include "PhysicsTools/Utilities/interface/RootMinuitResultPrinter.h"
#include <vector>
#include <cmath>
#include "TH1.h"
#include "TMath.h"

namespace fit {
  template <typename T>
  class HistoPoissonLikelihoodRatio {
  public:
    HistoPoissonLikelihoodRatio() {}
    HistoPoissonLikelihoodRatio(T &t, TH1 *histo, double rangeMin, double rangeMax)
        : t_(&t), rangeMin_(rangeMin), rangeMax_(rangeMax) {
      nBins_ = histo->GetNbinsX();
      xMin_ = histo->GetXaxis()->GetXmin();
      xMax_ = histo->GetXaxis()->GetXmax();
      deltaX_ = (xMax_ - xMin_) / nBins_;
      for (size_t i = 0; i < nBins_; ++i) {
        cont_.push_back(histo->GetBinContent(i + 1));
      }
    }
    double operator()() const {
      double chi2lambda = 0;
      for (size_t i = 0; i < nBins_; ++i) {
        double x = xMin_ + (i + .5) * deltaX_;
        if ((x > rangeMin_) && (x < rangeMax_)) {
          double nu = (*t_)(x);
          if (nu > 0 && cont_[i] > 0)
            chi2lambda += nu - cont_[i] + cont_[i] * log(cont_[i] / nu);
        }
      }
      chi2lambda *= 2;
      return chi2lambda;
    }
    void setHistos(TH1 *histo) {
      nBins_ = histo->GetNbinsX();
      xMin_ = histo->GetXaxis()->GetXmin();
      xMax_ = histo->GetXaxis()->GetXmax();
      deltaX_ = (xMax_ - xMin_) / nBins_;
    }
    size_t numberOfBins() const {
      size_t fullBins = 0;
      for (size_t i = 0; i < nBins_; ++i) {
        double x = xMin_ + (i + .5) * deltaX_;
        if ((x > rangeMin_) && (x < rangeMax_))
          fullBins++;
      }
      return fullBins;
    }
    T &function() { return *t_; }
    const T &function() const { return *t_; }

  private:
    T *t_;
    double rangeMin_, rangeMax_;
    size_t nBins_;
    double xMin_, xMax_, deltaX_;
    std::vector<double> cont_;
  };

  template <typename T>
  struct RootMinuitResultPrinter<HistoPoissonLikelihoodRatio<T> > {
    static void print(double amin, unsigned int numberOfFreeParameters, const HistoPoissonLikelihoodRatio<T> &f) {
      unsigned int ndof = f.numberOfBins() - numberOfFreeParameters;
      std::cout << "chi-squared/n.d.o.f. = " << amin << "/" << ndof << " = " << amin / ndof
                << "; prob: " << TMath::Prob(amin, ndof) << std::endl;
    }
  };
}  // namespace fit

#endif
