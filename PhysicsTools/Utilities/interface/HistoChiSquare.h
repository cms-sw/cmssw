#ifndef PhysicsTools_Utilities_HistoChiSquare_h
#define PhysicsTools_Utilities_HistoChiSquare_h
#include "PhysicsTools/Utilities/interface/RootMinuitResultPrinter.h"
#include <vector>
#include "TH1.h"
#include "TMath.h"

namespace fit {
  template <typename T>
  class HistoChiSquare {
  public:
    HistoChiSquare() {}
    HistoChiSquare(T &t, TH1 *histo, double rangeMin, double rangeMax)
        : t_(&t), rangeMin_(rangeMin), rangeMax_(rangeMax) {
      nBins_ = histo->GetNbinsX();
      xMin_ = histo->GetXaxis()->GetXmin();
      xMax_ = histo->GetXaxis()->GetXmax();
      deltaX_ = (xMax_ - xMin_) / nBins_;
      for (size_t i = 0; i < nBins_; ++i) {
        cont_.push_back(histo->GetBinContent(i + 1));
        err_.push_back(histo->GetBinError(i + 1));
      }
    }
    double operator()() const {
      double chi2 = 0;
      for (size_t i = 0; i < nBins_; ++i) {
        double x = xMin_ + (i + .5) * deltaX_;
        if ((x > rangeMin_) && (x < rangeMax_) && (err_[i] > 0)) {
          double r = (cont_[i] - (*t_)(x)) / err_[i];
          chi2 += (r * r);
        }
      }
      return chi2;
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
        if ((x > rangeMin_) && (x < rangeMax_) && (err_[i] > 0))
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
    std::vector<double> err_;
  };

  template <typename T>
  struct RootMinuitResultPrinter<HistoChiSquare<T> > {
    static void print(double amin, unsigned int numberOfFreeParameters, const HistoChiSquare<T> &f) {
      unsigned int ndof = f.numberOfBins() - numberOfFreeParameters;
      std::cout << "chi-squared/n.d.o.f. = " << amin << "/" << ndof << " = " << amin / ndof
                << "; prob: " << TMath::Prob(amin, ndof) << std::endl;
    }
  };
}  // namespace fit

#endif
