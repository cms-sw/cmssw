#ifndef PhysicsTools_Utilities_HistoPdf_h
#define PhysicsTools_Utilities_HistoPdf_h

#include "FWCore/Utilities/interface/EDMException.h"

#include <iostream>
#include "TH1.h"

namespace funct {
  class HistoPdf {
  public:
    template <typename Iterator>
    HistoPdf(double xMin, double xMax, const Iterator& begin, const Iterator& end)
        : xMin_(xMin), xMax_(xMax), delta_(xMax - xMin), binSize_(delta_ / (end - begin)), y_(end - begin) {
      double s = 0;
      unsigned int i = 0;
      for (Iterator it = begin; it != end; ++it)
        s += (y_[i++] = *it);
      for (std::vector<double>::iterator i = y_.begin(); i != y_.end(); ++i)
        *i /= s;
    }
    HistoPdf() {}
    template <typename Iterator>
    void init(double xMin, double xMax, const Iterator& begin, const Iterator& end) {
      xMin_ = xMin;
      xMax_ = xMax;
      delta_ = xMax - xMin;
      unsigned int n = end - begin;
      binSize_ = delta_ / n;
      y_.resize(n);
      double s = 0;
      unsigned int i = 0;
      for (Iterator it = begin; it != end; ++it)
        s += (y_[i++] = *it);
      for (std::vector<double>::iterator i = y_.begin(); i != y_.end(); ++i)
        *i /= s;
    }
    double operator()(double x) const {
      if (x < xMin_ || x > xMax_)
        return 0;
      double pdf = y_[static_cast<unsigned int>(((x - xMin_) / delta_) * y_.size())] / binSize_;
      return pdf;
    }
    void rebin(unsigned int r) {
      if (y_.size() % r != 0)
        throw edm::Exception(edm::errors::Configuration)
            << "HistoPdf: can't rebin histogram of " << y_.size() << " entries by " << r << "\n";
      unsigned int n = y_.size() / r;
      std::vector<double> y(n, 0);
      for (unsigned int i = 0, j = 0; i < n; ++i)
        for (unsigned int k = 0; k < r; ++k)
          y[i] += y_[j++];
      y_ = y;
      binSize_ *= r;
    }
    void dump() {
      std::cout << ">>> range: [" << xMin_ << ", " << xMax_ << "], bin size: " << delta_ << "/" << y_.size() << " = "
                << binSize_ << std::endl;
      double s = 0;
      for (unsigned int i = 0; i != y_.size(); ++i) {
        double x = xMin_ + (0.5 + i) * binSize_;
        double y = operator()(x);
        std::cout << ">>> pdf(" << x << ") = " << y << std::endl;
        s += y * binSize_;
      }
      std::cout << ">>>: PDF normalization is " << s << std::endl;
    }

  private:
    double xMin_, xMax_, delta_, binSize_;
    std::vector<double> y_;
  };

  class RootHistoPdf : public HistoPdf {
  public:
    explicit RootHistoPdf(const TH1& histo, double fMin, double fMax) {
      unsigned int nBins = histo.GetNbinsX();
      std::vector<double> y;
      y.reserve(nBins);
      double xMin = histo.GetXaxis()->GetXmin();
      double xMax = histo.GetXaxis()->GetXmax();
      double deltaX = (xMax - xMin) / nBins;
      for (unsigned int i = 0; i != nBins; ++i) {
        double x = xMin + (i + .5) * deltaX;
        if (x > fMin && x < fMax) {
          y.push_back(histo.GetBinContent(i + 1));
        }
      }
      init(fMin, fMax, y.begin(), y.end());
    }
  };

}  // namespace funct

#endif
