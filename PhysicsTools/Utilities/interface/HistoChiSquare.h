#ifndef PhysicsTools_Utilities_HistoChiSquare_h
#define PhysicsTools_Utilities_HistoChiSquare_h

#include "TH1.h"
#include <iostream>

using namespace std;

template<typename T>
class HistoChiSquare {
 public:
  HistoChiSquare() { }
  HistoChiSquare(T & t, TH1D *histo, const size_t nBins, double rangeMin, double rangeMax): 
    t_(&t), histo_(histo), nBins_(nBins), rangeMin_(rangeMin), rangeMax_(rangeMax) { 
  }
  double operator()() const { 
    double xMin_ = histo_->GetXaxis()->GetXmin();
    double xMax_ = histo_->GetXaxis()->GetXmax();
    double deltaX_ =(xMax_ - xMin_) / nBins_;
    double chi = 0;
    for(size_t i = 0; i < nBins_; ++i) { 
      double cont = histo_->GetBinContent(i+1);
      double err = histo_->GetBinError(i+1);
      double x = xMin_ + ( i +.5 ) * deltaX_;
      if((x > rangeMin_)&&(x < rangeMax_)&&(err > 0)) { 
	double r = ( cont - (*t_)(x) )/err;
	chi += (r * r);
      }
    }
    //cout << "chi = " << chi << endl;
    return chi;
  }
  void setHistos(TH1D *histo) { 
    histo_ = histo;
    //if(histo_ != 0) {
    //  xMin_ = histo_->GetXaxis()->GetXmin();
    //  xMax_ = histo_->GetXaxis()->GetXmax();
    //  deltaX_ =(xMax_ - xMin_) / nBins_;
      //cout << xMin_ << "  " << xMax_ << endl;
    //}
  }
  size_t degreesOfFreedom() const {
    double xMin_ = histo_->GetXaxis()->GetXmin();
    double xMax_ = histo_->GetXaxis()->GetXmax();
    double deltaX_ =(xMax_ - xMin_) / nBins_;
    size_t fullBins = 0;
    for(size_t i = 0; i < nBins_; ++i) { 
      double err = histo_->GetBinError(i+1);
      double x = xMin_ + ( i +.5 ) * deltaX_;
      if((x > rangeMin_)&&(x < rangeMax_)&&(err > 0))  
	fullBins++;
    }
    cout << "degrees of freedom = " << fullBins << endl;
    return fullBins; 
  }
  T & function() { return * t_; }
  const T & function() const { return * t_; }
 private:
  T * t_;
  TH1D *histo_;
  size_t nBins_;
  double rangeMin_, rangeMax_;
};

#endif
