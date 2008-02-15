#ifndef PhysicsTools_Utilities_HistoChiSquare_h
#define PhysicsTools_Utilities_HistoChiSquare_h

#include "TH1.h"
#include <iostream>

using namespace std;

namespace fit {
  template<typename T>
  class HistoChiSquare {
   public:
    HistoChiSquare() { }
    HistoChiSquare(T & t, TH1D *histo, double rangeMin, double rangeMax): 
      t_(&t), rangeMin_(rangeMin), rangeMax_(rangeMax) { 
      nBins_ = histo->GetNbinsX();
      xMin_ = histo->GetXaxis()->GetXmin();
      xMax_ = histo->GetXaxis()->GetXmax();
      deltaX_ =(xMax_ - xMin_) / nBins_;
      for(size_t i = 0; i < nBins_; ++i) { 
	cont_.push_back( histo->GetBinContent(i+1) );
	err_.push_back( histo->GetBinError(i+1) );
      }
    }
    double operator()() const { 
      double chi = 0;
      for(size_t i = 0; i < nBins_; ++i) { 
	double x = xMin_ + ( i +.5 ) * deltaX_;
	if((x > rangeMin_)&&(x < rangeMax_)&&(err_[i] > 0)) { 
	  double r = ( cont_[i] - (*t_)(x) )/err_[i];
	  chi += (r * r);
	}
      }
      //cout << "chi = " << chi << endl;
      return chi;
    }
    void setHistos(TH1D *histo) { 
      nBins_ = histo->GetNbinsX();
      xMin_ = histo->GetXaxis()->GetXmin();
      xMax_ = histo->GetXaxis()->GetXmax();
      deltaX_ =(xMax_ - xMin_) / nBins_;
      //cout << xMin_ << "  " << xMax_ << endl;
    }
    size_t degreesOfFreedom() const {
      size_t fullBins = 0;
      for(size_t i = 0; i < nBins_; ++i) { 
	double x = xMin_ + ( i +.5 ) * deltaX_;
	if((x > rangeMin_)&&(x < rangeMax_)&&(err_[i] > 0))  
	  fullBins++;
      }
      cout << "degrees of freedom = " << fullBins << endl;
      return fullBins; 
    }
    T & function() { return * t_; }
    const T & function() const { return * t_; }
  private:
    T * t_;
    double rangeMin_, rangeMax_;
    size_t nBins_;
    double xMin_, xMax_, deltaX_;
    vector<double> cont_;
    vector<double> err_;
  };
}

#endif
