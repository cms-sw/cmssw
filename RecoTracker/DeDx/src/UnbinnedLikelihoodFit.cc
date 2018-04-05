#include "RecoTracker/DeDx/interface/UnbinnedLikelihoodFit.h"
#include <TMath.h>

// a class to perform a likelihood fit
// Author: Christophe Delaere

/* Example of a Landau fit:
 * ------------------------
 * UnbinnedLikelihoodFit myfit;
 * double x[4] = {89,110,70,80};
 * myfit.setData(4,x);
 * TF1* myfunction = new TF1("myLandau","TMath::Landau(x,[0],[1],1)",0,255);
 * myfunction->SetParameters(100,10);
 * myfit.setFunction(myfunction);
 * myfit.fit();
 * myfit.getFunction()->Print();
 * double MPV = myfit.getFunction()->GetParameter(0);
 * double MPVerror = myfit.getFunction()->GetParError(0);
 */

// the function passed to minuit
void UnbinnedLL(Int_t&, Double_t*, Double_t &val, Double_t *par, Int_t) {
  // retrieve the data object (it's also the fitter)
  // - sign to have a minimum
  // factor 2 to have the right errors (see for example the pdg)
  val = -2*((dynamic_cast<const UnbinnedLikelihoodFit*>((TVirtualFitter::GetFitter())->GetObjectFit()))->logL(par));
}

// the constructor
UnbinnedLikelihoodFit::UnbinnedLikelihoodFit() {
  nparameters_ = 0;
  datasize_ = 0;
  x_ = nullptr;
  min = nullptr;
  tolerance_ = 0.01;
  maxIterations_ = 1000;
}

// the destructor
UnbinnedLikelihoodFit::~UnbinnedLikelihoodFit() {
}

// sets the data
// the class is not owner of the data... it only keeps a pointer to it.
void UnbinnedLikelihoodFit::setData(uint32_t n, double* x) {
  datasize_ = n;
  x_ = x;
}

// sets the function for the fit
void UnbinnedLikelihoodFit::setFunction(TF1* f) {
  function_ = f;
  nparameters_ = function_ ? function_->GetNpar() : 0;
}

// The fit itself
int32_t UnbinnedLikelihoodFit::fit(int32_t verbosity) {
  // creates a fitter 
  min = TVirtualFitter::Fitter(this,nparameters_);
  min->SetFCN(UnbinnedLL);
  
  // set print level: no output
  arglist_[0] = 0;
  min->ExecuteCommand("SET NOWarnings",arglist_,1);
  arglist_[0] = verbosity;
  min->ExecuteCommand("SET PRINT",arglist_,1);
  
  // initial values, error, range
  double parmin,parmax;
  for(uint32_t i=0;i<nparameters_;++i) {
    function_->GetParLimits(i, parmin, parmax);
    min->SetParameter(i,
                      function_->GetParName(i),
                      function_->GetParameter(i),
                      tolerance_,
                      parmin, parmax);
  }

  // run MIGRAD
  arglist_[0] = maxIterations_; // number of function calls
  arglist_[1] = tolerance_;     // tolerance
  int32_t status = min->ExecuteCommand("MIGRAD",arglist_,2);

  // get fit parameters and errors
  for(uint32_t i=0;i<nparameters_;++i) {
    function_->SetParameter(i, min->GetParameter(i));
    function_->SetParError( i, min->GetParError(i) );
  }

  // returns the status
  return status;
}

// the log-likelihood function
double UnbinnedLikelihoodFit::logL(const double* parameters) const {
  double val=0;
  if(!function_) return val;
  for (uint32_t i=0;i<datasize_;++i){
    val += TMath::Log(function_->EvalPar(&(x_[i]),parameters));
  }
  return val; 
} 

