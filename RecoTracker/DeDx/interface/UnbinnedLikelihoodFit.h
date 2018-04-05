#ifndef __UnbinnedLikelihoodFit_h_
#define __UnbinnedLikelihoodFit_h_
#include <TObject.h>
#include <TF1.h>
#include <TVirtualFitter.h>
#include <cstdint>

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
class UnbinnedLikelihoodFit : public TObject
{
  public:
    // Constructor and destructor
    UnbinnedLikelihoodFit();
    ~UnbinnedLikelihoodFit() override;

    // Set the data for the fit: a set of measurements
    void setData(uint32_t n, double* x);

    // Set the fit function
    void setFunction(TF1* f);

    // Set the fit options
    void setTolerance( double tol) { tolerance_ = tol; }
    void setMaxIterations( uint32_t n ) { maxIterations_ = n; }

    // Fit
    int32_t fit(int32_t verbosity = -1);
    int32_t fit(int32_t n, double* x, int32_t verbosity = -1) { setData(n,x); return fit(verbosity); }

    // Results a retrieved via the TF1
    TF1* getFunction() const { return function_; }
    double getParameterValue(uint32_t i) { return function_ ? function_->GetParameter(i) : 0; }
    double getParameterError(uint32_t i) { return function_ ? function_->GetParError(i)  : 0; }
    double* getParameterValues() { return function_ ? function_->GetParameters() : nullptr; }
    const double* getParameterErrors() { return function_ ? function_->GetParErrors()  : nullptr; }
  
  private:
    // input data
    uint32_t datasize_;
    double* x_;
    // the function
    TF1* function_;
    uint32_t nparameters_;
    // arguments for Minuit methods
    double arglist_[10];
    uint32_t maxIterations_;
    double tolerance_;
    // the minimizer (minuit)
    TVirtualFitter* min;

  private:
    // LL function
    double logL(const double* x) const;
    // The function to minimize
    friend void UnbinnedLL(Int_t &npar, Double_t *gin, Double_t &val, Double_t *par, Int_t iflag);
    
};

#endif

