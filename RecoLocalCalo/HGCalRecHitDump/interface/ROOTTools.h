#ifndef _root_tools_h_
#define _root_tools_h_

#include "RooRealVar.h"
#include "RooAbsPdf.h"

#include "TGraphErrors.h"

#include <vector>
#include <iostream>

// get effective sigma from cumulative distribution function (from Hgg analysis)
std::pair<float,float> getEffSigma(RooRealVar *var, 
				   RooAbsPdf *pdf, 
				   float wmin=-10,float wmax=10, float step=0.002, float epsilon=1e-4);


struct CircleParams_t
{
  float r,    r_err;
  float x0,   x0_err;
  float y0,   y0_err;
  float chi2, ndf;
  bool  isvalid;
};

CircleParams_t fitCircleTo(TGraphErrors *gr);
void circle_fcn(Int_t &, Double_t *, Double_t &f, Double_t *par, Int_t);


#endif
