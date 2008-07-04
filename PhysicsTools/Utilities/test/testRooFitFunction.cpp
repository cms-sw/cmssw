#include "PhysicsTools/Utilities/interface/RooFitFunction.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "PhysicsTools/Utilities/interface/Function.h"
#include "RooRealVar.h"
#include "RooPlot.h"
#include "TROOT.h"
#include "TCanvas.h"
#include <iostream>
using namespace RooFit;

int main() { 
  gROOT->SetStyle("Plain");
  try {
    funct::Parameter yield("Yield", 100);
    funct::Parameter mean("Mean", 0);
    funct::Parameter sigma("Sigma", 1);
    funct::Parameter c("C", 1./sqrt(2*M_PI)); 
    funct::X x;
    funct::Numerical<2> _2;
    
    funct::Expression f = yield * c * exp(-(((x-mean)/sigma)^_2)/_2);

    RooRealVar rX("x", "x", -5, 5);
    RooRealVar rYield(yield.name().c_str(), "Gaussian mean", yield, 0, 1000);
    RooRealVar rMean(mean.name().c_str(), "Gaussian mean", mean, -5, 5);
    RooRealVar rSigma(sigma.name().c_str(), "Gaussian sigma", sigma, 0, 5);
    
    TCanvas canvas;
    root::RooFitFunction<funct::X, funct::Expression> 
      rooFun("rooFun", "rooFun", f, rX, rYield, yield, rMean, mean, rSigma, sigma); 
    RooPlot * frame = rX.frame();
    rooFun.plotOn(frame);
    frame->Draw();
    canvas.SaveAs("rooPlot.eps");
  } catch(std::exception & err){
    std::cerr << "Exception caught:\n" << err.what() << std::endl;
    return 1;
  }
  
  return 0;
}
