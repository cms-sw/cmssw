#ifndef __RESONANCE_CALCULATORS_HH__
#define __RESONANCE_CALCULATORS_HH__


//
// author: J.P. Chou (Brown University)
//
// In this file, we define a number of signal pdfs, background pdfs, and signal
// widths which specify how the ResonanceCalculator is suppose to behave.
//
// RCGaussianSigPowerLawBkgConstRelSigWidth
//   implements a Gaussian signal pdf with fixed fractional width (10% of the
//   resonance mass by default).  The background pdf is a power law according to:
//   "pow(1.0-mass/roots,p1)/pow(mass/roots,p2+p3*log(mass/roots))"
//

#include "PhysicsTools/RooStatsCms/interface/ResonanceCalculatorAbs.hh"

#include "RooProdPdf.h"
#include "RooGaussian.h"
#include "RooWorkspace.h"
#include "RooRealVar.h"
#include "RooGenericPdf.h"
#include "RooConstVar.h"
#include "RooFormulaVar.h"
#include "RooProduct.h"
#include "RooBreitWigner.h"

////////////////////////////////////////////////////////////////////////////////
// Implements a simple resonance calculator with a steep power-law background,
// and gaussian signal with constant fractional width
////////////////////////////////////////////////////////////////////////////////


class SimpleResCalc : public ResonanceCalculatorAbs
{
public:
  // constructor sets up the workspace
  SimpleResCalc() : ResonanceCalculatorAbs() { setupWorkspace(); }
  ~SimpleResCalc() {}

  // optional user access to parameters
  RooRealVar* getBackgroundPar1(void) const { return getWorkspace()->var("p1"); }
  RooRealVar* getBackgroundPar2(void) const { return getWorkspace()->var("p2"); }
  RooRealVar* getBackgroundPar3(void) const { return getWorkspace()->var("p3"); }
  RooRealVar* getRootS(void) const { return getWorkspace()->var("roots"); }
  RooRealVar* getRelativeWidth(void) const { return getWorkspace()->var("relWidth"); }

protected:
  

  // setup a gaussian signal pdf
  RooAbsPdf* setupSignalPdf(void) {
    RooAbsPdf* signal = new RooGaussian("signal", "Signal pdf", *getObservable(), *getSignalMass(), *getSignalWidth());
    return signal;
  }
  
  // setup a power-law background pdf
  RooAbsPdf* setupBackgroundPdf(void) {
    RooRealVar* p1 = new RooRealVar("p1", "Background Parameter 1", 10., 0., 40.);
    RooRealVar* p2 = new RooRealVar("p2", "Background Parameter 2", 7., 0., 20.);
    RooRealVar* p3 = new RooRealVar("p3", "Background Parameter 3", 0.1, 0., 10.);
    RooConstVar* roots = new RooConstVar("roots","COM energy",7000.0);
    RooGenericPdf* background = new RooGenericPdf("background", "Background Pdf", "pow(1.0-@0/@4,@1)/pow(@0/@4,@2+@3*log(@0/@4))",
    						  RooArgList(*getObservable(), *p1, *p2, *p3, *roots));
    return background;
  }

  // seutp a constant fractional signal width
  RooAbsReal* setupSignalWidth(void) {
    RooConstVar* relWidth = new RooConstVar("relWidth", "Width relative to the resonance mass", 0.1);
    RooProduct *width = new RooProduct("sigwidth", "signal width", RooArgSet(*relWidth, *getSignalMass()));
    return width;
  }

};


////////////////////////////////////////////////////////////////////////////////
// Implements the Z'/RS graviton dimuon bump hunt
////////////////////////////////////////////////////////////////////////////////

class ZprimeDimuonResCalc : public ResonanceCalculatorAbs
{
public:
  // constructor sets up the workspace
  ZprimeDimuonResCalc() : ResonanceCalculatorAbs() { setupWorkspace(); }
  ~ZprimeDimuonResCalc() {}
  
  // optional user access to parameters
  RooRealVar* getGaussWidthPar0(void) const { return getWorkspace()->var("q0"); }
  RooRealVar* getGaussWidthPar1(void) const { return getWorkspace()->var("q1"); }
  RooRealVar* getGaussWidthPar2(void) const { return getWorkspace()->var("q2"); }
  RooRealVar* getSignalWidthPar0(void) const { return getWorkspace()->var("p0"); }
  RooRealVar* getSignalWidthPar1(void) const { return getWorkspace()->var("p1"); }
  RooRealVar* getBackgroundPar0(void) const { return getWorkspace()->var("b0"); }
  RooRealVar* getBackgroundPar1(void) const { return getWorkspace()->var("b1"); }

protected:

  // setup the signal pdf
  RooAbsPdf* setupSignalPdf(void) {
    RooConstVar* q0 = new RooConstVar("q0","Gaussian Width Parameter 0",0.0138);
    RooConstVar* q1 = new RooConstVar("q1","Gaussian Width Parameter 1",0.00009315);
    RooConstVar* q2 = new RooConstVar("q2","Gaussian Width Parameter 2",0.00000001077);
    RooFormulaVar* gausWidth=new RooFormulaVar("gausWidth", "@0*(@1+@2*@0+@3*@0*@0)", RooArgList(*getSignalMass(), *q0, *q1, *q2));
    RooConstVar* gausMean = new RooConstVar("gausMean", "Signal Gaussian Mean",0.0);
    RooBreitWigner* bw = new RooBreitWigner("bw", "Signal BW", *getObservable(), *getSignalMass(), *getSignalWidth());
    RooGaussian* gaus = new RooGaussian("gaus", "Signal Gaussian", *getObservable(), *gausMean, *gausWidth);
    RooProdPdf* signal = new RooProdPdf("signal", "Signal PDF", *bw, *gaus);
    return signal;
  }
  
  // setup the background
  RooAbsPdf* setupBackgroundPdf(void) {
    RooRealVar* b0 = new RooRealVar("b0", "Background Parameter 0", -0.006912, -1, 0.);
    RooRealVar* b1 = new RooRealVar("b1", "Background Parameter 1", -2.404, -5., 0.);
    RooGenericPdf* background = new RooGenericPdf("background", "Background Pdf", "exp(@0*@1)*pow(@1,@2)", RooArgList(*b0, *getObservable(), *b1));
    return background;
  }

  // setup the signal width
  RooAbsReal* setupSignalWidth(void) {
    RooConstVar* p0 = new RooConstVar("p0","Signal Width Parameter 0",-1.2979);
    RooConstVar* p1 = new RooConstVar("p1","Signal Width Parameter 1",0.0309338);
    RooFormulaVar* width = new RooFormulaVar("width", "@0+@1*@2",RooArgList(*p0, *p1, *getSignalMass()));
    return width;
  } 

};

#endif
