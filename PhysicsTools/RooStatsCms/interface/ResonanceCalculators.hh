#ifndef __RESONANCE_CALCULATORS_HH__
#define __RESONANCE_CALCULATORS_HH__


//
// author: J.P. Chou (Brown University)
//
// Define resonance calculators in various ways.  The most prominent is the FactoryResCalc
// which defines the shapes via the RooWorkspace::factory() interface.  Also defined are 
// some pared-down functions to call the calculator.
//
//

#include "PhysicsTools/RooStatsCms/interface/ResonanceCalculatorAbs.hh"

#include <string>

#include "RooProdPdf.h"
#include "RooGaussian.h"
#include "RooWorkspace.h"
#include "RooRealVar.h"
#include "RooGenericPdf.h"
#include "RooConstVar.h"
#include "RooFormulaVar.h"
#include "RooProduct.h"

// forward declarations
class TH1;

////////////////////////////////////////////////////////////////////////////////
// Implements a generic resonance calculator object which passes information
// via the RooWorkspace::factory() interface.  In order for this mechanism to work,
// certain variables must be named precisely:
//   * "obs" - the observable
//   * "signalmass" - the signal resonance mass
// The arguements in the constructor specify
//   * the name of the signal pdf
//   * the expression which instantiates the signal pdf
//   * the name of the background pdf
//   * the expression which instantiates the background pdf
//   * the name of the signal width variable
//   * the expression which instantiates the signal width
//
////////////////////////////////////////////////////////////////////////////////

class FactoryResCalc : public ResonanceCalculatorAbs
{
public:
  FactoryResCalc(const char *sigpdfname, const char* sigexpr,
		 const char *bkgpdfname, const char* bkgexpr,
		 const char *widthname, const char* widthexpr) : ResonanceCalculatorAbs() {
    setupWorkspaceViaFactory(sigpdfname, sigexpr, bkgpdfname, bkgexpr, widthname, widthexpr);
  }
  virtual ~FactoryResCalc() {}

protected:
  RooAbsPdf* setupSignalPdf(void) { assert(0); return 0; }
  RooAbsPdf* setupBackgroundPdf(void) { assert(0); return 0; }
  RooAbsReal* setupSignalWidth(void) { assert(0); return 0; }

};

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
  //  RooConstVar* getRelativeWidth(void) const { return getWorkspace()->var("relWidth"); }

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
    RooConstVar* relWidth = new RooConstVar("relWidth", "Width relative to the resonance mass", 0.02);
    RooProduct *width = new RooProduct("sigwidth", "signal width", RooArgSet(*relWidth, *getSignalMass()));
    return width;
  }

};

////////////////////////////////////////////////////////////////////////////////
// function calls
////////////////////////////////////////////////////////////////////////////////

void runResCalc(ResonanceCalculatorAbs& rc, const char* label);



#endif
