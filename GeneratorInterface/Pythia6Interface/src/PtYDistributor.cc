
#include "GeneratorInterface/Pythia6Interface/interface/PtYDistributor.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "TFile.h"
#include "TGraph.h"

#include "CLHEP/Random/RandGeneral.h"

#include <iostream>
#include <string>

using namespace gen;

PtYDistributor::PtYDistributor(const edm::FileInPath& fip,
                               double ptmax = 100,
                               double ptmin = 0,
                               double ymax = 10,
                               double ymin = -10,
                               int ptbins = 1000,
                               int ybins = 50)
    : ptmax_(ptmax), ptmin_(ptmin), ymax_(ymax), ymin_(ymin), ptbins_(ptbins), ybins_(ybins) {
  // edm::FileInPath f1(input);
  std::string fDataFile = fip.fullPath();

  std::cout << " File from " << fDataFile << std::endl;
  TFile f(fDataFile.c_str(), "READ");
  TGraph* yfunc = (TGraph*)f.Get("rapidity");
  TGraph* ptfunc = (TGraph*)f.Get("pt");

  if (!yfunc)
    throw edm::Exception(edm::errors::NullPointerError, "PtYDistributor")
        << "Rapidity distribution could not be found in file " << fDataFile;

  if (!ptfunc)
    throw edm::Exception(edm::errors::NullPointerError, "PtYDistributor")
        << "Pt distribution could not be found in file " << fDataFile;

  if (ptbins_ > 100000) {
    ptbins_ = 100000;
  }

  if (ybins_ > 100000) {
    ybins_ = 100000;
  }

  double aProbFunc1[100000];
  double aProbFunc2[100000];

  for (int i = 0; i < ybins_; ++i) {
    double xy = ymin_ + i * (ymax_ - ymin_) / ybins_;
    double yy = yfunc->Eval(xy);
    aProbFunc1[i] = yy;
  }

  for (int ip = 0; ip < ptbins_; ++ip) {
    double xpt = ptmin_ + ip * (ptmax_ - ptmin_) / ptbins_;
    double ypt = ptfunc->Eval(xpt);
    aProbFunc2[ip] = ypt;
  }

  fYGenerator = new CLHEP::RandGeneral(nullptr, aProbFunc1, ybins_);
  fPtGenerator = new CLHEP::RandGeneral(nullptr, aProbFunc2, ptbins_);

  f.Close();

}  // from file

double PtYDistributor::fireY(CLHEP::HepRandomEngine* engine) { return fireY(ymin_, ymax_, engine); }

double PtYDistributor::firePt(CLHEP::HepRandomEngine* engine) { return firePt(ptmin_, ptmax_, engine); }

double PtYDistributor::fireY(double ymin, double ymax, CLHEP::HepRandomEngine* engine) {
  double y = -999;

  if (fYGenerator) {
    while (y < ymin || y > ymax)
      y = ymin_ + (ymax_ - ymin_) * fYGenerator->shoot(engine);
  } else {
    throw edm::Exception(edm::errors::NullPointerError, "PtYDistributor")
        << "Random y requested but Random Number Generator for y not Initialized!";
  }
  return y;
}

double PtYDistributor::firePt(double ptmin, double ptmax, CLHEP::HepRandomEngine* engine) {
  double pt = -999;

  if (fPtGenerator) {
    while (pt < ptmin || pt > ptmax)
      pt = ptmin_ + (ptmax_ - ptmin_) * fPtGenerator->shoot(engine);
  } else {
    throw edm::Exception(edm::errors::NullPointerError, "PtYDistributor")
        << "Random pt requested but Random Number Generator for pt not Initialized!";
  }
  return pt;
}
