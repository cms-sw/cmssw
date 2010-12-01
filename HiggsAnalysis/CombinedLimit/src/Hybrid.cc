#include "HiggsAnalysis/CombinedLimit/interface/Hybrid.h"
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooStats/HybridCalculatorOriginal.h"
#include "RooAbsPdf.h"
#include "HiggsAnalysis/CombinedLimit/interface/Combine.h"

using namespace RooStats;

bool Hybrid::run(RooWorkspace *w, RooAbsData &data, double &limit) {
  RooRealVar *r = w->var("r"); r->setConstant(true);
  RooArgSet  poi(*r);
  w->loadSnapshot("clean");
  RooAbsPdf *altModel  = w->pdf("model_s"), *nullModel = w->pdf("model_b");
  
  HybridCalculatorOriginal* hc = new HybridCalculatorOriginal(data,*altModel,*nullModel);
  if (withSystematics) {
    if ((w->set("nuisances") == 0) || (w->pdf("nuisancePdf") == 0)) {
          std::cerr << "ERROR: nuisances or nuisancePdf not set. Perhaps you wanted to run with no systematics?\n" << std::endl;
          abort();
    }
    hc->UseNuisance(true);
    hc->SetNuisancePdf(*w->pdf("nuisancePdf"));
    hc->SetNuisanceParameters(*w->set("nuisances"));
  } else {
    hc->UseNuisance(false);
  }
  hc->SetTestStatistic(1); // 3 = TeV
  hc->PatchSetExtended(w->pdf("model_b")->canBeExtended()); // Number counting, each dataset has 1 entry 
  hc->SetNumberOfToys(nToys_);
  
  typedef std::pair<double,double> CLs_t;

  double clsTarget = 1 - cl; 
  CLs_t clsMin(1,0), clsMax(0,0);
  double rMin   = 0, rMax = r->getMax();
  
  std::cout << "Search for upper limit to the limit" << std::endl;
  HybridResult *hcResult = 0;
  for (;;) {
    r->setVal(r->getMax()); hcResult = hc->GetHypoTest();
    std::cout << "r = " << r->getVal() << ": CLs = " << hcResult->CLs() << " +/- " << hcResult->CLsError() << std::endl;
    if (hcResult->CLs() == 0) break;
    r->setMax(r->getMax()*2);
    if (r->getVal()/rMax >= 20) { 
      std::cerr << "Cannot set higher limit: at r = " << r->getVal() << " still get CLs = " << hcResult->CLs() << std::endl;
      return false;
    }
  }
  rMax = r->getMax();
  
  std::cout << "Now doing proper bracketing & bisection" << std::endl;
  bool lucky = false;
  do {
    CLs_t clsMid = eval(r, 0.5*(rMin+rMax), hc, clsTarget);
    if (clsMid.second == -1) {
      std::cerr << "Hypotest failed" << std::endl;
      return false;
    }
    if (fabs(clsMid.first-clsTarget) <= clsAccuracy_) {
      std::cout << "reached accuracy." << std::endl;
      lucky = true;
      break;
    }
    if ((clsMid.first>clsTarget) == (clsMax.first>clsTarget)) {
      rMax = r->getVal(); clsMax = clsMid;
    } else {
      rMin = r->getVal(); clsMin = clsMid;
    }
  } while (rMax-rMin > std::max(rAbsAccuracy_, rRelAccuracy_ * r->getVal()));

  limit = r->getVal();
  if (lucky && rInterval_) {
      std::cout << "\n -- HypoTestInverter (before determining interval) -- \n";
      std::cout << "Limit: r < " << limit << " +/- " << (rMax - rMin) << " @ " <<cl * 100<<"% CL\n";

      double rBoundLow  = limit - 0.5*std::max(rAbsAccuracy_, rRelAccuracy_ * limit);
      for (r->setVal(rMin); r->getVal() < rBoundLow  && (fabs(clsMin.first-clsTarget) >= clsAccuracy_); rMin = r->getVal()) {
        clsMax = eval(r, 0.5*(r->getVal()+limit), hc, clsTarget);
      }

      double rBoundHigh = limit + 0.5*std::max(rAbsAccuracy_, rRelAccuracy_ * limit);
      for (r->setVal(rMax); r->getVal() > rBoundHigh && (fabs(clsMax.first-clsTarget) >= clsAccuracy_); rMax = r->getVal()) {
        clsMax = eval(r, 0.5*(r->getVal()+limit), hc, clsTarget);
      }
  } 
  std::cout << "\n -- HypoTestInverter -- \n";
  std::cout << "Limit: r < " << limit << " +/- " << 0.5*(rMax - rMin) << " @ " <<cl * 100<<"% CL\n";
  return true;
}

std::pair<double, double> Hybrid::eval(RooRealVar *r, double rVal, RooStats::HybridCalculatorOriginal *hc, double clsTarget, bool adaptive) {
    using namespace RooStats;
    r->setVal(rVal);
    std::auto_ptr<HybridResult> hcResult(hc->GetHypoTest());
    if (hcResult.get() == 0) {
        std::cerr << "Hypotest failed" << std::endl;
        return std::pair<double, double>(-1,-1);
    }
    double clsMid = hcResult->CLs(), clsMidErr = hcResult->CLsError();
    std::cout << "r = " << rVal << ": CLs = " << clsMid << " +/- " << clsMidErr << std::endl;
    if (adaptive) {
        while (fabs(clsMid-clsTarget) < 3*clsMidErr && clsMidErr >= clsAccuracy_) {
            std::auto_ptr<HybridResult> more(hc->GetHypoTest());
            hcResult->Add(more.get());
            clsMid    = hcResult->CLs(); 
            clsMidErr = hcResult->CLsError();
            std::cout << "r = " << rVal << ": CLs = " << clsMid << " +/- " << clsMidErr << std::endl;
        }
    }
    if (verbose) {
        std::cout << "r = " << r->getVal() << ": \n" <<
            "\tCLs      = " << hcResult->CLs()      << " +/- " << hcResult->CLsError()      << "\n" <<
            "\tCLb      = " << hcResult->CLb()      << " +/- " << hcResult->CLbError()      << "\n" <<
            "\tCLsplusb = " << hcResult->CLsplusb() << " +/- " << hcResult->CLsplusbError() << "\n" <<
            std::endl;
    }
    return std::pair<double, double>(clsMid, clsMidErr);
} 

