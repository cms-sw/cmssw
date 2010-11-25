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
  hc->PatchSetExtended(false); // Number counting, each dataset has 1 entry 
  hc->SetNumberOfToys(nToys_);
  
  double clsTarget = 1 - cl, clsAcc  = 0.005 /* ??? */ , rAcc = 0.1, rRelAcc = 1 - cl; 
  double clsMin = 1, clsMax = 0, clsMinErr = 0, clsMaxErr = 0;
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
  do {
    r->setVal(0.5*(rMin+rMax));
    hcResult = hc->GetHypoTest();
    if (hcResult == 0) {
      std::cerr << "Hypotest failed" << std::endl;
      return false;
    }
    double clsMid = hcResult->CLs(), clsMidErr = hcResult->CLsError();
    std::cout << "r = " << r->getVal() << ": CLs = " << clsMid << " +/- " << clsMidErr << std::endl;
    while (fabs(clsMid-clsTarget) < 3*clsMidErr && clsMidErr >= clsAcc) {
      HybridResult *more = hc->GetHypoTest();
      hcResult->Add(more);
      clsMid    = hcResult->CLs(); 
      clsMidErr = hcResult->CLsError();
      std::cout << "r = " << r->getVal() << ": CLs = " << clsMid << " +/- " << clsMidErr << std::endl;
    }
    if (verbose) {
      std::cout << "r = " << r->getVal() << ": \n" <<
	"\tCLs      = " << hcResult->CLs()      << " +/- " << hcResult->CLsError()      << "\n" <<
	"\tCLb      = " << hcResult->CLb()      << " +/- " << hcResult->CLbError()      << "\n" <<
	"\tCLsplusb = " << hcResult->CLsplusb() << " +/- " << hcResult->CLsplusbError() << "\n" <<
	std::endl;
    }
    if (fabs(clsMid-clsTarget) <= clsAcc) {
      std::cout << "reached accuracy." << std::endl;
      break;
    }
    if ((clsMid>clsTarget) == (clsMax>clsTarget)) {
      rMax = r->getVal(); clsMax = clsMid; clsMaxErr = clsMidErr;
    } else {
      rMin = r->getVal(); clsMin = clsMid; clsMinErr = clsMidErr;
    }
  } while (rMax-rMin > std::max(rAcc, rRelAcc * r->getVal()));
  if (clsMinErr == 0 && clsMaxErr == 0) {
    std::cerr << "Error: both boundaries have no passing/failing toys.\n";
    return false;
  }
  limit = r->getVal();
  std::cout << "\n -- HypoTestInverter -- \n";
  std::cout << "Limit: r < " << limit << " +/- " << (rMax - rMin) << "@" <<cl * 100<<"% CL\n";
  
  return true;
}
