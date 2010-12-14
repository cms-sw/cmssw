#include "HiggsAnalysis/CombinedLimit/interface/HybridNew.h"
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooAbsPdf.h"
#include <RooStats/ModelConfig.h>
#include <RooStats/HybridCalculator.h>
#include <RooStats/SimpleLikelihoodRatioTestStat.h>
#include <RooStats/RatioOfProfiledLikelihoodsTestStat.h>
#include <RooStats/ProfileLikelihoodTestStat.h>
#include <RooStats/ToyMCSampler.h>
#include "HiggsAnalysis/CombinedLimit/interface/Combine.h"

using namespace RooStats;

HybridNew::HybridNew() : 
LimitAlgo("HybridNew specific options") {
    options_.add_options()
        ("toysH,T", boost::program_options::value<unsigned int>()->default_value(500),    "Number of Toy MC extractions to compute CLs+b, CLb and CLs")
        ("clsAcc",  boost::program_options::value<double>( )->default_value(0.005), "Absolute accuracy on CLs to reach to terminate the scan")
        ("rAbsAcc", boost::program_options::value<double>()->default_value(0.1),   "Absolute accuracy on r to reach to terminate the scan")
        ("rRelAcc", boost::program_options::value<double>()->default_value(0.05),  "Relative accuracy on r to reach to terminate the scan")
        ("rule",    boost::program_options::value<std::string>()->default_value("CLs"),    "Rule to use: CLs, CLsplusb")
        ("testStat",boost::program_options::value<std::string>()->default_value("LEP"),"Test statistics: LEP, TEV, Atlas.")
        ("rInterval", "Always try to compute an interval on r even after having found a point satisfiying the CL")
    ;
}

void HybridNew::applyOptions(const boost::program_options::variables_map &vm) {
    nToys_       = vm["toysH"].as<unsigned int>();
    clsAccuracy_  = vm["clsAcc" ].as<double>();
    rAbsAccuracy_ = vm["rAbsAcc"].as<double>();
    rRelAccuracy_ = vm["rRelAcc"].as<double>();
    rule_     = vm["rule"].as<std::string>();
    testStat_ = vm["testStat"].as<std::string>();
    if (rule_ == "CLs") {
        CLs_ = true;
    } else if (rule_ == "CLsplusb") {
        CLs_ = false;
    } else {
        std::cerr << "ERROR: rule must be either 'CLs' or 'CLsplusb'" << std::endl;
        abort();
    }
    rInterval_ = vm.count("rInterval");
    if (testStat_ != "LEP" && testStat_ != "TEV" && testStat_ != "Atlas") {
        std::cerr << "Error: test statistics should be one of 'LEP' or 'TEV' or 'Atlas', and not '" << testStat_ << "'" << std::endl;
        abort();
    }
}

bool HybridNew::run(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint) {
  RooRealVar *r = w->var("r"); r->setConstant(true);
  w->loadSnapshot("clean");

  if ((hint != 0) && (*hint > r->getMin())) {
    r->setMax(std::min<double>(3*(*hint), r->getMax()));
  }
  
  typedef std::pair<double,double> CLs_t;

  double clsTarget = 1 - cl; 
  CLs_t clsMin(1,0), clsMax(0,0);
  double rMin = 0, rMax = r->getMax();

  std::cout << "Search for upper limit to the limit" << std::endl;
  for (;;) {
    CLs_t clsMax = eval(w, data, r, r->getMax());
    if (clsMax.first == 0 || clsMax.first + 3 * fabs(clsMax.second) < cl ) break;
    r->setMax(r->getMax()*2);
    if (r->getVal()/rMax >= 20) { 
      std::cerr << "Cannot set higher limit: at r = " << r->getVal() << " still get " << (CLs_ ? "CLs" : "CLsplusb") << " = " << clsMax.first << std::endl;
      return false;
    }
  }
  rMax = r->getMax();
  
  std::cout << "Now doing proper bracketing & bisection" << std::endl;
  bool lucky = false;
  do {
    CLs_t clsMid = eval(w, data, r, 0.5*(rMin+rMax), true, clsTarget);
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

  if (lucky) {
    limit = r->getVal();
    if (rInterval_) {
      std::cout << "\n -- HypoTestInverter (before determining interval) -- \n";
      std::cout << "Limit: r < " << limit << " +/- " << 0.5*(rMax - rMin) << " @ " <<cl * 100<<"% CL\n";

      double rBoundLow  = limit - 0.5*std::max(rAbsAccuracy_, rRelAccuracy_ * limit);
      for (r->setVal(rMin); r->getVal() < rBoundLow  && (fabs(clsMin.first-clsTarget) >= clsAccuracy_); rMin = r->getVal()) {
        clsMax = eval(w, data, r, 0.5*(r->getVal()+limit), true, clsTarget);
      }

      double rBoundHigh = limit + 0.5*std::max(rAbsAccuracy_, rRelAccuracy_ * limit);
      for (r->setVal(rMax); r->getVal() > rBoundHigh && (fabs(clsMax.first-clsTarget) >= clsAccuracy_); rMax = r->getVal()) {
        clsMax = eval(w, data, r, 0.5*(r->getVal()+limit), true, clsTarget);
      }
    }
  } else {
    limit = 0.5*(rMax+rMin);
  }
  std::cout << "\n -- HypoTestInverter -- \n";
  std::cout << "Limit: r < " << limit << " +/- " << 0.5*(rMax - rMin) << " @ " <<cl * 100<<"% CL\n";
  return true;
}

std::pair<double, double> HybridNew::eval(RooWorkspace *w, RooAbsData &data, RooRealVar *r, double rVal, bool adaptive, double clsTarget) {
    using namespace RooStats;
    RooFit::MsgLevel globalKill = RooMsgService::instance().globalKillBelow();
    RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING);

    const RooArgSet &obs = *w->set("observables");
    const RooArgSet &poi = *w->set("POI");

    r->setVal(rVal); 
    ModelConfig modelConfig("sb_model", w);
    modelConfig.SetPdf(*w->pdf("model_s"));
    modelConfig.SetObservables(obs);
    modelConfig.SetParametersOfInterest(poi);
    modelConfig.SetNuisanceParameters(*w->set("nuisances"));
    modelConfig.SetSnapshot(poi);

    ModelConfig modelConfig_bonly("b_model", w);
    modelConfig_bonly.SetPdf(*w->pdf("model_b"));
    modelConfig_bonly.SetObservables(obs);
    modelConfig_bonly.SetParametersOfInterest(poi);
    modelConfig_bonly.SetNuisanceParameters(*w->set("nuisances"));
    modelConfig_bonly.SetSnapshot(poi);

    std::auto_ptr<TestStatistic> qvar;
    if (testStat_ == "LEP") {
        qvar.reset(new SimpleLikelihoodRatioTestStat(*modelConfig_bonly.GetPdf(),*modelConfig.GetPdf()));
    } else if (testStat_ == "TEV") {
        qvar.reset(new RatioOfProfiledLikelihoodsTestStat(*modelConfig_bonly.GetPdf(),*modelConfig.GetPdf(), modelConfig.GetSnapshot()));
        ((RatioOfProfiledLikelihoodsTestStat&)*qvar).SetSubtractMLE(false);
    } else if (testStat_ == "Atlas") {
        modelConfig_bonly.SetPdf(*w->pdf("model_s"));
        RooArgSet nullPOI; nullPOI.addClone(*r); 
        ((RooRealVar &)nullPOI["r"]).setVal(0);
        modelConfig_bonly.SetSnapshot(nullPOI);
        qvar.reset(new ProfileLikelihoodTestStat(*modelConfig_bonly.GetPdf()));
    }

    ToyMCSampler toymcsampler(*qvar, nToys_);
    if (!w->pdf("model_b")->canBeExtended()) toymcsampler.SetNEventsPerToy(1);

    std::auto_ptr<HybridCalculator> hc(new HybridCalculator(data,modelConfig, modelConfig_bonly, &toymcsampler));
    hc->ForcePriorNuisanceNull(*w->pdf("nuisancePdf"));
    hc->ForcePriorNuisanceAlt(*w->pdf("nuisancePdf"));

    std::auto_ptr<HypoTestResult> hcResult(hc->GetHypoTest());
    if (hcResult.get() == 0) {
        std::cerr << "Hypotest failed" << std::endl;
        RooMsgService::instance().setGlobalKillBelow(globalKill);
        return std::pair<double, double>(-1,-1);
    }
    double clsMid    = (CLs_ ? hcResult->CLs()      : hcResult->CLsplusb());
    double clsMidErr = (CLs_ ? hcResult->CLsError() : hcResult->CLsplusbError());
    std::cout << "r = " << rVal << (CLs_ ? ": CLs = " : ": CLsplusb = ") << clsMid << " +/- " << clsMidErr << std::endl;
    if (adaptive) {
        while (fabs(clsMid-clsTarget) < 3*clsMidErr && clsMidErr >= clsAccuracy_) {
            std::auto_ptr<HypoTestResult> more(hc->GetHypoTest());
            hcResult->Append(more.get());
            clsMid    = (CLs_ ? hcResult->CLs()      : hcResult->CLsplusb());
            clsMidErr = (CLs_ ? hcResult->CLsError() : hcResult->CLsplusbError());
            std::cout << "r = " << rVal << (CLs_ ? ": CLs = " : ": CLsplusb = ") << clsMid << " +/- " << clsMidErr << std::endl;
        }
    }
    if (verbose > 0) {
        std::cout << "r = " << r->getVal() << ": \n" <<
            "\tCLs      = " << hcResult->CLs()      << " +/- " << hcResult->CLsError()      << "\n" <<
            "\tCLb      = " << hcResult->CLb()      << " +/- " << hcResult->CLbError()      << "\n" <<
            "\tCLsplusb = " << hcResult->CLsplusb() << " +/- " << hcResult->CLsplusbError() << "\n" <<
            std::endl;
    }
    RooMsgService::instance().setGlobalKillBelow(globalKill);
    return std::pair<double, double>(clsMid, clsMidErr);
} 

