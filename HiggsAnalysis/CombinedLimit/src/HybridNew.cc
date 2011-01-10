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
#include "HiggsAnalysis/CombinedLimit/interface/RooFitGlobalKillSentry.h"

using namespace RooStats;

HybridNew::HybridNew() : 
LimitAlgo("HybridNew specific options") {
    // NOTE: we do NOT re-declare options which are in common with Hybrid method
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,28,0)
    options_.add_options()
    /*
        ("toysH", boost::program_options::value<unsigned int>()->default_value(500),    "Number of Toy MC extractions to compute CLs+b, CLb and CLs")
        ("clsAcc",  boost::program_options::value<double>( )->default_value(0.005), "Absolute accuracy on CLs to reach to terminate the scan")
        ("rAbsAcc", boost::program_options::value<double>()->default_value(0.1),   "Absolute accuracy on r to reach to terminate the scan")
        ("rRelAcc", boost::program_options::value<double>()->default_value(0.05),  "Relative accuracy on r to reach to terminate the scan")
        ("rule",    boost::program_options::value<std::string>()->default_value("CLs"),    "Rule to use: CLs, CLsplusb")
        ("testStat",boost::program_options::value<std::string>()->default_value("LEP"),"Test statistics: LEP, TEV, Atlas.")
        ("rInterval", "Always try to compute an interval on r even after having found a point satisfiying the CL")
    */
        ("nCPU", boost::program_options::value<unsigned int>()->default_value(0), "Use N CPUs with PROOF Lite")
    ;
#endif
}

void HybridNew::applyOptions(const boost::program_options::variables_map &vm) {
    nToys_       = vm["toysH"].as<unsigned int>();
    clsAccuracy_  = vm["clsAcc" ].as<double>();
    rAbsAccuracy_ = vm["rAbsAcc"].as<double>();
    rRelAccuracy_ = vm["rRelAcc"].as<double>();
    rule_     = vm["rule"].as<std::string>();
    testStat_ = vm["testStat"].as<std::string>();
    nCpu_     = vm["nCPU"].as<unsigned int>();
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
    RooFitGlobalKillSentry silence(RooFit::WARNING);
    bool ret = doSignificance_ ? runSignificance(w, data, limit, hint) : runLimit(w, data, limit, hint);
    return ret;
}

bool HybridNew::runSignificance(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint) {
    RooRealVar *r = w->var("r"); r->setConstant(true);
    HybridNew::Setup setup;
    std::auto_ptr<RooStats::HybridCalculator> hc(create(w, data, r, 1.0, setup));
    std::auto_ptr<HypoTestResult> hcResult(hc->GetHypoTest());
    if (hcResult.get() == 0) {
        std::cerr << "Hypotest failed" << std::endl;
        return false;
    }
    limit = hcResult->Significance();
    double sigHi = RooStats::PValueToSignificance( 1 - (hcResult->CLb() + hcResult->CLbError()) ) - limit;
    double sigLo = RooStats::PValueToSignificance( 1 - (hcResult->CLb() - hcResult->CLbError()) ) - limit;
    std::cout << "\n -- Hybrid New -- \n";
    std::cout << "Significance: " << limit << "  " << sigLo << "/+" << sigHi << " (CLb " << hcResult->CLb() << " +/- " << hcResult->CLbError() << ")\n";
    return isfinite(limit);
}

bool HybridNew::runLimit(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint) {
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
    HybridNew::Setup setup;
    std::auto_ptr<RooStats::HybridCalculator> hc(create(w, data, r, rVal, setup));
    if (verbose) std::cout << "  r = " << rVal << std::endl;
    std::pair<double, double> ret = eval(*hc, adaptive, clsTarget);
    return ret;
}

std::auto_ptr<RooStats::HybridCalculator> HybridNew::create(RooWorkspace *w, RooAbsData &data, RooRealVar *r, double rVal, HybridNew::Setup &setup) {
    using namespace RooStats;

    const RooArgSet &obs = *w->set("observables");
    const RooArgSet &poi = *w->set("POI");

    r->setVal(rVal); 
    setup.modelConfig = ModelConfig("sb_model", w);
    setup.modelConfig.SetPdf(*w->pdf("model_s"));
    setup.modelConfig.SetObservables(obs);
    setup.modelConfig.SetParametersOfInterest(poi);
    if (withSystematics) setup.modelConfig.SetNuisanceParameters(*w->set("nuisances"));
    setup.modelConfig.SetSnapshot(poi);

    setup.modelConfig_bonly = ModelConfig("b_model", w);
    setup.modelConfig_bonly.SetPdf(*w->pdf("model_b"));
    setup.modelConfig_bonly.SetObservables(obs);
    setup.modelConfig_bonly.SetParametersOfInterest(poi);
    if (withSystematics) setup.modelConfig_bonly.SetNuisanceParameters(*w->set("nuisances"));
    setup.modelConfig_bonly.SetSnapshot(poi);

    if (testStat_ == "LEP") {
        setup.qvar.reset(new SimpleLikelihoodRatioTestStat(*setup.modelConfig_bonly.GetPdf(),*setup.modelConfig.GetPdf()));
    } else if (testStat_ == "TEV") {
        setup.qvar.reset(new RatioOfProfiledLikelihoodsTestStat(*setup.modelConfig_bonly.GetPdf(),*setup.modelConfig.GetPdf(), setup.modelConfig.GetSnapshot()));
        ((RatioOfProfiledLikelihoodsTestStat&)*setup.qvar).SetSubtractMLE(false);
    } else if (testStat_ == "Atlas") {
        setup.modelConfig_bonly.SetPdf(*w->pdf("model_s"));
        RooArgSet nullPOI; nullPOI.addClone(*r); 
        ((RooRealVar &)nullPOI["r"]).setVal(0);
        setup.modelConfig_bonly.SetSnapshot(nullPOI);
        setup.qvar.reset(new ProfileLikelihoodTestStat(*setup.modelConfig_bonly.GetPdf()));
    }

    setup.toymcsampler = ToyMCSampler(*setup.qvar, nToys_);
    if (!w->pdf("model_b")->canBeExtended()) setup.toymcsampler.SetNEventsPerToy(1);

#if ROOT_VERSION_CODE >= ROOT_VERSION(5,28,0)
    if (nCpu_ > 0) {
        if (verbose > 1) std::cout << "  Will use " << nCpu_ << " CPUs." << std::endl;
        setup.pc.reset(new ProofConfig(*w, nCpu_, "", kFALSE)); 
        setup.toymcsampler.SetProofConfig(setup.pc.get());
    }   
#endif

    std::auto_ptr<HybridCalculator> hc(new HybridCalculator(data,setup.modelConfig, setup.modelConfig_bonly, &setup.toymcsampler));
    if (withSystematics) {
        hc->ForcePriorNuisanceNull(*w->pdf("nuisancePdf"));
        hc->ForcePriorNuisanceAlt(*w->pdf("nuisancePdf"));
    }
    return hc;
}

std::pair<double,double> 
HybridNew::eval(RooStats::HybridCalculator &hc, bool adaptive, double clsTarget) {
    std::auto_ptr<HypoTestResult> hcResult(hc.GetHypoTest());
    if (hcResult.get() == 0) {
        std::cerr << "Hypotest failed" << std::endl;
        return std::pair<double, double>(-1,-1);
    }
    double clsMid    = (CLs_ ? hcResult->CLs()      : hcResult->CLsplusb());
    double clsMidErr = (CLs_ ? hcResult->CLsError() : hcResult->CLsplusbError());
    if (verbose) std::cout << (CLs_ ? "\tCLs = " : "\tCLsplusb = ") << clsMid << " +/- " << clsMidErr << std::endl;
    if (adaptive) {
        while (fabs(clsMid-clsTarget) < 3*clsMidErr && clsMidErr >= clsAccuracy_) {
            std::auto_ptr<HypoTestResult> more(hc.GetHypoTest());
            hcResult->Append(more.get());
            clsMid    = (CLs_ ? hcResult->CLs()      : hcResult->CLsplusb());
            clsMidErr = (CLs_ ? hcResult->CLsError() : hcResult->CLsplusbError());
            if (verbose) std::cout << (CLs_ ? "\tCLs = " : "\tCLsplusb = ") << clsMid << " +/- " << clsMidErr << std::endl;
        }
    }
    if (verbose > 0) {
        std::cout <<
            "\tCLs      = " << hcResult->CLs()      << " +/- " << hcResult->CLsError()      << "\n" <<
            "\tCLb      = " << hcResult->CLb()      << " +/- " << hcResult->CLbError()      << "\n" <<
            "\tCLsplusb = " << hcResult->CLsplusb() << " +/- " << hcResult->CLsplusbError() << "\n" <<
            std::endl;
    }
    return std::pair<double, double>(clsMid, clsMidErr);
} 

