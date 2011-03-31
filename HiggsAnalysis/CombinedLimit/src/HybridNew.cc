#include <stdexcept>
#include <cstdio>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>

#include "HiggsAnalysis/CombinedLimit/interface/HybridNew.h"
#include <TFile.h>
#include <TF1.h>
#include <TKey.h>
#include <TLine.h>
#include <TCanvas.h>
#include <TGraphErrors.h>
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooAbsPdf.h"
#include "RooRandom.h"
#include "RooAddPdf.h"
#include "RooConstVar.h"
#include <RooStats/ModelConfig.h>
#include <RooStats/HybridCalculator.h>
#include <RooStats/SimpleLikelihoodRatioTestStat.h>
#include <RooStats/RatioOfProfiledLikelihoodsTestStat.h>
#include <RooStats/ProfileLikelihoodTestStat.h>
#include <RooStats/ToyMCSampler.h>
#include <RooStats/HypoTestPlot.h>
#include "HiggsAnalysis/CombinedLimit/interface/Combine.h"
#include "HiggsAnalysis/CombinedLimit/interface/RooFitGlobalKillSentry.h"
#include "HiggsAnalysis/CombinedLimit/interface/SimplerLikelihoodRatioTestStat.h"
#include "HiggsAnalysis/CombinedLimit/interface/ProfiledLikelihoodRatioTestStat.h"
#include "HiggsAnalysis/CombinedLimit/interface/utils.h"

using namespace RooStats;

HybridNew::WorkingMode HybridNew::workingMode_ = MakeLimit;
unsigned int HybridNew::nToys_ = 500;
double HybridNew::clsAccuracy_ = 0.05;
double HybridNew::rAbsAccuracy_ = 0.1;
double HybridNew::rRelAccuracy_ = 0.05;
std::string HybridNew::rule_ = "CLs";
std::string HybridNew::testStat_ = "LEP";
unsigned int HybridNew::nCpu_ = 0; // proof-lite mode
unsigned int HybridNew::fork_ = 1; // fork mode
double HybridNew::rValue_  = 1.0;
bool HybridNew::CLs_ = false;
bool HybridNew::saveHybridResult_  = false;
bool HybridNew::readHybridResults_ = false; 
bool HybridNew::importanceSamplingNull_ = false;
bool HybridNew::importanceSamplingAlt_  = false;
std::string HybridNew::algo_ = "logSecant";
bool HybridNew::optimizeProductPdf_     = false;
bool HybridNew::optimizeTestStatistics_ = true;
std::string HybridNew::plot_;
 
HybridNew::HybridNew() : 
LimitAlgo("HybridNew specific options") {
    options_.add_options()
        ("rule",    boost::program_options::value<std::string>(&rule_)->default_value(rule_),            "Rule to use: CLs, CLsplusb")
        ("testStat",boost::program_options::value<std::string>(&testStat_)->default_value(testStat_),    "Test statistics: LEP, TEV, Atlas.")
        ("singlePoint",  boost::program_options::value<float>(),  "Just compute CLs for the given value of r")
        ("onlyTestStat", "Just compute test statistics, not actual p-values (works only with --singlePoint)")
        ("searchAlgo", boost::program_options::value<std::string>(&algo_)->default_value(algo_),         "Algorithm to use to search for the limit (bisection, logSecant)")
        ("toysH,T", boost::program_options::value<unsigned int>(&nToys_)->default_value(nToys_),         "Number of Toy MC extractions to compute CLs+b, CLb and CLs")
        ("clsAcc",  boost::program_options::value<double>(&clsAccuracy_ )->default_value(clsAccuracy_),  "Absolute accuracy on CLs to reach to terminate the scan")
        ("rAbsAcc", boost::program_options::value<double>(&rAbsAccuracy_)->default_value(rAbsAccuracy_), "Absolute accuracy on r to reach to terminate the scan")
        ("rRelAcc", boost::program_options::value<double>(&rRelAccuracy_)->default_value(rRelAccuracy_), "Relative accuracy on r to reach to terminate the scan")
        ("fork",    boost::program_options::value<unsigned int>(&fork_)->default_value(fork_),           "Fork to N processes before running the toys (set to 0 for debugging)")
        ("nCPU",    boost::program_options::value<unsigned int>(&nCpu_)->default_value(nCpu_),           "Use N CPUs with PROOF Lite (experimental!)")
        ("saveHybridResult",  "Save result in the output file  (option saveToys must be enabled)")
        ("readHybridResults", "Read and merge results from file (option toysFile must be enabled)")
        ("importanceSamplingNull", boost::program_options::value<bool>(&importanceSamplingNull_)->default_value(importanceSamplingNull_),  
                                   "Enable importance sampling for null hypothesis (background only)") 
        ("importanceSamplingAlt",  boost::program_options::value<bool>(&importanceSamplingAlt_)->default_value(importanceSamplingAlt_),    
                                   "Enable importance sampling for alternative hypothesis (signal plus background)") 
        ("optimizeTestStatistics", boost::program_options::value<bool>(&optimizeTestStatistics_)->default_value(optimizeTestStatistics_), 
                                   "Use optimized test statistics if the likelihood is not extended (works for LEP and TEV test statistics).")
        ("optimizeProductPdf",     boost::program_options::value<bool>(&optimizeProductPdf_)->default_value(optimizeProductPdf_),      
                                   "Optimize the code factorizing pdf (experimental)")
        ("plot",   boost::program_options::value<std::string>(), "Save a plot of the result (test statistics distributions or limit scan)")
    ;
}

void HybridNew::applyOptions(const boost::program_options::variables_map &vm) {
    if (vm.count("singlePoint")) {
        if (doSignificance_) throw std::invalid_argument("HybridNew: Can't use --significance and --singlePoint at the same time");
        rValue_ = vm["singlePoint"].as<float>();
        workingMode_ = ( vm.count("onlyTestStat") ? MakeTestStatistics : MakePValues );
        rValue_ = vm["singlePoint"].as<float>();
    } else if (vm.count("onlyTestStat")) {
        throw std::invalid_argument("HybridNew: --onlyTestStat works only with --singlePoint");
    } else if (doSignificance_) {
        workingMode_ = MakeSignificance;
    } else {
        workingMode_ = MakeLimit;
    }
    saveHybridResult_ = vm.count("saveHybridResult");
    readHybridResults_ = vm.count("readHybridResults");
    plot_ = vm.count("plot") ? vm["plot"].as<std::string>() : std::string();
    validateOptions(); 
}

void HybridNew::applyDefaultOptions() { 
    workingMode_ = MakeLimit;
    validateOptions(); 
}

void HybridNew::validateOptions() {
    if (fork_ > 1) nToys_ /= fork_; // makes more sense
    if (rule_ == "CLs") {
        CLs_ = true;
    } else if (rule_ == "CLsplusb") {
        CLs_ = false;
    } else {
        throw std::invalid_argument("HybridNew: Rule must be either 'CLs' or 'CLsplusb'");
    }
    if (testStat_ != "LEP" && testStat_ != "TEV" && testStat_ != "Atlas" && testStat_ != "Profile") {
        throw std::invalid_argument("HybridNew: Test statistics should be one of 'LEP' or 'TEV' or 'Atlas' or 'Profile'");
    }
}

bool HybridNew::run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
    RooFitGlobalKillSentry silence(verbose <= 1 ? RooFit::WARNING : RooFit::DEBUG);
    perf_totalToysRun_ = 0; // reset performance counter
    switch (workingMode_) {
        case MakeLimit:            return runLimit(w, mc_s, mc_b, data, limit, limitErr, hint);
        case MakeSignificance:     return runSignificance(w, mc_s, mc_b, data, limit, limitErr, hint);
        case MakePValues:          return runSinglePoint(w, mc_s, mc_b, data, limit, limitErr, hint);
        case MakeTestStatistics:   return runTestStatistics(w, mc_s, mc_b, data, limit, limitErr, hint);
    }
    assert("Shouldn't get here" == 0);
}

bool HybridNew::runSignificance(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
    HybridNew::Setup setup;
    std::auto_ptr<RooStats::HybridCalculator> hc(create(w, mc_s, mc_b, data, 1.0, setup));
    std::auto_ptr<HypoTestResult> hcResult(readHybridResults_ ? readToysFromFile() : (fork_ ? evalWithFork(*hc) : hc->GetHypoTest()));
    if (hcResult.get() == 0) {
        std::cerr << "Hypotest failed" << std::endl;
        return false;
    }
    if (saveHybridResult_) {
        if (writeToysHere == 0) throw std::logic_error("Option saveToys must be enabled to turn on saveHypoTestResult");
        TString name = TString::Format("HypoTestResult_r%g_%u", 0., RooRandom::integer(std::numeric_limits<UInt_t>::max() - 1));
        writeToysHere->WriteTObject(new HypoTestResult(*hcResult), name);
        if (verbose) std::cout << "Hybrid result saved as " << name << " in " << writeToysHere->GetFile()->GetName() << " : " << writeToysHere->GetPath() << std::endl;
    }
    if (testStat_ == "Atlas" || testStat_ == "Profile") {
        // I need to flip the P-values
        hcResult->SetPValueIsRightTail(!hcResult->GetPValueIsRightTail());
        hcResult->SetTestStatisticData(hcResult->GetTestStatisticData()-1e-9); // issue with < vs <= in discrete models
    } else {
        hcResult->SetTestStatisticData(hcResult->GetTestStatisticData()+1e-9); // issue with < vs <= in discrete models
    }
    limit = hcResult->Significance();
    double sigHi = RooStats::PValueToSignificance( 1 - (hcResult->CLb() + hcResult->CLbError()) ) - limit;
    double sigLo = RooStats::PValueToSignificance( 1 - (hcResult->CLb() - hcResult->CLbError()) ) - limit;
    limitErr = std::max(sigHi,-sigLo);
    std::cout << "\n -- Hybrid New -- \n";
    std::cout << "Significance: " << limit << "  " << sigLo << "/+" << sigHi << " (CLb " << hcResult->CLb() << " +/- " << hcResult->CLbError() << ")\n";
    return isfinite(limit);
}

bool HybridNew::runLimit(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
  RooRealVar *r = dynamic_cast<RooRealVar *>(mc_s->GetParametersOfInterest()->first()); r->setConstant(true);
  w->loadSnapshot("clean");
  if (!plot_.empty()) limitPlot_.reset(new TGraphErrors());

  if ((hint != 0) && (*hint > r->getMin())) {
    r->setMax(std::min<double>(3.0 * (*hint), r->getMax()));
    r->setMin(std::max<double>(0.3 * (*hint), r->getMin()));
  }
  
  typedef std::pair<double,double> CLs_t;

  double clsTarget = 1 - cl; 
  CLs_t clsMin(1,0), clsMax(0,0), clsMid(0,0);
  double rMin = r->getMin(), rMax = r->getMax(); 
  limitErr = 0.5*(rMax - rMin);

  if (verbose > 0) std::cout << "Search for upper limit to the limit" << std::endl;
  for (int tries = 0; tries < 6; ++tries) {
    clsMax = eval(w, mc_s, mc_b, data, rMax);
    if (clsMax.first == 0 || clsMax.first + 3 * fabs(clsMax.second) < clsTarget ) break;
    rMax += rMax;
    if (tries == 5) { 
      std::cerr << "Cannot set higher limit: at " << r->GetName() << " = " << rMax << " still get " << (CLs_ ? "CLs" : "CLsplusb") << " = " << clsMax.first << std::endl;
      return false;
    }
  }
  if (verbose > 0) std::cout << "Search for lower limit to the limit" << std::endl;
  clsMin = eval(w, mc_s, mc_b, data, rMin);
  if (clsMin.first != 1 && clsMin.first - 3 * fabs(clsMin.second) < clsTarget) {
      if (CLs_) { 
          rMin = 0;
          clsMin = CLs_t(1,0); // this is always true for CLs
      } else {
          rMin = -rMax / 4;
          for (int tries = 0; tries < 6; ++tries) {
              clsMin = eval(w, mc_s, mc_b, data, rMin);
              if (clsMin.first == 1 || clsMin.first - 3 * fabs(clsMin.second) > clsTarget) break;
              rMin += rMin;
              if (tries == 5) { 
                  std::cerr << "Cannot set lower limit: at " << r->GetName() << " = " << rMin << " still get " << (CLs_ ? "CLs" : "CLsplusb") << " = " << clsMin.first << std::endl;
                  return false;
              }
          }
      }
  }
  
  if (verbose > 0) std::cout << "Now doing proper bracketing & bisection" << std::endl;
  bool done = false;
  do {
    // determine point by bisection or interpolation
    limit = 0.5*(rMin+rMax); limitErr = 0.5*(rMax-rMin);
    if (algo_ == "logSecant" && clsMax.first != 0) {
        double logMin = log(clsMin.first), logMax = log(clsMax.first), logTarget = log(clsTarget);
        limit = rMin + (rMax-rMin) * (logTarget - logMin)/(logMax - logMin);
        if (clsMax.second != 0 && clsMin.second != 0) {
            limitErr = hypot((logTarget-logMax) * (clsMin.second/clsMin.first), (logTarget-logMin) * (clsMax.second/clsMax.first));
            limitErr *= (rMax-rMin)/((logMax-logMin)*(logMax-logMin));
        }
    }
    r->setError(limitErr);

    // exit if reached accuracy on r 
    if (limitErr < std::max(rAbsAccuracy_, rRelAccuracy_ * limit)) {
        if (verbose > 1) std::cout << "  reached accuracy " << limitErr << " below " << std::max(rAbsAccuracy_, rRelAccuracy_ * limit) << std::endl;
        done = true; break;
    }

    // evaluate point 
    clsMid = eval(w, mc_s, mc_b, data, limit, true, clsTarget);
    if (clsMid.second == -1) {
      std::cerr << "Hypotest failed" << std::endl;
      return false;
    }

    // if sufficiently far away, drop one of the points
    if (fabs(clsMid.first-clsTarget) >= 2*clsMid.second) {
        if ((clsMid.first>clsTarget) == (clsMax.first>clsTarget)) {
          rMax = limit; clsMax = clsMid;
        } else {
          rMin = limit; clsMin = clsMid;
        }
    } else {
        if (verbose > 0) std::cout << "Trying to move the interval edges closer" << std::endl;
        double rMinBound = rMin, rMaxBound = rMax;
        // try to reduce the size of the interval 
        while (clsMin.second == 0 || fabs(rMin-limit) > std::max(rAbsAccuracy_, rRelAccuracy_ * limit)) {
            rMin = 0.5*(rMin+limit); 
            clsMin = eval(w, mc_s, mc_b, data, rMin, true, clsTarget); 
            if (fabs(clsMin.first-clsTarget) <= 2*clsMin.second) break;
            rMinBound = rMin;
        } 
        while (clsMax.second == 0 || fabs(rMax-limit) > std::max(rAbsAccuracy_, rRelAccuracy_ * limit)) {
            rMax = 0.5*(rMax+limit); 
            clsMax = eval(w, mc_s, mc_b, data, rMax, true, clsTarget); 
            if (fabs(clsMax.first-clsTarget) <= 2*clsMax.second) break;
            rMaxBound = rMax;
        } 
        break;
    }
  } while (true);


  if (!done) {
      if (verbose) {
          std::cout << "\n -- HybridNew, before fit -- \n";
          std::cout << "Limit: " << r->GetName() << " < " << limit << " +/- " << limitErr << " [" << rMin << ", " << rMax << "]\n";
      }

      TF1 expoFit("expoFit","[0]*exp([1]*(x-[2]))", rMin, rMax);
      expoFit.FixParameter(0,clsTarget);
      expoFit.SetParameter(1,log(clsMax.first/clsMin.first)/(rMax-rMin));
      expoFit.SetParameter(2,limit);
      TGraphErrors graph(3);
      graph.SetPoint(0, rMin,  clsMin.first); graph.SetPointError(0, 0, clsMin.second);
      graph.SetPoint(1, limit, clsMid.first); graph.SetPointError(1, 0, clsMid.second);
      graph.SetPoint(2, rMax,  clsMax.first); graph.SetPointError(2, 0, clsMax.second);
      graph.Fit(&expoFit,(verbose <= 1 ? "QNR EX0" : "NR EXO"));
     
      if ((rMin < expoFit.GetParameter(2))  && (expoFit.GetParameter(2) < rMax) && 
          (expoFit.GetParError(2) < limitErr) && (expoFit.GetParError(2) < 0.5*(rMax-rMin))) { 
          // sanity check fit result
          limit = expoFit.GetParameter(2);
          limitErr = expoFit.GetParError(2);
      } else if (0.5*(rMax - rMin) < limitErr) {
          limit  = 0.5*(rMax+rMin);
          limitErr = 0.5*(rMax-rMin);
      }
  }

  if (limitPlot_.get()) {
      TCanvas *c1 = new TCanvas("c1","c1");
      limitPlot_->Sort();
      limitPlot_->SetLineWidth(2);
      limitPlot_->Draw("APC");
      TLine line(limitPlot_->GetX()[0], clsTarget, limitPlot_->GetX()[limitPlot_->GetN()-1], clsTarget);
      line.SetLineColor(kRed); line.SetLineWidth(2); line.Draw();
      line.DrawLine(limit, 0, limit, limitPlot_->GetY()[0]);
      line.SetLineWidth(1); line.SetLineStyle(2);
      line.DrawLine(limit-limitErr, 0, limit-limitErr, limitPlot_->GetY()[0]);
      line.DrawLine(limit+limitErr, 0, limit+limitErr, limitPlot_->GetY()[0]);
      c1->Print(plot_.c_str());
  }

  std::cout << "\n -- Hybrid New -- \n";
  std::cout << "Limit: " << r->GetName() << " < " << limit << " +/- " << limitErr << " @ " << cl * 100 << "% CL\n";
  if (verbose > 1) std::cout << "Total toys: " << perf_totalToysRun_ << std::endl;
  return true;
}

bool HybridNew::runSinglePoint(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
    RooRealVar *r = dynamic_cast<RooRealVar *>(mc_s->GetParametersOfInterest()->first()); r->setConstant(true);
    std::pair<double, double> result = eval(w, mc_s, mc_b, data, rValue_, true);
    std::cout << "\n -- Hybrid New -- \n";
    std::cout << (CLs_ ? "CLs = " : "CLsplusb = ") << result.first << " +/- " << result.second << std::endl;
    limit = result.first;
    limitErr = result.second;
    return true;
}

bool HybridNew::runTestStatistics(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
    RooRealVar *r = dynamic_cast<RooRealVar *>(mc_s->GetParametersOfInterest()->first());
    HybridNew::Setup setup;
    std::auto_ptr<RooStats::HybridCalculator> hc(create(w, mc_s, mc_b, data, rValue_, setup));
    RooArgSet nullPOI(*setup.modelConfig_bonly.GetSnapshot());
    if (testStat_ == "Atlas" || testStat_ == "Profile") nullPOI.setRealValue(r->GetName(), rValue_);
    limit = -2 * setup.qvar->Evaluate(data, nullPOI);
    if (testStat_ == "Atlas" || testStat_ == "Profile") limit = -limit; // there's a sign flip for these two
    std::cout << "\n -- Hybrid New -- \n";
    std::cout << "-2 ln Q_{"<< testStat_<<"} = " << limit << std::endl;
    return true;
}

std::pair<double, double> HybridNew::eval(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double rVal, bool adaptive, double clsTarget) {
    if (readHybridResults_) {
        std::auto_ptr<RooStats::HypoTestResult> result(readToysFromFile(rVal));
        std::pair<double, double> ret(-1,-1);
        if (result.get() == 0) { 
            std::cerr << "HypoTestResults for r = " << rVal << " not found in file" << std::endl;
        } else {
            ret.first  = CLs_ ? result->CLs()      : result->CLsplusb();
            ret.second = CLs_ ? result->CLsError() : result->CLsplusbError();
        }
        return ret;
    }

    HybridNew::Setup setup;
    RooRealVar *r = dynamic_cast<RooRealVar *>(mc_s->GetParametersOfInterest()->first());
    r->setVal(rVal);
    if (verbose) std::cout << "  " << r->GetName() << " = " << rVal << " +/- " << r->getError() << std::endl;
    std::auto_ptr<RooStats::HybridCalculator> hc(create(w, mc_s, mc_b, data, rVal, setup));
    std::pair<double, double> ret = eval(*hc, rVal, adaptive, clsTarget);

    // add to plot 
    if (limitPlot_.get()) { 
        limitPlot_->Set(limitPlot_->GetN()+1);
        limitPlot_->SetPoint(limitPlot_->GetN()-1, rVal, ret.first); 
        limitPlot_->SetPointError(limitPlot_->GetN()-1, 0, ret.second);
    }

    return ret;
}



std::auto_ptr<RooStats::HybridCalculator> HybridNew::create(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double rVal, HybridNew::Setup &setup) {
  using namespace RooStats;
  
  RooArgSet  poi(*mc_s->GetParametersOfInterest());
  RooRealVar *r = dynamic_cast<RooRealVar *>(poi.first());
  
  r->setVal(rVal); 
  if (testStat_ == "Atlas" || testStat_ == "Profile") {
    r->setConstant(false); r->setMin(0);
  } else {
    r->setConstant(true);
  }
  setup.modelConfig = ModelConfig(*mc_s);
  setup.modelConfig.SetGlobalObservables(RooArgSet()); // NOT for Hybrid
  
  setup.modelConfig_bonly = ModelConfig(*mc_b);
  setup.modelConfig_bonly.SetGlobalObservables(RooArgSet()); // NOT for Hybrid

  if (testStat_ == "Atlas" || testStat_ == "Profile") {
      // these need the S+B snapshot for both
      // must set it here and not later because calling SetSnapshot more than once does not work properly
      setup.modelConfig_bonly.SetSnapshot(poi);
  } else {
      RooArgSet poiZero; 
      setup.modelConfig_bonly.SetSnapshot(poiZero);
  }

  if (testStat_ == "LEP") {
      //SLR is evaluated using the central value of the nuisance parameters, so I believe we have to put them in the snapshots
      RooArgSet snapS; snapS.addClone(poi); 
      if (withSystematics) snapS.addClone(*mc_s->GetNuisanceParameters());
      RooArgSet snapB; snapB.addClone(snapS);
      snapS.setRealValue(r->GetName(), rVal);
      snapB.setRealValue(r->GetName(),    0);
      if (optimizeTestStatistics_ && !mc_s->GetPdf()->canBeExtended()) {
          // FIXME
          //if (withSystematics && optimizeProductPdf_) {
          //    if (w->pdf("modelObs_b") == 0 || w->pdf("modelObs_s") == 0) 
          //        throw std::invalid_argument("HybridNew: you can't use 'optimizeProduct' if the module does not define 'modelObs_s', 'modelObs_b'");
          //    setup.qvar.reset(new SimplerLikelihoodRatioTestStat(*w->pdf("modelObs_b"), *w->pdf("modelObs_s"), snapB, snapS));
          //} else {
          setup.qvar.reset(new SimplerLikelihoodRatioTestStat(*setup.modelConfig_bonly.GetPdf(),*setup.modelConfig.GetPdf(), snapB, snapS));
          //}
      } else {
          // FIXME
          //if (withSystematics && optimizeProductPdf_) {
          //    if (w->pdf("modelObs_b") == 0 || w->pdf("modelObs_s") == 0) 
          //        throw std::invalid_argument("HybridNew: you can't use 'optimizeProduct' if the module does not define 'modelObs_s', 'modelObs_b'");
          //    setup.qvar.reset(new SimpleLikelihoodRatioTestStat(*w->pdf("modelObs_b"), *w->pdf("modelObs_s")));
          //} else {
          setup.qvar.reset(new SimpleLikelihoodRatioTestStat(*setup.modelConfig_bonly.GetPdf(),*setup.modelConfig.GetPdf()));
          //}
          ((SimpleLikelihoodRatioTestStat&)*setup.qvar).SetNullParameters(snapB); // Null is B
          ((SimpleLikelihoodRatioTestStat&)*setup.qvar).SetAltParameters(snapS);
      }
  } else if (testStat_ == "TEV") {
      // FIXME
    /*if (optimizeTestStatistics_ && !w->pdf("model_s")->canBeExtended()) {
        setup.qvar.reset(new ProfiledLikelihoodRatioTestStat(*setup.modelConfig_bonly.GetPdf(),*setup.modelConfig.GetPdf(), 
                                                             withSystematics ? w->set("nuisances") : 0, poiZero, poi));
    } else {*/   // turn this off for now, it does not work properly
        setup.qvar.reset(new RatioOfProfiledLikelihoodsTestStat(*setup.modelConfig_bonly.GetPdf(),*setup.modelConfig.GetPdf(), setup.modelConfig.GetSnapshot()));
        ((RatioOfProfiledLikelihoodsTestStat&)*setup.qvar).SetSubtractMLE(false);
    //}
  } else if (testStat_ == "Atlas" || testStat_ == "Profile") {
    setup.qvar.reset(new ProfileLikelihoodTestStat(*setup.modelConfig.GetPdf()));
    if (testStat_ == "Atlas") {
       ((ProfileLikelihoodTestStat&)*setup.qvar).SetOneSided(true);
    }
  }
  
  // FIXME
  //if (withSystematics && optimizeProductPdf_) {
  //    if (w->pdf("modelObs_b") == 0 || w->pdf("modelObs_s") == 0) 
  //        throw std::invalid_argument("HybridNew: you can't use 'optimizeProduct' if the module does not define 'modelObs_s', 'modelObs_b'");
  //   setup.modelConfig.SetPdf(*w->pdf("modelObs_s"));
  //   setup.modelConfig_bonly.SetPdf(*w->pdf("modelObs_b"));
  //} 

  setup.toymcsampler.reset(new ToyMCSampler(*setup.qvar, nToys_));

  if (!mc_b->GetPdf()->canBeExtended()) setup.toymcsampler->SetNEventsPerToy(1);
  
  if (nCpu_ > 0) {
    if (verbose > 1) std::cout << "  Will use " << nCpu_ << " CPUs." << std::endl;
    setup.pc.reset(new ProofConfig(*w, nCpu_, "", kFALSE)); 
    setup.toymcsampler->SetProofConfig(setup.pc.get());
  }   
  
  std::auto_ptr<HybridCalculator> hc(new HybridCalculator(data,setup.modelConfig, setup.modelConfig_bonly, setup.toymcsampler.get()));
  if (withSystematics) {
    setup.nuisancePdf.reset(utils::makeNuisancePdf(*mc_s));
    hc->ForcePriorNuisanceNull(*setup.nuisancePdf);
    hc->ForcePriorNuisanceAlt(*setup.nuisancePdf);
  }

  // we need less B toys than S toys
  if (workingMode_ == MakeSignificance) {
      // need only B toys. just keep a few S+B ones to avoid possible divide-by-zero errors somewhere
      hc->SetToys(nToys_, int(0.01*nToys_)+1);
  } else if (!CLs_) {
      // we need only S+B toys to compute CLs+b
      hc->SetToys(int(0.01*nToys_)+1, nToys_);
  } else {
      // need both, but more S+B than B 
      hc->SetToys(int(0.25*nToys_)+1, nToys_);
  }

  static const char * istr = "__HybridNew__importanceSamplingDensity";
  if(importanceSamplingNull_) {
    if(verbose > 1) std::cout << ">>> Enabling importance sampling for null hyp." << std::endl;
    if(!withSystematics) {
      throw std::invalid_argument("Importance sampling is not available without systematics");
    }
    RooArgSet importanceSnapshot;
    importanceSnapshot.addClone(poi);
    importanceSnapshot.addClone(*mc_s->GetNuisanceParameters());
    if (verbose > 2) importanceSnapshot.Print("V");
    hc->SetNullImportanceDensity(mc_b->GetPdf(), &importanceSnapshot);
  }
  if(importanceSamplingAlt_) {
    if(verbose > 1) std::cout << ">>> Enabling importance sampling for alt. hyp." << std::endl;
    if(!withSystematics) {
      throw std::invalid_argument("Importance sampling is not available without systematics");
    }
    if (w->pdf(istr) == 0) {
      w->factory("__oneHalf__[0.5]");
      RooAddPdf *sum = new RooAddPdf(istr, "fifty-fifty", *mc_s->GetPdf(), *mc_b->GetPdf(), *w->var("__oneHalf__"));
      w->import(*sum); 
    }
    RooArgSet importanceSnapshot;
    importanceSnapshot.addClone(poi);
    importanceSnapshot.addClone(*mc_s->GetNuisanceParameters());
    if (verbose > 2) importanceSnapshot.Print("V");
    hc->SetAltImportanceDensity(w->pdf(istr), &importanceSnapshot);
  }

  return hc;
}

std::pair<double,double> 
HybridNew::eval(RooStats::HybridCalculator &hc, double rVal, bool adaptive, double clsTarget) {
    std::auto_ptr<HypoTestResult> hcResult(fork_ ? evalWithFork(hc) : hc.GetHypoTest());
    if (hcResult.get() == 0) {
        std::cerr << "Hypotest failed" << std::endl;
        return std::pair<double, double>(-1,-1);
    }
    if (testStat_ == "Atlas" || testStat_ == "Profile") {
        // I need to flip the P-values
        hcResult->SetPValueIsRightTail(!hcResult->GetPValueIsRightTail());
        hcResult->SetTestStatisticData(hcResult->GetTestStatisticData()-1e-9); // issue with < vs <= in discrete models
    } else {
        hcResult->SetTestStatisticData(hcResult->GetTestStatisticData()+1e-9); // issue with < vs <= in discrete models
    }
    double clsMid    = (CLs_ ? hcResult->CLs()      : hcResult->CLsplusb());
    double clsMidErr = (CLs_ ? hcResult->CLsError() : hcResult->CLsplusbError());
    if (verbose) std::cout << (CLs_ ? "\tCLs = " : "\tCLsplusb = ") << clsMid << " +/- " << clsMidErr << std::endl;
    if (adaptive) {
        hc.SetToys(CLs_ ? nToys_ : 1, 4*nToys_);
        while (clsMidErr >= clsAccuracy_ && (clsTarget == -1 || fabs(clsMid-clsTarget) < 3*clsMidErr) ) {
            std::auto_ptr<HypoTestResult> more(fork_ ? evalWithFork(hc) : hc.GetHypoTest());
            if (testStat_ == "Atlas" || testStat_ == "Profile") more->SetPValueIsRightTail(!more->GetPValueIsRightTail());
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
    perf_totalToysRun_ += (hcResult->GetAltDistribution()->GetSize() + hcResult->GetNullDistribution()->GetSize());

    if (!plot_.empty() && workingMode_ != MakeLimit) {
        HypoTestPlot plot(*hcResult, 30);
        TCanvas *c1 = new TCanvas("c1","c1");
        plot.Draw();
        c1->Print(plot_.c_str());
        delete c1;
    }
    if (saveHybridResult_) {
        if (writeToysHere == 0) throw std::logic_error("Option saveToys must be enabled to turn on saveHypoTestResult");
        TString name = TString::Format("HypoTestResult_r%g_%u", rVal, RooRandom::integer(std::numeric_limits<UInt_t>::max() - 1));
        writeToysHere->WriteTObject(new HypoTestResult(*hcResult), name);
        if (verbose) std::cout << "Hybrid result saved as " << name << " in " << writeToysHere->GetFile()->GetName() << " : " << writeToysHere->GetPath() << std::endl;
    }

    return std::pair<double, double>(clsMid, clsMidErr);
} 

RooStats::HypoTestResult * HybridNew::evalWithFork(RooStats::HybridCalculator &hc) {
    std::auto_ptr<RooStats::HypoTestResult> result(0);
    char *tmpfile = tempnam(NULL,"rstat");
    unsigned int ich = 0;
    std::vector<UInt_t> newSeeds(fork_);
    for (ich = 0; ich < fork_; ++ich) {
        newSeeds[ich] = RooRandom::integer(std::numeric_limits<UInt_t>::max()-1);
        if (!fork()) break; // spawn children (but only in the parent thread)
    }
    if (ich == fork_) { // if i'm the parent
        int cstatus, ret;
        do {
            do { ret = waitpid(-1, &cstatus, 0); } while (ret == -1 && errno == EINTR);
        } while (ret != -1);
        if (ret == -1 && errno != ECHILD) throw std::runtime_error("Didn't wait for child");
        for (ich = 0; ich < fork_; ++ich) {
            TFile *f = TFile::Open(TString::Format("%s.%d.root", tmpfile, ich));
            if (f == 0) throw std::runtime_error(TString::Format("Child didn't leave output file %s.%d.root", tmpfile, ich).Data());
            RooStats::HypoTestResult *res = dynamic_cast<RooStats::HypoTestResult *>(f->Get("result"));
            if (res == 0)  throw std::runtime_error(TString::Format("Child output file %s.%d.root is corrupted", tmpfile, ich).Data());
            if (result.get()) result->Append(res); else result.reset(dynamic_cast<RooStats::HypoTestResult *>(res->Clone()));
            f->Close();
            unlink(TString::Format("%s.%d.root",    tmpfile, ich).Data());
            unlink(TString::Format("%s.%d.out.txt", tmpfile, ich).Data());
            unlink(TString::Format("%s.%d.err.txt", tmpfile, ich).Data());
        }
    } else {
        RooRandom::randomGenerator()->SetSeed(newSeeds[ich]); 
        freopen(TString::Format("%s.%d.out.txt", tmpfile, ich).Data(), "w", stdout);
        freopen(TString::Format("%s.%d.err.txt", tmpfile, ich).Data(), "w", stderr);
        std::cout << " I'm child " << ich << ", seed " << newSeeds[ich] << std::endl;
        RooStats::HypoTestResult *hcResult = hc.GetHypoTest();
        TFile *f = TFile::Open(TString::Format("%s.%d.root", tmpfile, ich), "RECREATE");
        f->WriteTObject(hcResult, "result");
        f->ls();
        f->Close();
        fflush(stdout); fflush(stderr);
        std::cout << "And I'm done" << std::endl;
        throw std::runtime_error("done"); // I have to throw instead of exiting, otherwise there's no proper stack unwinding
                                          // and deleting of intermediate objects, and when the statics get deleted it crashes
                                          // in 5.27.06 (but not in 5.28)
    }
    free(tmpfile);
    return result.release();
}

RooStats::HypoTestResult * HybridNew::readToysFromFile(double rValue) {
    if (!readToysFromHere) throw std::logic_error("Cannot use readHypoTestResult: option toysFile not specified, or input file empty");
    TDirectory *toyDir = readToysFromHere->GetDirectory("toys");
    if (!toyDir) throw std::logic_error("Cannot use readHypoTestResult: option toysFile not specified, or input file empty");
    if (verbose) std::cout << "Reading toys for r = " << rValue << std::endl;
    TString prefix = TString::Format("HypoTestResult_r%g_",rValue);
    std::auto_ptr<RooStats::HypoTestResult> ret;
    TIter next(toyDir->GetListOfKeys()); TKey *k;
    while ((k = (TKey *) next()) != 0) {
        if (TString(k->GetName()).Index(prefix) != 0) continue;
        RooStats::HypoTestResult *toy = dynamic_cast<RooStats::HypoTestResult *>(toyDir->Get(k->GetName()));
        if (toy == 0) continue;
        if (verbose) std::cout << " - " << k->GetName() << std::endl;
        if (ret.get() == 0) {
            ret.reset(new RooStats::HypoTestResult(*toy));
        } else {
            ret->Append(toy);
        }
    }

    return ret.release();
}

