#include <stdexcept>
#include <cstdio>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>

#include "HiggsAnalysis/CombinedLimit/interface/HybridNew.h"
#include <TFile.h>
#include <TKey.h>
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooAbsPdf.h"
#include "RooRandom.h"
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
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,28,0)
        ("nCPU", boost::program_options::value<unsigned int>()->default_value(0), "Use N CPUs with PROOF Lite")
#endif
    ;
}

void HybridNew::applyOptions(const boost::program_options::variables_map &vm) {
    nToys_        = vm.count("toysH")   ? vm["toysH"].as<unsigned int>() : 500;
    clsAccuracy_  = vm.count("clsAcc" ) ? vm["clsAcc" ].as<double>() : 0.005;
    rAbsAccuracy_ = vm.count("rAbsAcc") ? vm["rAbsAcc"].as<double>() : 0.1;
    rRelAccuracy_ = vm.count("rRelAcc") ? vm["rRelAcc"].as<double>() : 0.05;
    rule_     = vm.count("rule")     ? vm["rule"].as<std::string>()     : "CLs";
    testStat_ = vm.count("testStat") ? vm["testStat"].as<std::string>() : "LEP";
    nCpu_     = vm.count("nCPU") ? vm["nCPU"].as<unsigned int>() : 0;
    fork_     = vm.count("fork") ? vm["fork"].as<unsigned int>() : 0;
    if (rule_ == "CLs") {
        CLs_ = true;
    } else if (rule_ == "CLsplusb") {
        CLs_ = false;
    } else {
        throw std::invalid_argument("HybridNew: Rule must be either 'CLs' or 'CLsplusb'");
    }
    rInterval_ = vm.count("rInterval");
    if (testStat_ != "LEP" && testStat_ != "TEV" && testStat_ != "Atlas") {
        throw std::invalid_argument("HybridNew: Test statistics should be one of 'LEP' or 'TEV' or 'Atlas'");
    }
    if ((singlePointScan_ = vm.count("singlePoint"))) {
        rValue_ = vm["singlePoint"].as<float>();
    }
}

bool HybridNew::run(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint) {
    RooFitGlobalKillSentry silence(RooFit::WARNING);
    if (singlePointScan_) return runSinglePoint(w, data, limit, hint);
    bool ret = doSignificance_ ? runSignificance(w, data, limit, hint) : runLimit(w, data, limit, hint);
    return ret;
}

bool HybridNew::runSignificance(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint) {
    RooRealVar *r = w->var("r"); r->setConstant(true);
    HybridNew::Setup setup;
    std::auto_ptr<RooStats::HybridCalculator> hc(create(w, data, r, 1.0, setup));
    std::auto_ptr<HypoTestResult> hcResult(readHybridResults_ ? readToysFromFile() : (fork_ ? evalWithFork(*hc) : hc->GetHypoTest()));
    if (hcResult.get() == 0) {
        std::cerr << "Hypotest failed" << std::endl;
        return false;
    }
    if (saveHybridResult_) {
        if (writeToysHere == 0) throw std::logic_error("Option saveToys must be enabled to turn on saveHypoTestResult");
        TString name = TString::Format("HypoTestResult_%u", RooRandom::integer(std::numeric_limits<UInt_t>::max() - 1));
        writeToysHere->WriteTObject(new HypoTestResult(*hcResult), name);
        if (verbose) std::cout << "Hybrid result saved as " << name << " in " << writeToysHere->GetFile()->GetName() << " : " << writeToysHere->GetPath() << std::endl;
    }
    hcResult->SetTestStatisticData(hcResult->GetTestStatisticData()+1e-9); // issue with < vs <= in discrete models
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

bool HybridNew::runSinglePoint(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint) {
    RooRealVar *r = w->var("r"); r->setConstant(true);
    std::pair<double, double> result = eval(w, data, r, rValue_, true);
    std::cout << "\n -- Hybrid New -- \n";
    std::cout << (CLs_ ? "\tCLs = " : "\tCLsplusb = ") << result.first << " +/- " << result.second << std::endl;
    limit = result.first;
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
        ((SimpleLikelihoodRatioTestStat&)*setup.qvar).SetNullParameters(*setup.modelConfig_bonly.GetSnapshot());
        ((SimpleLikelihoodRatioTestStat&)*setup.qvar).SetAltParameters( *setup.modelConfig.GetSnapshot());
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

    setup.toymcsampler.reset(new ToyMCSampler(*setup.qvar, nToys_));
    if (!w->pdf("model_b")->canBeExtended()) setup.toymcsampler->SetNEventsPerToy(1);

#if ROOT_VERSION_CODE >= ROOT_VERSION(5,28,0)
    if (nCpu_ > 0) {
        if (verbose > 1) std::cout << "  Will use " << nCpu_ << " CPUs." << std::endl;
        setup.pc.reset(new ProofConfig(*w, nCpu_, "", kFALSE)); 
        setup.toymcsampler->SetProofConfig(setup.pc.get());
    }   
#endif

    std::auto_ptr<HybridCalculator> hc(new HybridCalculator(data,setup.modelConfig, setup.modelConfig_bonly, setup.toymcsampler.get()));
    if (withSystematics) {
        hc->ForcePriorNuisanceNull(*w->pdf("nuisancePdf"));
        hc->ForcePriorNuisanceAlt(*w->pdf("nuisancePdf"));
    }
    return hc;
}

std::pair<double,double> 
HybridNew::eval(RooStats::HybridCalculator &hc, bool adaptive, double clsTarget) {
    std::auto_ptr<HypoTestResult> hcResult(fork_ ? evalWithFork(hc) : hc.GetHypoTest());
    if (hcResult.get() == 0) {
        std::cerr << "Hypotest failed" << std::endl;
        return std::pair<double, double>(-1,-1);
    }
    hcResult->SetTestStatisticData(hcResult->GetTestStatisticData()+1e-9); // issue with < vs <= in discrete models
    double clsMid    = (CLs_ ? hcResult->CLs()      : hcResult->CLsplusb());
    double clsMidErr = (CLs_ ? hcResult->CLsError() : hcResult->CLsplusbError());
    if (verbose) std::cout << (CLs_ ? "\tCLs = " : "\tCLsplusb = ") << clsMid << " +/- " << clsMidErr << std::endl;
    if (adaptive) {
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,28,0)
        hc.SetToys(nToys_, 4*nToys_);
#else
        static_cast<ToyMCSampler*>(hc.GetTestStatSampler())->SetNToys(4*nToys_);
#endif
        while (clsMidErr >= clsAccuracy_ && (clsTarget == -1 || fabs(clsMid-clsTarget) < 3*clsMidErr) ) {
            std::auto_ptr<HypoTestResult> more(fork_ ? evalWithFork(hc) : hc.GetHypoTest());
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
        exit(0);
    }
    free(tmpfile);
    return result.release();
}

RooStats::HypoTestResult * HybridNew::readToysFromFile() {
    if (!readToysFromHere) throw std::logic_error("Cannot use readHypoTestResult: option toysFile not specified, or input file empty");
    TDirectory *toyDir = readToysFromHere->GetDirectory("toys");
    if (!toyDir) throw std::logic_error("Cannot use readHypoTestResult: option toysFile not specified, or input file empty");
    if (verbose) std::cout << "Reading toys" << std::endl;

    std::auto_ptr<RooStats::HypoTestResult> ret;
    TIter next(toyDir->GetListOfKeys()); TKey *k;
    while ((k = (TKey *) next()) != 0) {
        if (TString(k->GetName()).Index("HypoTestResult_") != 0) continue;
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

