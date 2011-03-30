#include <stdexcept>
#include <cstdio>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>

#include "HiggsAnalysis/CombinedLimit/interface/Hybrid.h"
#include <TFile.h>
#include <TKey.h>
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooStats/HybridCalculatorOriginal.h"
#include "RooAbsPdf.h"
#include "RooRandom.h"
#include "HiggsAnalysis/CombinedLimit/interface/Combine.h"
#include "HiggsAnalysis/CombinedLimit/interface/RooFitGlobalKillSentry.h"

using namespace RooStats;

unsigned int Hybrid::nToys_ = 500;
double Hybrid::clsAccuracy_ = 0.05;
double Hybrid::rAbsAccuracy_ = 0.1;
double Hybrid::rRelAccuracy_ = 0.05;
std::string Hybrid::rule_ = "CLs";
std::string Hybrid::testStat_ = "LEP";
unsigned int Hybrid::fork_ = 1;
bool Hybrid::rInterval_ = false;
double Hybrid::rValue_  = 1.0;
bool Hybrid::CLs_ = false;
bool Hybrid::saveHybridResult_  = false;
bool Hybrid::readHybridResults_ = false; 
bool Hybrid::singlePointScan_   = false; 

Hybrid::Hybrid() : 
LimitAlgo("Hybrid specific options") {
    options_.add_options()
        ("toysH,T", boost::program_options::value<unsigned int>(&nToys_)->default_value(nToys_),         "Number of Toy MC extractions to compute CLs+b, CLb and CLs")
        ("clsAcc",  boost::program_options::value<double>(&clsAccuracy_ )->default_value(clsAccuracy_),  "Absolute accuracy on CLs to reach to terminate the scan")
        ("rAbsAcc", boost::program_options::value<double>(&rAbsAccuracy_)->default_value(rAbsAccuracy_), "Absolute accuracy on r to reach to terminate the scan")
        ("rRelAcc", boost::program_options::value<double>(&rRelAccuracy_)->default_value(rRelAccuracy_), "Relative accuracy on r to reach to terminate the scan")
        ("rule",    boost::program_options::value<std::string>(&rule_)->default_value(rule_),            "Rule to use: CLs, CLsplusb")
        ("testStat",boost::program_options::value<std::string>(&testStat_)->default_value(testStat_),    "Test statistics: LEP, TEV, Atlas.")
        ("fork",    boost::program_options::value<unsigned int>(&fork_)->default_value(fork_),           "Fork to N processes before running the toys (set to 0 for debugging)")
        ("singlePoint",  boost::program_options::value<float>(),  "Just compute CLs for the given value of r")
        ("rInterval",         "Always try to compute an interval on r even after having found a point satisfiying the CL")
        ("saveHybridResult",  "Save result in the output file  (option saveToys must be enabled)")
        ("readHybridResults", "Read and merge results from file (option toysFile must be enabled)")
    ;
}

void Hybrid::applyOptions(const boost::program_options::variables_map &vm) {
  rInterval_ = vm.count("rInterval");
  saveHybridResult_ = vm.count("saveHybridResult");
  readHybridResults_ = vm.count("readHybridResults");
  if ((singlePointScan_ = vm.count("singlePoint"))) {
    rValue_ = vm["singlePoint"].as<float>();
  }
  validateOptions();
}

void Hybrid::applyDefaultOptions() { validateOptions(); }

void Hybrid::validateOptions() {
  if (rule_ == "CLs") {
    CLs_ = true;
  } else if (rule_ == "CLsplusb") {
    CLs_ = false;
  } else {
    throw std::invalid_argument("Hybrid: Rule should be one of 'CLs' or 'CLsplusb'");
  }
  if (testStat_ != "LEP" && testStat_ != "TEV"/* && testStat_ != "Atlas"*/) { // no Atlas for this, it has bugs.
    throw std::invalid_argument("Hybrid: Test statistics should be one of 'LEP' or 'TEV'"); //or 'Atlas'
  }
}

bool Hybrid::run(RooWorkspace *w, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
  RooFitGlobalKillSentry silence(RooFit::WARNING);
  RooRealVar *r = w->var("r"); r->setConstant(true);
  RooArgSet  poi(*r);
  w->loadSnapshot("clean");
  RooAbsPdf *altModel  = w->pdf("model_s"), *nullModel = w->pdf("model_b");
  
  HybridCalculatorOriginal hc(data,*altModel,*nullModel);
  if (withSystematics) {
    if ((w->set("nuisances") == 0) || (w->pdf("nuisancePdf") == 0)) {
      throw std::logic_error("Hybrid: running with systematics enabled, but nuisances or nuisancePdf not defined.");
    }
    hc.UseNuisance(true);
    hc.SetNuisancePdf(*w->pdf("nuisancePdf"));
    hc.SetNuisanceParameters(*w->set("nuisances"));
  } else {
    hc.UseNuisance(false);
  }
  if (testStat_ == "LEP") {
    hc.SetTestStatistic(1);
    r->setConstant(true);
  } else if (testStat_ == "TEV") {
    hc.SetTestStatistic(3);
    r->setConstant(true);
  } else if (testStat_ == "Atlas") {
    hc.SetTestStatistic(3);
    r->setConstant(false);
  }
  hc.PatchSetExtended(w->pdf("model_b")->canBeExtended()); // Number counting, each dataset has 1 entry 
  hc.SetNumberOfToys(nToys_);

  if (singlePointScan_) return runSinglePoint(hc, w, data, limit, limitErr, hint);
  return doSignificance_ ? runSignificance(hc, w, data, limit, limitErr, hint) : runLimit(hc, w, data, limit, limitErr, hint);
}
  
bool Hybrid::runLimit(HybridCalculatorOriginal& hc, RooWorkspace *w, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
  RooRealVar *r = w->var("r"); r->setConstant(true);
  if ((hint != 0) && (*hint > r->getMin())) {
    r->setMax(std::min<double>(3*(*hint), r->getMax()));
  }
  
  typedef std::pair<double,double> CLs_t;

  double clsTarget = 1 - cl; 
  CLs_t clsMin(1,0), clsMax(0,0);
  double rMin = 0, rMax = r->getMax();

  std::cout << "Search for upper limit to the limit" << std::endl;
  for (;;) {
    CLs_t clsMax = eval(r, r->getMax(), hc);
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
    CLs_t clsMid = eval(r, 0.5*(rMin+rMax), hc, true, clsTarget);
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
      std::cout << "\n -- Hybrid (before determining interval) -- \n";
      std::cout << "Limit: r < " << limit << " +/- " << 0.5*(rMax - rMin) << " @ " <<cl * 100<<"% CL\n";

      double rBoundLow  = limit - 0.5*std::max(rAbsAccuracy_, rRelAccuracy_ * limit);
      for (r->setVal(rMin); r->getVal() < rBoundLow  && (fabs(clsMin.first-clsTarget) >= clsAccuracy_); rMin = r->getVal()) {
        clsMax = eval(r, 0.5*(r->getVal()+limit), hc, true, clsTarget);
      }

      double rBoundHigh = limit + 0.5*std::max(rAbsAccuracy_, rRelAccuracy_ * limit);
      for (r->setVal(rMax); r->getVal() > rBoundHigh && (fabs(clsMax.first-clsTarget) >= clsAccuracy_); rMax = r->getVal()) {
        clsMax = eval(r, 0.5*(r->getVal()+limit), hc, true, clsTarget);
      }
    }
  } else {
    limit = 0.5*(rMax+rMin);
  }
  limitErr = 0.5*(rMax - rMin);

  std::cout << "\n -- Hybrid -- \n";
  std::cout << "Limit: r < " << limit << " +/- " << limitErr << " @ " <<cl * 100<<"% CL\n";
  return true;
}

bool Hybrid::runSignificance(HybridCalculatorOriginal& hc, RooWorkspace *w, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
    using namespace RooStats;
    RooRealVar *r = w->var("r"); 
    r->setVal(1);
    r->setConstant(true);
    std::auto_ptr<HybridResult> hcResult(readHybridResults_ ? readToysFromFile() : (fork_ ? evalWithFork(hc) : hc.GetHypoTest()));
    if (hcResult.get() == 0) {
        std::cerr << "Hypotest failed" << std::endl;
        return false;
    }
    if (saveHybridResult_) {
        if (writeToysHere == 0) throw std::logic_error("Option saveToys must be enabled to turn on saveHybridResult");
        TString name = TString::Format("HybridResult_%u", RooRandom::integer(std::numeric_limits<UInt_t>::max() - 1));
        writeToysHere->WriteTObject(new HybridResult(*hcResult), name);
        if (verbose) std::cout << "Hybrid result saved as " << name << " in " << writeToysHere->GetFile()->GetName() << " : " << writeToysHere->GetPath() << std::endl;
    }
    limit = hcResult->Significance();
    double sigHi = RooStats::PValueToSignificance( 1 - (hcResult->CLb() + hcResult->CLbError()) ) - limit;
    double sigLo = RooStats::PValueToSignificance( 1 - (hcResult->CLb() - hcResult->CLbError()) ) - limit;
    limitErr = std::max(sigHi,-sigLo);
    std::cout << "\n -- Hybrid -- \n";
    std::cout << "Significance: " << limit << "  " << sigLo << "/+" << sigHi << " (CLb " << hcResult->CLb() << " +/- " << hcResult->CLbError() << ")\n";
    return isfinite(limit);
}

bool Hybrid::runSinglePoint(HybridCalculatorOriginal & hc, RooWorkspace *w, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
    std::pair<double, double> result = eval(w->var("r"), rValue_, hc, true);
    std::cout << "\n -- Hybrid -- \n";
    std::cout << (CLs_ ? "\tCLs = " : "\tCLsplusb = ") << result.first << " +/- " << result.second << std::endl;
    limit = result.first;
    limitErr = result.second;
    return true;
}

RooStats::HybridResult * Hybrid::readToysFromFile() {
    if (!readToysFromHere) throw std::logic_error("Cannot use readHybridResult: option toysFile not specified, or input file empty");
    TDirectory *toyDir = readToysFromHere->GetDirectory("toys");
    if (!toyDir) throw std::logic_error("Cannot use readHybridResult: option toysFile not specified, or input file empty");
    if (verbose) std::cout << "Reading toys" << std::endl;

    std::auto_ptr<RooStats::HybridResult> ret;
    TIter next(toyDir->GetListOfKeys()); TKey *k;
    while ((k = (TKey *) next()) != 0) {
        if (TString(k->GetName()).Index("HybridResult_") != 0) continue;
        RooStats::HybridResult *toy = dynamic_cast<RooStats::HybridResult *>(toyDir->Get(k->GetName()));
        if (toy == 0) continue;
        if (verbose) std::cout << " - " << k->GetName() << std::endl;
        if (ret.get() == 0) {
            ret.reset(new RooStats::HybridResult(*toy));
        } else {
            ret->Append(toy);
        }
    }
    return ret.release();
}

std::pair<double, double> Hybrid::eval(RooRealVar *r, double rVal, RooStats::HybridCalculatorOriginal &hc, bool adaptive, double clsTarget) {
    using namespace RooStats;
    r->setVal(rVal);
    std::auto_ptr<HybridResult> hcResult(fork_ ? evalWithFork(hc) : hc.GetHypoTest());
    if (hcResult.get() == 0) {
        std::cerr << "Hypotest failed" << std::endl;
        return std::pair<double, double>(-1,-1);
    }
    double clsMid    = (CLs_ ? hcResult->CLs()      : hcResult->CLsplusb());
    double clsMidErr = (CLs_ ? hcResult->CLsError() : hcResult->CLsplusbError());
    std::cout << "r = " << rVal << (CLs_ ? ": CLs = " : ": CLsplusb = ") << clsMid << " +/- " << clsMidErr << std::endl;
    if (adaptive) {
        while (clsMidErr >= clsAccuracy_ && (clsTarget == -1 || fabs(clsMid-clsTarget) < 3*clsMidErr) ) {
            std::auto_ptr<HybridResult> more(fork_ ? evalWithFork(hc) : hc.GetHypoTest());
            hcResult->Add(more.get());
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
    return std::pair<double, double>(clsMid, clsMidErr);
} 

RooStats::HybridResult * Hybrid::evalWithFork(RooStats::HybridCalculatorOriginal &hc) {
    std::auto_ptr<RooStats::HybridResult> result(0);
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
            RooStats::HybridResult *res = dynamic_cast<RooStats::HybridResult *>(f->Get("result"));
            if (res == 0)  throw std::runtime_error(TString::Format("Child output file %s.%d.root is corrupted", tmpfile, ich).Data());
            if (result.get()) result->Append(res); else result.reset(dynamic_cast<RooStats::HybridResult *>(res->Clone()));
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
        RooStats::HybridResult *hcResult = hc.GetHypoTest();
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
