#include <stdexcept>
#include <cstdio>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>

#include "../interface/HybridNew.h"
#include <TFile.h>
#include <TF1.h>
#include <TKey.h>
#include <TLine.h>
#include <TCanvas.h>
#include <TGraphErrors.h>
#include <TStopwatch.h>
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooAbsPdf.h"
#include "RooFitResult.h"
#include "RooRandom.h"
#include "RooAddPdf.h"
#include "RooConstVar.h"
#include "RooMsgService.h"
#include <RooStats/ModelConfig.h>
#include <RooStats/FrequentistCalculator.h>
#include <RooStats/HybridCalculator.h>
#include <RooStats/SimpleLikelihoodRatioTestStat.h>
#include <RooStats/RatioOfProfiledLikelihoodsTestStat.h>
#include <RooStats/ProfileLikelihoodTestStat.h>
#include <RooStats/ToyMCSampler.h>
#include <RooStats/HypoTestPlot.h>
#include "../interface/Combine.h"
#include "../interface/CloseCoutSentry.h"
#include "../interface/RooFitGlobalKillSentry.h"
#include "../interface/SimplerLikelihoodRatioTestStat.h"
#include "../interface/ProfiledLikelihoodRatioTestStat.h"
#include "../interface/SimplerLikelihoodRatioTestStatExt.h"
#include "../interface/ProfiledLikelihoodRatioTestStatExt.h"
#include "../interface/BestFitSigmaTestStat.h"
#include "../interface/ToyMCSamplerOpt.h"
#include "../interface/utils.h"
#include "../interface/ProfileLikelihood.h"
#include "../interface/ProfilingTools.h"


#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

using namespace RooStats;
using namespace std;

HybridNew::WorkingMode HybridNew::workingMode_ = MakeLimit;
unsigned int HybridNew::nToys_ = 500;
double HybridNew::clsAccuracy_ = 0.005;
double HybridNew::rAbsAccuracy_ = 0.1;
double HybridNew::rRelAccuracy_ = 0.05;
double HybridNew::interpAccuracy_ = 0.2;
std::string HybridNew::rule_ = "CLs";
std::string HybridNew::testStat_ = "LEP";
bool HybridNew::genNuisances_ = true;
bool HybridNew::genGlobalObs_ = false;
bool HybridNew::fitNuisances_ = false;
unsigned int HybridNew::iterations_ = 1;
unsigned int HybridNew::nCpu_ = 0; // proof-lite mode
unsigned int HybridNew::fork_ = 1; // fork mode
std::string         HybridNew::rValue_   = "1.0";
RooArgSet           HybridNew::rValues_;
bool HybridNew::CLs_ = false;
bool HybridNew::saveHybridResult_  = false;
bool HybridNew::readHybridResults_ = false; 
bool  HybridNew::expectedFromGrid_ = false; 
bool  HybridNew::clsQuantiles_ = true; 
float HybridNew::quantileForExpectedFromGrid_ = 0.5; 
bool  HybridNew::fullBToys_ = false; 
bool  HybridNew::fullGrid_ = false; 
bool  HybridNew::saveGrid_ = false; 
bool  HybridNew::noUpdateGrid_ = false; 
std::string HybridNew::gridFile_ = "";
std::string HybridNew::scaleAndConfidenceSelection_ ="0.68,0.95";
bool HybridNew::importanceSamplingNull_ = false;
bool HybridNew::importanceSamplingAlt_  = false;
std::string HybridNew::algo_ = "logSecant";
bool HybridNew::optimizeProductPdf_     = true;
bool HybridNew::optimizeTestStatistics_ = true;
bool HybridNew::newToyMCSampler_        = true;
bool HybridNew::rMinSet_                = false; 
bool HybridNew::rMaxSet_                = false; 
std::string HybridNew::plot_;
std::string HybridNew::minimizerAlgo_ = "Minuit2";
float       HybridNew::minimizerTolerance_ = 1e-2;
float       HybridNew::adaptiveToys_ = -1;
bool        HybridNew::reportPVal_ = false;
float HybridNew::confidenceToleranceForToyScaling_ = 0.2;
float HybridNew::maxProbability_ = 0.999;
#define EPS 1e-9
 
HybridNew::HybridNew() : 
LimitAlgo("HybridNew specific options") {
    options_.add_options()
        ("rule",    boost::program_options::value<std::string>(&rule_)->default_value(rule_),            "Rule to use: CLs, CLsplusb")
        ("testStat",boost::program_options::value<std::string>(&testStat_)->default_value(testStat_),    "Test statistics: LEP, TEV, LHC (previously known as Atlas), Profile.")
        ("singlePoint",  boost::program_options::value<std::string>(&rValue_)->default_value(rValue_),  "Just compute CLs for the given value of the parameter of interest. In case of multiple parameters, use a syntax 'name=value,name2=value2,...'")
        ("onlyTestStat", "Just compute test statistics, not actual p-values (works only with --singlePoint)")
        ("generateNuisances",            boost::program_options::value<bool>(&genNuisances_)->default_value(genNuisances_), "Generate nuisance parameters for each toy")
        ("generateExternalMeasurements", boost::program_options::value<bool>(&genGlobalObs_)->default_value(genGlobalObs_), "Generate external measurements for each toy, taken from the GlobalObservables of the ModelConfig")
        ("fitNuisances", boost::program_options::value<bool>(&fitNuisances_)->default_value(fitNuisances_), "Fit the nuisances, when not generating them.")
        ("searchAlgo", boost::program_options::value<std::string>(&algo_)->default_value(algo_),         "Algorithm to use to search for the limit (bisection, logSecant)")
        ("toysH,T", boost::program_options::value<unsigned int>(&nToys_)->default_value(nToys_),         "Number of Toy MC extractions to compute CLs+b, CLb and CLs")
        ("clsAcc",  boost::program_options::value<double>(&clsAccuracy_ )->default_value(clsAccuracy_),  "Absolute accuracy on CLs to reach to terminate the scan")
        ("rAbsAcc", boost::program_options::value<double>(&rAbsAccuracy_)->default_value(rAbsAccuracy_), "Absolute accuracy on r to reach to terminate the scan")
        ("rRelAcc", boost::program_options::value<double>(&rRelAccuracy_)->default_value(rRelAccuracy_), "Relative accuracy on r to reach to terminate the scan")
        ("interpAcc", boost::program_options::value<double>(&interpAccuracy_)->default_value(interpAccuracy_), "Minimum uncertainty from interpolation delta(x)/(max(x)-min(x))")
        ("iterations,i", boost::program_options::value<unsigned int>(&iterations_)->default_value(iterations_), "Number of times to throw 'toysH' toys to compute the p-values (for --singlePoint if clsAcc is set to zero disabling adaptive generation)")
        ("fork",    boost::program_options::value<unsigned int>(&fork_)->default_value(fork_),           "Fork to N processes before running the toys (set to 0 for debugging)")
        ("nCPU",    boost::program_options::value<unsigned int>(&nCpu_)->default_value(nCpu_),           "Use N CPUs with PROOF Lite (experimental!)")
        ("saveHybridResult",  "Save result in the output file")
        ("readHybridResults", "Read and merge results from file (requires 'toysFile' or 'grid')")
        ("grid",    boost::program_options::value<std::string>(&gridFile_),            "Use the specified file containing a grid of SamplingDistributions for the limit (implies readHybridResults).\n For --singlePoint or --signif use --toysFile=x.root --readHybridResult instead of this.")
        ("expectedFromGrid", boost::program_options::value<float>(&quantileForExpectedFromGrid_)->default_value(0.5), "Use the grid to compute the expected limit for this quantile")
        ("signalForSignificance", boost::program_options::value<std::string>()->default_value("1"), "Use this value of the parameter of interest when generating signal toys for expected significance (same syntax as --singlePoint)")
        ("clsQuantiles", boost::program_options::value<bool>(&clsQuantiles_)->default_value(clsQuantiles_), "Compute correct quantiles of CLs or CLsplusb instead of assuming they're the same as CLb ones")
        //("importanceSamplingNull", boost::program_options::value<bool>(&importanceSamplingNull_)->default_value(importanceSamplingNull_),  
        //                           "Enable importance sampling for null hypothesis (background only)") 
        //("importanceSamplingAlt",  boost::program_options::value<bool>(&importanceSamplingAlt_)->default_value(importanceSamplingAlt_),    
        //                           "Enable importance sampling for alternative hypothesis (signal plus background)") 
        ("optimizeTestStatistics", boost::program_options::value<bool>(&optimizeTestStatistics_)->default_value(optimizeTestStatistics_), 
                                   "Use optimized test statistics if the likelihood is not extended (works for LEP and TEV test statistics).")
        ("optimizeProductPdf",     boost::program_options::value<bool>(&optimizeProductPdf_)->default_value(optimizeProductPdf_),      
                                   "Optimize the code factorizing pdfs")
        ("minimizerAlgo",      boost::program_options::value<std::string>(&minimizerAlgo_)->default_value(minimizerAlgo_), "Choice of minimizer used for profiling (Minuit vs Minuit2)")
        ("minimizerTolerance", boost::program_options::value<float>(&minimizerTolerance_)->default_value(minimizerTolerance_),  "Tolerance for minimizer used for profiling")
        ("plot",   boost::program_options::value<std::string>(&plot_), "Save a plot of the result (test statistics distributions or limit scan)")
        ("frequentist", "Shortcut to switch to Frequentist mode (--generateNuisances=0 --generateExternalMeasurements=1 --fitNuisances=1)")
        ("newToyMCSampler", boost::program_options::value<bool>(&newToyMCSampler_)->default_value(newToyMCSampler_), "Use new ToyMC sampler with support for mixed binned-unbinned generation. On by default, you can turn it off if it doesn't work for your workspace.")
        ("fullGrid", "Evaluate p-values at all grid points, without optimitations")
        ("saveGrid", "Save CLs or (or FC p-value) at all grid points in the output tree. The value of 'r' is saved in the 'limit' branch, while the CLs or p-value in the 'quantileExpected' branch and the uncertainty on 'limitErr' (since there's no quantileExpectedErr)")
        ("noUpdateGrid", "Do not update test statistics at grid points")
        ("fullBToys", "Run as many B toys as S ones (default is to run 1/4 of b-only toys)")
        ("pvalue", "Report p-value instead of significance (when running with --significance)")
        ("adaptiveToys",boost::program_options::value<float>(&adaptiveToys_)->default_value(adaptiveToys_), "Throw less toys far from interesting contours , --toysH scaled by scale when prob is far from any of CL_i = {importanceContours} ")
        ("importantContours",boost::program_options::value<std::string>(&scaleAndConfidenceSelection_)->default_value(scaleAndConfidenceSelection_), "Throw less toys far from interesting contours , format : CL_1,CL_2,..CL_N (--toysH scaled down when prob is far from any of CL_i) ")
        ("maxProbability", boost::program_options::value<float>(&maxProbability_)->default_value(maxProbability_),  "when point is >  maxProbability countour, don't bother throwing toys")
        ("confidenceTolerance", boost::program_options::value<float>(&confidenceToleranceForToyScaling_)->default_value(confidenceToleranceForToyScaling_),  "Determine what 'far' means for adatptiveToys. (relative in terms of (1-cl))")
	
    ;
}

void HybridNew::applyOptions(const boost::program_options::variables_map &vm) {
    rMinSet_ = vm.count("rMin")>0; rMaxSet_ = vm.count("rMax")>0;
    if (vm.count("expectedFromGrid") && !vm["expectedFromGrid"].defaulted()) {
        //if (!vm.count("grid")) throw std::invalid_argument("HybridNew: Can't use --expectedFromGrid without --grid!");
        if (quantileForExpectedFromGrid_ <= 0 || quantileForExpectedFromGrid_ >= 1.0) throw std::invalid_argument("HybridNew: the quantile for the expected limit must be between 0 and 1");
        expectedFromGrid_ = true;
        g_quantileExpected_ = quantileForExpectedFromGrid_;
    }


    if (vm.count("frequentist")) {
        genNuisances_ = 0; genGlobalObs_ = withSystematics; fitNuisances_ = withSystematics;
        if (vm["testStat"].defaulted()) testStat_ = "LHC";
        if (vm["toys"].as<int>() > 0 and vm.count("toysFrequentist")) {
            if (vm["fitNuisances"].defaulted() && withSystematics) {
                std::cout << "When tossing frequenst toys outside the HybridNew, the nuisances will not be refitted for each toy by default. This can be changed by specifying explicitly the fitNuisances option" << std::endl;
                fitNuisances_ = false;
            }
        }
    }
    if (genGlobalObs_ && genNuisances_) {
        std::cerr << "ALERT: generating both global observables and nuisance parameters at the same time is not validated." << std::endl;
    }
    if (!vm["singlePoint"].defaulted()) {
        if (doSignificance_) throw std::invalid_argument("HybridNew: Can't use --significance and --singlePoint at the same time");
        workingMode_ = ( vm.count("onlyTestStat") ? MakeTestStatistics : MakePValues );
    } else if (vm.count("onlyTestStat")) {
        if (doSignificance_) workingMode_ = MakeSignificanceTestStatistics;
        else throw std::invalid_argument("HybridNew: --onlyTestStat works only with --singlePoint or --significance");
    } else if (doSignificance_) {
        workingMode_ = MakeSignificance;
        rValue_ = vm["signalForSignificance"].as<std::string>();
    } else {
        workingMode_ = MakeLimit;
    }
    saveHybridResult_ = vm.count("saveHybridResult");
    readHybridResults_ = vm.count("readHybridResults") || vm.count("grid");
    if (readHybridResults_ && !(vm.count("toysFile") || vm.count("grid")))     throw std::invalid_argument("HybridNew: must have 'toysFile' or 'grid' option to have 'readHybridResults'\n");
    mass_ = vm["mass"].as<float>();
    fullGrid_ = vm.count("fullGrid");
    saveGrid_ = vm.count("saveGrid");
    fullBToys_ = vm.count("fullBToys");
    noUpdateGrid_ = vm.count("noUpdateGrid");
    reportPVal_ = vm.count("pvalue");
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
    } else if (rule_ == "FC") {
        CLs_ = false;
    } else {
        throw std::invalid_argument("HybridNew: Rule must be either 'CLs' or 'CLsplusb'");
    }
    if (testStat_ == "SimpleLikelihoodRatio"      || testStat_ == "SLR" ) { testStat_ = "LEP";     }
    if (testStat_ == "RatioOfProfiledLikelihoods" || testStat_ == "ROPL") { testStat_ = "TEV";     }
    if (testStat_ == "ProfileLikelihood"          || testStat_ == "PL")   { testStat_ = "Profile"; }
    if (testStat_ == "ModifiedProfileLikelihood"  || testStat_ == "MPL")  { testStat_ = "LHC";     }
    if (testStat_ == "SignFlipProfileLikelihood"  || testStat_ == "SFPL") { testStat_ = "LHCFC";   }
    if (testStat_ == "Atlas") { testStat_ = "LHC"; std::cout << "Note: the Atlas test statistics is now known as LHC test statistics.\n" << std::endl; }
    if (testStat_ != "LEP" && testStat_ != "TEV" && testStat_ != "LHC"  && testStat_ != "LHCFC" && testStat_ != "Profile" && testStat_ != "MLZ") {
        throw std::invalid_argument("HybridNew: Test statistics should be one of 'LEP' or 'TEV' or 'LHC' (previously known as 'Atlas') or 'Profile'");
    }
    if (verbose) {
        if (testStat_ == "LEP")     std::cout << ">>> using the Simple Likelihood Ratio test statistics (Q_LEP)" << std::endl;
        if (testStat_ == "TEV")     std::cout << ">>> using the Ratio of Profiled Likelihoods test statistics (Q_TEV)" << std::endl;
        if (testStat_ == "LHC")     std::cout << ">>> using the Profile Likelihood test statistics modified for upper limits (Q_LHC)" << std::endl;
        if (testStat_ == "LHCFC")   std::cout << ">>> using the Profile Likelihood test statistics modified for upper limits and Feldman-Cousins (Q_LHCFC)" << std::endl;
        if (testStat_ == "Profile") std::cout << ">>> using the Profile Likelihood test statistics not modified for upper limits (Q_Profile)" << std::endl;
        if (testStat_ == "MLZ")     std::cout << ">>> using the Maximum likelihood estimator of the signal strength as test statistics" << std::endl;
    }
    if (readHybridResults_ || workingMode_ == MakeTestStatistics || workingMode_ == MakeSignificanceTestStatistics) {
        // If not generating toys, don't need to fit nuisance parameters
        fitNuisances_ = false;
    }
    if (reportPVal_ && workingMode_ != MakeSignificance) throw std::invalid_argument("HybridNew: option --pvalue must go together with --significance");
}

void HybridNew::setupPOI(RooStats::ModelConfig *mc_s) {
    RooArgSet POI(*mc_s->GetParametersOfInterest());
    if (rValue_.find("=") == std::string::npos) {
        if (POI.getSize() != 1) throw std::invalid_argument("Error: the argument to --singlePoint or --signalForSignificance is a single value, but there are multiple POIs");
        POI.snapshot(rValues_);
        errno = 0; // check for errors in str->float conversion
        ((RooRealVar*)rValues_.first())->setVal(strtod(rValue_.c_str(),NULL));
        if (errno != 0) std::invalid_argument("Error: the argument to --singlePoint or --signalForSignificance is not a valid number.");
    } else {
        std::string::size_type eqidx = 0, colidx = 0, colidx2;
        do {
            eqidx   = rValue_.find("=", colidx);
            colidx2 = rValue_.find(",", colidx+1);
            if (eqidx == std::string::npos || (colidx2 != std::string::npos && colidx2 < eqidx)) {
                throw std::invalid_argument("Error: the argument to --singlePoint or --signalForSignificance is not in the forms 'value' or 'name1=value1,name2=value2,...'\n");
            }
            std::string poiName = rValue_.substr(colidx, eqidx-colidx);
            std::string poiVal  = rValue_.substr(eqidx+1, (colidx2 == std::string::npos ? std::string::npos : colidx2 - eqidx - 1));
            RooAbsArg *poi = POI.find(poiName.c_str());
            if (poi == 0) throw std::invalid_argument("Error: unknown parameter '"+poiName+"' passed to --singlePoint or --signalForSignificance.");
            rValues_.addClone(*poi);
            errno = 0;
            rValues_.setRealValue(poi->GetName(), strtod(poiVal.c_str(),NULL));
            if (errno != 0) throw std::invalid_argument("Error: invalid value '"+poiVal+"' for parameter '"+poiName+"' passed to --singlePoint or --signalForSignificance.");
            colidx = colidx2+1;
        } while (colidx2 != std::string::npos);
        if (rValues_.getSize() != POI.getSize()) {
            throw std::invalid_argument("Error: not all parameters of interest specified in  --singlePoint or --signalForSignificance");
        }
    }
}

bool HybridNew::run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
    RooFitGlobalKillSentry silence(verbose <= 1 ? RooFit::WARNING : RooFit::DEBUG);
    ProfileLikelihood::MinimizerSentry minimizerConfig(minimizerAlgo_, minimizerTolerance_);
    perf_totalToysRun_ = 0; // reset performance counter
    if (rValues_.getSize() == 0) setupPOI(mc_s);
    switch (workingMode_) {
        case MakeLimit:            return runLimit(w, mc_s, mc_b, data, limit, limitErr, hint);
        case MakeSignificance:     return runSignificance(w, mc_s, mc_b, data, limit, limitErr, hint);
        case MakePValues:          return runSinglePoint(w, mc_s, mc_b, data, limit, limitErr, hint);
        case MakeTestStatistics:   
        case MakeSignificanceTestStatistics: 
                                   return runTestStatistics(w, mc_s, mc_b, data, limit, limitErr, hint);
    }
    assert("Shouldn't get here" == 0);
}

bool HybridNew::runSignificance(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
    HybridNew::Setup setup;
    std::auto_ptr<RooStats::HybridCalculator> hc(create(w, mc_s, mc_b, data, rValues_, setup));
    std::auto_ptr<HypoTestResult> hcResult;
    if (readHybridResults_) {
        hcResult.reset(readToysFromFile(rValues_));
    } else {
        hcResult.reset(evalGeneric(*hc));
        for (unsigned int i = 1; i < iterations_; ++i) {
            std::auto_ptr<HypoTestResult> more(evalGeneric(*hc));
            hcResult->Append(more.get());
            if (verbose) std::cout << "\t1 - CLb = " << hcResult->CLb() << " +/- " << hcResult->CLbError() << std::endl;
        }
    }
    if (hcResult.get() == 0) {
        std::cerr << "Hypotest failed" << std::endl;
        return false;
    }
    if (saveHybridResult_) {
        TString name = TString::Format("HypoTestResult_mh%g",mass_);
        RooLinkedListIter it = rValues_.iterator();
        for (RooRealVar *rIn = (RooRealVar*) it.Next(); rIn != 0; rIn = (RooRealVar*) it.Next()) {
            name += Form("_%s%g", rIn->GetName(), rIn->getVal());
        }
        name += Form("_%u", RooRandom::integer(std::numeric_limits<UInt_t>::max() - 1));
        writeToysHere->WriteTObject(new HypoTestResult(*hcResult), name);
        if (verbose) std::cout << "Hybrid result saved as " << name << " in " << writeToysHere->GetFile()->GetName() << " : " << writeToysHere->GetPath() << std::endl;
    }
    if (verbose > 1) {
        std::cout << "Observed test statistics in data: " << hcResult->GetTestStatisticData() << std::endl;
        std::cout << "Background-only toys sampled:     " << hcResult->GetNullDistribution()->GetSize() << std::endl;
    }
    if (expectedFromGrid_) applyExpectedQuantile(*hcResult);
    // I don't need to flip the P-values for significances, only for limits
    hcResult->SetTestStatisticData(hcResult->GetTestStatisticData()+EPS); // issue with < vs <= in discrete models
    double sig   = hcResult->Significance();
    double sigHi = RooStats::PValueToSignificance( (hcResult->CLb() - hcResult->CLbError()) ) - sig;
    double sigLo = RooStats::PValueToSignificance( (hcResult->CLb() + hcResult->CLbError()) ) - sig;
    if (reportPVal_) {
        limit    = hcResult->NullPValue();
        limitErr = hcResult->NullPValueError();
    } else {
        limit = sig;
        limitErr = std::max(sigHi,-sigLo);
    }
    std::cout << "\n -- Hybrid New -- \n";
    std::cout << "Significance: " << sig << "  " << sigLo << "/+" << sigHi << "\n";
    std::cout << "Null p-value: " << hcResult->NullPValue() << " +/- " << hcResult->NullPValueError() << "\n";
    return isfinite(limit);
}

bool HybridNew::runLimit(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
  if (mc_s->GetParametersOfInterest()->getSize() != 1) throw std::logic_error("Cannot run limits in more than one dimension, for the moment.");
  RooRealVar *r = dynamic_cast<RooRealVar *>(mc_s->GetParametersOfInterest()->first()); r->setConstant(true);
  w->loadSnapshot("clean");

  if ((hint != 0) && (*hint > r->getMin())) {
      r->setMax(std::min<double>(3.0 * (*hint), r->getMax()));
      r->setMin(std::max<double>(0.3 * (*hint), r->getMin()));
  }
  
  typedef std::pair<double,double> CLs_t;

  double clsTarget = 1 - cl; 
  CLs_t clsMin(1,0), clsMax(0,0), clsMid(0,0);
  double rMin = r->getMin(), rMax = r->getMax(); 
  limit    = 0.5*(rMax + rMin);
  limitErr = 0.5*(rMax - rMin);
  bool done = false;

  TF1 expoFit("expoFit","[0]*exp([1]*(x-[2]))", rMin, rMax);

  if (readHybridResults_) { 
      if (verbose > 0) std::cout << "Search for upper limit using pre-computed grid of p-values" << std::endl;

      if (!gridFile_.empty()) {
        if (grid_.empty()) {
            std::auto_ptr<TFile> gridFile(TFile::Open(gridFile_.c_str()));
            if (gridFile.get() == 0) throw std::runtime_error(("Can't open grid file "+gridFile_).c_str());
            TDirectory *toyDir = gridFile->GetDirectory("toys");
            if (!toyDir) throw std::logic_error("Cannot use readHypoTestResult: empty toy dir in input file empty");
            readGrid(toyDir, rMinSet_ ? rMin : -99e99, rMaxSet_ ? rMax :+99e99);
        }
        if (grid_.size() <= 1) throw std::logic_error("The grid must contain at least 2 points."); 
        if (noUpdateGrid_) {
            if (testStat_ == "LHCFC" && rule_ == "FC" && (saveGrid_ || lowerLimit_)) {
                std::cout << "Will have to re-run points for which the test statistics was set to zero" << std::endl;
                updateGridDataFC(w, mc_s, mc_b, data, !fullGrid_, clsTarget);
            } else {
                std::cout << "Will use the test statistics that had already been computed" << std::endl;
            }
        } else {
            updateGridData(w, mc_s, mc_b, data, !fullGrid_, clsTarget);
        }
      } else throw std::logic_error("When setting a limit reading results from file, a grid file must be specified with option --grid");
      if (grid_.size() <= 1) throw std::logic_error("The grid must contain at least 2 points."); 

      useGrid();

      double minDist=1e3;
      double nearest = 0;
      rMin = limitPlot_->GetX()[0]; 
      rMax = limitPlot_->GetX()[limitPlot_->GetN()-1];
      for (int i = 0, n = limitPlot_->GetN(); i < n; ++i) {
          double x = limitPlot_->GetX()[i], y = limitPlot_->GetY()[i], ey = limitPlot_->GetErrorY(i);
          if (verbose > 0) printf("  r %.2f: %s = %6.4f +/- %6.4f\n", x, CLs_ ? "CLs" : "CLsplusb", y, ey);
          if (saveGrid_) { limit = x; limitErr = ey; Combine::commitPoint(false, y); }
          if (y-3*max(ey,0.01) >= clsTarget) { rMin = x; clsMin = CLs_t(y,ey); }
          if (fabs(y-clsTarget) < minDist) { nearest = x; minDist = fabs(y-clsTarget); }
          rMax = x; clsMax = CLs_t(y,ey);    
          if (y+3*max(ey,0.01) <= clsTarget && !fullGrid_) break; 
      }
      limit = nearest;
      if (verbose > 0) std::cout << " after scan x ~ " << limit << ", bounds [ " << rMin << ", " << rMax << "]" << std::endl;
      limitErr = std::max(limit-rMin, rMax-limit);
      expoFit.SetRange(rMin,rMax);

      if (limitErr < std::max(rAbsAccuracy_, rRelAccuracy_ * limit)) {
          if (verbose > 1) std::cout << "  reached accuracy " << limitErr << " below " << std::max(rAbsAccuracy_, rRelAccuracy_ * limit) << std::endl;
          done = true; 
      }
  } else {
      limitPlot_.reset(new TGraphErrors());

      if (verbose > 0) std::cout << "Search for upper limit to the limit" << std::endl;
      for (int tries = 0; tries < 6; ++tries) {
          clsMax = eval(w, mc_s, mc_b, data, rMax);
          if (lowerLimit_) break; // we can't search for lower limits this way
          if (clsMax.first == 0 || clsMax.first + 3 * fabs(clsMax.second) < clsTarget ) break;
          rMax += rMax;
          if (tries == 5) { 
              std::cerr << "Cannot set higher limit: at " << r->GetName() << " = " << rMax << " still get " << (CLs_ ? "CLs" : "CLsplusb") << " = " << clsMax.first << std::endl;
              return false;
          }
      }
      if (verbose > 0) std::cout << "Search for lower limit to the limit" << std::endl;
      clsMin = (CLs_ && rMin == 0 ? CLs_t(1,0) : eval(w, mc_s, mc_b, data, rMin));
      if (!lowerLimit_ && clsMin.first != 1 && clsMin.first - 3 * fabs(clsMin.second) < clsTarget) {
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
      do {
          // determine point by bisection or interpolation
          limit = 0.5*(rMin+rMax); limitErr = 0.5*(rMax-rMin);
          if (algo_ == "logSecant" && clsMax.first != 0 && clsMin.first != 0) {
              double logMin = log(clsMin.first), logMax = log(clsMax.first), logTarget = log(clsTarget);
              limit = rMin + (rMax-rMin) * (logTarget - logMin)/(logMax - logMin);
              if (clsMax.second != 0 && clsMin.second != 0) {
                  limitErr = hypot((logTarget-logMax) * (clsMin.second/clsMin.first), (logTarget-logMin) * (clsMax.second/clsMax.first));
                  limitErr *= (rMax-rMin)/((logMax-logMin)*(logMax-logMin));
              }
              // disallow "too precise" interpolations
              if (limitErr < interpAccuracy_ * (rMax-rMin)) limitErr = interpAccuracy_ * (rMax-rMin);
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
              expoFit.SetRange(rMinBound,rMaxBound);
              break;
          }
      } while (true);

  }

  if (!done) { // didn't reach accuracy with scan, now do fit
      double rMinBound, rMaxBound; expoFit.GetRange(rMinBound, rMaxBound);
      if (verbose) {
          std::cout << "\n -- HybridNew, before fit -- \n";
          std::cout << "Limit: " << r->GetName() << " < " << limit << " +/- " << limitErr << " [" << rMinBound << ", " << rMaxBound << "]\n";
      }

      expoFit.FixParameter(0,clsTarget);
      double clsmaxfirst = clsMax.first;
      if ( clsmaxfirst == 0.0 ) clsmaxfirst = 0.005;
      double par1guess = log(clsmaxfirst/clsMin.first)/(rMax-rMin);
      expoFit.SetParameter(1,par1guess);
      expoFit.SetParameter(2,limit);
      limitErr = std::max(fabs(rMinBound-limit), fabs(rMaxBound-limit));
      int npoints = 0; 
      for (int j = 0; j < limitPlot_->GetN(); ++j) { 
          if (limitPlot_->GetX()[j] >= rMinBound && limitPlot_->GetX()[j] <= rMaxBound) npoints++; 
      }
      for (int i = 0, imax = (readHybridResults_ ? 0 : 8); i <= imax; ++i, ++npoints) {
          limitPlot_->Sort();
          limitPlot_->Fit(&expoFit,(verbose <= 1 ? "QNR EX0" : "NR EX0"));
          if (verbose) {
              std::cout << "Fit to " << npoints << " points: " << expoFit.GetParameter(2) << " +/- " << expoFit.GetParError(2) << std::endl;
          }
          if ((rMinBound < expoFit.GetParameter(2))  && (expoFit.GetParameter(2) < rMaxBound) && (expoFit.GetParError(2) < 0.5*(rMaxBound-rMinBound))) { 
              // sanity check fit result
              limit = expoFit.GetParameter(2);
              limitErr = expoFit.GetParError(2);
              if (limitErr < std::max(rAbsAccuracy_, rRelAccuracy_ * limit)) break;
          }
          // add one point in the interval. 
          double rTry = RooRandom::uniform()*(rMaxBound-rMinBound)+rMinBound; 
          if (i != imax) eval(w, mc_s, mc_b, data, rTry, true, clsTarget);
      } 
  }
 
  if (!plot_.empty() && limitPlot_.get()) {
      TCanvas *c1 = new TCanvas("c1","c1");
      limitPlot_->Sort();
      limitPlot_->SetLineWidth(2);
      double xmin = r->getMin(), xmax = r->getMax();
      for (int j = 0; j < limitPlot_->GetN(); ++j) {
        if (limitPlot_->GetY()[j] > 1.4*clsTarget || limitPlot_->GetY()[j] < 0.6*clsTarget) continue;
        xmin = std::min(limitPlot_->GetX()[j], xmin);
        xmax = std::max(limitPlot_->GetX()[j], xmax);
      }
      limitPlot_->GetXaxis()->SetRangeUser(xmin,xmax);
      limitPlot_->GetYaxis()->SetRangeUser(0.5*clsTarget, 1.5*clsTarget);
      limitPlot_->Draw("AP");
      expoFit.Draw("SAME");
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
    std::pair<double, double> result = eval(w, mc_s, mc_b, data, rValues_, clsAccuracy_ != 0);
    std::cout << "\n -- Hybrid New -- \n";
    std::cout << (CLs_ ? "CLs = " : "CLsplusb = ") << result.first << " +/- " << result.second << std::endl;
    if (verbose > 1) std::cout << "Total toys: " << perf_totalToysRun_ << std::endl;
    limit = result.first;
    limitErr = result.second;
    return true;
}

bool HybridNew::runTestStatistics(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) {
    bool isProfile = (testStat_ == "LHC" || testStat_ == "LHCFC"  || testStat_ == "Profile");
    if (readHybridResults_ && expectedFromGrid_) {
        std::auto_ptr<RooStats::HypoTestResult> result(readToysFromFile(rValues_));
        applyExpectedQuantile(*result);
        limit = -2 * result->GetTestStatisticData();
    } else {    
        HybridNew::Setup setup;
        std::auto_ptr<RooStats::HybridCalculator> hc(create(w, mc_s, mc_b, data, rValues_, setup));
        RooArgSet nullPOI(*setup.modelConfig_bonly.GetSnapshot());
        if (isProfile) { 
            /// Probably useless, but should double-check before deleting this.
            nullPOI.assignValueOnly(rValues_);
        }
        limit = -2 * setup.qvar->Evaluate(data, nullPOI);
    }
    if (isProfile) limit = -limit; // there's a sign flip for these two
    std::cout << "\n -- Hybrid New -- \n";
    std::cout << "-2 ln Q_{"<< testStat_<<"} = " << limit << std::endl;
    return true;
}

std::pair<double, double> HybridNew::eval(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double rVal, bool adaptive, double clsTarget) {
    RooArgSet rValues;
    mc_s->GetParametersOfInterest()->snapshot(rValues);
    RooRealVar *r = dynamic_cast<RooRealVar*>(rValues.first());
    if (rVal > r->getMax()) r->setMax(2*rVal);
    r->setVal(rVal);
    return eval(w,mc_s,mc_b,data,rValues,adaptive,clsTarget);
}

std::pair<double, double> HybridNew::eval(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, const RooAbsCollection & rVals, bool adaptive, double clsTarget) {
    if (readHybridResults_) {
        bool isProfile = (testStat_ == "LHC" || testStat_ == "LHCFC"  || testStat_ == "Profile");
        std::auto_ptr<RooStats::HypoTestResult> result(readToysFromFile(rVals));
        std::pair<double, double> ret(-1,-1);
        assert(result.get() != 0 && "Null result in HybridNew::eval(Workspace,...) after readToysFromFile");
        if (expectedFromGrid_) {
            applyExpectedQuantile(*result);
            result->SetTestStatisticData(result->GetTestStatisticData() + (isProfile ? -EPS : EPS));
        } else if (!noUpdateGrid_) {
            Setup setup;
            std::auto_ptr<RooStats::HybridCalculator> hc = create(w, mc_s, mc_b, data, rVals, setup);
            RooArgSet nullPOI(*setup.modelConfig_bonly.GetSnapshot());
            if (isProfile) nullPOI.assignValueOnly(rVals);
            double testStat = setup.qvar->Evaluate(data, nullPOI);
            result->SetTestStatisticData(testStat + (isProfile ? -EPS : EPS));
        }
        ret = eval(*result, rVals);
        return ret;
    }

    HybridNew::Setup setup;
    RooLinkedListIter it = rVals.iterator();
    for (RooRealVar *rIn = (RooRealVar*) it.Next(); rIn != 0; rIn = (RooRealVar*) it.Next()) {
        RooRealVar *r = dynamic_cast<RooRealVar *>(mc_s->GetParametersOfInterest()->find(rIn->GetName()));
        r->setVal(rIn->getVal());
        if (verbose) std::cout << "  " << r->GetName() << " = " << rIn->getVal() << " +/- " << r->getError() << std::endl;
    } 
    std::auto_ptr<RooStats::HybridCalculator> hc(create(w, mc_s, mc_b, data, rVals, setup));
    std::pair<double, double> ret = eval(*hc, rVals, adaptive, clsTarget);

    // add to plot 
    if (limitPlot_.get()) { 
        limitPlot_->Set(limitPlot_->GetN()+1);
        limitPlot_->SetPoint(limitPlot_->GetN()-1, ((RooAbsReal*)rVals.first())->getVal(), ret.first); 
        limitPlot_->SetPointError(limitPlot_->GetN()-1, 0, ret.second);
    }

    return ret;
}



std::auto_ptr<RooStats::HybridCalculator> HybridNew::create(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double rVal, HybridNew::Setup &setup) {
    RooArgSet rValues;
    mc_s->GetParametersOfInterest()->snapshot(rValues);
    RooRealVar *r = dynamic_cast<RooRealVar*>(rValues.first());
    if (rVal > r->getMax()) r->setMax(2*rVal);
    r->setVal(rVal);
    return create(w,mc_s,mc_b,data,rValues,setup);
}

std::auto_ptr<RooStats::HybridCalculator> HybridNew::create(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, const RooAbsCollection & rVals, HybridNew::Setup &setup) {
  using namespace RooStats;
  
  w->loadSnapshot("clean");
  // realData_ = &data;  

  RooArgSet  poi(*mc_s->GetParametersOfInterest()), params(poi);
  RooRealVar *r = dynamic_cast<RooRealVar *>(poi.first());

  if (poi.getSize() == 1) { // here things are a bit more convoluted, although they could probably be cleaned up
      double rVal = ((RooAbsReal*)rVals.first())->getVal();
      if (testStat_ != "MLZ") r->setMax(rVal); 
      r->setVal(rVal); 
      if (testStat_ == "LHC" || testStat_ == "Profile") {
        r->setConstant(false); r->setMin(0); 
        if (workingMode_ == MakeSignificance || workingMode_ == MakeSignificanceTestStatistics) {
            r->setVal(0);
            // r->removeMax(); // NO, this is done within the test statistics, and knowing the scale of the variable is useful
        }
      } else {
        r->setConstant(true);
      }
  } else {
      if (testStat_ == "Profile") utils::setAllConstant(poi, false);
      else if (testStat_ == "LEP" || testStat_ == "TEV") utils::setAllConstant(poi, true);
      poi.assignValueOnly(rVals);
  }

  //set value of "globalConstrained" nuisances to the value of the corresponding global observable and set them constant
  //also fill the RooArgLists which can be passed to the test statistic  in order to produce the correct behaviour
  //this is only supported currently for the optimized version of the LHC-type test statistic
  RooArgList gobsParams;
  RooArgList gobs;
  if (withSystematics && testStat_ == "LHC" && optimizeTestStatistics_) {
    RooArgList allnuis(*mc_s->GetNuisanceParameters());
    RooArgList allgobs(*mc_s->GetGlobalObservables());
    for (int i=0; i<allnuis.getSize(); ++i) {
      RooRealVar *nuis = (RooRealVar*)allnuis.at(i);
      if (nuis->getAttribute("globalConstrained")) {
        RooRealVar *glob = (RooRealVar*)allgobs.find(TString::Format("%s_In",nuis->GetName()));
        if (glob) {
          nuis->setVal(glob->getVal());
          nuis->setConstant();
          gobsParams.add(*nuis);
          gobs.add(*glob);
        }
      }
    }
  }

  utils::CheapValueSnapshot fitMu, fitZero;
  std::auto_ptr<RooArgSet> paramsToFit;
  if (fitNuisances_ && mc_s->GetNuisanceParameters() && withSystematics) {
    TStopwatch timer;
    bool isExt = mc_s->GetPdf()->canBeExtended();
    utils::setAllConstant(poi, true);
    paramsToFit.reset(mc_s->GetPdf()->getParameters(data));
    RooStats::RemoveConstantParameters(&*paramsToFit);
    /// Background-only fit. In 1D case, can use model_s with POI set to zero, but in nD must use model_b.
    RooAbsPdf *pdfB = mc_b->GetPdf();
    if (poi.getSize() == 1) {
        r->setVal(0); pdfB = mc_s->GetPdf();
    }
    timer.Start();
    {
        CloseCoutSentry sentry(verbose < 3);
        pdfB->fitTo(data, RooFit::Minimizer("Minuit2","minimize"), RooFit::Strategy(1), RooFit::Hesse(0), RooFit::Extended(isExt), RooFit::Constrain(*mc_s->GetNuisanceParameters()));
        fitZero.readFrom(*paramsToFit);
    }
    if (verbose > 1) { std::cout << "Zero signal fit" << std::endl; fitZero.Print("V"); }
    if (verbose > 1) { std::cout << "Fitting of the background hypothesis done in " << timer.RealTime() << " s" << std::endl; }
    poi.assignValueOnly(rVals);
    timer.Start();
    {
       CloseCoutSentry sentry(verbose < 3);
       mc_s->GetPdf()->fitTo(data, RooFit::Minimizer("Minuit2","minimize"), RooFit::Strategy(1), RooFit::Hesse(0), RooFit::Extended(isExt), RooFit::Constrain(*mc_s->GetNuisanceParameters()));
       fitMu.readFrom(*paramsToFit);
    }
    if (verbose > 1) { std::cout << "Reference signal fit" << std::endl; fitMu.Print("V"); }
    if (verbose > 1) { std::cout << "Fitting of the signal-plus-background hypothesis done in " << timer.RealTime() << " s" << std::endl; }
  } else { fitNuisances_ = false; }

  // since ModelConfig cannot allow re-setting sets, we have to re-make everything 
  setup.modelConfig = ModelConfig("HybridNew_mc_s", w);
  setup.modelConfig.SetPdf(*mc_s->GetPdf());
  setup.modelConfig.SetObservables(*mc_s->GetObservables());
  setup.modelConfig.SetParametersOfInterest(*mc_s->GetParametersOfInterest());
  if (withSystematics) {
    if (genNuisances_ && mc_s->GetNuisanceParameters()) setup.modelConfig.SetNuisanceParameters(*mc_s->GetNuisanceParameters());
    if (genGlobalObs_ && mc_s->GetGlobalObservables())  setup.modelConfig.SetGlobalObservables(*mc_s->GetGlobalObservables());
    // if (genGlobalObs_ && mc_s->GetGlobalObservables())  snapGlobalObs_.reset(mc_s->GetGlobalObservables()->snapshot()); 
  }

  setup.modelConfig_bonly = ModelConfig("HybridNew_mc_b", w);
  setup.modelConfig_bonly.SetPdf(*mc_b->GetPdf());
  setup.modelConfig_bonly.SetObservables(*mc_b->GetObservables());
  setup.modelConfig_bonly.SetParametersOfInterest(*mc_b->GetParametersOfInterest());
  if (withSystematics) {
    if (genNuisances_ && mc_b->GetNuisanceParameters()) setup.modelConfig_bonly.SetNuisanceParameters(*mc_b->GetNuisanceParameters());
    if (genGlobalObs_ && mc_b->GetGlobalObservables())  setup.modelConfig_bonly.SetGlobalObservables(*mc_b->GetGlobalObservables());
  }

  if (withSystematics && !genNuisances_) {
      // The pdf will contain non-const parameters which are not observables
      // and the HybridCalculator will assume they're nuisances and try to generate them
      // to avoid this, we force him to generate a fake nuisance instead
      if (w->var("__HybridNew_fake_nuis__") == 0) { 
        w->factory("__HybridNew_fake_nuis__[0.5,0,1]");
        w->factory("Uniform::__HybridNew_fake_nuisPdf__(__HybridNew_fake_nuis__)");
      }
      setup.modelConfig.SetNuisanceParameters(RooArgSet(*w->var("__HybridNew_fake_nuis__")));
      setup.modelConfig_bonly.SetNuisanceParameters(RooArgSet(*w->var("__HybridNew_fake_nuis__")));
  }
 

  // create snapshots
  RooArgSet paramsZero; 
  if (poi.getSize() == 1) { // in the trivial 1D case, the background has POI=0.
    paramsZero.addClone(*rVals.first()); 
    paramsZero.setRealValue(rVals.first()->GetName(), 0);
    if (testStat_ == "LEP" || testStat_ == "TEV") { 
        ((RooRealVar&)paramsZero[rVals.first()->GetName()]).setConstant(true);
    }
  }
  if (fitNuisances_ && paramsToFit.get()) { params.add(*paramsToFit); fitMu.writeTo(params); }
  if (fitNuisances_ && paramsToFit.get()) { paramsZero.addClone(*paramsToFit); fitZero.writeTo(paramsZero); }
  setup.modelConfig.SetSnapshot(params);
  setup.modelConfig_bonly.SetSnapshot(paramsZero);
  TString paramsSnapName  = TString::Format("%s_%s_snapshot", setup.modelConfig.GetName(), params.GetName());
  TString paramsZSnapName = TString::Format("%s__snapshot",   setup.modelConfig_bonly.GetName());
  RooFit::MsgLevel level = RooMsgService::instance().globalKillBelow();
  RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
  w->defineSet(paramsSnapName,  params ,    true);
  w->defineSet(paramsZSnapName, paramsZero ,true);
  RooMsgService::instance().setGlobalKillBelow(level);

  // Create pdfs without nusiances terms, can be used for LEP tests statistics and
  // for generating toys when not generating global observables
  RooAbsPdf *factorizedPdf_s = setup.modelConfig.GetPdf(), *factorizedPdf_b = setup.modelConfig_bonly.GetPdf();
  if (withSystematics && optimizeProductPdf_ && !genGlobalObs_) {
        RooArgList constraints;
        RooAbsPdf *factorizedPdf_s = utils::factorizePdf(*mc_s->GetObservables(), *mc_s->GetPdf(), constraints);
        RooAbsPdf *factorizedPdf_b = utils::factorizePdf(*mc_b->GetObservables(), *mc_b->GetPdf(), constraints);
        if (factorizedPdf_s != mc_s->GetPdf()) setup.cleanupList.addOwned(*factorizedPdf_s);
        if (factorizedPdf_b != mc_b->GetPdf()) setup.cleanupList.addOwned(*factorizedPdf_b);
        setup.modelConfig.SetPdf(*factorizedPdf_s);
        setup.modelConfig_bonly.SetPdf(*factorizedPdf_b);
  }

  if (testStat_ == "LEP") {
      //SLR is evaluated using the central value of the nuisance parameters, so we have to put them in the parameter sets
      if (withSystematics) {
          if (!fitNuisances_) {
              params.add(*mc_s->GetNuisanceParameters(), true);
              paramsZero.addClone(*mc_b->GetNuisanceParameters(), true);
          } else {
              std::cerr << "ALERT: using LEP test statistics with --fitNuisances is not validated and most likely broken" << std::endl;
              params.assignValueOnly(*mc_s->GetNuisanceParameters());
              paramsZero.assignValueOnly(*mc_s->GetNuisanceParameters());
          }
      } 
      RooAbsPdf *pdfB = factorizedPdf_b; 
      if (poi.getSize() == 1) pdfB = factorizedPdf_s; // in this case we can remove the arbitrary constant from the test statistics.
      if (optimizeTestStatistics_) {
          if (!mc_s->GetPdf()->canBeExtended()) {
              setup.qvar.reset(new SimplerLikelihoodRatioTestStat(*pdfB,*factorizedPdf_s, paramsZero, params));
          } else {
              setup.qvar.reset(new SimplerLikelihoodRatioTestStatOpt(*mc_s->GetObservables(), *pdfB, *factorizedPdf_s, paramsZero, params, withSystematics));
          }
      } else {
          std::cerr << "ALERT: LEP test statistics without optimization not validated." << std::endl;
          RooArgSet paramsSnap; params.snapshot(paramsSnap); // needs a snapshot
          setup.qvar.reset(new SimpleLikelihoodRatioTestStat(*pdfB,*factorizedPdf_s));
          ((SimpleLikelihoodRatioTestStat&)*setup.qvar).SetNullParameters(paramsZero); // Null is B
          ((SimpleLikelihoodRatioTestStat&)*setup.qvar).SetAltParameters(paramsSnap);
      }
  } else if (testStat_ == "TEV") {
      std::cerr << "ALERT: Tevatron test statistics not yet validated." << std::endl;
      RooAbsPdf *pdfB = factorizedPdf_b; 
      if (poi.getSize() == 1) pdfB = factorizedPdf_s; // in this case we can remove the arbitrary constant from the test statistics.
      if (optimizeTestStatistics_) {
          setup.qvar.reset(new ProfiledLikelihoodRatioTestStatOpt(*mc_s->GetObservables(), *pdfB, *mc_s->GetPdf(), mc_s->GetNuisanceParameters(), paramsZero, params));
          ((ProfiledLikelihoodRatioTestStatOpt&)*setup.qvar).setPrintLevel(verbose);
      } else {   
          setup.qvar.reset(new RatioOfProfiledLikelihoodsTestStat(*mc_s->GetPdf(), *pdfB, setup.modelConfig.GetSnapshot()));
          ((RatioOfProfiledLikelihoodsTestStat&)*setup.qvar).SetSubtractMLE(false);
      }
  } else if (testStat_ == "LHC" || testStat_ == "LHCFC" || testStat_ == "Profile") {
      if (poi.getSize() != 1 && testStat_ != "Profile") {
        throw std::logic_error("ERROR: modified profile likelihood definitions (LHC, LHCFC) do not make sense in more than one dimension");
      }
      if (optimizeTestStatistics_) {
          ProfiledLikelihoodTestStatOpt::OneSidedness side = ProfiledLikelihoodTestStatOpt::oneSidedDef;
          if (testStat_ == "LHCFC")   side = ProfiledLikelihoodTestStatOpt::signFlipDef;
          if (testStat_ == "Profile") side = ProfiledLikelihoodTestStatOpt::twoSidedDef;
          if (workingMode_ == MakeSignificance) r->setVal(0.0);
          setup.qvar.reset(new ProfiledLikelihoodTestStatOpt(*mc_s->GetObservables(), *mc_s->GetPdf(), mc_s->GetNuisanceParameters(),  params, poi, gobsParams,gobs, verbose, side));
      } else {
          std::cerr << "ALERT: LHC test statistics without optimization not validated." << std::endl;
          setup.qvar.reset(new ProfileLikelihoodTestStat(*mc_s->GetPdf()));
          if (testStat_ == "LHC") {
              ((ProfileLikelihoodTestStat&)*setup.qvar).SetOneSided(true);
          } else if (testStat_ == "LHCFC") {
              throw std::invalid_argument("Test statistics LHCFC is not supported without optimization");
          }
      }
  } else if (testStat_ == "MLZ") {
      if (workingMode_ == MakeSignificance) r->setVal(0.0);
      setup.qvar.reset(new BestFitSigmaTestStat(*mc_s->GetObservables(), *mc_s->GetPdf(), mc_s->GetNuisanceParameters(),  params, verbose));
  }

  RooAbsPdf *nuisancePdf = 0;
  if (withSystematics && (genNuisances_ || (newToyMCSampler_ && genGlobalObs_))) {
    nuisancePdf = utils::makeNuisancePdf(*mc_s);
    if (nuisancePdf) setup.cleanupList.addOwned(*nuisancePdf);
  }
  if (newToyMCSampler_) { 
    setup.toymcsampler.reset(new ToyMCSamplerOpt(*setup.qvar, nToys_, nuisancePdf, genNuisances_));
  } else {
    std::cerr << "ALERT: running with newToyMC=0 not validated." << std::endl;
    setup.toymcsampler.reset(new ToyMCSampler(*setup.qvar, nToys_));
  }

  if (!mc_b->GetPdf()->canBeExtended()) setup.toymcsampler->SetNEventsPerToy(1);
  
  if (nCpu_ > 0) {
    std::cerr << "ALERT: running with proof not validated." << std::endl;
    if (verbose > 1) std::cout << "  Will use " << nCpu_ << " CPUs." << std::endl;
    setup.pc.reset(new ProofConfig(*w, nCpu_, "", kFALSE)); 
    setup.toymcsampler->SetProofConfig(setup.pc.get());
  }   


  std::auto_ptr<HybridCalculator> hc(new HybridCalculator(data, setup.modelConfig, setup.modelConfig_bonly, setup.toymcsampler.get()));
  if (genNuisances_ || !genGlobalObs_) {
      if (withSystematics) {
          setup.toymcsampler->SetGlobalObservables(*setup.modelConfig.GetNuisanceParameters());
          (static_cast<HybridCalculator&>(*hc)).ForcePriorNuisanceNull(*nuisancePdf);
          (static_cast<HybridCalculator&>(*hc)).ForcePriorNuisanceAlt(*nuisancePdf);
      }  
  } else if (genGlobalObs_ && !genNuisances_) {
      setup.toymcsampler->SetGlobalObservables(*setup.modelConfig.GetGlobalObservables());
      hc->ForcePriorNuisanceNull(*w->pdf("__HybridNew_fake_nuisPdf__"));
      hc->ForcePriorNuisanceAlt(*w->pdf("__HybridNew_fake_nuisPdf__"));
  }

  // we need less B toys than S toys
  if (workingMode_ == MakeSignificance) {
      // need only B toys. just keep a few S+B ones to avoid possible divide-by-zero errors somewhere
      hc->SetToys(nToys_, int(0.01*nToys_)+1);
      if (fullBToys_) {
        hc->SetToys(nToys_, nToys_);
      }      
  } else if (!CLs_) {

      if (adaptiveToys_>0.){
      	double qN = 2*(setup.toymcsampler->EvaluateTestStatistic(data,poi));
      	double prob = ROOT::Math::chisquared_cdf_c(qN,poi.getSize());

	std::vector<float>scaleAndConfidences;
  	std::vector<std::string> scaleAndConfidencesList;  
    	boost::split(scaleAndConfidencesList,scaleAndConfidenceSelection_ , boost::is_any_of(","));

  	for (UInt_t p = 0; p < scaleAndConfidencesList.size(); ++p) {
		
		scaleAndConfidences.push_back(atof(scaleAndConfidencesList[p].c_str()));
	}
	
    	int nCL_ = scaleAndConfidences.size();
        float scaleNumberOfToys = adaptiveToys_;
	int nToyssc = nToys_;

	if ((1.-prob) > maxProbability_) nToyssc=1.;
	else {
	    for (int CL_i=0;CL_i<nCL_;CL_i++){
		bool isClose = fabs(prob-(1-scaleAndConfidences[CL_i])) < confidenceToleranceForToyScaling_*(1-scaleAndConfidences[CL_i]); 
		if (isClose) scaleNumberOfToys = 1.;
	    }
	}

	nToyssc = (int) nToyssc*scaleNumberOfToys; nToyssc = nToyssc>0 ? nToyssc:1;

        hc->SetToys(fullBToys_ ? nToyssc : 1, nToyssc);
      }
      else {
        // we need only S+B toys to compute CLs+b
        hc->SetToys(fullBToys_ ? nToys_ : int(0.01*nToys_)+1, nToys_);
        //for two sigma bands need an equal number of B
        if (expectedFromGrid_ && (fabs(0.5-quantileForExpectedFromGrid_)>=0.4) ) {
          hc->SetToys(nToys_, nToys_);
        }      
      }	
    
  } else {
      // need both, but more S+B than B 
      hc->SetToys(fullBToys_ ? nToys_ : int(0.25*nToys_), nToys_);
      //for two sigma bands need an equal number of B
      if (expectedFromGrid_ && (fabs(0.5-quantileForExpectedFromGrid_)>=0.4) ) {
        hc->SetToys(nToys_, nToys_);
      }
  }

#if ROOT_VERSION_CODE <  ROOT_VERSION(5,34,00)
  static const char * istr = "__HybridNew__importanceSamplingDensity";
  if(importanceSamplingNull_) {
    std::cerr << "ALERT: running with importance sampling not validated (and probably not working)." << std::endl;
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
    std::cerr << "ALERT: running with importance sampling not validated (and probably not working)." << std::endl;
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
#endif

  return hc;
}

std::pair<double,double> 
HybridNew::eval(RooStats::HybridCalculator &hc, const RooAbsCollection & rVals, bool adaptive, double clsTarget) {
    std::auto_ptr<HypoTestResult> hcResult(evalGeneric(hc));
    if (expectedFromGrid_) applyExpectedQuantile(*hcResult);
    if (hcResult.get() == 0) {
        std::cerr << "Hypotest failed" << std::endl;
        return std::pair<double, double>(-1,-1);
    }
    if (testStat_ == "LHC" || testStat_ == "LHCFC" || testStat_ == "Profile") {
        // I need to flip the P-values
        hcResult->SetTestStatisticData(hcResult->GetTestStatisticData()-EPS); // issue with < vs <= in discrete models
    } else {
        hcResult->SetTestStatisticData(hcResult->GetTestStatisticData()+EPS); // issue with < vs <= in discrete models
        hcResult->SetPValueIsRightTail(!hcResult->GetPValueIsRightTail());
    }
    std::pair<double,double> cls = eval(*hcResult, rVals);
    if (verbose) std::cout << (CLs_ ? "\tCLs = " : "\tCLsplusb = ") << cls.first << " +/- " << cls.second << std::endl;
    if (adaptive) {
        if (CLs_) {
          hc.SetToys(int(0.25*nToys_ + 1), nToys_);
        }
        else {
          hc.SetToys(1, nToys_);
        }
        //for two sigma bands need an equal number of B
        if (expectedFromGrid_ && (fabs(0.5-quantileForExpectedFromGrid_)>=0.4) ) {
          hc.SetToys(nToys_, nToys_);
        }
        while (cls.second >= clsAccuracy_ && (clsTarget == -1 || fabs(cls.first-clsTarget) < 3*cls.second) ) {
            std::auto_ptr<HypoTestResult> more(evalGeneric(hc));
            more->SetBackgroundAsAlt(false);
            if (testStat_ == "LHC" || testStat_ == "LHCFC"  || testStat_ == "Profile") more->SetPValueIsRightTail(!more->GetPValueIsRightTail());
            hcResult->Append(more.get());
            if (expectedFromGrid_) applyExpectedQuantile(*hcResult);
            cls = eval(*hcResult, rVals);
            if (verbose) std::cout << (CLs_ ? "\tCLs = " : "\tCLsplusb = ") << cls.first << " +/- " << cls.second << std::endl;
        }
    } else if (iterations_ > 1) {
        for (unsigned int i = 1; i < iterations_; ++i) {
            std::auto_ptr<HypoTestResult> more(evalGeneric(hc));
            more->SetBackgroundAsAlt(false);
            if (testStat_ == "LHC" || testStat_ == "LHCFC"  || testStat_ == "Profile") more->SetPValueIsRightTail(!more->GetPValueIsRightTail());
            hcResult->Append(more.get());
            if (expectedFromGrid_) applyExpectedQuantile(*hcResult);
            cls = eval(*hcResult, rVals);
            if (verbose) std::cout << (CLs_ ? "\tCLs = " : "\tCLsplusb = ") << cls.first << " +/- " << cls.second << std::endl;
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
        TString name = TString::Format("HypoTestResult_mh%g",mass_);
        RooLinkedListIter it = rVals.iterator();
        for (RooRealVar *rIn = (RooRealVar*) it.Next(); rIn != 0; rIn = (RooRealVar*) it.Next()) {
            name += Form("_%s%g", rIn->GetName(), rIn->getVal());
        }
        name += Form("_%u", RooRandom::integer(std::numeric_limits<UInt_t>::max() - 1));
        writeToysHere->WriteTObject(new HypoTestResult(*hcResult), name);
        if (verbose) std::cout << "Hybrid result saved as " << name << " in " << writeToysHere->GetFile()->GetName() << " : " << writeToysHere->GetPath() << std::endl;
    }

    return cls;
} 

std::pair<double,double> HybridNew::eval(const RooStats::HypoTestResult &hcres, const RooAbsCollection & rVals) 
{
    double rVal = ((RooAbsReal*)rVals.first())->getVal();
    return eval(hcres,rVal);
}

std::pair<double,double> HybridNew::eval(const RooStats::HypoTestResult &hcres, double rVal) 
{
    if (testStat_ == "LHCFC") {
        RooStats::SamplingDistribution * bDistribution = hcres.GetNullDistribution(), * sDistribution = hcres.GetAltDistribution();
        const std::vector<Double_t> & bdist   = bDistribution->GetSamplingDistribution();
        const std::vector<Double_t> & bweight = bDistribution->GetSampleWeights();
        const std::vector<Double_t> & sdist   = sDistribution->GetSamplingDistribution();
        const std::vector<Double_t> & sweight = sDistribution->GetSampleWeights();
        Double_t data =  hcres.GetTestStatisticData();
        std::vector<Double_t> absbdist(bdist.size()), abssdist(sdist.size());
        std::vector<Double_t> absbweight(bweight), abssweight(sweight);
        Double_t absdata;
        if (rule_ == "FC") {
            for (int i = 0, n = absbdist.size(); i < n; ++i) absbdist[i] = fabs(bdist[i]);
            for (int i = 0, n = abssdist.size(); i < n; ++i) abssdist[i] = fabs(sdist[i]);
            absdata = fabs(data)-2*minimizerTolerance_; // needed here since zeros are not exact by more than tolerance
        } else {
            for (int i = 0, n = absbdist.size(); i < n; ++i) absbdist[i] = max(0., bdist[i]);
            for (int i = 0, n = abssdist.size(); i < n; ++i) abssdist[i] = max(0., sdist[i]);
            absdata = max(0., data) - EPS;
        }
        if (rVal == 0) { // S+B toys are equal to B ones!
            abssdist.reserve(absbdist.size() + abssdist.size());
            abssdist.insert(abssdist.end(), absbdist.begin(), absbdist.end());
            abssweight.reserve(absbweight.size() + abssweight.size());
            abssweight.insert(abssweight.end(), absbweight.begin(), absbweight.end());
        }
        RooStats::HypoTestResult result;
        RooStats::SamplingDistribution *abssDist = new RooStats::SamplingDistribution("s","s",abssdist,abssweight);
        RooStats::SamplingDistribution *absbDist = new RooStats::SamplingDistribution("b","b",absbdist,absbweight);
        result.SetNullDistribution(absbDist);
        result.SetAltDistribution(abssDist);
        result.SetTestStatisticData(absdata);
        result.SetPValueIsRightTail(!result.GetPValueIsRightTail());
        if (CLs_) {
            return std::pair<double,double>(result.CLs(), result.CLsError());
        } else {
            return std::pair<double,double>(result.CLsplusb(), result.CLsplusbError());
        }
    } else {
        if (CLs_) {
            return std::pair<double,double>(hcres.CLs(), hcres.CLsError());
        } else {
            return std::pair<double,double>(hcres.CLsplusb(), hcres.CLsplusbError());
        }
    }
}

void HybridNew::applyExpectedQuantile(RooStats::HypoTestResult &hcres) {
  if (expectedFromGrid_) {
      if (workingMode_ == MakeSignificance || workingMode_ == MakeSignificanceTestStatistics) {
          applySignalQuantile(hcres); 
      } else if (clsQuantiles_) {
          applyClsQuantile(hcres);
      } else {
          std::vector<Double_t> btoys = hcres.GetNullDistribution()->GetSamplingDistribution();
          std::sort(btoys.begin(), btoys.end());
          Double_t testStat = btoys[std::min<int>(floor((1.-quantileForExpectedFromGrid_) * btoys.size()+0.5), btoys.size())];
          if (verbose > 0) std::cout << "Text statistics for " << quantileForExpectedFromGrid_ << " quantile: " << testStat << std::endl;
          hcres.SetTestStatisticData(testStat);
          //std::cout << "CLs quantile = " << (CLs_ ? hcres.CLs() : hcres.CLsplusb()) << " for test stat = " << testStat << std::endl;
      }
  }
}
void HybridNew::applyClsQuantile(RooStats::HypoTestResult &hcres) {
    RooStats::SamplingDistribution * bDistribution = hcres.GetNullDistribution(), * sDistribution = hcres.GetAltDistribution();
    const std::vector<Double_t> & bdist   = bDistribution->GetSamplingDistribution();
    const std::vector<Double_t> & bweight = bDistribution->GetSampleWeights();
    const std::vector<Double_t> & sdist   = sDistribution->GetSamplingDistribution();
    const std::vector<Double_t> & sweight = sDistribution->GetSampleWeights();
    TStopwatch timer;

    /** New test implementation, scales as N*log(N) */
    timer.Start();
    std::vector<std::pair<double,double> > bcumul; bcumul.reserve(bdist.size()); 
    std::vector<std::pair<double,double> > scumul; scumul.reserve(sdist.size());
    double btot = 0, stot = 0;
    for (std::vector<Double_t>::const_iterator it = bdist.begin(), ed = bdist.end(), itw = bweight.begin(); it != ed; ++it, ++itw) {
        bcumul.push_back(std::pair<double,double>(*it, *itw));
        btot += *itw;
    }
    for (std::vector<Double_t>::const_iterator it = sdist.begin(), ed = sdist.end(), itw = sweight.begin(); it != ed; ++it, ++itw) {
        scumul.push_back(std::pair<double,double>(*it, *itw));
        stot += *itw;
    }
    double sinv = 1.0/stot, binv = 1.0/btot, runningSum;
    // now compute integral distribution of Q(s+b data) so that we can quickly compute the CL_{s+b} for all test stats.
    std::sort(scumul.begin(), scumul.end());
    runningSum = 0;
    for (std::vector<std::pair<double,double> >::reverse_iterator it = scumul.rbegin(), ed = scumul.rend(); it != ed; ++it) {
        runningSum += it->second; 
        it->second = runningSum * sinv;
    }
    std::sort(bcumul.begin(), bcumul.end());
    std::vector<std::pair<double,std::pair<double,double> > > xcumul; xcumul.reserve(bdist.size());
    runningSum = 0;
    std::vector<std::pair<double,double> >::const_iterator sbegin = scumul.begin(), send = scumul.end();
    //int k = 0;
    for (std::vector<std::pair<double,double> >::const_reverse_iterator it = bcumul.rbegin(), ed = bcumul.rend(); it != ed; ++it) {
        runningSum += it->second; 
        std::vector<std::pair<double,double> >::const_iterator match = std::upper_bound(sbegin, send, std::pair<double,double>(it->first, 0));
        if (match == send) {
            //std::cout << "Did not find match, for it->first == " << it->first << ", as back = ( " << scumul.back().first << " , " << scumul.back().second << " ) " << std::endl;
            double clsb = (scumul.back().second > 0.5 ? 1.0 : 0.0), clb = runningSum*binv, cls = clsb / clb;
            xcumul.push_back(std::make_pair(CLs_ ? cls : clsb, *it));
        } else {
            double clsb = match->second, clb = runningSum*binv, cls = clsb / clb;
            //if ((++k) % 100 == 0) printf("At %+8.5f  CLb = %6.4f, CLsplusb = %6.4f, CLs =%7.4f\n", it->first, clb, clsb, cls);
            xcumul.push_back(std::make_pair(CLs_ ? cls : clsb, *it));
        }
    }
    // sort 
    std::sort(xcumul.begin(), xcumul.end()); 
    // get quantile
    runningSum = 0; double cut = quantileForExpectedFromGrid_ * btot;
    for (std::vector<std::pair<double,std::pair<double,double> > >::const_iterator it = xcumul.begin(), ed = xcumul.end(); it != ed; ++it) {
        runningSum += it->second.second; 
        if (runningSum >= cut) {
            hcres.SetTestStatisticData(it->second.first);
            //std::cout << "CLs quantile = " << it->first << " for test stat = " << it->second.first << std::endl;
            break;
        }
    }
    //std::cout << "CLs quantile = " << (CLs_ ? hcres.CLs() : hcres.CLsplusb()) << std::endl;
    //std::cout << "Computed quantiles in " << timer.RealTime() << " s" << std::endl; 
#if 0
    /** Implementation in RooStats 5.30: scales as N^2, inefficient */
    timer.Start();
    std::vector<std::pair<double, double> > values(bdist.size()); 
    for (int i = 0, n = bdist.size(); i < n; ++i) { 
        hcres.SetTestStatisticData( bdist[i] );
        values[i] = std::pair<double, double>(CLs_ ? hcres.CLs() : hcres.CLsplusb(), bdist[i]);
    }
    std::sort(values.begin(), values.end());
    int index = std::min<int>(floor((1.-quantileForExpectedFromGrid_) * values.size()+0.5), values.size());
    std::cout << "CLs quantile = " << values[index].first << " for test stat = " << values[index].second << std::endl;
    hcres.SetTestStatisticData(values[index].second);
    std::cout << "CLs quantile = " << (CLs_ ? hcres.CLs() : hcres.CLsplusb()) << " for test stat = " << values[index].second << std::endl;
    std::cout << "Computed quantiles in " << timer.RealTime() << " s" << std::endl; 
#endif
}

void HybridNew::applySignalQuantile(RooStats::HypoTestResult &hcres) {
    std::vector<Double_t> stoys = hcres.GetAltDistribution()->GetSamplingDistribution();
    std::sort(stoys.begin(), stoys.end());
    Double_t testStat = stoys[std::min<int>(floor(quantileForExpectedFromGrid_ * stoys.size()+0.5), stoys.size())];
    if (verbose > 0) std::cout << "Text statistics for " << quantileForExpectedFromGrid_ << " quantile: " << testStat << std::endl;
    hcres.SetTestStatisticData(testStat);
}

RooStats::HypoTestResult * HybridNew::evalGeneric(RooStats::HybridCalculator &hc, bool noFork) {
    if (fork_ && !noFork) return evalWithFork(hc);
    else {
        TStopwatch timer; timer.Start();
        RooStats::HypoTestResult * ret = hc.GetHypoTest();
        if (runtimedef::get("HybridNew_Timing")) std::cout << "Evaluated toys in " << timer.RealTime() << " s " <<  std::endl;
        return ret;
    }
}

RooStats::HypoTestResult * HybridNew::evalWithFork(RooStats::HybridCalculator &hc) {
    TStopwatch timer;
    std::auto_ptr<RooStats::HypoTestResult> result(0);
    char tmpfile[999]; snprintf(tmpfile, 998, "%s/rstats-XXXXXX", P_tmpdir);
    int fd = mkstemp(tmpfile); close(fd);
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
        RooStats::HypoTestResult *hcResult = evalGeneric(hc, /*noFork=*/true);
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
    if (verbose > 1) { std::cout << "      Evaluation of p-values done in  " << timer.RealTime() << " s" << std::endl; }
    return result.release();
}

#if 0
/// Another implementation of frequentist toy tossing without RooStats.
/// Can use as a cross-check if needed

RooStats::HypoTestResult * HybridNew::evalFrequentist(RooStats::HybridCalculator &hc) {
    int toysSB = (workingMode_ == MakeSignificance ? 1 : nToys_);
    int toysB  = (workingMode_ == MakeSignificance ? nToys_ : (CLs_ ? nToys_/4+1 : 1));
    RooArgSet obs(*hc.GetAlternateModel()->GetObservables());
    RooArgSet nuis(*hc.GetAlternateModel()->GetNuisanceParameters());
    RooArgSet gobs(*hc.GetAlternateModel()->GetGlobalObservables());
    RooArgSet nullPoi(*hc.GetNullModel()->GetSnapshot());
    std::auto_ptr<RooAbsCollection> parS(hc.GetAlternateModel()->GetPdf()->getParameters(obs));
    std::auto_ptr<RooAbsCollection> parB(hc.GetNullModel()->GetPdf()->getParameters(obs));
    RooArgList constraintsS, constraintsB;
    RooAbsPdf *factorS = hc.GetAlternateModel()->GetPdf();
    RooAbsPdf *factorB = hc.GetNullModel()->GetPdf();
    //std::auto_ptr<RooAbsPdf> factorS(utils::factorizePdf(obs, *hc.GetAlternateModel()->GetPdf(),  constraintsS));
    //std::auto_ptr<RooAbsPdf> factorB(utils::factorizePdf(obs, *hc.GetNullModel()->GetPdf(), constraintsB));
    std::auto_ptr<RooAbsPdf> nuisPdf(utils::makeNuisancePdf(const_cast<RooStats::ModelConfig&>(*hc.GetAlternateModel())));
    std::vector<Double_t> distSB, distB;
    *parS = *snapGlobalObs_;
    Double_t tsData = hc.GetTestStatSampler()->GetTestStatistic()->Evaluate(*realData_, nullPoi);
    if (verbose > 2) std::cout << "Test statistics on data: " << tsData << std::endl;
    for (int i = 0; i < toysSB; ++i) {
       // Initialize parameters to snapshot
       *parS = *hc.GetAlternateModel()->GetSnapshot(); 
       // Throw global observables, and set them
       if (verbose > 2) { std::cout << "Generating global obs starting from " << std::endl; parS->Print("V"); }
       std::auto_ptr<RooDataSet> gdata(nuisPdf->generate(gobs, 1));
       *parS = *gdata->get(0);
       if (verbose > 2) { std::cout << "Generated global obs" << std::endl; utils::printRAD(&*gdata); }
       // Throw observables
       if (verbose > 2) { std::cout << "Generating obs starting from " << std::endl; parS->Print("V"); }
       std::auto_ptr<RooDataSet> data(factorS->generate(obs, RooFit::Extended()));
       if (verbose > 2) { std::cout << "Generated obs" << std::endl; utils::printRAD(&*data); }
       // Evaluate T.S.
       distSB.push_back(hc.GetTestStatSampler()->GetTestStatistic()->Evaluate(*data, nullPoi));
       //if std::cout << "Test statistics on S+B : " << distSB.back() << std::endl;
    }
    for (int i = 0; i < toysB; ++i) {
       // Initialize parameters to snapshot
       *parB = *hc.GetNullModel()->GetSnapshot();
       //*parB = *hc.GetAlternateModel()->GetSnapshot(); 
       // Throw global observables, and set them
       if (verbose > 2) { std::cout << "Generating global obs starting from " << std::endl; parB->Print("V"); }
       std::auto_ptr<RooDataSet> gdata(nuisPdf->generate(gobs, 1));
       *parB = *gdata->get(0);
       if (verbose > 2) { std::cout << "Generated global obs" << std::endl; utils::printRAD(&*gdata); }
       // Throw observables
       if (verbose > 2) { std::cout << "Generating obs starting from " << std::endl; parB->Print("V"); }
       std::auto_ptr<RooDataSet> data(factorB->generate(obs, RooFit::Extended()));
       if (verbose > 2) { std::cout << "Generated obs" << std::endl; utils::printRAD(&*data); }
       // Evaluate T.S.
       distB.push_back(hc.GetTestStatSampler()->GetTestStatistic()->Evaluate(*data, nullPoi));
       //std::cout << "Test statistics on B   : " << distB.back() << std::endl;
    }
    // Load reference global observables
    RooStats::HypoTestResult *ret = new RooStats::HypoTestResult();
    ret->SetTestStatisticData(tsData);
    ret->SetAltDistribution(new RooStats::SamplingDistribution("sb","sb",distSB));
    ret->SetNullDistribution(new RooStats::SamplingDistribution("b","b",distB));
    return ret;
}
#endif

RooStats::HypoTestResult * HybridNew::readToysFromFile(const RooAbsCollection & rVals) {
    if (!readToysFromHere) throw std::logic_error("Cannot use readHypoTestResult: option toysFile not specified, or input file empty");
    TDirectory *toyDir = readToysFromHere->GetDirectory("toys");
    if (!toyDir) throw std::logic_error("Cannot use readHypoTestResult: empty toy dir in input file empty");
    if (verbose) std::cout << "Reading toys for ";
    TString prefix1 = TString::Format("HypoTestResult_mh%g",mass_);
    TString prefix2 = TString::Format("HypoTestResult");
    RooLinkedListIter it = rVals.iterator();
    for (RooRealVar *rIn = (RooRealVar*) it.Next(); rIn != 0; rIn = (RooRealVar*) it.Next()) {
        if (verbose) std::cout << rIn->GetName() << " = " << rIn->getVal() << "   ";
        prefix1 += Form("_%s%g", rIn->GetName(), rIn->getVal());
        prefix2 += Form("_%s%g", rIn->GetName(), rIn->getVal());
    }
    if (verbose) std::cout << std::endl;
    std::auto_ptr<RooStats::HypoTestResult> ret;
    TIter next(toyDir->GetListOfKeys()); TKey *k;
    while ((k = (TKey *) next()) != 0) {
        if (TString(k->GetName()).Index(prefix1) != 0 && TString(k->GetName()).Index(prefix2) != 0) continue;
        RooStats::HypoTestResult *toy = dynamic_cast<RooStats::HypoTestResult *>(toyDir->Get(k->GetName()));
        if (toy == 0) continue;
        if (verbose > 1) std::cout << " - " << k->GetName() << std::endl;
        if (ret.get() == 0) {
            ret.reset(new RooStats::HypoTestResult(*toy));
        } else {
            ret->Append(toy);
        }
    }

    if (ret.get() == 0) {
        std::cout << "ERROR: parameter point not found in input root file.\n";
        rVals.Print("V");
        if (verbose > 0) toyDir->ls();
        std::cout << "ERROR: parameter point not found in input root file" << std::endl;
        throw std::invalid_argument("Missing input");
    }
    if (verbose > 0) {
        std::cout <<
            "\tCLs      = " << ret->CLs()      << " +/- " << ret->CLsError()      << "\n" <<
            "\tCLb      = " << ret->CLb()      << " +/- " << ret->CLbError()      << "\n" <<
            "\tCLsplusb = " << ret->CLsplusb() << " +/- " << ret->CLsplusbError() << "\n" <<
            std::endl;
        if (!plot_.empty() && workingMode_ != MakeLimit) {
            HypoTestPlot plot(*ret, 30);
            TCanvas *c1 = new TCanvas("c1","c1");
            plot.Draw();
            c1->Print(plot_.c_str());
            delete c1;
        }
    }
    return ret.release();
}

void HybridNew::readGrid(TDirectory *toyDir, double rMin, double rMax) {
    if (rValues_.getSize() != 1) throw std::runtime_error("Running limits with grid only works in one dimension for the moment");
    clearGrid();

    TIter next(toyDir->GetListOfKeys()); TKey *k; const char *poiName = rValues_.first()->GetName();
    while ((k = (TKey *) next()) != 0) {
        TString name(k->GetName());
        if (name.Index("HypoTestResult_mh") == 0) {
            if (name.Index(TString::Format("HypoTestResult_mh%g_%s",mass_,poiName)) != 0 || name.Index("_", name.Index("_")+1) == -1) continue;
            name.ReplaceAll(TString::Format("HypoTestResult_mh%g_%s",mass_,poiName),"");  // remove the prefix
            if (name.Index("_") == -1) continue;                                          // check if it has the _<uniqueId> postfix
            name.Remove(name.Index("_"),name.Length());                                   // remove it before calling atof
        } else if (name.Index("HypoTestResult_") == 0) {
            // let's put a warning here, since results of this form were supported in the past
            std::cout << "HybridNew::readGrid: HypoTestResult with non-conformant name " << name << " will be skipped" << std::endl;
            continue;
        } else continue;
        double rVal = atof(name.Data());
        if (rVal < rMin || rVal > rMax) continue;
        if (verbose > 2) std::cout << "  Do " << k->GetName() << " -> " << name << " --> " << rVal << std::endl;
        RooStats::HypoTestResult *toy = dynamic_cast<RooStats::HypoTestResult *>(toyDir->Get(k->GetName()));
        RooStats::HypoTestResult *&merge = grid_[rVal];
        if (merge == 0) merge = new RooStats::HypoTestResult(*toy);
        else merge->Append(toy);
        merge->ResetBit(1);
    }
    if (verbose > 1) {
        std::cout << "GRID, as is." << std::endl;
        typedef std::map<double, RooStats::HypoTestResult *>::iterator point;
        for (point it = grid_.begin(), ed = grid_.end(); it != ed; ++it) {
            std::cout << "  - " << it->first << "  (CLs = " << it->second->CLs() << " +/- " << it->second->CLsError() << ")" << std::endl;
        }
    }
}
void HybridNew::updateGridData(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, bool smart, double clsTarget_) {
    typedef std::map<double, RooStats::HypoTestResult *>::iterator point;
    if (!smart) {
        for (point it = grid_.begin(), ed = grid_.end(); it != ed; ++it) {
            it->second->ResetBit(1);
            updateGridPoint(w, mc_s, mc_b, data, it);
        }
    } else {
        typedef std::pair<double,double> CLs_t;
        std::vector<point> points; points.reserve(grid_.size()); 
        std::vector<CLs_t> values; values.reserve(grid_.size());
        for (point it = grid_.begin(), ed = grid_.end(); it != ed; ++it) { points.push_back(it); values.push_back(CLs_t(-99, -99)); }
        int iMin = 0, iMax = points.size()-1;
        while (iMax-iMin > 3) {
            if (verbose > 1) std::cout << "Bisecting range [" << iMin << ", " << iMax << "]" << std::endl; 
            int iMid = (iMin+iMax)/2;
            CLs_t clsMid = values[iMid] = updateGridPoint(w, mc_s, mc_b, data, points[iMid]);
            if (verbose > 1) std::cout << "    Midpoint " << iMid << " value " << clsMid.first << " +/- " << clsMid.second << std::endl; 
            if (clsMid.first - 3*max(clsMid.second,0.01) > clsTarget_) { 
                if (verbose > 1) std::cout << "    Replacing Min" << std::endl; 
                iMin = iMid; continue;
            } else if (clsMid.first + 3*max(clsMid.second,0.01) < clsTarget_) {
                if (verbose > 1) std::cout << "    Replacing Max" << std::endl; 
                iMax = iMid; continue;
            } else {
                if (verbose > 1) std::cout << "    Tightening Range" << std::endl; 
                while (iMin < iMid-1) {
                    int iLo = (iMin+iMid)/2;
                    CLs_t clsLo = values[iLo] = updateGridPoint(w, mc_s, mc_b, data, points[iLo]);
                    if (verbose > 1) std::cout << "        Lowpoint " << iLo << " value " << clsLo.first << " +/- " << clsLo.second << std::endl; 
                    if (clsLo.first - 3*max(clsLo.second,0.01) > clsTarget_) iMin = iLo; 
                    else break;
                }
                while (iMax > iMid+1) {
                    int iHi = (iMax+iMid)/2;
                    CLs_t clsHi = values[iHi] = updateGridPoint(w, mc_s, mc_b, data, points[iHi]);
                    if (verbose > 1) std::cout << "        Highpoint " << iHi << " value " << clsHi.first << " +/- " << clsHi.second << std::endl; 
                    if (clsHi.first + 3*max(clsHi.second,0.01) < clsTarget_) iMax = iHi; 
                    else break;
                }
                break;
            }
        }
        if (verbose > 1) std::cout << "Final range [" << iMin << ", " << iMax << "]" << std::endl; 
        for (int i = 0; i < iMin; ++i) {
            points[i]->second->SetBit(1);
            if (verbose > 1) std::cout << "  Will not use point " << i << " (r " << points[i]->first << ")" << std::endl;
        }
        for (int i = iMin; i <= iMax; ++i) {
            points[i]->second->ResetBit(1);
            if (values[i].first < -2) {
                if (verbose > 1) std::cout << "   Updaing point " << i << " (r " << points[i]->first << ")" << std::endl; 
                updateGridPoint(w, mc_s, mc_b, data, points[i]);
            }
            else if (verbose > 1) std::cout << "   Point " << i << " (r " << points[i]->first << ") was already updated during search." << std::endl; 
        }
        for (int i = iMax+1, n = points.size(); i < n; ++i) {
            points[i]->second->SetBit(1);
            if (verbose > 1) std::cout << "  Will not use point " << i << " (r " << points[i]->first << ")" << std::endl;
        }
    }
}
void HybridNew::updateGridDataFC(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, bool smart, double clsTarget_) {
    typedef std::map<double, RooStats::HypoTestResult *>::iterator point;
    std::vector<Double_t> rToUpdate; std::vector<point> pointToUpdate;
    for (point it = grid_.begin(), ed = grid_.end(); it != ed; ++it) {
        it->second->ResetBit(1);
        if (it->first == 0 || fabs(it->second->GetTestStatisticData()) <= 2*minimizerTolerance_) {
            rToUpdate.push_back(it->first);
            pointToUpdate.push_back(it);
        }
    }
    if (verbose > 0) std::cout << "A total of " << rToUpdate.size() << " points will be updated." << std::endl;
    Setup setup;
    std::auto_ptr<RooStats::HybridCalculator> hc = create(w, mc_s, mc_b, data, rToUpdate.back(), setup);
    RooArgSet nullPOI(*setup.modelConfig_bonly.GetSnapshot());
    std::vector<Double_t> qVals = ((ProfiledLikelihoodTestStatOpt&)(*setup.qvar)).Evaluate(data, nullPOI, rToUpdate);
    for (int i = 0, n = rToUpdate.size(); i < n; ++i) {
        pointToUpdate[i]->second->SetTestStatisticData(qVals[i] - EPS);
    }
    if (verbose > 0) std::cout << "All points have been updated." << std::endl;
}

std::pair<double,double> HybridNew::updateGridPoint(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, std::map<double, RooStats::HypoTestResult *>::iterator point) {
    typedef std::pair<double,double> CLs_t;
    bool isProfile = (testStat_ == "LHC" || testStat_ == "LHCFC"  || testStat_ == "Profile");
    if (point->first == 0 && CLs_) return std::pair<double,double>(1,0);
    RooArgSet  poi(*mc_s->GetParametersOfInterest());
    RooRealVar *r = dynamic_cast<RooRealVar *>(poi.first());
    if (expectedFromGrid_) {
        applyExpectedQuantile(*point->second);
        point->second->SetTestStatisticData(point->second->GetTestStatisticData() + (isProfile ? EPS : EPS));
    } else {
        Setup setup;
        std::auto_ptr<RooStats::HybridCalculator> hc = create(w, mc_s, mc_b, data, point->first, setup);
        RooArgSet nullPOI(*setup.modelConfig_bonly.GetSnapshot());
        if (isProfile) nullPOI.setRealValue(r->GetName(), point->first);
        double testStat = setup.qvar->Evaluate(data, nullPOI);
        point->second->SetTestStatisticData(testStat + (isProfile ? EPS : EPS));
    }
    if (verbose > 1) {
        std::cout << "At " << r->GetName() << " = " << point->first << ":\n" << 
            "\tCLs      = " << point->second->CLs()      << " +/- " << point->second->CLsError()      << "\n" <<
            "\tCLb      = " << point->second->CLb()      << " +/- " << point->second->CLbError()      << "\n" <<
            "\tCLsplusb = " << point->second->CLsplusb() << " +/- " << point->second->CLsplusbError() << "\n" <<
            std::endl;
    }
    
    return eval(*point->second, point->first);
}
void HybridNew::useGrid() {
    typedef std::pair<double,double> CLs_t;
    int i = 0, n = 0;
    limitPlot_.reset(new TGraphErrors(1));
    for (std::map<double, RooStats::HypoTestResult *>::iterator itg = grid_.begin(), edg = grid_.end(); itg != edg; ++itg, ++i) {
        if (itg->second->TestBit(1)) continue;
        CLs_t val(1,0);
        if (CLs_) {
            if (itg->first > 0) val = eval(*itg->second, itg->first);
        } else {
            val = eval(*itg->second, itg->first);
        }
        if (val.first == -1) continue;
        if (val.second == 0 && (val.first != 1 && val.first != 0)) continue;
        limitPlot_->Set(n+1);
        limitPlot_->SetPoint(     n, itg->first, val.first); 
        limitPlot_->SetPointError(n, 0,          val.second);
        n++;
    }
}
void HybridNew::clearGrid() {
    for (std::map<double, RooStats::HypoTestResult *>::iterator it = grid_.begin(), ed = grid_.end(); it != ed; ++it) {
        delete it->second;
    }
    grid_.clear();
}

