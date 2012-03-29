#include "../interface/CascadeMinimizer.h"
#include "../interface/ProfiledLikelihoodRatioTestStatExt.h"
#include "../interface/ProfileLikelihood.h"
#include "../interface/CloseCoutSentry.h"
#include "../interface/utils.h"

#include <Math/MinimizerOptions.h>

boost::program_options::options_description CascadeMinimizer::options_("Cascade Minimizer options");
std::vector<CascadeMinimizer::Algo> CascadeMinimizer::fallbacks_;
bool CascadeMinimizer::preScan_;
bool CascadeMinimizer::poiOnlyFit_;
bool CascadeMinimizer::singleNuisFit_;
bool CascadeMinimizer::setZeroPoint_;
bool CascadeMinimizer::oldFallback_ = true;

CascadeMinimizer::CascadeMinimizer(RooAbsReal &nll, Mode mode, RooRealVar *poi, int initialStrategy) :
    nll_(nll),
    minimizer_(nll_),
    mode_(mode),
    strategy_(initialStrategy),
    poi_(poi)
{
}

bool CascadeMinimizer::improve(int verbose, bool cascade) 
{
    if (setZeroPoint_) {
        cacheutils::CachingSimNLL *simnll = dynamic_cast<cacheutils::CachingSimNLL *>(&nll_);
        if (simnll) { 
            simnll->setZeroPoint();
        }
    }
    minimizer_.setPrintLevel(verbose-2);  
    minimizer_.setStrategy(strategy_);
    bool outcome = improveOnce(verbose-2);
    if (cascade && !outcome && !fallbacks_.empty()) {
        std::string nominalType(ROOT::Math::MinimizerOptions::DefaultMinimizerType());
        std::string nominalAlgo(ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo());
        float       nominalTol(ROOT::Math::MinimizerOptions::DefaultTolerance());
        if (verbose > 0) std::cerr << "Failed minimization with " << nominalType << "," << nominalAlgo << " and tolerance " << nominalTol << std::endl;
        for (std::vector<Algo>::const_iterator it = fallbacks_.begin(), ed = fallbacks_.end(); it != ed; ++it) {
            ProfileLikelihood::MinimizerSentry minimizerConfig(it->algo, it->tolerance != -1.f ? it->tolerance : nominalTol);
            if (nominalType != ROOT::Math::MinimizerOptions::DefaultMinimizerType() ||
                nominalAlgo != ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo() ||
                nominalTol  != ROOT::Math::MinimizerOptions::DefaultTolerance()) {
                if (verbose > 0) std::cerr << "Will fallback to minimization using " << it->algo << " and tolerance " << it->tolerance << std::endl;
                outcome = improveOnce(verbose-2);
                if (outcome) break;
            }
        }
    }
    if (setZeroPoint_) {
        cacheutils::CachingSimNLL *simnll = dynamic_cast<cacheutils::CachingSimNLL *>(&nll_);
        if (simnll) simnll->clearZeroPoint();
    }
    return outcome;
}

bool CascadeMinimizer::improveOnce(int verbose) 
{
    std::string myType(ROOT::Math::MinimizerOptions::DefaultMinimizerType());
    std::string myAlgo(ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo());
    bool outcome = false;
    if (oldFallback_){
        outcome = nllutils::robustMinimize(nll_, minimizer_, verbose);
    } else {
        int status = minimizer_.minimize(myType.c_str(), myAlgo.c_str());
        outcome = (status == 0);
    }
    return outcome;
}

bool CascadeMinimizer::minimize(int verbose, bool cascade) 
{
    minimizer_.setPrintLevel(verbose-2);  
    minimizer_.setStrategy(strategy_);
    if (preScan_) minimizer_.minimize("Minuit2","Scan");
    // // FIXME to be ported later
    //if (mode_ == Unconstrained && poiOnlyFit_) {
    //    OneDimMinimizer min1D(&nll_, poi_);
    //    min1D.minimize(100,ROOT::Math::MinimizerOptions::DefaultTolerance());
    //}
    return improve(verbose, cascade);
}


void CascadeMinimizer::initOptions() 
{
    options_.add_options()
        ("cminPoiOnlyFit",  "Do first a fit floating only the parameter of interest")
        ("cminPreScan",  "Do a scan before first minimization")
        ("cminSingleNuisFit", "Do first a minimization of each nuisance parameter individually")
        ("cminFallbackAlgo", boost::program_options::value<std::vector<std::string> >(), "Fallback algorithms if the default minimizer fails (can use multiple ones). Syntax is algo[,subalgo][:tolerance]")
        ("cminSetZeroPoint", "Change the reference point of the NLL to be zero during minimization")
        ("cminOldRobustMinimize", boost::program_options::value<bool>(&oldFallback_)->default_value(oldFallback_), "Use the old 'robustMinimize' logic in addition to the cascade")
        ;
}

void CascadeMinimizer::applyOptions(const boost::program_options::variables_map &vm) 
{
    using namespace std;

    preScan_ = vm.count("cminPreScan");
    poiOnlyFit_ = vm.count("cminPoiOnlyFit");
    singleNuisFit_ = vm.count("cminSingleNuisFit");
    setZeroPoint_  = vm.count("cminSetZeroPoint");
    if (vm.count("cminFallbackAlgo")) {
        vector<string> falls(vm["cminFallbackAlgo"].as<vector<string> >());
        for (vector<string>::const_iterator it = falls.begin(), ed = falls.end(); it != ed; ++it) {
            const string & str = *it;
            string::size_type idx = std::min(str.find(";"), str.find(":"));
            if (idx != string::npos && idx < str.length()) {
                 fallbacks_.push_back(Algo(str.substr(0,idx), atof(str.substr(idx+1).c_str())));       
                 std::cout << "Configured fallback algorithm " << fallbacks_.back().algo << ", tolerance " << fallbacks_.back().tolerance << std::endl;
            } else {
                 fallbacks_.push_back(Algo(str));
                 std::cout << "Configured fallback algorithm " << fallbacks_.back().algo << ", default tolerance"  << std::endl;
            }
        }
    }
}
