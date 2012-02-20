#include "../interface/CascadeMinimizer.h"
#include "../interface/ProfiledLikelihoodRatioTestStatExt.h"
#include "../interface/ProfileLikelihood.h"
#include "../interface/CloseCoutSentry.h"
#include "../interface/utils.h"

#include "RooMinimizer.h"
#include <Math/MinimizerOptions.h>

boost::program_options::options_description CascadeMinimizer::options_("Cascade Minimizer options");
std::vector<CascadeMinimizer::Algo> CascadeMinimizer::fallbacks_;
bool CascadeMinimizer::poiOnlyFit_;
bool CascadeMinimizer::singleNuisFit_;

CascadeMinimizer::CascadeMinimizer(RooAbsReal &nll, Mode mode, RooRealVar *poi, int initialStrategy) :
    nll_(nll),
    minimizer_(nll_),
    mode_(mode),
    strategy_(initialStrategy),
    poi_(poi)
{
}

bool CascadeMinimizer::minimize(int verbose) 
{
    minimizer_.setPrintLevel(verbose-2);  
    minimizer_.setStrategy(strategy_);
    bool nominal = nllutils::robustMinimize(nll_, minimizer_, verbose);
    if (!nominal) {
        CloseCoutSentry::breakFree();
        std::string nominalType(ROOT::Math::MinimizerOptions::DefaultMinimizerType());
        std::string nominalAlgo(ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo());
        float       nominalTol(ROOT::Math::MinimizerOptions::DefaultTolerance());
        if (verbose > -2) std::cerr << "Failed minimization with " << nominalType << "," << nominalAlgo << " and tolerance " << nominalTol << std::endl;
        for (std::vector<Algo>::const_iterator it = fallbacks_.begin(), ed = fallbacks_.end(); it != ed; ++it) {
            ProfileLikelihood::MinimizerSentry minimizerConfig(it->algo, it->tolerance != -1.f ? it->tolerance : nominalTol);
            if (nominalType != ROOT::Math::MinimizerOptions::DefaultMinimizerType() ||
                nominalAlgo != ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo() ||
                nominalTol  != ROOT::Math::MinimizerOptions::DefaultTolerance()) {
                if (verbose > -2) std::cerr << "Will fallback to minimization using " << it->algo << " and tolerance " << it->tolerance << std::endl;
                bool thisTry = nllutils::robustMinimize(nll_, minimizer_, verbose);
                if (thisTry) return true;
            }
        }
    }
    return nominal;
}

void CascadeMinimizer::initOptions() 
{
    options_.add_options()
        ("cminPoiOnlyFit",  "Do first a fit floating only the parameter of interest")
        ("cminSingleNuisFit", "Do first a minimization of each nuisance parameter individually")
        ("cminFallbackAlgo", boost::program_options::value<std::vector<std::string> >(), "Fallback algorithms if the default minimizer fails (can use multiple ones). Syntax is algo[,subalgo][;tolerance]")
        ;
}

void CascadeMinimizer::applyOptions(const boost::program_options::variables_map &vm) 
{
    using namespace std;

    poiOnlyFit_ = vm.count("cminPoiOnlyFit");
    singleNuisFit_ = vm.count("cminSingleNuisFit");
    if (vm.count("cminFallbackAlgo")) {
        vector<string> falls(vm["cminFallbackAlgo"].as<vector<string> >());
        for (vector<string>::const_iterator it = falls.begin(), ed = falls.end(); it != ed; ++it) {
            const string & str = *it;
            string::size_type idx = str.find(";");
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
