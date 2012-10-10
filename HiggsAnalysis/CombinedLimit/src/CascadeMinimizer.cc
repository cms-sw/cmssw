#include "../interface/CascadeMinimizer.h"
#include "../interface/ProfiledLikelihoodRatioTestStatExt.h"
#include "../interface/ProfileLikelihood.h"
#include "../interface/CloseCoutSentry.h"
#include "../interface/utils.h"

#include <Math/MinimizerOptions.h>
#include <RooCategory.h>
#include <RooNumIntConfig.h>


boost::program_options::options_description CascadeMinimizer::options_("Cascade Minimizer options");
std::vector<CascadeMinimizer::Algo> CascadeMinimizer::fallbacks_;
bool CascadeMinimizer::preScan_;
bool CascadeMinimizer::poiOnlyFit_;
bool CascadeMinimizer::singleNuisFit_;
bool CascadeMinimizer::setZeroPoint_ = true;
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
    minimizer_.setPrintLevel(verbose-2);  
    minimizer_.setStrategy(strategy_);
    bool outcome = improveOnce(verbose-2);
    if (cascade && !outcome && !fallbacks_.empty()) {
        std::string nominalType(ROOT::Math::MinimizerOptions::DefaultMinimizerType());
        std::string nominalAlgo(ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo());
        float       nominalTol(ROOT::Math::MinimizerOptions::DefaultTolerance());
        int         nominalStrat(strategy_);
        if (verbose > 0) std::cerr << "Failed minimization with " << nominalType << "," << nominalAlgo << " and tolerance " << nominalTol << std::endl;
        for (std::vector<Algo>::const_iterator it = fallbacks_.begin(), ed = fallbacks_.end(); it != ed; ++it) {
            ProfileLikelihood::MinimizerSentry minimizerConfig(it->algo, it->tolerance != Algo::default_tolerance() ? it->tolerance : nominalTol);
            int myStrategy = it->strategy; if (myStrategy == Algo::default_strategy()) myStrategy = nominalStrat;
            if (nominalType != ROOT::Math::MinimizerOptions::DefaultMinimizerType() ||
                nominalAlgo != ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo() ||
                nominalTol  != ROOT::Math::MinimizerOptions::DefaultTolerance()     ||
                myStrategy  != nominalStrat) {
                if (verbose > 0) std::cerr << "Will fallback to minimization using " << it->algo << ", strategy " << myStrategy << " and tolerance " << it->tolerance << std::endl;
                minimizer_.setStrategy(myStrategy);
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
    if (setZeroPoint_) {
        cacheutils::CachingSimNLL *simnll = dynamic_cast<cacheutils::CachingSimNLL *>(&nll_);
        if (simnll) { 
            simnll->setZeroPoint();
        }
    }
    bool outcome = false;
    if (oldFallback_){
        outcome = nllutils::robustMinimize(nll_, minimizer_, verbose);
    } else {
        int status = minimizer_.minimize(myType.c_str(), myAlgo.c_str());
        outcome = (status == 0);
    }
    if (setZeroPoint_) {
        cacheutils::CachingSimNLL *simnll = dynamic_cast<cacheutils::CachingSimNLL *>(&nll_);
        if (simnll) simnll->clearZeroPoint();
    }
    return outcome;
}

bool CascadeMinimizer::minimize(int verbose, bool cascade) 
{
    minimizer_.setPrintLevel(verbose-2);  
    minimizer_.setStrategy(strategy_);
    if (preScan_) minimizer_.minimize("Minuit2","Scan");
     // FIXME can be made smarter than this
    if (mode_ == Unconstrained && poiOnlyFit_) {
        trivialMinimize(nll_, *poi_, 200);
    }
    return improve(verbose, cascade);
}


void CascadeMinimizer::initOptions() 
{
    options_.add_options()
        ("cminPoiOnlyFit",  "Do first a fit floating only the parameter of interest")
        ("cminPreScan",  "Do a scan before first minimization")
        ("cminSingleNuisFit", "Do first a minimization of each nuisance parameter individually")
        ("cminFallbackAlgo", boost::program_options::value<std::vector<std::string> >(), "Fallback algorithms if the default minimizer fails (can use multiple ones). Syntax is algo[,subalgo][,strategy][:tolerance]")
        ("cminSetZeroPoint", boost::program_options::value<bool>(&setZeroPoint_)->default_value(setZeroPoint_), "Change the reference point of the NLL to be zero during minimization")
        ("cminOldRobustMinimize", boost::program_options::value<bool>(&oldFallback_)->default_value(oldFallback_), "Use the old 'robustMinimize' logic in addition to the cascade")
        //("cminDefaultIntegratorEpsAbs", boost::program_options::value<double>(), "RooAbsReal::defaultIntegratorConfig()->setEpsAbs(x)")
        //("cminDefaultIntegratorEpsRel", boost::program_options::value<double>(), "RooAbsReal::defaultIntegratorConfig()->setEpsRel(x)")
        //("cminDefaultIntegrator1D", boost::program_options::value<std::string>(), "RooAbsReal::defaultIntegratorConfig()->method1D().setLabel(x)")
        //("cminDefaultIntegrator1DOpen", boost::program_options::value<std::string>(), "RooAbsReal::defaultIntegratorConfig()->method1DOpen().setLabel(x)")
        //("cminDefaultIntegrator2D", boost::program_options::value<std::string>(), "RooAbsReal::defaultIntegratorConfig()->method2D().setLabel(x)")
        //("cminDefaultIntegrator2DOpen", boost::program_options::value<std::string>(), "RooAbsReal::defaultIntegratorConfig()->method2DOpen().setLabel(x)")
        //("cminDefaultIntegratorND", boost::program_options::value<std::string>(), "RooAbsReal::defaultIntegratorConfig()->methodND().setLabel(x)")
        //("cminDefaultIntegratorNDOpen", boost::program_options::value<std::string>(), "RooAbsReal::defaultIntegratorConfig()->methodNDOpen().setLabel(x)")
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
            std::string algo = *it;
            float tolerance = Algo::default_tolerance(); 
            int   strategy = Algo::default_strategy(); 
            string::size_type idx = std::min(algo.find(";"), algo.find(":"));
            if (idx != string::npos && idx < algo.length()) {
                 tolerance = atof(algo.substr(idx+1).c_str());
                 algo      = algo.substr(0,idx); // DON'T SWAP THESE TWO LINES
            }
            idx = algo.find(",");
            if (idx != string::npos && idx < algo.length()) {
                // if after the comma there's a number, then it's a strategy
                if ( '0' <= algo[idx+1] && algo[idx+1] <= '9' ) {
                    strategy = atoi(algo.substr(idx+1).c_str());
                    algo     = algo.substr(0,idx); // DON'T SWAP THESE TWO LINES
                } else {
                // otherwise, it could be Name,subname,strategy
                    idx = algo.find(",",idx+1);
                    if (idx != string::npos && idx < algo.length()) {
                        strategy = atoi(algo.substr(idx+1).c_str());
                        algo     = algo.substr(0,idx); // DON'T SWAP THESE TWO LINES
                    }
                }
            }
            fallbacks_.push_back(Algo(algo, tolerance, strategy));
            std::cout << "Configured fallback algorithm " << fallbacks_.back().algo << 
                            ", strategy " << fallbacks_.back().strategy   << 
                            ", tolerance " << fallbacks_.back().tolerance << std::endl;
        }
    }
    //if (vm.count("cminDefaultIntegratorEpsAbs")) RooAbsReal::defaultIntegratorConfig()->setEpsAbs(vm["cminDefaultIntegratorEpsAbs"].as<double>());
    //if (vm.count("cminDefaultIntegratorEpsRel")) RooAbsReal::defaultIntegratorConfig()->setEpsRel(vm["cminDefaultIntegratorEpsRel"].as<double>());
    //if (vm.count("cminDefaultIntegrator1D")) setDefaultIntegrator(RooAbsReal::defaultIntegratorConfig()->method1D(), vm["cminDefaultIntegrator1D"].as<std::string>());
    //if (vm.count("cminDefaultIntegrator1DOpen")) setDefaultIntegrator(RooAbsReal::defaultIntegratorConfig()->method1DOpen(), vm["cminDefaultIntegrator1DOpen"].as<std::string>());
    //if (vm.count("cminDefaultIntegrator2D")) setDefaultIntegrator(RooAbsReal::defaultIntegratorConfig()->method2D(), vm["cminDefaultIntegrator2D"].as<std::string>());
    //if (vm.count("cminDefaultIntegrator2DOpen")) setDefaultIntegrator(RooAbsReal::defaultIntegratorConfig()->method2DOpen(), vm["cminDefaultIntegrator2DOpen"].as<std::string>());
    //if (vm.count("cminDefaultIntegratorND")) setDefaultIntegrator(RooAbsReal::defaultIntegratorConfig()->methodND(), vm["cminDefaultIntegratorND"].as<std::string>());
    //if (vm.count("cminDefaultIntegratorNDOpen")) setDefaultIntegrator(RooAbsReal::defaultIntegratorConfig()->methodNDOpen(), vm["cminDefaultIntegratorNDOpen"].as<std::string>());
}

//void CascadeMinimizer::setDefaultIntegrator(RooCategory &cat, const std::string & val) {
//    if (val == "list") {
//        std::cout << "States for " << cat.GetName() << std::endl;
//        int i0 = cat.getBin();
//        for (int i = 0, n = cat.numBins((const char *)0); i < n; ++i) {
//            cat.setBin(i); std::cout << " - " << cat.getLabel() <<  ( i == i0 ? " (current default)" : "") << std::endl;
//        }
//        std::cout << std::endl;
//        cat.setBin(i0);
//    } else {
//        cat.setLabel(val.c_str()); 
//    }
//}


void CascadeMinimizer::trivialMinimize(const RooAbsReal &nll, RooRealVar &r, int points) const {
    double rMin = r.getMin(), rMax = r.getMax(), rStep = (rMax-rMin)/(points-1);
    int iMin = -1; double minnll = 0;
    for (int i = 0; i < points; ++i) {
        double x = rMin + (i+0.5)*rStep;
        r.setVal(x);
        double y = nll.getVal();
        if (iMin == -1 || y < minnll) { minnll = y; iMin = i; }
    }
    r.setVal( rMin + (iMin+0.5)*rStep );
}
