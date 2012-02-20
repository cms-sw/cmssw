#ifndef HiggsAnalysis_CombinedLimit_CascadeMinimizer_h
#define HiggsAnalysis_CombinedLimit_CascadeMinimizer_h

class RooAbsReal;
class RooMinimizer;
class RooArgSet;
class RooRealVar;
#include <RooArgSet.h>
#include <RooMinimizer.h>
#include <boost/program_options.hpp>

class CascadeMinimizer {
    public:
        enum Mode { Constrained, Unconstrained };
        CascadeMinimizer(RooAbsReal &nll, Mode mode, RooRealVar *poi=0, int initialStrategy=0) ;
        bool minimize(int verbose=0);
        RooMinimizer & minimizer() { return minimizer_; }
        RooFitResult *save() { return minimizer().save(); }
        void  setStrategy(int strategy) { strategy_ = strategy; }
        void  setErrorLevel(float errorLevel) { minimizer_.setErrorLevel(errorLevel); }
        static void  initOptions() ;
        static void  applyOptions(const boost::program_options::variables_map &vm) ;
        static const boost::program_options::options_description & options() { return options_; }
    private:
        RooAbsReal & nll_;
        RooMinimizer minimizer_;
        Mode         mode_;
        int          strategy_;
        RooRealVar * poi_; 

        /// options configured from command line
        static boost::program_options::options_description options_;
        /// compact information about an algorithm
        struct Algo { 
            Algo() : algo(), tolerance() {}
            Algo(const std::string &str, float tol=-1.f) : algo(str), tolerance(tol) {}
            std::string algo; float tolerance;
        };
        /// list of algorithms to run if the default one fails
        static std::vector<Algo> fallbacks_;
        /// do first a fit of only the POI
        static bool poiOnlyFit_;
        /// do first a minimization of each nuisance individually 
        static bool singleNuisFit_;
};

#endif
