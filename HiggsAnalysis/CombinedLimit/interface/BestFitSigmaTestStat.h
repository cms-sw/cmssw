#ifndef HiggsAnalysis_CombinedLimit_BestFitSigmaTestStat
#define HiggsAnalysis_CombinedLimit_BestFitSigmaTestStat

#include <memory>
#include <vector>

class RooMinimizer;
#include <RooAbsPdf.h>
#include <RooAbsData.h>
#include <RooArgSet.h>
#include <RooStats/TestStatistic.h>
#include "../interface/RooSimultaneousOpt.h"
#include "../interface/CachingNLL.h"

class BestFitSigmaTestStat : public RooStats::TestStatistic {
    public:
        BestFitSigmaTestStat(const RooArgSet & observables,
                RooAbsPdf &pdf, 
                const RooArgSet *nuisances, 
                const RooArgSet & params, int verbosity=0) ; 

        virtual Double_t Evaluate(RooAbsData& data, RooArgSet& nullPOI) ;

        virtual const TString GetVarName() const { return "mu-hat`"; }

        // Verbosity (default: 0)
        void setPrintLevel(Int_t level) { verbosity_ = level; }
    private:

        RooAbsPdf *pdf_;
        RooArgSet snap_, poi_, nuisances_; 
        std::auto_ptr<RooArgSet> params_;
        std::auto_ptr<RooAbsReal> nll_;
        Int_t verbosity_;

        // create NLL. if returns true, it can be kept, if false it should be deleted at the end of Evaluate
        bool createNLL(RooAbsPdf &pdf, RooAbsData &data) ;
        double minNLL(bool constrained, RooRealVar *r=0) ;
}; // TestSimpleStatistics


#endif
