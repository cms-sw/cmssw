#ifndef HiggsAnalysis_CombinedLimit_ProfiledLikelihoodRatioTestStat_h
#define HiggsAnalysis_CombinedLimit_ProfiledLikelihoodRatioTestStat_h

#include <memory>
#include <RooAbsPdf.h>
#include <RooAbsData.h>
#include <RooArgSet.h>
#include <RooStats/TestStatistic.h>

class ProfiledLikelihoodRatioTestStat : public RooStats::TestStatistic {
    public:
        ProfiledLikelihoodRatioTestStat(RooAbsPdf &pdfNull, RooAbsPdf &pdfAlt, const RooArgSet *nuisances, const RooArgSet & paramsNull = RooArgSet(), const RooArgSet & paramsAlt = RooArgSet()) : 
            pdfNull_(&pdfNull), pdfAlt_(&pdfAlt),
            paramsNull_(pdfNull_->getVariables()), 
            paramsAlt_(pdfAlt_->getVariables()) 
        {
            snapNull_.addClone(paramsNull);
            snapAlt_.addClone(paramsAlt);
            if (nuisances) nuisances_.addClone(*nuisances);
        }

        virtual Double_t Evaluate(RooAbsData& data, RooArgSet& nullPOI) ;

        virtual const TString GetVarName() const {
            return TString::Format("-log(%s/%s)", pdfNull_->GetName(), pdfAlt_->GetName()); 
        }

    private:
        RooAbsPdf *pdfNull_, *pdfAlt_;
        RooArgSet snapNull_, snapAlt_; 
        RooArgSet nuisances_; 
        std::auto_ptr<RooArgSet> paramsNull_, paramsAlt_;
}; // TestSimpleStatistics


#endif
