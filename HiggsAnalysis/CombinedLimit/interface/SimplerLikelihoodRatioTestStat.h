#ifndef HiggsAnalysis_CombinedLimit_SimplerLikelihoodRatioTestStat_h
#define HiggsAnalysis_CombinedLimit_SimplerLikelihoodRatioTestStat_h
#include <memory>
#include <stdexcept>

#include <RooAbsPdf.h>
#include <RooAbsData.h>
#include <RooArgSet.h>
#include <RooStats/TestStatistic.h>

class SimplerLikelihoodRatioTestStat : public RooStats::TestStatistic {
    public:
        SimplerLikelihoodRatioTestStat(RooAbsPdf &pdfNull, RooAbsPdf &pdfAlt, const RooArgSet & paramsNull = RooArgSet(), const RooArgSet & paramsAlt = RooArgSet()) : 
            pdfNull_(&pdfNull), pdfAlt_(&pdfAlt),
            paramsNull_(pdfNull_->getVariables()), 
            paramsAlt_(pdfAlt_->getVariables()) 
        {
            snapNull_.addClone(paramsNull);
            snapAlt_.addClone(paramsAlt);
        }

        virtual Double_t Evaluate(RooAbsData& data, RooArgSet& nullPOI) 
        {
            if (data.numEntries() != 1) throw std::invalid_argument("HybridNew::TestSimpleStatistics: dataset doesn't have exactly 1 entry.");
            const RooArgSet *entry = data.get(0);
            *paramsNull_ = *entry;
            *paramsNull_ = snapNull_;
            *paramsNull_ = nullPOI;
            double nullNLL = pdfNull_->getVal();

            *paramsAlt_ = *entry;
            *paramsAlt_ = snapAlt_;
            double altNLL = pdfAlt_->getVal();

            return -log(nullNLL/altNLL);
        }

        virtual const TString GetVarName() const {
            return TString::Format("-log(%s/%s)", pdfNull_->GetName(), pdfAlt_->GetName()); 
        }

    private:
        RooAbsPdf *pdfNull_, *pdfAlt_;
        RooArgSet snapNull_, snapAlt_; 
        std::auto_ptr<RooArgSet> paramsNull_, paramsAlt_;
}; // TestSimpleStatistics

#endif
