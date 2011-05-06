#ifndef HiggsAnalysis_CombinedLimit_SimplerLikelihoodRatioTestStatExt_h
#define HiggsAnalysis_CombinedLimit_SimplerLikelihoodRatioTestStatExt_h

/**
  BEGIN_HTML
  <p>
  SimplerLikelihoodRatioTestStatOpt: an optimized implementaion of the simple likelihood ratio 
     test statistics Q = - ln( L(data|null) / L(data|alt) )
  </p>

  <p>Optimization is obtained through the following methods:</p>
  <ul>
  <li>Constraints not depending on observables are factorized away at construction time.</li>
  <li>The nodes that directly depend on the observables are identified at construction time, and so only those nodes have their servers redirected when changing dataset.</li>
  <li>Simultaneous pdfs are split at construction time; when evaluating the likelihood, no lookups of components by name are performed.</li>
  <li>The amount of objects created and destroyed for each evaluation is reduced to a minimum.</li>
  <li>Only one loop is performed on the dataset (since only one is necessary), and the data is not copied.</li>

  </ul>
  Author: Giovanni Petrucciani (UCSD/CMS/CERN), May 2011
  </p>
  END_HTML
*/

#include <memory>
#include <stdexcept>
#include <RooAbsPdf.h>
#include <RooAbsData.h>
#include <RooSimultaneous.h>
#include <RooArgSet.h>
#include <RooStats/TestStatistic.h>


class SimplerLikelihoodRatioTestStatOpt : public RooStats::TestStatistic {
    public:
        /// Create a SimplerLikelihoodRatioTestStatOpt. 
        ///   obs = the observables on which the two pdfs depend
        ///   pdfNull, pdfAlt = pdfs of the two models
        ///   paramsNull, paramsAlt = values of the parameters that should be set before evaluating the pdfs
        ///   factorize: if set to true, the constraint terms not depending on the observables will be removed 
        SimplerLikelihoodRatioTestStatOpt(const RooArgSet &obs, RooAbsPdf &pdfNull, RooAbsPdf &pdfAlt, const RooArgSet & paramsNull = RooArgSet(), const RooArgSet & paramsAlt = RooArgSet(), bool factorize=true) ;

        virtual ~SimplerLikelihoodRatioTestStatOpt() ;

        virtual Double_t Evaluate(RooAbsData& data, RooArgSet& nullPOI) ;

        virtual const TString GetVarName() const {
            return TString::Format("-log(%s/%s)", pdfNull_->GetName(), pdfAlt_->GetName()); 
        }
    private:
        /// observables (global argset)
        const RooArgSet *obs_; 
        /// pdfs (cloned, and with constraints factorized away)
        RooAbsPdf *pdfNull_, *pdfAlt_;
        /// snapshot with all branch nodes of the pdfs before factorization
        RooArgSet pdfCompNull_, pdfCompAlt_;
        /// nodes which depend directly on observables, on which one has to do redirectServers
        RooArgList pdfDepObs_;
        /// snapshots of parameters for the two pdfs
        RooArgSet snapNull_, snapAlt_; 
        /// parameter sets to apply snapshots to
        std::auto_ptr<RooArgSet> paramsNull_, paramsAlt_;
        /// owned copy of the pdfs after factorizing
        std::auto_ptr<RooAbsPdf> pdfNullOwned_, pdfAltOwned_;
        /// pdfNull, pdfAlt cast to sympdf (may be null) after factorization
        RooSimultaneous *simPdfNull_, *simPdfAlt_;
        /// components of the sim pdfs after factorization, for each bin in sim. category. can contain nulls
        std::vector<RooAbsPdf *> simPdfComponentsNull_, simPdfComponentsAlt_;

        double evalSimNLL(RooAbsData &data,  RooSimultaneous *pdf, std::vector<RooAbsPdf *> &components);
        double evalSimpleNLL(RooAbsData &data,  RooAbsPdf *pdf);
        void unrollSimPdf(RooSimultaneous *pdf, std::vector<RooAbsPdf *> &out);

}; // 

// ===== This below is identical to the RooStats::SimpleLikelihoodRatioTestStat also in implementation
//       I've made a copy here just to be able to put some debug hooks inside.
#if 0
class SimplerLikelihoodRatioTestStatExt : public RooStats::TestStatistic {
    public:
        SimplerLikelihoodRatioTestStatExt(const RooArgSet &obs, RooAbsPdf &pdfNull, RooAbsPdf &pdfAlt, const RooArgSet & paramsNull = RooArgSet(), const RooArgSet & paramsAlt = RooArgSet()) ;

        virtual ~SimplerLikelihoodRatioTestStatExt() ;

        virtual Double_t Evaluate(RooAbsData& data, RooArgSet& nullPOI) ;

        virtual const TString GetVarName() const {
            return TString::Format("-log(%s/%s)", pdfNull_->GetName(), pdfAlt_->GetName()); 
        }

    private:
        RooAbsPdf *pdfNull_, *pdfAlt_;
        RooArgSet snapNull_, snapAlt_; 
        std::auto_ptr<RooArgSet> paramsNull_, paramsAlt_;
        std::auto_ptr<RooAbsPdf> pdfNullOwned_, pdfAltOwned_;
}; // TestSimpleStatistics
#endif

#endif
