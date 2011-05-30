#ifndef HiggsAnalysis_CombinedLimit_ProfiledLikelihoodRatioTestStatExt_h
#define HiggsAnalysis_CombinedLimit_ProfiledLikelihoodRatioTestStatExt_h

#include <memory>
#include <RooAbsPdf.h>
#include <RooAbsData.h>
#include <RooArgSet.h>
#include <RooArgList.h>
#include <RooSimultaneous.h>
#include <RooSetProxy.h>
#include <RooStats/TestStatistic.h>

/// Class that holds a set of variables in a vector
class OptimizedDatasetForNLL {
    public:
        OptimizedDatasetForNLL(const RooArgSet &vars, const RooAbsData *data, int firstEntry=0, int lastEntry=-1) ;
        ~OptimizedDatasetForNLL();
        void addEntries(const RooAbsData *data, int firstEntry=0, int lastEntry=-1);
        int numEntries() const { return numEntries_; }
        double sumWeights() const { return sumw_; }
        const RooArgSet & obs() const { return obs_; }
        double get(int i) const ;
    private:
        struct Var { RooRealVar *var; std::vector<double> vals; };
        int numEntries_; double sumw_;
        RooArgSet obs_;
        mutable std::vector<Var> vars_;
        mutable std::vector<double> weight_;
        mutable const RooAbsData *dataset_;
        mutable std::vector<RooRealVar *> datasetVars_;
};

/// Class to compute NLL for a pdf which is not RooSimultaneous and has no constraints 
class OptimizedSimpleNLL : public RooAbsReal {
    public:
        OptimizedSimpleNLL(RooAbsPdf *pdf) ; 
        OptimizedSimpleNLL(const OptimizedSimpleNLL &other, const char *name=0);
        ~OptimizedSimpleNLL() {} 
        virtual OptimizedSimpleNLL *clone(const char *name) const { return new OptimizedSimpleNLL(*this, name); }
        virtual Double_t evaluate() const ;
        virtual Bool_t isDerived() const { return kTRUE; }
        virtual Double_t defaultErrorLevel() const { return 0.5; }
        void setData(const RooAbsData &data, int firstEntry=0, int lastEntry=-1) ;
        void addData(const RooAbsData &data, int firstEntry=0, int lastEntry=-1) ;
        void setNoData() { data_.reset(); setValueDirty(); }
        const OptimizedDatasetForNLL *data() const { return data_.get(); }
        virtual RooArgSet* getObservables(const RooArgSet* depList, Bool_t valueOnly = kTRUE) const ;
        virtual RooArgSet* getParameters(const RooArgSet* depList, Bool_t stripDisconnected = kTRUE) const ;
    private:
        RooAbsPdf *pdf_;
        RooArgSet obs_;
        RooSetProxy params_;
        std::auto_ptr<OptimizedDatasetForNLL> data_;
};

/// Class to compute NLL for a pdf that can be a RooSimultaneous
class OptimizedSimNLL : public RooAbsReal {
    public:
        OptimizedSimNLL(const RooArgSet &obs, const RooArgSet &nuis, RooAbsPdf *pdf);
        OptimizedSimNLL(const OptimizedSimNLL &other, const char *name=0);
        ~OptimizedSimNLL();
        virtual OptimizedSimNLL *clone(const char *name) const { return new OptimizedSimNLL(*this, name); }
        virtual Bool_t isDerived() const { return kTRUE; }
        virtual Double_t defaultErrorLevel() const { return 0.5; }
        void setData(const RooAbsData &data) ;
        virtual Double_t evaluate() const;
        using RooAbsReal::getParameters;
        using RooAbsReal::getObservables;
        virtual RooArgSet* getObservables(const RooArgSet* depList, Bool_t valueOnly = kTRUE) const ;
        virtual RooArgSet* getParameters(const RooArgSet* depList, Bool_t stripDisconnected = kTRUE) const ;
    private:
        RooAbsPdf *originalPdf_;
        RooArgSet   obs_, nuis_;
        RooSetProxy params_;
        RooArgSet   pdfClonePieces_;
        std::auto_ptr<RooSimultaneous> simpdf_;
        std::auto_ptr<RooAbsPdf>       nonsimpdf_;
        std::vector<OptimizedSimpleNLL *> nlls_;
        int                               simCount_;
        std::vector<RooAbsPdf *> constraints_;
        std::vector<RooArgSet>   constraintNSets_;

        void init(const RooArgSet &obs, const RooArgSet &nuis, RooAbsPdf *pdf) ;
};


class ProfiledLikelihoodRatioTestStatExt : public RooStats::TestStatistic {
    public:
        ProfiledLikelihoodRatioTestStatExt(const RooArgSet &obs, RooAbsPdf &pdfNull, RooAbsPdf &pdfAlt, const RooArgSet *nuisances, const RooArgSet & paramsNull = RooArgSet(), const RooArgSet & paramsAlt = RooArgSet()) ;
 
        virtual Double_t Evaluate(RooAbsData& data, RooArgSet& nullPOI) ;

        virtual const TString GetVarName() const {
            return TString::Format("-log(%s/%s)", pdfNull_->GetName(), pdfAlt_->GetName()); 
        }

        // Verbosity (default: 0)
        void setPrintLevel(Int_t level) { verbosity_ = level; }

    private:
        RooAbsPdf *pdfNull_, *pdfAlt_;
        RooArgSet snapNull_, snapAlt_; 
        RooArgSet nuisances_; 
        std::auto_ptr<RooArgSet> paramsNull_, paramsAlt_;
        Int_t verbosity_;

        double minNLL(RooAbsPdf &pdf, RooAbsData &data) ;
}; // TestSimpleStatistics

class ProfiledLikelihoodRatioTestStatOpt : public RooStats::TestStatistic {
    public:
        ProfiledLikelihoodRatioTestStatOpt(const RooArgSet & observables,
                RooAbsPdf &pdfNull, RooAbsPdf &pdfAlt, 
                const RooArgSet *nuisances, 
                const RooArgSet & paramsNull = RooArgSet(), const RooArgSet & paramsAlt = RooArgSet()) ; 
        virtual Double_t Evaluate(RooAbsData& data, RooArgSet& nullPOI) ;

        virtual const TString GetVarName() const {
            return TString::Format("-log(%s/%s)", pdfNull_->GetName(), pdfAlt_->GetName()); 
        }

        // Verbosity (default: 0)
        void setPrintLevel(Int_t level) { verbosity_ = level; }

    private:
        RooAbsPdf *pdfNull_, *pdfAlt_;
        RooArgSet snapNull_, snapAlt_; 
        RooArgSet nuisances_; 
        std::auto_ptr<RooArgSet> paramsNull_, paramsAlt_;
        std::auto_ptr<OptimizedSimNLL> nllNull_, nllAlt_;
        Int_t verbosity_;

        double minNLL(OptimizedSimNLL &nll, RooAbsData &data) ;
}; // TestSimpleStatistics

class ProfiledLikelihoodTestStatOpt : public RooStats::TestStatistic {
    public:
        ProfiledLikelihoodTestStatOpt(const RooArgSet & observables,
                RooAbsPdf &pdf, 
                const RooArgSet *nuisances, 
                const RooArgSet *globalObs, 
                const RooArgSet & params) ; 
        virtual Double_t Evaluate(RooAbsData& data, RooArgSet& nullPOI) ;

        virtual const TString GetVarName() const { return "boh"; }

        // Verbosity (default: 0)
        void setPrintLevel(Int_t level) { verbosity_ = level; }

    private:
        RooAbsPdf *pdf_;
        RooArgSet snap_, poi_, nuisances_, globalObs_; 
        std::auto_ptr<RooArgSet> params_;
        std::auto_ptr<OptimizedSimNLL> nll_;
        Int_t verbosity_;

        double minNLL(RooAbsPdf &pdf, RooAbsData &data) ;
        double minNLL(OptimizedSimNLL &nll, RooAbsData &data) ;
}; // TestSimpleStatistics


#endif
