#ifndef HiggsAnalysis_CombinedLimit_CachingNLL_h
#define HiggsAnalysis_CombinedLimit_CachingNLL_h

#include <memory>
#include <RooAbsPdf.h>
#include <RooAddPdf.h>
#include <RooProdPdf.h>
#include <RooAbsData.h>
#include <RooArgSet.h>
#include <RooSetProxy.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>

// Part zero: ArgSet checker
namespace cacheutils {
    class ArgSetChecker {
        public:
            ArgSetChecker() {}
            ArgSetChecker(const RooAbsCollection *set) ;
            bool changed(bool updateIfChanged=false) ;
        private:
            std::vector<RooRealVar *> vars_;
            std::vector<double> vals_;
    };

// Part one: cache all values of a pdf
class CachingPdf {
    public:
        CachingPdf(RooAbsPdf *pdf, const RooArgSet *obs) ;
        CachingPdf(const CachingPdf &other) ;
        ~CachingPdf() ;
        const std::vector<Double_t> & eval(const RooAbsData &data) ;
        const RooAbsPdf *pdf() const { return pdf_; }
    private:
        const RooArgSet *obs_;
        RooAbsPdf *pdfOriginal_;
        RooArgSet  pdfPieces_;
        RooAbsPdf *pdf_;
        ArgSetChecker checker_;
        const RooAbsData *lastData_;
        std::vector<Double_t> vals_;
        void realFill_(const RooAbsData &data) ;
};

class CachingAddNLL : public RooAbsReal {
    public:
        CachingAddNLL(const char *name, const char *title, RooAddPdf *pdf, RooAbsData *data) ;
        CachingAddNLL(const CachingAddNLL &other, const char *name = 0) ;
        virtual CachingAddNLL *clone(const char *name = 0) const ;
        virtual Double_t evaluate() const ;
        virtual Bool_t isDerived() const { return kTRUE; }
        virtual Double_t defaultErrorLevel() const { return 0.5; }
        void setData(const RooAbsData &data) ;
        virtual RooArgSet* getObservables(const RooArgSet* depList, Bool_t valueOnly = kTRUE) const ;
        virtual RooArgSet* getParameters(const RooArgSet* depList, Bool_t stripDisconnected = kTRUE) const ;
        double  sumWeights() const { return sumWeights_; }
    private:
        void setup_();
        RooAddPdf *pdf_;
        RooSetProxy params_;
        const RooAbsData *data_;
        std::vector<double>  weights_;
        double               sumWeights_;
        mutable std::vector<RooAbsReal*> coeffs_;
        mutable std::vector<CachingPdf>  pdfs_;
        mutable std::vector<double> partialSum_;
};

class CachingSimNLL  : public RooAbsReal {
    public:
        CachingSimNLL(RooSimultaneous *pdf, RooAbsData *data) ;
        CachingSimNLL(const CachingSimNLL &other, const char *name = 0) ;
        virtual CachingSimNLL *clone(const char *name = 0) const ;
        virtual Double_t evaluate() const ;
        virtual Bool_t isDerived() const { return kTRUE; }
        virtual Double_t defaultErrorLevel() const { return 0.5; }
        void setData(const RooAbsData &data) ;
        virtual RooArgSet* getObservables(const RooArgSet* depList, Bool_t valueOnly = kTRUE) const ;
        virtual RooArgSet* getParameters(const RooArgSet* depList, Bool_t stripDisconnected = kTRUE) const ;
    private:
        void setup_();
        RooSimultaneous   *pdfOriginal_;
        const RooAbsData  *dataOriginal_;
        RooSetProxy        params_;
        RooArgSet piecesForCloning_;
        std::auto_ptr<RooSimultaneous>  factorizedPdf_;
        std::auto_ptr<RooProdPdf>       constrainPdf_;
        std::auto_ptr<TList>            dataSets_;
        std::vector<CachingAddNLL*>     pdfs_;
};

}
#endif
