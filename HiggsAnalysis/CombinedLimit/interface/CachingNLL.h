#ifndef HiggsAnalysis_CombinedLimit_CachingNLL_h
#define HiggsAnalysis_CombinedLimit_CachingNLL_h

#include <memory>
#include <RooAbsPdf.h>
#include <RooAddPdf.h>
#include <RooRealSumPdf.h>
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
        CachingPdf(RooAbsReal *pdf, const RooArgSet *obs) ;
        CachingPdf(const CachingPdf &other) ;
        ~CachingPdf() ;
        const std::vector<Double_t> & eval(const RooAbsData &data) ;
        const RooAbsReal *pdf() const { return pdf_; }
        void  setDataDirty() { lastData_ = 0; }
    private:
        const RooArgSet *obs_;
        RooAbsReal *pdfOriginal_;
        RooArgSet  pdfPieces_;
        RooAbsReal *pdf_;
        ArgSetChecker checker_;
        const RooAbsData *lastData_;
        std::vector<Double_t> vals_;
        void realFill_(const RooAbsData &data) ;
};

class CachingAddNLL : public RooAbsReal {
    public:
        CachingAddNLL(const char *name, const char *title, RooAbsPdf *pdf, RooAbsData *data) ;
        CachingAddNLL(const CachingAddNLL &other, const char *name = 0) ;
        virtual ~CachingAddNLL() ;
        virtual CachingAddNLL *clone(const char *name = 0) const ;
        virtual Double_t evaluate() const ;
        virtual Bool_t isDerived() const { return kTRUE; }
        virtual Double_t defaultErrorLevel() const { return 0.5; }
        void setData(const RooAbsData &data) ;
        virtual RooArgSet* getObservables(const RooArgSet* depList, Bool_t valueOnly = kTRUE) const ;
        virtual RooArgSet* getParameters(const RooArgSet* depList, Bool_t stripDisconnected = kTRUE) const ;
        double  sumWeights() const { return sumWeights_; }
        const RooAbsPdf *pdf() const { return pdf_; }
    private:
        void setup_();
        RooAbsPdf *pdf_;
        RooSetProxy params_;
        const RooAbsData *data_;
        std::vector<double>  weights_;
        double               sumWeights_;
        mutable std::vector<RooAbsReal*> coeffs_;
        mutable std::vector<CachingPdf>  pdfs_;
        mutable std::vector<RooAbsReal*> integrals_;
        mutable std::vector<double> partialSum_;
        mutable bool isRooRealSum_;
};

class CachingSimNLL  : public RooAbsReal {
    public:
        CachingSimNLL(RooSimultaneous *pdf, RooAbsData *data, const RooArgSet *nuis=0) ;
        CachingSimNLL(const CachingSimNLL &other, const char *name = 0) ;
        virtual CachingSimNLL *clone(const char *name = 0) const ;
        virtual Double_t evaluate() const ;
        virtual Bool_t isDerived() const { return kTRUE; }
        virtual Double_t defaultErrorLevel() const { return 0.5; }
        void setData(const RooAbsData &data) ;
        virtual RooArgSet* getObservables(const RooArgSet* depList, Bool_t valueOnly = kTRUE) const ;
        virtual RooArgSet* getParameters(const RooArgSet* depList, Bool_t stripDisconnected = kTRUE) const ;
        void splitWithWeights(const RooAbsData &data, const RooAbsCategory& splitCat, Bool_t createEmptyDataSets) ;
    private:
        void setup_();
        RooSimultaneous   *pdfOriginal_;
        const RooAbsData  *dataOriginal_;
        const RooArgSet   *nuis_;
        RooSetProxy        params_;
        RooArgSet piecesForCloning_;
        std::auto_ptr<RooSimultaneous>  factorizedPdf_;
        std::vector<RooAbsPdf *>        constrainPdfs_;
        std::vector<CachingAddNLL*>     pdfs_;
        std::auto_ptr<TList>            dataSets_;
        std::vector<RooDataSet *>       datasets_;
};

}
#endif
