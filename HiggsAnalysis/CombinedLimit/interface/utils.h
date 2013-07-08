#ifndef HiggsAnalysis_CombinedLimit_utils_h
#define HiggsAnalysis_CombinedLimit_utils_h

#include <vector>
#include <string>
struct RooDataHist;
struct RooAbsData;
struct RooAbsPdf;
struct RooAbsReal;
struct RooAbsArg;
struct RooArgSet;
struct RooArgList;
struct RooSimultaneous;
struct RooAbsCollection;
struct RooWorkspace;
struct RooPlot;
namespace RooStats { class ModelConfig; }
namespace utils {
    void printRDH(RooAbsData *data) ;
    void printRAD(const RooAbsData *d) ;
    void printPdf(RooAbsPdf *pdf) ;
    void printPdf(RooStats::ModelConfig &model) ;
    void printPdf(RooWorkspace *w, const char *pdfName) ;

    // Clone a pdf and all it's branch nodes. on request, clone also leaf nodes (i.e. RooRealVars)
    RooAbsPdf *fullClonePdf(const RooAbsPdf *pdf, RooArgSet &holder, bool cloneLeafNodes=false) ;
    // Clone a function and all it's branch nodes. on request, clone also leaf nodes (i.e. RooRealVars)
    RooAbsReal *fullCloneFunc(const RooAbsReal *pdf, RooArgSet &holder, bool cloneLeafNodes=false) ;

    /// Create a pdf which depends only on observables, and collect the other constraint terms
    /// Will return 0 if it's all constraints, &pdf if it's all observables, or a new pdf if it's something mixed
    /// In the last case, you're the owner of the returned pdf.
    RooAbsPdf *factorizePdf(const RooArgSet &observables, RooAbsPdf &pdf, RooArgList &constraints);

    /// collect factors depending on observables in obsTerms, and all others in constraints
    void factorizePdf(RooStats::ModelConfig &model, RooAbsPdf &pdf, RooArgList &obsTerms, RooArgList &constraints, bool debug=false);
    void factorizePdf(const RooArgSet &observables, RooAbsPdf &pdf, RooArgList &obsTerms, RooArgList &constraints, bool debug=false);
    RooAbsPdf *makeNuisancePdf(RooStats::ModelConfig &model, const char *name="nuisancePdf") ;
    RooAbsPdf *makeNuisancePdf(RooAbsPdf &pdf, const RooArgSet &observables, const char *name="nuisancePdf") ;

    /// factorize a RooAbsReal
    void factorizeFunc(const RooArgSet &observables, RooAbsReal &pdf, RooArgList &obsTerms, RooArgList &otherTerms, bool debug=false);

    /// Note: doesn't recompose Simultaneous pdfs properly, for that use factorizePdf method
    RooAbsPdf *makeObsOnlyPdf(RooStats::ModelConfig &model, const char *name="obsPdf") ;

    /// add to 'clients' all object within allObjects that *directly* depend on values
    void getClients(const RooAbsCollection &values, const RooAbsCollection &allObjects, RooAbsCollection &clients) ;

    /// set all RooRealVars to constants. return true if at least one changed status
    bool setAllConstant(const RooAbsCollection &coll, bool constant=true) ;

    /// Performs the following checks:
    ///  - global observables, if any, are RooRealVars and are const
    ///  - nuisances, if any, are RooRealVars and are floating
    ///  - parameters of interest are all RooRealVars and are floating
    ///  - there are no other floating parameters except observables, nuisances and POI
    bool checkModel(const RooStats::ModelConfig &model, bool throwOnFail=false) ;

    RooSimultaneous * rebuildSimPdf(const RooArgSet &observables, RooSimultaneous *pdf) ;

    void copyAttributes(const RooAbsArg &from, RooAbsArg &to) ;

    void guessChannelMode(RooSimultaneous &simPdf, RooAbsData &simData, bool verbose=false) ;

    /// set style for plots
    void tdrStyle() ;
    
    /// make plots, if possible
    std::vector<RooPlot *> makePlots(const RooAbsPdf &pdf, const RooAbsData &data, const char *signalSel=0, const char *backgroundSel=0, float rebinFactor=1.0);

    struct CheapValueSnapshot {
        public:
            CheapValueSnapshot() : src_(0) {}
            CheapValueSnapshot(const RooAbsCollection &src) : src_(0), values_() { readFrom(src); }
            void readFrom(const RooAbsCollection &) ;
            void writeTo(const RooAbsCollection &) const ;
            void clear() { src_ = 0; values_.clear(); }
            const RooAbsCollection &src() const { return *src_; }
            bool  empty() const { return src_ == 0; }
            void  Print(const char *fmt) const ;
        private:
            const RooAbsCollection *src_;
            std::vector<double> values_;
    };
}

#endif
