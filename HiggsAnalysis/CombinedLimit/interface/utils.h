#ifndef HiggsAnalysis_CombinedLimit_utils_h
#define HiggsAnalysis_CombinedLimit_utils_h

struct RooDataHist;
struct RooAbsData;
struct RooAbsPdf;
struct RooAbsArg;
struct RooArgSet;
struct RooArgList;
struct RooSimultaneous;
struct RooAbsCollection;
struct RooWorkspace;
namespace RooStats { class ModelConfig; }
namespace utils {
    void printRDH(RooAbsData *data) ;
    void printRAD(const RooAbsData *d) ;
    void printPdf(RooAbsPdf *pdf) ;
    void printPdf(RooStats::ModelConfig &model) ;
    void printPdf(RooWorkspace *w, const char *pdfName) ;

    // Clone a pdf and all it's branch nodes. on request, clone also leaf nodes (i.e. RooRealVars)
    RooAbsPdf *fullClonePdf(const RooAbsPdf *pdf, RooArgSet &holder, bool cloneLeafNodes=false) ;

    /// Create a pdf which depends only on observables, and collect the other constraint terms
    /// Will return 0 if it's all constraints, &pdf if it's all observables, or a new pdf if it's something mixed
    /// In the last case, you're the owner of the returned pdf.
    RooAbsPdf *factorizePdf(const RooArgSet &observables, RooAbsPdf &pdf, RooArgList &constraints);

    /// collect factors depending on observables in obsTerms, and all others in constraints
    void factorizePdf(RooStats::ModelConfig &model, RooAbsPdf &pdf, RooArgList &obsTerms, RooArgList &constraints, bool debug=false);
    void factorizePdf(const RooArgSet &observables, RooAbsPdf &pdf, RooArgList &obsTerms, RooArgList &constraints, bool debug=false);
    RooAbsPdf *makeNuisancePdf(RooStats::ModelConfig &model, const char *name="nuisancePdf") ;
    RooAbsPdf *makeNuisancePdf(RooAbsPdf &pdf, const RooArgSet &observables, const char *name="nuisancePdf") ;

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
}
#endif
