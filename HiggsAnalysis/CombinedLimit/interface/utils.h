#ifndef HiggsAnalysis_CombinedLimit_utils_h
#define HiggsAnalysis_CombinedLimit_utils_h

struct RooDataHist;
struct RooAbsData;
struct RooAbsPdf;
struct RooArgList;
struct RooWorkspace;
namespace RooStats { class ModelConfig; }
namespace utils {
    void printRDH(RooAbsData *data) ;
    void printRAD(const RooAbsData *d) ;
    void printPdf(RooAbsPdf *pdf) ;
    void printPdf(RooStats::ModelConfig &model) ;
    void printPdf(RooWorkspace *w, const char *pdfName) ;

    /// collect factors depending on observables in obsTerms, and all others in constraints
    void factorizePdf(RooStats::ModelConfig &model, RooAbsPdf &pdf, RooArgList &obsTerms, RooArgList &constraints, bool debug=false);
    RooAbsPdf *makeNuisancePdf(RooStats::ModelConfig &model, const char *name="nuisancePdf") ;
    RooAbsPdf *makeObsOnlyPdf(RooStats::ModelConfig &model, const char *name="obsPdf") ;
}
#endif
