#ifndef HiggsAnalysis_CombinedLimit_PdfDiagonalizer_h
#define HiggsAnalysis_CombinedLimit_PdfDiagonalizer_h

class RooWorkspace;
struct RooFitResult;
struct RooAbsPdf;

#include <string>
#include <RooArgList.h>

class PdfDiagonalizer {
    public:
        PdfDiagonalizer(const char *name, RooWorkspace *w, RooFitResult &result);

        RooAbsPdf *diagonalize(RooAbsPdf &pdf) ;
        const RooArgList & originalParams() { return parameters_; }
        const RooArgList & diagonalParams() { return eigenVars_; }
    private:
        std::string name_;
        RooArgList  parameters_;
        RooArgList  eigenVars_;
        RooArgList  replacements_;
};

#endif
