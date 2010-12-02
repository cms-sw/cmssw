#ifndef HiggsAnalysis_CombinedLimit_utils_h
#define HiggsAnalysis_CombinedLimit_utils_h

struct RooDataHist;
struct RooAbsData;
struct RooWorkspace;

namespace utils {
    void printRDH(RooDataHist *data) ;
    void printRAD(const RooAbsData *d) ;
    void printPdf(RooWorkspace *w, const char *pdfName) ;
}
#endif
