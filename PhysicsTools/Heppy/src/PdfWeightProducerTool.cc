#include "PhysicsTools/Heppy/interface/PdfWeightProducerTool.h"
#include <cassert>

namespace LHAPDF {
      void initPDFSet(int nset, const std::string& filename, int member=0);
      int numberPDF(int nset);
      void usePDFMember(int nset, int member);
      double xfx(int nset, double x, double Q, int fl);
      double getXmin(int nset, int member);
      double getXmax(int nset, int member);
      double getQ2min(int nset, int member);
      double getQ2max(int nset, int member);
      void extrapolate(bool extrapolate=true);
}

namespace heppy {

void PdfWeightProducerTool::addPdfSet(const std::string &name) {
    pdfs_.push_back(name);
    weights_[name] = std::vector<double>();
}

void PdfWeightProducerTool::beginJob() {
    for (unsigned int i = 0, n = pdfs_.size(); i < n; ++i) {
        LHAPDF::initPDFSet(i+1, pdfs_[i]);
    }
}

void PdfWeightProducerTool::processEvent(const GenEventInfoProduct & pdfstuff) {
    float Q = pdfstuff.pdf()->scalePDF;

    int id1 = pdfstuff.pdf()->id.first;
    double x1 = pdfstuff.pdf()->x.first;
    double pdf1 = pdfstuff.pdf()->xPDF.first;

    int id2 = pdfstuff.pdf()->id.second;
    double x2 = pdfstuff.pdf()->x.second;
    double pdf2 = pdfstuff.pdf()->xPDF.second; 

    for (unsigned int i = 0, n = pdfs_.size(); i < n; ++i) {
        std::vector<double> & weights = weights_[pdfs_[i]];
        unsigned int nweights = 1;
        if (LHAPDF::numberPDF(i+1)>1) nweights += LHAPDF::numberPDF(i+1);
        weights.resize(nweights);

        for (unsigned int j = 0; j < nweights; ++j) { 
            LHAPDF::usePDFMember(i+1,j);
            double newpdf1 = LHAPDF::xfx(i+1, x1, Q, id1)/x1;
            double newpdf2 = LHAPDF::xfx(i+1, x2, Q, id2)/x2;
            weights[j] = newpdf1/pdf1*newpdf2/pdf2;
        }
    }
}

const std::vector<double> & PdfWeightProducerTool::getWeights(const std::string &name) const {
    std::map<std::string, std::vector<double> >::const_iterator match = weights_.find(name);
    assert(match != weights_.end()); 
    return match->second;   
}


}
