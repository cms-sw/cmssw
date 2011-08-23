#include <cmath>
#include <cstdio>
#include <TMath.h>
#include <TFile.h>
#include <TStopwatch.h>
#include <RooWorkspace.h>
#include <RooRealVar.h>
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooAbsPdf.h>
#include <RooRandom.h>
#include <RooMinimizer.h>
#include <RooFitResult.h>
#include <RooHistPdf.h>
#include <TH1.h>
#include "HiggsAnalysis/CombinedLimit/interface/VerticalInterpHistPdf.h"

RooWorkspace *w;


void runStep(const char *pdfname, const char *xname, const char *thetaname) {
    RooRealVar *x     = w->var(xname);
    RooRealVar *theta = w->var(thetaname);
    VerticalInterpHistPdf *pdf   = (VerticalInterpHistPdf *) w->pdf(pdfname);
    pdf->getVal();
    x->setVal(x->getVal() * 1.05);
    //std::cout << "Step x: value dirty " << pdf->isValueDirty() << ", shape dirty " << pdf->isShapeDirty() << std::endl;
    pdf->getVal();
    x->setVal(x->getVal() * 0.95);
    //std::cout << "Step x: value dirty " << pdf->isValueDirty() << ", shape dirty " << pdf->isShapeDirty() << std::endl;
    pdf->getVal();
    theta->setVal(theta->getVal() + 0.1);
    //std::cout << "Step theta: value dirty " << pdf->isValueDirty() << ", shape dirty " << pdf->isShapeDirty() << std::endl;
    pdf->getVal();
    theta->setVal(theta->getVal() - 0.1);
    //std::cout << "Step theta: value dirty " << pdf->isValueDirty() << ", shape dirty " << pdf->isShapeDirty() << std::endl;
    pdf->getVal();
}


int main(int argc, char **argv) {
    w = new RooWorkspace("w","w");
    w->factory("x[0,10]"); w->var("x")->setBins(10);
    w->factory("a[-5,5]");
    w->factory("CBShape::cb(x,cbm[1.7],cbs[0.5],cba[-1],cbn[2])");
    double refval = -1, hival = -0.8, loval = -1.5;
    const char *pdf = "cb", *var = "cba"; 
    w->var(var)->setVal(refval); TH1 *h_nominal = w->pdf(pdf)->createHistogram("x");
    w->var(var)->setVal( hival); TH1 *h_hi      = w->pdf(pdf)->createHistogram("x");
    w->var(var)->setVal( loval); TH1 *h_lo      = w->pdf(pdf)->createHistogram("x");

    RooArgList hobs(*w->var("x"));
    RooArgSet  obs(*w->var("x"));
    RooHistPdf *p_nominal = new RooHistPdf("p_nominal", "", obs, *(new RooDataHist("d_nominal","",hobs,h_nominal)));
    RooHistPdf *p_hi      = new RooHistPdf("p_hi",      "", obs, *(new RooDataHist("d_hi",     "",hobs,h_hi)));
    RooHistPdf *p_lo      = new RooHistPdf("p_lo",      "", obs, *(new RooDataHist("d_lo",     "",hobs,h_lo)));

    w->var("a")->setVal(0.3);
    RooArgList pdfs, coeffs(*w->var("a"));
    pdfs.add(*p_nominal);
    pdfs.add(*p_hi);
    pdfs.add(*p_lo);
    VerticalInterpHistPdf *morph = new VerticalInterpHistPdf("m","m",*w->var("x"),pdfs,coeffs,0.5,+1);
    w->import(*morph); 
    runStep("m", "x", "a");
}
