#include <cmath>
#include <cstdio>
#include <RooWorkspace.h>
#include <RooRealVar.h>
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooAbsPdf.h>
#include <RooAddition.h>
#include <RooRandom.h>
#include <RooCustomizer.h>
#include <RooFitResult.h>
#include <RooMsgService.h>
#include <TMatrixDSym.h>
#include <TMatrixDSymEigen.h>
#include <TCanvas.h>
#include <RooPlot.h>
#include "../../interface/PdfDiagonalizer.h"

void runDiago(RooWorkspace *w, RooDataSet *data, RooFitResult *result, const char *fit) {
    PdfDiagonalizer diago("eig", w, *result);
    RooAbsPdf *newfit = diago.diagonalize(*w->pdf(fit));
    w->import(*newfit, RooFit::RecycleConflictNodes());
    RooFitResult *res2 = newfit->fitTo(*data, RooFit::Save(1), RooFit::Minimizer("Minuit2"), RooFit::PrintLevel(-1));
    res2->Print("V");
}


void runTest(RooWorkspace *w, const char *gen, const char *fit, bool diago, int N=500) {
    RooArgSet obs(*w->set("obs"));
    RooDataSet *data = w->pdf(gen)->generate(obs, N); data->SetName("data");
    RooFitResult *result = w->pdf(fit)->fitTo(*data, RooFit::Save(1), RooFit::Minimizer("Minuit2"), RooFit::PrintLevel(-1));
    result->Print("V");
    runDiago(w,data,result,fit);
    w->import(*data);
}

void testOne() {
    RooWorkspace *w = new RooWorkspace("w","w");
    w->factory("x[-5,5]"); 
    w->factory("a[-3,3]");
    w->factory("b[-3,3]");
    w->factory("Gaussian::gen(x, 0, 1)");
    w->factory("Gaussian::fit0(x, a, sum(1,b) )");
    w->factory("Gaussian::fit(x, sum::somma(a,b), sum::differenza(1,a,prod::meno(b,-1)))");
    w->defineSet("obs", "x");
    w->defineSet("vars", "a,b");
    std::cout << "Fit in uncorrelated basis" << std::endl;
    runTest(w, "gen", "fit0", true);
    std::cout << "Fit in correlated basis" << std::endl;
    runTest(w, "gen", "fit", true);
}

void testTwo() {
    RooWorkspace *w = new RooWorkspace("w","w");
    w->factory("x[0,1]"); 
    w->factory("a[-1,1]");
    w->factory("b[-1,1]");
    w->factory("Exponential::gen(x, -0.2)");
    w->factory("Polynomial::fit(x, {a,b})");
    w->defineSet("obs", "x");
    std::cout << "Fit in correlated basis" << std::endl;
    runTest(w, "gen", "fit", true, 5000);
    TCanvas *c1 = new TCanvas("c1","c1");
    RooPlot *frame = w->var("x")->frame(RooFit::Bins(20));
    w->data("data")->plotOn(frame);
    w->pdf("fit")->plotOn(frame, RooFit::LineColor(kBlue));
    w->pdf("fit_eig")->plotOn(frame, RooFit::LineColor(kRed));
    frame->Draw();
    c1->Print("~/public_html/drop/plot.png");
}

void testThree() {
    RooWorkspace *w = new RooWorkspace("w","w");
    w->factory("x[0,1]"); 
    w->factory("SUM::gen(0.7*Exponential::bg(x, -0.5), Gaussian::pk(x, 0.5, 0.1))");
    w->factory("SUM::fit0(c0[0.5,0.3,1]*Exponential(x, slope[-0.6,-1,0]), Gaussian(x, mu[0.4,0.2,.8], sig[0.1,0.05,0.2]))");
    w->defineSet("obs", "x");
    std::cout << "Fit in current basis" << std::endl;
    runTest(w, "gen", "fit0", true, 5000);
    TCanvas *c1 = new TCanvas("c1","c1");
    RooPlot *frame = w->var("x")->frame(RooFit::Bins(20));
    w->data("data")->plotOn(frame);
    w->pdf("fit0")->plotOn(frame, RooFit::LineColor(kBlue));
    w->pdf("fit0_eig")->plotOn(frame, RooFit::LineColor(kRed));
    frame->Draw();
    c1->Print("~/public_html/drop/plot.png");
}


int main(int argc, char **argv) {
    RooRandom::randomGenerator()->SetSeed(42); 
    RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
    //testOne();
    testTwo();
    testThree();
}
