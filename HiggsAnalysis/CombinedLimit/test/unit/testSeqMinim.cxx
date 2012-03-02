#include <cmath>
#include <cstdio>
#include <TMath.h>
#include <TFile.h>
#include <TStopwatch.h>
#include <RooWorkspace.h>
#include <RooRealVar.h>
#include <RooDataSet.h>
#include <RooAbsPdf.h>
#include <RooRandom.h>
#include <RooMinimizer.h>
#include <RooFitResult.h>
#include <RooStats/RooStatsUtils.h>
#include <RooStats/ModelConfig.h>
#include <Math/MinimizerOptions.h>

#include "HiggsAnalysis/CombinedLimit/interface/ProfileLikelihood.h"
#include "HiggsAnalysis/CombinedLimit/interface/RooSimultaneousOpt.h"
#include "HiggsAnalysis/CombinedLimit/interface/SequentialMinimizer.h"
#include "HiggsAnalysis/CombinedLimit/interface/ProfiledLikelihoodRatioTestStatExt.h"
#include "HiggsAnalysis/CombinedLimit/interface/ProfilingTools.h"



RooWorkspace *w;
SequentialMinimizer *smin;

double runSeqMin(RooAbsReal *nll, RooRealVar *poi, bool improve) {
    if (!improve) smin = new SequentialMinimizer(nll,poi);
    std::cout << "Initial NLL is " << nll->getVal() << std::endl;
    bool ret = (improve ? smin->improve() : smin->minimize());
    std::cout << "Final NLL is " << nll->getVal() << std::endl;
    std::cout << "Result is " << (ret ? "OK" : "FAIL") << std::endl;
    return  nll->getVal();
}

double runRooMin(RooAbsReal *nll, RooRealVar *poi, bool improve) {
    std::cout << "Initial NLL is " << nll->getVal() << std::endl;
    RooMinimizer minim(*nll);
    bool ret = nllutils::robustMinimize(*nll, minim, 0);
    std::cout << "Final NLL is " << nll->getVal() << std::endl;
    std::cout << "Result is " << (ret ? "OK" : "FAIL") << std::endl;
    return  nll->getVal();
}

template<typename Func>
double testTwoPoints(Func f, RooAbsReal *nll, RooRealVar *poi, double x0, double x1) {
    poi->setConstant(true);
    poi->setVal(x0);
    bool ok0 = f(nll,NULL,false);
    double y0 = nll->getVal();
    poi->setVal(x1);
    bool ok1 = f(nll,NULL,true);
    double y1 = nll->getVal();
    return 2*(y1-y0);
}

template<typename Func>
double testProfile(Func f, RooAbsReal *nll, RooRealVar *poi, double &xmin, double x1, double &ymin, double &y1) {
    poi->setConstant(false);
    poi->setVal(xmin);
    bool ok0 = f(nll,poi,false);
    xmin = poi->getVal();
    ymin = nll->getVal();
    poi->setConstant(true);
    poi->setVal(x1);
    bool ok1 = f(nll,NULL,false);
    y1 = nll->getVal();
    return 2*(y1-ymin);
}



void runExternal(const char *file, double mh, const char *algo, double tol, const char *wsp, const char *datan, const char *mcn) {
    runtimedef::set("DEBUG_SMD",1);
    ProfileLikelihood::MinimizerSentry cfg(algo,tol);
    TFile *f = TFile::Open(file); if (f == 0) return;
    w = (RooWorkspace *) f->Get(wsp); if (w == 0) return;
    if (w->var("MH")) w->var("MH")->setVal(mh);
    //w->Print("V");
    RooAbsData *data = w->data(datan); if (data == 0) return;
    RooStats::ModelConfig *mc = (RooStats::ModelConfig *) w->genobj(mcn); if (mc == 0) return;
    RooArgSet nuis(*mc->GetNuisanceParameters());

    RooRealVar *poi = (RooRealVar *) mc->GetParametersOfInterest()->first();
    cacheutils::CachingSimNLL nll((RooSimultaneous*)mc->GetPdf(), data, &nuis);
    RooArgSet clean; nuis.snapshot(clean);
    nll.setZeroPoint();
#if 0
    std::cout << "NLL value = " << nll.getVal() << std::endl;
    poi->setVal(0); nuis = clean;
    double dnR = testTwoPoints(runRooMin, &nll, poi, 0, 0.5);
    //double dnR = 0;
    poi->setVal(0); nuis = clean;
    double dnS = testTwoPoints(runSeqMin, &nll, poi, 0, 0.5);
   
    std::cout << "dnR = " << dnR << ", dnS = " << dnS << ", diff = " << dnR - dnS << std::endl; 
#else

    double xminR = 1.0, xminS = 1.0, yminR = 0, yminS, y1R = 0, y1S, q0R = 0, q0S;
#ifdef CMP
    nuis = clean;
    q0R = testProfile(runRooMin, &nll, poi, xminR, 0.0, yminR, y1R);
#endif
    nuis = clean;
    q0S = testProfile(runSeqMin, &nll, poi, xminS, 0.0, yminS, y1S);
    printf("RooFit     minimizer: minimum at r = %.6f, NLL(min) = %+16.8f, NLL(0) = %+16.8f, q0 = %8.5f\n", xminR, yminR, y1R, q0R);
    printf("Sequential minimizer: minimum at r = %.6f, NLL(min) = %+16.8f, NLL(0) = %+16.8f, q0 = %8.5f\n", xminS, yminS, y1S, q0S);
#endif
}

void runPerf(const char *opt, const char *n, const char *file, const char *wsp, const char *datan, const char *mcn) {
/*
    TFile *f = TFile::Open(file); if (f == 0) return;
    w = (RooWorkspace *) f->Get(wsp); if (w == 0) return;
    RooAbsData *data = w->data(datan); if (data == 0) return;
    RooStats::ModelConfig *mc = (RooStats::ModelConfig *) w->genobj(mcn); if (mc == 0) return;
    //runPerf(strstr(opt,"opt") != NULL, *mc, data, atoi(n));
    runPerfMulti(strstr(opt,"opt") != NULL, *mc, data, atoi(n));
*/
}


int main(int argc, char **argv) {
    RooRandom::randomGenerator()->SetSeed(42);
    if (argc >= 2) {
        if (strstr(argv[1],"root")) {
            runExternal(argv[1], 
                        argc >= 3 ? atof(argv[2]) : 120.,  
                        argc >= 4 ? argv[3] : "Minuit2",  
                        argc >= 5 ? atof(argv[4]) : 0.0001,  
                        "w",  
                        "data_obs", 
                        "ModelConfig");
        } else if (argc >= 4){
            runPerf(argv[1], argv[2], argv[3],
                    argc >= 5 ? argv[4] : "w",  
                    argc >= 6 ? argv[5] : "data_obs", 
                    argc >= 7 ? argv[6] : "ModelConfig");
        }
    }
    return 0;
}
