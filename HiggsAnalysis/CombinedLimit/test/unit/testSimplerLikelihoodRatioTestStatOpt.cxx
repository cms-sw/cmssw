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
#include <RooMsgService.h>
#include <RooStats/RooStatsUtils.h>
#include <RooStats/ModelConfig.h>
#include <RooStats/SimpleLikelihoodRatioTestStat.h>
#include "HiggsAnalysis/CombinedLimit/interface/SimplerLikelihoodRatioTestStatExt.h"
#include "HiggsAnalysis/CombinedLimit/interface/utils.h"

RooWorkspace *w;

int runReuseTS(RooStats::ModelConfig &mc, RooAbsData *data, int tries=0) {
    mc.GetPdf()->graphVizTree("model.dot", "\\n");
    RooArgSet obs(*mc.GetObservables()), poi(*mc.GetParametersOfInterest());
    ((RooRealVar *)poi.first())->setConstant(true);
    ((RooRealVar *)poi.first())->setVal(0);
    RooArgSet snapB; snapB.addClone(poi); 
    if (mc.GetNuisanceParameters()) snapB.addClone(*mc.GetNuisanceParameters());
    ((RooRealVar *)poi.first())->setVal(1.0);
    RooArgSet snapS; snapS.addClone(poi); 
    if (mc.GetNuisanceParameters()) snapS.addClone(*mc.GetNuisanceParameters());
    w->saveSnapshot("start", w->allVars());
    RooStats::SimpleLikelihoodRatioTestStat plainTS(*mc.GetPdf(), *mc.GetPdf(), snapB, snapS);
    //SimplerLikelihoodRatioTestStatExt     plainTS(*mc.GetObservables(), *mc.GetPdf(), *mc.GetPdf(), snapB, snapS);
    SimplerLikelihoodRatioTestStatOpt         optTS(*mc.GetObservables(), *mc.GetPdf(), *mc.GetPdf(), snapB, snapS);
    std::vector<double> plain, opt, plainTimes, optTimes;
    for (int i = 0; i <= tries; ++i) {
        if (i) {
            ((RooRealVar *)poi.first())->setVal(0);
            data = mc.GetPdf()->generate(obs, RooFit::Extended());
            ((RooRealVar *)poi.first())->setVal(1.0);
        }
        TStopwatch timer;
        w->loadSnapshot("start"); timer.Start();
        plain.push_back(plainTS.Evaluate(*data, snapB));
        plainTimes.push_back(timer.RealTime()); 
        w->loadSnapshot("start"); timer.Start();
        opt.push_back(optTS.Evaluate(*data, snapB));
        optTimes.push_back(timer.RealTime()); 
        if (i) delete data;
    }
    std::cout << "Run " << tries << " times with pdf = " << mc.GetPdf()->GetName() << std::endl;
    for (int i = 0; i < plain.size(); ++i) {
        printf("try %4d: plain %+10.5f    opt %+10.5f  (diff %+8.5f)\n", plain[i], opt[i],  opt[i]-plain[i]);
    }
    for (int i = 0; i < plainTimes.size(); ++i) {
	std::cout << " time " << i << ": plain = " << plainTimes[i] << 
                                     ", opt = "   << optTimes[i] << 
                                     ", ratio = "  << plainTimes[i]/optTimes[i] << std::endl;
    }
}

int runPerfTS(bool opt, RooStats::ModelConfig &mc, RooAbsData *data, int tries=0) {
    RooArgSet obs(*mc.GetObservables()), poi(*mc.GetParametersOfInterest());
    ((RooRealVar *)poi.first())->setConstant(true);
    ((RooRealVar *)poi.first())->setVal(0);
    RooArgSet snapB; snapB.addClone(poi);
    ((RooRealVar *)poi.first())->setVal(1.0);
    RooArgSet snapS; snapS.addClone(poi);
    RooStats::TestStatistic *ts = 0;
    if (opt) ts = new SimplerLikelihoodRatioTestStatOpt(*mc.GetObservables(), *mc.GetPdf(), *mc.GetPdf(), snapB, snapS);
    else     ts = new RooStats::SimpleLikelihoodRatioTestStat(*mc.GetPdf(), *mc.GetPdf(), snapB, snapS);
    TStopwatch timer; timer.Stop(); timer.Reset();
    double totalTime = 0.0;
    for (int i = 0; i <= tries; ++i) {
        std::cout << "iter " << i << std::endl;
        if (i) {
            ((RooRealVar *)poi.first())->setVal(0);
            data = mc.GetPdf()->generate(obs, RooFit::Extended());
            ((RooRealVar *)poi.first())->setVal(1.0);
        }
        timer.Start(false);
        double val = ts->Evaluate(*data, snapB);
        timer.Stop();
        if (i) delete data;
    }
    double total = timer.RealTime();
    std::cout << " total: time = " << total  << std::endl;
}


void runExternal(const char *file, int n, const char *wsp, const char *datan, const char *mcn) {
    TFile *f = TFile::Open(file); if (f == 0) return;
    w = (RooWorkspace *) f->Get(wsp); if (w == 0) return;
    RooAbsData *data = w->data(datan); if (data == 0) return;
    RooStats::ModelConfig *mc = (RooStats::ModelConfig *) w->genobj(mcn); if (mc == 0) return;
    runReuseTS(*mc, data, n);
}

void runPerf(const char *opt, const char *n, const char *file, const char *wsp, const char *datan, const char *mcn) {
    TFile *f = TFile::Open(file); if (f == 0) return;
    w = (RooWorkspace *) f->Get(wsp); if (w == 0) return;
    RooAbsData *data = w->data(datan); if (data == 0) return;
    RooStats::ModelConfig *mc = (RooStats::ModelConfig *) w->genobj(mcn); if (mc == 0) return;
    runPerfTS(strstr(opt,"opt") != NULL, *mc, data, atoi(n));
}


int main(int argc, char **argv) {
    RooRandom::randomGenerator()->SetSeed(42);
    RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING);
    if (argc >= 2) {
        if (strstr(argv[1],"root")) {
            runExternal(argv[1], 
                        argc >= 3 ? atoi(argv[2]) : 10,
                        argc >= 4 ? argv[3] : "w",  
                        argc >= 5 ? argv[4] : "data_obs", 
                        argc >= 6 ? argv[5] : "ModelConfig");
        } else if (argc >= 4){
            runPerf(argv[1], argv[2], argv[3],
                    argc >= 5 ? argv[4] : "w",  
                    argc >= 6 ? argv[5] : "data_obs", 
                    argc >= 7 ? argv[6] : "ModelConfig");
        }
    } else {
        printf("usage: \n");
        printf("   - testSimplerLikelihoodRatioTestStatOpt rootfile [ N workspaceName dataName  modelConfig ]\n");
        printf("     run N times both test statistics, and compare values and times\n");
        printf("   - testSimplerLikelihoodRatioTestStatOpt (opt|plain) N rootfile [ workspaceName dataName  modelConfig ]\n");
        printf("     run N times the opt or plain test statistics, and print out total time\n");
        printf("defaults: N = 10, workspaceName = 'w', dataName = 'data_obs', modelConfig = 'ModelConfig'\n");
    }
    return 0;
}
