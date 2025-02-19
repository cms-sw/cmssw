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
#include <RooStats/RatioOfProfiledLikelihoodsTestStat.h>
#include "HiggsAnalysis/CombinedLimit/interface/CachingNLL.h"
#include "HiggsAnalysis/CombinedLimit/interface/utils.h"
#include "HiggsAnalysis/CombinedLimit/interface/ProfiledLikelihoodRatioTestStatExt.h"
#include "HiggsAnalysis/CombinedLimit/interface/VerticalInterpHistPdf.h"
#include "HiggsAnalysis/CombinedLimit/interface/RooSimultaneousOpt.h"

RooWorkspace *w;


void testFastHistPdf(VerticalInterpHistPdf &slow, FastVerticalInterpHistPdf &fast, RooAbsData &data, RooAbsData &nuisdata) 
{
    RooArgSet * obs  = slow.getObservables(data);
    RooArgSet * pars = slow.getParameters(data);
    unsigned int ntry = 0, nfail = 0;
    for (int j = 0, nj = nuisdata.numEntries(); j < nj; ++j) {
        *pars = *nuisdata.get(j);
        for (int i = 0, n = data.numEntries(); i < n; ++i) {
            *obs = *data.get(i);
            double yslow = slow.getVal(obs), yfast = fast.getVal(obs);
            double ok = fabs((yslow - yfast)/(yslow+yfast+1e-2)) < 1e-5;
            ntry++;
            if (!ok) {
                printf("slow %8.4f, fast %8.4f, reldiff %6.4f\n", yslow, yfast, fabs((yslow-yfast)/(yslow+yfast+0.01)));
                nfail++;
            }
        }
    }
    printf("%u attempts, %u failures\n", ntry, nfail);
    delete obs;
    delete pars;
}

template<typename PdfT>
double testPerformances(PdfT &pdf, RooAbsData &data, RooAbsData &nuisdata) {
    RooArgSet * obs  = pdf.getObservables(data);
    RooArgSet * pars = pdf.getParameters(data);
    unsigned int ntry = 0;
    double dummy = 0;
    TStopwatch timer; timer.Start();
    for (int j = 0, nj = nuisdata.numEntries(); j < nj; ++j) {
        *pars = *nuisdata.get(j);
        for (int i = 0, n = data.numEntries(); i < n; ++i) {
            *obs = *data.get(i);
            double ypdf = pdf.getVal(obs);
            ntry++;
        }
    }
    double time = timer.RealTime();
    printf("%u attempts, total time %8.3f s for %s\n", ntry, time, pdf.ClassName());
    return time;
}

void testPdfs(RooStats::ModelConfig &mc, RooAbsData *data, int tries, bool performances) {
    RooAbsPdf *nuispdf = utils::makeNuisancePdf(mc);
    RooAbsData *nuisdata = nuispdf->generate(*mc.GetNuisanceParameters(), tries);
    RooArgList allPdfs(mc.GetWS()->allPdfs());
    const RooArgSet  *obs = data->get();
    for (int i = 0; i < allPdfs.getSize(); ++i) {
        RooAbsPdf *pdf = (RooAbsPdf *) allPdfs.at(i);
        if (pdf->dependsOn(*obs) && pdf->InheritsFrom("VerticalInterpHistPdf")) {
            std::cout << "Will use pdf " << pdf->GetName() << " (" << pdf->ClassName() << ")" << std::endl;
            VerticalInterpHistPdf &oldpdf = dynamic_cast<VerticalInterpHistPdf &>(*pdf);
            FastVerticalInterpHistPdf newpdf(oldpdf);
            if (performances) {
                double tslow = testPerformances(oldpdf,*data,*nuisdata);
                double tfast = testPerformances(newpdf,*data,*nuisdata);
                printf("total time: slow %8.3f s, fast %8.3f s, ratio %5.2f\n", tslow, tfast, tslow/tfast);
            } else {
                testFastHistPdf(oldpdf, newpdf, *data, *nuisdata);
            }
        }
    }
}

void run(const char *file, int n, const char *wsp, const char *datan, const char *mcn) {
    TFile *f = TFile::Open(file); if (f == 0) return;
    w = (RooWorkspace *) f->Get(wsp); if (w == 0) return;
    if (w->var("MH")) w->var("MH")->setVal(130.);
    //w->Print("V");
    RooAbsData *data = w->data(datan); if (data == 0) return;
    RooStats::ModelConfig *mc = (RooStats::ModelConfig *) w->genobj(mcn); if (mc == 0) return;
    testPdfs(*mc, data, abs(n), n < 0);  
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
            run(argv[1], 
                        argc >= 3 ? atoi(argv[2]) : 10,  
                        argc >= 4 ? argv[3] : "w",  
                        argc >= 5 ? argv[4] : "data_obs", 
                        argc >= 6 ? argv[5] : "ModelConfig");
        } 
    }
    return 0;
}
