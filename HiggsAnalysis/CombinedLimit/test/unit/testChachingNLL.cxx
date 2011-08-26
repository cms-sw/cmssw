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
#include "HiggsAnalysis/CombinedLimit/interface/RooSimultaneousOpt.h"

RooWorkspace *w;

void testChecker(RooStats::ModelConfig &mc) {
    const RooArgSet *nuis = mc.GetNuisanceParameters();
    cacheutils::ArgSetChecker check(*nuis);
    RooArgList nuisParams(*nuis);
    for (int i = 0; i < 30; ++i) {
        std::cout << "Test #1: " << (check.changed() == false ? "OK" : "FAIL") << std::endl; 
        RooRealVar *v = (RooRealVar *) nuisParams.at(i %  nuisParams.getSize());
        if (v->isConstant()) continue;
        //std::cout << "Changing " << v->GetName()  << std::endl;
        v->setVal(v->getVal() +  0.0001);
        std::cout << "Test #2: "<< (check.changed()  != false ? "OK" : "FAIL") << std::endl; // must change
        std::cout << "Test #3: "<< (check.changed()  != false ? "OK" : "FAIL") << std::endl; // must still be changed
        std::cout << "Test #4: "<< (check.changed(1) != false ? "OK" : "FAIL") << std::endl; // again (but should fix)
        std::cout << "Test #5: "<< (check.changed()  == false ? "OK" : "FAIL") << std::endl; // should be ok
        Double_t backup =  v->getVal();
        v->setVal(v->getVal() +  0.0001);
        std::cout << "Test #6: "<< (check.changed()  != false ? "OK" : "FAIL") << std::endl; // must change
        v->setVal(backup);
        std::cout << "Test #7: "<< (check.changed()  == false ? "OK" : "FAIL") << std::endl; // should be ok
    }
}

void testCachingPdf(RooAbsPdf *pdf, RooAbsData *data) 
{
    const RooArgSet  *obs = data->get();
    RooArgSet *params = pdf->getVariables();
    std::vector<Double_t> refvals;
    for (int i = 0; i < data->numEntries(); ++i) {
        //refvals.push_back(0); continue;
        *params = *data->get(i);
        refvals.push_back(pdf->getVal(obs));
    }
    cacheutils::CachingPdf cpdf(pdf, data->get());
    const std::vector<Double_t> &vals = cpdf.eval(*data);
    for (int i = 0; i < data->numEntries(); ++i) {
        std::cout << "plain: " << refvals[i] << ", opt: " << vals[i] << ", " << (refvals[i] == vals[i] ? "OK" : "FAIL") << std::endl;
    }
    const std::vector<Double_t> &vals2 = cpdf.eval(*data);
    for (int i = 0; i < data->numEntries(); ++i) {
        std::cout << "plain: " << refvals[i] << ", opt: " << vals2[i] << ", " << (refvals[i] == vals2[i] ? "OK" : "FAIL") << std::endl;
    }
    
}

void testCachingAddNLL(RooAddPdf *pdf, RooAbsData *data) 
{
    RooAbsReal *nll = pdf->createNLL(*data);
    RooArgList params(*pdf->getParameters(*data));
    cacheutils::CachingAddNLL onll("","", pdf, data);
    std::vector<double> plain, alter;
    plain.push_back(nll->getVal()); 
    alter.push_back(onll.getVal());
    for (int i = 0; i < 30; ++i) {
        RooRealVar *v = (RooRealVar *) params.at(i %  params.getSize());
        if (v->isConstant()) continue;
        v->setVal(v->getVal() +  0.05);
        plain.push_back(nll->getVal()); 
        alter.push_back(onll.getVal());
    }
    for (int i = 0,  n = plain.size(); i < n; ++i) {
        printf("plain % 10.4f  alter % 10.4f   diff % 10.4f\n", plain[i], alter[i], alter[i]-plain[i]);
    }
}

void testCachingSimNLL(RooSimultaneous *pdf, RooAbsData *data, RooArgSet *constraints, int nattempts) 
{
    RooAbsReal *nll = pdf->createNLL(*data, RooFit::Constrain(*constraints));
    RooArgList params(*pdf->getParameters(*data));
    TStopwatch timer;
    double plainTime = 0, alterTime = 0;
    cacheutils::CachingSimNLL onll(pdf, data, constraints);
    std::vector<double> plain, alter; 
    timer.Start(); plain.push_back(nll->getVal()); plainTime += timer.RealTime();
    timer.Start(); alter.push_back(onll.getVal()); alterTime += timer.RealTime();
    for (int i = 0; i < nattempts; ++i) {
        RooRealVar *v = (RooRealVar *) params.at(i %  params.getSize());
        if (v->isConstant()) continue;
        double val = v->getVal();
        v->randomize();
        timer.Start(); plain.push_back(nll->getVal()); plainTime += timer.RealTime();
        timer.Start(); alter.push_back(onll.getVal()); alterTime += timer.RealTime();
        v->setVal(val);
    }
    for (int i = 0,  n = plain.size(); i < n; ++i) {
        printf("plain % 10.4f  alter % 10.4f   diff % 10.4f\n", plain[i], alter[i], alter[i]-plain[i]);
    }
    std::cout << "Run " << plain.size() << " times with pdf = " << pdf->GetName() << std::endl;
    std::cout << "   plain = " << plainTime/plain.size() << " s/eval" << std::endl;
    std::cout << "   alter = " << alterTime/plain.size() << " s/eval" << std::endl;
    std::cout << "   ratio = " << plainTime/alterTime << std::endl;

}

template<int ID>
RooFitResult *fitNLL(RooAbsReal &nll) {
    RooMinimizer minim(nll);
    minim.setStrategy(0);
    minim.setPrintLevel(0);
    minim.minimize("Minuit2","migrad");
    return minim.save();
}

void testCachingSimFit(RooSimultaneous *pdf, RooAbsData *data, RooArgSet *constraints, int nattempts) 
{
    RooArgSet *allParams = pdf->getParameters(*data);
    RooArgSet savParams; allParams->snapshot(savParams);
    double plainTime = 0, alterTime = 0;
    TStopwatch timer;
    RooFitResult *plainFit, *alterFit;
    cacheutils::CachingSimNLL onll(pdf, data, constraints);
    for (int i = 0; i < nattempts; ++i) {
        *allParams = savParams;
        timer.Start();
        RooAbsReal *nll = pdf->createNLL(*data, RooFit::Constrain(*constraints));
        plainFit = fitNLL<0>(*nll);
        delete nll;
        plainTime += timer.RealTime();

        *allParams = savParams;
        timer.Start();
        onll.setData(*data);
        alterFit = fitNLL<1>(onll);
        alterTime += timer.RealTime();
    } 
    std::cout << "Done fits. plain fit result: " << std::endl; plainFit->Print("V");
    std::cout << "Done fits. alter fit result: " << std::endl; alterFit->Print("V");
    std::cout << "   plain = " << plainTime/nattempts << " s/eval" << std::endl;
    std::cout << "   alter = " << alterTime/nattempts << " s/eval" << std::endl;
    std::cout << "   ratio = " << plainTime/alterTime << std::endl;

}

void testCachingSimTestStat(RooStats::ModelConfig &mc, RooAbsData *data, int nattempts) 
{
    RooSimultaneous *pdf = (RooSimultaneous *) mc.GetPdf();
    if (nattempts > 0) pdf = new RooSimultaneousOpt(*pdf, "opt");
    RooArgSet snap;
    mc.GetNuisanceParameters()->snapshot(snap);
    mc.GetParametersOfInterest()->snapshot(snap);
    snap.setRealValue("r", 1);
    snap.Print("V");
    ProfiledLikelihoodTestStatOpt testStat(*mc.GetObservables(), *pdf, mc.GetNuisanceParameters(), snap, RooArgList(), RooArgList(), 0);
    std::cout << "value: " << testStat.Evaluate(*data, snap) << std::endl;
}


void testCachingPdf(RooStats::ModelConfig &mc, RooAbsData *data, int tries=0) {
    RooArgList allPdfs(mc.GetWS()->allPdfs());
    const RooArgSet  *obs = data->get();
    for (int i = 0; i < allPdfs.getSize(); ++i) {
        RooAbsPdf *pdf = (RooAbsPdf *) allPdfs.at(i);
        if (pdf->dependsOn(*obs) && 
            !pdf->InheritsFrom("RooSimultaneous") && !pdf->InheritsFrom("RooAddPdf") && !pdf->InheritsFrom("RooProdPdf")) {
            std::cout << "Will use pdf " << pdf->GetName() << " (" << pdf->ClassName() << ")" << std::endl;
            testCachingPdf(pdf, data);
            if (tries-- == 0) break;
        } else {
            //std::cout << "Will NOT use pdf " << pdf->GetName() << " (" << pdf->ClassName() << ")" << std::endl;
        }
    }
}

void testCachingAddNLL(RooStats::ModelConfig &mc, RooAbsData *data, int tries=0) {
    RooArgList allPdfs(mc.GetWS()->allPdfs());
    const RooArgSet  *obs = data->get();
    for (int i = 0; i < allPdfs.getSize(); ++i) {
        RooAbsPdf *pdf = (RooAbsPdf *) allPdfs.at(i);
        if (pdf->dependsOn(*obs) && pdf->InheritsFrom("RooAddPdf")) {
            std::cout << "Will use pdf " << pdf->GetName() << " (" << pdf->ClassName() << ")" << std::endl;
            testCachingAddNLL((RooAddPdf*)pdf, data);
            if (tries-- == 0) break;
        } else {
            //std::cout << "Will NOT use pdf " << pdf->GetName() << " (" << pdf->ClassName() << ")" << std::endl;
        }
    }
}

void testCachingSimNLL(RooStats::ModelConfig &mc, RooAbsData *data, int tries=0) {
    RooArgSet nuis(*mc.GetNuisanceParameters());
    RooArgList allPdfs(mc.GetWS()->allPdfs());
    const RooArgSet  *obs = data->get();
    for (int i = 0; i < allPdfs.getSize(); ++i) {
        RooAbsPdf *pdf = (RooAbsPdf *) allPdfs.at(i);
        if (pdf->dependsOn(*obs) && pdf->InheritsFrom("RooSimultaneous")) {
            std::cout << "Will use pdf " << pdf->GetName() << " (" << pdf->ClassName() << ")" << std::endl;
            if (tries > 0) testCachingSimNLL((RooSimultaneous*)pdf, data, &nuis, tries);
            else testCachingSimFit((RooSimultaneous*)pdf, data, &nuis, 1-tries);
            //if (tries-- == 0) break;
        } else {
            //std::cout << "Will NOT use pdf " << pdf->GetName() << " (" << pdf->ClassName() << ")" << std::endl;
        }
    }
}
/*
    RooArgSet obs(*mc.GetObservables()), poi(*mc.GetParametersOfInterest());
    RooArgSet nuisSet; if (nuis) nuisSet.add(*nuis);
    ((RooRealVar *)poi.first())->setConstant(true);
    OptimizedSimNLL nll(obs, nuisSet, mc.GetPdf());
    nll.setData(*data);
    RooArgList nuisParams(nuisSet);
    double val = nll.getVal(0);
    for (int i = 0; i < nuisParams.getSize(); ++i) {
        RooRealVar *v = (RooRealVar *) nuisParams.at(i);
        v->setVal(v->getVal() +  0.0001);
        std::cout << "Changed " << v->GetName() << std::endl;
        val = nll.getVal(0);
    }
*/

void runExternal(const char *file, int n, const char *wsp, const char *datan, const char *mcn) {
    TFile *f = TFile::Open(file); if (f == 0) return;
    w = (RooWorkspace *) f->Get(wsp); if (w == 0) return;
    if (w->var("MH")) w->var("MH")->setVal(130.);
    w->Print("V");
    RooAbsData *data = w->data(datan); if (data == 0) return;
    RooStats::ModelConfig *mc = (RooStats::ModelConfig *) w->genobj(mcn); if (mc == 0) return;
    //testChecker(*mc);  
    //testCachingPdf(*mc, data, n);  
    //testCachingAddNLL(*mc, data, n);  
    testCachingSimNLL(*mc, data, n);  
    //testCachingSimTestStat(*mc, data, n);  
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
    }
    return 0;
}
