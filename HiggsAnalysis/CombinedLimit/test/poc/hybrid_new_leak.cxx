#include <RooWorkspace.h>
#include <RooRealVar.h>
#include <RooDataSet.h>
#include <RooStats/ModelConfig.h>
#include <RooStats/HybridCalculator.h>
#include <RooStats/SimpleLikelihoodRatioTestStat.h>
#include <RooStats/ToyMCSampler.h>
#include <TStopwatch.h>
#include <iostream>

// compile with gcc hybrid_new_leak.cxx -o hybrid_new_leak.exe $(root-config --ldflags --cflags --libs)  -lRooFitCore -lRooStats
int main(int argc, char **argv) {
    using namespace RooStats;
    RooWorkspace *w = new RooWorkspace("w","w");
    w->factory("nobs[10,0,1000]");
    w->factory("r[1,0,100]");
    w->factory("Gaussian::nuisPdf(nB[0,20], 10, 1)");
    w->factory("sum::nSB( prod::nS(r,20), nB )");
    w->factory("PROD::model_s(Poisson::model_s_stat(nobs, nSB), nuisPdf)");
    w->factory("PROD::model_b(Poisson::model_b_stat(nobs, nB ), nuisPdf)");
    w->factory("set::nuis(nB)");

    RooArgSet obs(*w->var("nobs")), poi(*w->var("r"));
    RooDataSet data("data","data", obs); data.add(obs);
    w->import(data);


    ModelConfig s("s", w);
    s.SetPdf(*w->pdf("model_s"));
    s.SetObservables(obs);
    s.SetParametersOfInterest(poi);
    s.SetNuisanceParameters(*w->set("nuis"));
    w->var("r")->setVal(1.0);
    s.SetSnapshot(poi);

    ModelConfig b("b", w);
    b.SetPdf(*w->pdf("model_b"));
    b.SetObservables(obs);
    b.SetParametersOfInterest(poi);
    b.SetNuisanceParameters(*w->set("nuis"));
    w->var("r")->setVal(0.0);
    b.SetSnapshot(poi);

    SimpleLikelihoodRatioTestStat q(*w->pdf("model_b"), *w->pdf("model_s"));
    ToyMCSampler toymc(q, argc > 1 ? atoi(argv[1]) : 500);
    toymc.SetNEventsPerToy(1);

    HybridCalculator hc(data, s, b, &toymc);
    hc.ForcePriorNuisanceNull(*w->pdf("nuisPdf"));
    hc.ForcePriorNuisanceAlt(*w->pdf("nuisPdf"));

    std::cout << "Running with " << (argc > 1 ? atoi(argv[1]) : 500) << " toys" << std::endl;
    TStopwatch timer;
    HypoTestResult *result = hc.GetHypoTest();
    timer.Stop();
    std::cout << "Done in " <<  timer.RealTime()/60. << " min " << std::endl;
}

