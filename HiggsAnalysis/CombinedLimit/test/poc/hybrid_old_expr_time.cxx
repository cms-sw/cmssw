#include <RooWorkspace.h>
#include <RooRealVar.h>
#include <RooDataSet.h>
#include <RooStats/HybridCalculatorOriginal.h>
#include <RooStats/HybridResult.h>
#include <TStopwatch.h>
#include <iostream>

// compile with gcc hybrid_old_expr_time.cxx -o hybrid_old_expr_time.exe $(root-config --ldflags --cflags --libs)  -lRooFitCore -lRooStats
int main(int argc, char **argv) {
    using namespace RooStats;
    RooWorkspace *w = new RooWorkspace("w","w");
    w->factory("nobs[10,0,1000]");
    w->factory("r[1,0,100]");
    w->factory("Gaussian::nuisPdf(thetaB[-5,5], 0, 1)");
    w->factory("expr::nB('10*pow(1.1,thetaB)', thetaB)");
    w->factory("expr::nSB('10*pow(1.1,thetaB) + r*20', thetaB, r)");
    w->factory("PROD::model_s(Poisson::model_s_stat(nobs, nSB), nuisPdf)");
    w->factory("PROD::model_b(Poisson::model_b_stat(nobs, nB ), nuisPdf)");
    w->factory("set::nuis(thetaB)");

    RooArgSet obs(*w->var("nobs"));
    RooDataSet data("data","data", obs); data.add(obs);
    w->import(data);

    HybridCalculatorOriginal* hc = new HybridCalculatorOriginal(data,*w->pdf("model_s"),*w->pdf("model_b"));
    hc->UseNuisance(true);
    hc->SetNuisancePdf(*w->pdf("nuisPdf"));
    hc->SetNuisanceParameters(*w->set("nuis"));
    hc->SetTestStatistic(1);
    w->var("r")->setConstant(true);
    hc->PatchSetExtended(false);
    hc->SetNumberOfToys(argc > 1 ? atoi(argv[1]) : 500);

    std::cout << "Running with " << (argc > 1 ? atoi(argv[1]) : 500) << " toys" << std::endl;
    TStopwatch timer;
    HybridResult *result = hc->GetHypoTest();
    timer.Stop();
    std::cout << "Done in " <<  timer.RealTime()/60. << " min " << std::endl;
}

