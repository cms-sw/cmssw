#include <RooWorkspace.h>
#include <RooRealVar.h>
#include <RooDataSet.h>
#include <RooStats/HybridCalculatorOriginal.h>
#include <RooStats/HybridResult.h>
#include <TGraphAsymmErrors.h>
#include <TFile.h>
#include <TStopwatch.h>
#include <iostream>
#include <sys/types.h>
#include <unistd.h>


// compile with gcc hybrid_old_scan.cxx -o hybrid_old_scan.exe $(root-config --ldflags --cflags --libs)  -lRooFitCore -lRooStats
int main(int argc, char **argv) {
    using namespace RooStats;
    RooWorkspace *w = new RooWorkspace("w","w");
    w->factory("nobs[10,0,1000]");
    w->factory("r[1,0,100]");
    w->factory("Gaussian::nuisPdf(nB[0,20], 10, 1)");
    w->factory("sum::nSB( prod::nS(r,5), nB )");
    w->factory("PROD::model_s(Poisson::model_s_stat(nobs, nSB), nuisPdf)");
    w->factory("PROD::model_b(Poisson::model_b_stat(nobs, nB ), nuisPdf)");
    w->factory("set::nuis(nB)");

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
    double r = 1.5;
    TGraphAsymmErrors *asme = new TGraphAsymmErrors(20);
    for (int i = 0; i < 20; ++i) {
        r += 0.01; w->var("r")->setVal(r);
        timer.Start();
        HybridResult *result = hc->GetHypoTest();
        asme->SetPoint(i, r, result->CLs());
        asme->SetPointError(i, 0, 0, result->CLsError(), result->CLsError());
        timer.Stop();
        std::cout << "Memory: " << std::flush; 
        char buff[255]; sprintf(buff,"awk '{ORS=\"\"; print $23/1024/1024}' /proc/%d/stat", getpid());
        system(buff);
        std::cout << " MB. Time " <<  timer.RealTime() << " sec " << std::endl;
    }
    TFile *out = TFile::Open("hybrid_old_scan.root", "RECREATE");
    out->WriteTObject(asme, "scan");
    out->Close();
}

