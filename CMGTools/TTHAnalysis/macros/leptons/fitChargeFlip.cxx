#include <cmath>
#include <cstdio>
#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2F.h>
#include <TLorentzVector.h>
#include <Math/Functor.h>
#include <Minuit2/Minuit2Minimizer.h>

const int nbins = 6;
double nss[nbins][nbins], nos[nbins][nbins];

double matchv(const double *x, bool verbose) {
    double w = 0;
    for (int i = 0; i < nbins; ++i) {
        for (int j = 0; j < nbins; ++j) {
            if (nos[i][j] == 0) continue;
            double nobs  = nss[i][j];
            double npred = nos[i][j]*(x[i]+x[j]);
            double den   = std::sqrt(npred); //std::max(std::sqrt(npred),0.05*npred);
            //double den   = std::max(std::sqrt(npred),0.05*npred);
            double inc = std::pow((nobs-npred)/den,2);
            w += inc;
            if (verbose) {
                printf("   bin (%d,%d): sum cc id %.5f, pred = %7.1f, obs = %7.1f, chi2 = %6.2f\n", i,j,(x[i]+x[j]),npred,nobs,inc);
            }
        }
    }
    printf("chi2 for %.5f/%.5f/%.5f/%.5f/%.5f = %7.1f\n", x[0],x[1],x[2],x[3],x[4], w);
    return w;
}
double match(const double *x) {
    return matchv(x,0);
}

void fitChargeFlip(TString process="DY") {
    TFile *fos = (TFile*) gROOT->GetListOfFiles()->At(0);
    TFile *fss = (TFile*) gROOT->GetListOfFiles()->At(1);
    TH1 *hss = (TH1*) fss->Get("bin_"+process);
    TH1 *hos = (TH1*) fos->Get("bin_"+process);
    for (int i = 0; i < nbins*nbins; ++i) {
        nss[i/nbins][i%nbins] = hss->GetBinContent(i+1);
        nos[i/nbins][i%nbins] = hos->GetBinContent(i+1);
    }

    ROOT::Minuit2::Minuit2Minimizer min ( ROOT::Minuit2::kMigrad );
    min.SetMaxFunctionCalls(1000000);
    min.SetMaxIterations(100000);
    min.SetTolerance(0.001);

    ROOT::Math::Functor f(&match,nbins); 
    min.SetFunction(f);
    for (int i = 0; i < nbins; ++i) {
        min.SetLimitedVariable(i,Form("bin%d",i),i*0.0001,1e-5,0,0.1);
    }
    min.SetErrorDef(1.0);
    min.Minimize();

    const double *xs = min.X(), *xe = min.Errors(); 
    matchv(xs,1);
    for (int i = 0; i < nbins; ++i) {
        printf("Charge flip in bin %d: %.5f +/- %.5f\n", i, xs[i], xe[i]);
    }

    TFile *fOut = new TFile(Form("QF_%s_el.root",process.Data()),"RECREATE");
    const int npt = 3, neta = 2;
    double ptbins[npt+1] = { 5.0, 20.0, 50.0, 100.0 };
    double etabins[neta+1] = { 0.0, 1.479, 2.5 };

    TH2F *QF_el = new TH2F("QF_el_"+process,"",npt,ptbins,neta,etabins);
    QF_el->SetBinContent(1,1, xs[0]); // low pt barrel
    QF_el->SetBinContent(2,1, xs[1]); // high pt barrel
    QF_el->SetBinContent(3,1, xs[2]); // high pt barrel (same as above)
    QF_el->SetBinContent(1,2, xs[3]); // low pt endcap
    QF_el->SetBinContent(2,2, xs[4]); // medium pt endcap
    QF_el->SetBinContent(3,2, xs[5]); // high pt endcap

    QF_el->SetBinError(1,1, xe[0]); // low pt barrel
    QF_el->SetBinError(2,1, xe[1]); // high pt barrel
    QF_el->SetBinError(3,1, xe[2]); // high pt barrel (same as above)
    QF_el->SetBinError(1,2, xe[3]); // low pt endcap
    QF_el->SetBinError(2,2, xe[4]); // medium pt endcap
    QF_el->SetBinError(3,2, xe[5]); // high pt endcap

    QF_el->Write();
    printf("Written to %s\n", fOut->GetName());
    fOut->Close();

 
}
