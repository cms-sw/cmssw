TString gTreePath = "/data/gpetrucc/8TeV/ttH/TREES_270413_HADD/%s/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root";
TFile *fOut = 0;

void fillDen(TString hist, TString cut, TString compName, int maxJets) {
    TDirectory *root = gDirectory;
    TFile *f = TFile::Open(Form(gTreePath.Data(),compName.Data()));
    TTree *tree = (TTree*) f->Get("ttHLepTreeProducerBase");
    fOut->cd();
    for (int i = 1; i <= maxJets; ++i) {
        TString mycut = cut;   mycut.ReplaceAll("%d", Form("%d",i));
        tree->Draw(Form("abs(Jet%d_eta):min(Jet%d_pt,149.9)>>+%s", i,i, hist.Data()),
                   Form("%s", mycut.Data()));
    }
    f->Close();
}
void fillNum(TString hist, TString cut, TString pass, TString compName, int maxJets) {
    TDirectory *root = gDirectory;
    TFile *f = TFile::Open(Form(gTreePath.Data(),compName.Data()));
    TTree *tree = (TTree*) f->Get("ttHLepTreeProducerBase");
    fOut->cd();
    for (int i = 1; i <= maxJets; ++i) {
        TString mycut = cut;   mycut.ReplaceAll("%d", Form("%d",i));
        TString mypass = pass; mypass.ReplaceAll("%d", Form("%d",i));
        tree->Draw(Form("abs(Jet%d_eta):min(Jet%d_pt,149.9)>>+%s", i,i, hist.Data()),
                   Form("(%s) && (%s)", mycut.Data(), mypass.Data()));
    }
    f->Close();
}
void fillEff(TString hist, TString histDen, TString histNum) {
    fOut->cd();
    TH2 *den = (TH2*) gROOT->FindObject(histDen); 
    TH2 *num = (TH2*) gROOT->FindObject(histNum); 
    if (histNum.Contains("CSVL")) den->Write();
    num->Write();
    TH2 *ratio = num->Clone(hist);
    ratio->Divide(num,den,1,1,"B");
    ratio->Write();
}

void fillTrivialEfficienciessFromMC() {
    const int npt = 8, neta = 10;
    double ptbins[npt+1] = { 25, 30, 40, 50, 60, 75, 90, 120, 150 };
    double etabins[neta+1] = { 0.0, 0.5, 1.0, 1.5, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 5.0 };

    fOut = TFile::Open("bTagEff_TTMC.root", "RECREATE");
    const int  nsels = 3;
    const char *sels[nsels] = { "CSVL", "CSVM", "CSVT" };
    const float selsop[nsels] = { 0.244, 0.679, 0.898 };
    const int  nflavs = 3;
    const char *flavs[nflavs] = { "b", "c", "l" };
    const int  iflavs[nflavs] = {  5,   4,   0  };
    for (int ifl = 0; ifl < nflavs; ++ifl) {
        TH2F *eff_den = new TH2F(Form("%s_den",flavs[ifl]),"",npt,ptbins,neta,etabins);
        eff_den->Sumw2();
        for (int is = 0; is < nsels; ++is) {
            TH2F *eff_num = new TH2F(Form("%s_num_%s",flavs[ifl],sels[is]),"",npt,ptbins,neta,etabins);
            eff_num->Sumw2();
        }
    }
    TString sample = "TTJets";
    const char *samples[4] = { "TTJets", "TTLep", "TtW", "TbartW" };
    for (int id = 0; id < 4; ++id) { 
        sample = TString(samples[id]);

        for (int ifl = 0; ifl < nflavs; ++ifl) {
            std::cout << "Processing denominator for " << flavs[ifl] << " on " << sample << std::endl;

            TString baseCut = Form("Jet%%d_pt > 0 && Jet%%d_mcFlavour == %d ", iflavs[ifl]);
            fillDen(Form("%s_den",flavs[ifl]), baseCut, sample, 8);

            for (int is = 0; is < nsels; ++is) {
                std::cout << "Processing numerator " << sels[is] << " for " << flavs[ifl] << " on " << sample << std::endl;
                TString numCut = Form("Jet%%d_btagCSV > %f", selsop[is]);
                fillNum(Form("%s_num_%s",flavs[ifl],sels[is]), baseCut, numCut, sample, 8);
            }
        }
    }

    for (int ifl = 0; ifl < nflavs; ++ifl) {
        for (int is = 0; is < nsels; ++is) {
            std::cout << "Processing efficiencies for " << sels[is] << " for " << flavs[ifl] << std::endl;
            fillEff(Form("%s_eff_%s",flavs[ifl],sels[is]), Form("%s_den",flavs[ifl]), Form("%s_num_%s",flavs[ifl],sels[is]));
        }
    }
    fOut->Close();
}
