TString gTreePath = "/afs/cern.ch/user/g/gpetrucc/w/SusyFakes/TREES_SIGNAL_120514/%s/ttHLepTreeProducerSusyFR/ttHLepTreeProducerSusyFR_tree.root";
void fillFR(TString hist, TString cut, TString pass, TString compName, int maxLep) {
    TDirectory *root = gDirectory;
    TFile *f = TFile::Open(Form(gTreePath.Data(),compName.Data()));
    TTree *tree = (TTree*) f->Get("ttHLepTreeProducerSusyFR");
    root->cd();
    for (int i = 0; i < maxLep; ++i) {
        TString mycut = cut;   mycut.ReplaceAll("%d", Form("%d",i));
        TString mypass = pass; mypass.ReplaceAll("%d", Form("%d",i));
        tree->Draw(Form("abs(LepGood_eta[%d]):min(LepGood_pt[%d],49.9)>>+%s_den", i,i, hist.Data()),
                   Form("%s", mycut.Data()));
        tree->Draw(Form("abs(LepGood_eta[%d]):min(LepGood_pt[%d],49.9)>>+%s_num", i,i, hist.Data()),
                   Form("(%s) && (%s)", mycut.Data(), mypass.Data()));
    }
    f->Close();
    TH2 *den = (TH2*) gROOT->FindObject(hist+"_den"); den->Sumw2(); den->Write();
    TH2 *num = (TH2*) gROOT->FindObject(hist+"_num"); num->Sumw2(); num->Write();
    TH2 *ratio = num->Clone(hist);
    ratio->Divide(num,den,1,1,"B");
    ratio->Write();
}

void fillTrivialFakeRatesFromMC(int triggering=1) {
    const int npt_mu = 7, npt_el = 7, neta_mu = 5, neta_el = 5;
    double ptbins_mu[npt_mu+1] = { 10, 15, 20, 25, 30, 35, 45, 50 };
    double ptbins_el[npt_el+1] = { 10, 15, 20, 25, 30, 35, 45, 50 };
    double etabins_mu[neta_mu+1] = { 0.0, 0.5, 1.0, 1.5, 2.0, 2.5 };
    double etabins_el[neta_el+1] = { 0.0, 0.5, 1.0, 1.5, 2.0, 2.5 };


    gROOT->ProcessLine(".L ../../python/plotter/fakeRate.cc+");

    TFile *fOut = TFile::Open(triggering ? "fakeRates_TTJets_MC.root" :  "fakeRates_TTJets_MC_NonTrig.root", "RECREATE");
    //TFile *fOut = TFile::Open("fakeRates_TTLep_MC.root", "RECREATE");
    const int  nsels = 1;
    const char *sels[nsels] = { "FR" };
    for (int is = 0; is < nsels; ++is) {
        TH2F *FR_tight_el_den = new TH2F(Form("%s_tight_el_den",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_tight_el_num = new TH2F(Form("%s_tight_el_num",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_tight_mu_den = new TH2F(Form("%s_tight_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_tight_mu_num = new TH2F(Form("%s_tight_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
    }

    TString baseCut = " ";
    if (triggering) baseCut += "nLepGood10 == 2 && ";
    baseCut += "minMllAFAS > 12 && ";
    baseCut += " (LepGood_pdgId[0]*LepGood_pdgId[1] > 0) && ";

    TString sample = "TTJets";
    const char *samples[7] = { "TTJets", "TTLep", "TtW", "TbartW", "TTJetsLep", "TTJetsSem", "TTJetsHad" };
    for (int id = 0; id < 1; ++id) { 
        sample = TString(samples[id]);
        //if (sample != "T?TJetsSem") continue;
        //fillBaseWeights(?"W_btag_el", baseCut + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 11", "LepGood%d_pt > 10 && LepGood%d_mva < 0.25", sample, 4);
        //fillBaseWeights(?"W_btag_mu", baseCut + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 13", "LepGood%d_pt > 10 && LepGood%d_mva < 0.25", sample, 4);

        std::cout << "Processing MVA selection on " << sample << std::endl;
        fillFR("FR_tight_mu",  baseCut + "LepGood_mcMatchId[%d] == 0 && abs(LepGood_pdgId[%d]) == 13", "LepGood_tightFakeId[%d] >= 0.70", sample, 3);
        fillFR("FR_tight_el",  baseCut + "LepGood_mcMatchId[%d] == 0 && abs(LepGood_pdgId[%d]) == 11", "LepGood_tightFakeId[%d] >= 0.70", sample, 3);

    }

    fOut->Close();
}
