TString gTreePath = "/data/b/botta/TTHAnalysis/trees/TREES_250513_HADD/%s/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root";
void fillFR(TString hist, TString cut, TString pass, TString compName, int maxLep) {
    TDirectory *root = gDirectory;
    TFile *f = TFile::Open(Form(gTreePath.Data(),compName.Data()));
    TTree *tree = (TTree*) f->Get("ttHLepTreeProducerBase");
    root->cd();
    for (int i = 3; i <= maxLep; ++i) {
        TString mycut = cut;   mycut.ReplaceAll("%d", Form("%d",i));
        TString mypass = pass; mypass.ReplaceAll("%d", Form("%d",i));
        tree->Draw(Form("abs(LepGood%d_eta):min(LepGood%d_pt,99.9)>>+%s_den", i,i, hist.Data()),
                   Form("%s", mycut.Data()));
        tree->Draw(Form("abs(LepGood%d_eta):min(LepGood%d_pt,99.9)>>+%s_num", i,i, hist.Data()),
                   Form("(%s) && (%s)", mycut.Data(), mypass.Data()));
    }
    f->Close();
    TH2 *den = (TH2*) gROOT->FindObject(hist+"_den"); den->Sumw2(); den->Write();
    TH2 *num = (TH2*) gROOT->FindObject(hist+"_num"); num->Sumw2(); num->Write();
    TH2 *ratio = num->Clone(hist);
    ratio->Divide(num,den,1,1,"B");
    ratio->Write();
}
void fillBaseWeights(TString hist, TString cut, TString pass, TString compName, int maxLep) {
    TDirectory *root = gDirectory;
    TFile *f = TFile::Open(Form(gTreePath.Data(),compName.Data()));
    TTree *tree = (TTree*) f->Get("ttHLepTreeProducerBase");
    root->cd();
    for (int i = 1; i <= maxLep; ++i) {
        TString mycut = cut;   mycut.ReplaceAll("%d", Form("%d",i));
        TString mypass = pass; mypass.ReplaceAll("%d", Form("%d",i));
        tree->Draw(Form("TMath::Max(TMath::Min(LepGood%d_jetBTagCSV,0.999),0)>>+%s", i, hist.Data()),  mycut.Data());
    }
    TH2 *h = (TH2*) gROOT->FindObject(hist); 
    f->Close();
    h->Write();
}

void fillZFakeRatesFromMC(int withb=0) {
    gROOT->ProcessLine(".L ../../python/plotter/functions.cc+");
    gROOT->ProcessLine(".L ../../python/plotter/fakeRate.cc+");

    const int npt_mu = 7, npt_el = 5, neta_mu = 2, neta_el = 3;
    double ptbins_mu[npt_mu+1] = { 5.0, 6.0, 7.0, 8.5, 10, 15, 20, 30 };
    double ptbins_el[npt_el+1] = {           7.0, 8.5, 10, 15, 20, 30 };
    //double etabins_mu[neta+1] = { 0.0, 0.7, 1.5,   2.0,  2.5 };
    //double etabins_el[neta+1] = { 0.0, 0.7, 1.479, 2.0,  2.5 };
    double etabins_mu[neta_mu+1] = { 0.0, 1.5,   2.5 };
    double etabins_el[neta_el+1] = { 0.0, 0.8, 1.479, 2.5 };

    TFile *fOut = TFile::Open(withb ? "fakeRates_Zb_DYJets_MC.root" : "fakeRates_Z_DYJets_MC.root", "RECREATE");
    const int  nsels = 3;
    const char *sels[nsels] = { "FR", "FRC", "FRH" };
    for (int is = 0; is < nsels; ++is) {
        TH2F *FR_mu_den = new TH2F(Form("%s_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_mu_num = new TH2F(Form("%s_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_el_den = new TH2F(Form("%s_el_den",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_el_num = new TH2F(Form("%s_el_num",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_tight_mu_den = new TH2F(Form("%s_tight_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_tight_mu_num = new TH2F(Form("%s_tight_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_loose_mu_den = new TH2F(Form("%s_loose_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_loose_mu_num = new TH2F(Form("%s_loose_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_tight_el_den = new TH2F(Form("%s_tight_el_den",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_tight_el_num = new TH2F(Form("%s_tight_el_num",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_loose_el_den = new TH2F(Form("%s_loose_el_den",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_loose_el_num = new TH2F(Form("%s_loose_el_num",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
    }
    //TH1 *w_el = new TH1F("W_btag_el", "CSV", 20, 0, 1);
    //TH1 *w_el = new TH1F("W_btag_mu", "CSV", 20, 0, 1);

    TString baseCut = " nLepGood == 3 && LepGood1_pdgId+LepGood2_pdgId == 0 && ";
    baseCut += "minMllAFAS > 12 && abs(mZ1-91.2) < 10 && ";
    //baseCut += "met*0.00397 + mhtJet25*0.00265 < 0.15 && ";
    baseCut += "met*0.00397 + mhtJet25*0.00265 < 0.3 && ";
    baseCut += "mtw_wz3l(LepGood1_pt,LepGood1_eta,LepGood1_phi,LepGood1_mass,LepGood2_pt,LepGood2_eta,LepGood2_phi,LepGood2_mass,LepGood3_pt,LepGood3_eta,LepGood3_phi,LepGood3_mass,mZ1,met,met_phi) < 40 && ";
    baseCut += "abs(mass_2(LepGood1_pt,LepGood1_eta,LepGood1_phi,LepGood1_mass,LepGood2_pt,LepGood2_eta,LepGood2_phi,LepGood2_mass)-mZ1) < 0.01 && ";
    //baseCut += "nBJetMedium25 == 1 && ";
    if (withb) baseCut += "nBJetMedium25 == 1 && ";
    //baseCut += "LepGood%d_mcMatchAny == 2 && ";

    TString baseCutT = baseCut; //;
    baseCutT += "LepGood%d_innerHits*(abs(LepGood%d_pdgId) == 11) == 0 && "; // require to be zero if the lepton is an electron
    baseCutT += "(LepGood%d_convVeto==0)*(abs(LepGood%d_pdgId) == 11) == 0 && ";
    TString baseCutTC = baseCutT + " (LepGood%d_tightCharge > (abs(LepGood%d_pdgId) == 11)) && ";
    TString baseCutTCB = baseCutTC+ " (LepGood%d_sip3d < 4) && (abs(LepGood%d_pdgId) == 11 || LepGood%d_tightId) && (abs(LepGood%d_pdgId) == 13 || passEgammaTightMVA(LepGood%d_pt,LepGood%d_eta,LepGood%d_tightId)) && ";

    TString sample = "";
    const char *samples[4] = { "DYJetsM50", "DY1JetsM50", "DY2JetsM50", "DY3JetsM50" };
    for (int id = 0; id < 4; ++id) { 
        sample = TString(samples[id]);
        //fillBaseWeights("W_btag_el", baseCut + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 11", "LepGood%d_pt > 10 && LepGood%d_mva < 0.25", sample, 4);
        //fillBaseWeights("W_btag_mu", baseCut + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 13", "LepGood%d_pt > 10 && LepGood%d_mva < 0.25", sample, 4);

        std::cout << "Processing MVA selection on " << sample << std::endl;
        fillFR("FR_el", baseCutTC + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 11", "LepGood%d_mva >= -0.3", sample, 3);
        fillFR("FR_mu", baseCutTC + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 13", "LepGood%d_mva >= -0.3", sample, 3);
        fillFR("FR_tight_el", baseCutTC + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 11", "LepGood%d_mva >= 0.70", sample, 3);
        fillFR("FR_tight_mu", baseCutTC + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 13", "LepGood%d_mva >= 0.70", sample, 3);
        fillFR("FR_loose_el", baseCut   + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 11", "LepGood%d_mva >= -0.3", sample, 3);
        fillFR("FR_loose_mu", baseCut   + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 13", "LepGood%d_mva >= -0.3", sample, 3);
        //fillFR("FR_tight2_el",baseCutTC+ "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 11", "LepGood%d_mva >= 0.70", sample, 3);
        fillFR("FRC_tight_el", baseCutTCB + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 11", "LepGood%d_relIso < 0.12", sample, 3);
        fillFR("FRC_tight_mu", baseCutTCB + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 13", "LepGood%d_relIso < 0.12", sample, 3);
#if 0
        std::cout << "Processing cut-based selection on " << sample << std::endl;
        fillFR("FRC_el", baseCut + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 11 && (abs(LepGood%d_eta)<1.4442 || abs(LepGood%d_eta)>1.5660)", "LepGood%d_relIso < 0.25 && LepGood%d_tightId > 0.0 && abs(LepGood%d_dxy) < 0.04 && abs(LepGood%d_innerHits) <= 0", sample, triggering ? 2 : 4);
        fillFR("FRC_mu", baseCut + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 13", "LepGood%d_relIso < 0.2", sample, triggering ? 2 : 4);
        fillFR("FRC_tight_el", baseCutT + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 11 && (abs(LepGood%d_eta)<1.4442 || abs(LepGood%d_eta)>1.5660)", "LepGood%d_relIso < 0.15 && LepGood%d_tightId > 0.0 && abs(LepGood%d_dxy) < 0.02 && abs(LepGood%d_innerHits) <= 0", sample, triggering ? 2 : 4);
        fillFR("FRC_tight_mu", baseCutT + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 13 && abs(LepGood%d_eta) < 2.1", "LepGood%d_relIso < 0.12 && LepGood%d_tightId   && abs(LepGood%d_dxy) < 0.2 && abs(LepGood%d_dz) < 0.5", sample, triggering ? 2 : 4);


        std::cout << "Processing hybrid selection on " << sample << std::endl;
        fillFR("FRH_el", baseCut + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 11", "LepGood%d_relIso < 0.25 && LepGood%d_sip3d < 6 && LepGood%d_jetBTagCSV < 0.3", sample, 4);
        fillFR("FRH_mu", baseCut + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 13", "LepGood%d_relIso < 0.25 && LepGood%d_sip3d < 6 && LepGood%d_jetBTagCSV < 0.3", sample, 4);
        fillFR("FRH_tight_el", baseCutT + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 11", "LepGood%d_relIso < 0.2 && LepGood%d_sip3d < 4 && LepGood%d_jetBTagCSV < 0.3", sample, 4);
        fillFR("FRH_tight_mu", baseCutT + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 13", "LepGood%d_relIso < 0.2 && LepGood%d_sip3d < 4 && LepGood%d_jetBTagCSV < 0.3", sample, 4);
        fillFR("FRH_el", baseCut + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 11", "LepGood%d_sip3d < 10 && LepGood%d_jetBTagCSV < 0.3", sample, 4);
        fillFR("FRH_mu", baseCut + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 13", "LepGood%d_sip3d < 10 && LepGood%d_jetBTagCSV < 0.3", sample, 4);
        fillFR("FRH_tight_el", baseCutT + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 11", "LepGood%d_sip3d < 3 && LepGood%d_jetBTagCSV < 20.2", sample, 4);
        fillFR("FRH_tight_mu", baseCutT + "LepGood%d_mcMatchId == 0 && abs(LepGood%d_pdgId) == 13", "LepGood%d_sip3d < 3 && LepGood%d_jetBTagCSV < 20.2", sample, 4);
#else
#endif
    }

    fOut->Close();
}
