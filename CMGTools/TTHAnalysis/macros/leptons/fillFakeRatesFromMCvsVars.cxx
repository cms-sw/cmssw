TString gTreePath = "/data/b/botta/TTHAnalysis/trees/TREES_250513_HADD/%s/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root";
void fillFR(TString hist, TString xvar, TString cut, TString pass, TString compName, int maxLep) {
    TDirectory *root = gDirectory;
    TFile *f = TFile::Open(Form(gTreePath.Data(),compName.Data()));
    TTree *tree = (TTree*) f->Get("ttHLepTreeProducerBase");
    root->cd();
    for (int i = 1; i <= maxLep; ++i) {
        TString mycut = cut;   mycut.ReplaceAll("%d", Form("%d",i));
        TString mypass = pass; mypass.ReplaceAll("%d", Form("%d",i));
        tree->Draw(Form("abs(LepGood%d_eta):%s>>+%s_den", i, xvar.Data(), hist.Data()),
                   Form("%s", mycut.Data()));
        tree->Draw(Form("abs(LepGood%d_eta):%s>>+%s_num", i, xvar.Data(), hist.Data()),
                   Form("(%s) && (%s)", mycut.Data(), mypass.Data()));
    }
    f->Close();
    TH2 *den = (TH2*) gROOT->FindObject(hist+"_den"); den->Sumw2(); den->Write();
    TH2 *num = (TH2*) gROOT->FindObject(hist+"_num"); num->Sumw2(); num->Write();
    TH2 *ratio = num->Clone(hist);
    ratio->Divide(num,den,1,1,"B");
    ratio->Write();
}

void fillFakeRatesFromMCvsVars(int triggering=1) {
    TFile *fOut = TFile::Open("fakeRates_TTJets_Vars.root", "RECREATE");

    const int neta_mu = 2, neta_el = 3;
    double etabins_mu[neta_mu+1] = { 0.0, 1.5,   2.5 };
    double etabins_el[neta_el+1] = { 0.0, 0.8, 1.479, 2.5 };

    const int nvtx = 5;
    double vtxbins[nvtx+1] = {  0, 10, 15, 20, 25, 35 };
    const int njet = 4;
    double jetbins[njet+1] = {  2.5, 3.5, 4.5, 5.5, 6.5 };
    const int nbjet = 3;
    double bjetbins[nbjet+1] = {  0.5, 1.5, 2.5, 3.5 };

    const int  nsels = 2;
    const char *sels[nsels] = { "FR_tight", "FRC_tight" };
    const char *scut[nsels] = { "LepGood%d_mva >= 0.70", 
                                "LepGood%d_relIso < 0.12" };
    const int  nlep = 4;
    const char *leps[nlep] = { "el_b", "el_l", "mu_b", "mu_l" };
    const char *lcut[nlep] = { "abs(LepGood%d_pdgId) == 11 && LepGood%d_mcMatchAny == 2",
                               "abs(LepGood%d_pdgId) == 11 && (LepGood%d_mcMatchAny == 1 || LepGood%d_mcMatchAny == 0)",
                               "abs(LepGood%d_pdgId) == 13 && LepGood%d_mcMatchAny == 2",
                               "abs(LepGood%d_pdgId) == 13 && (LepGood%d_mcMatchAny == 1 || LepGood%d_mcMatchAny == 0)" };
    const char *kinds[2] = { "num", "den" };
    for (int is = 0; is < nsels; ++is) {
      for (int k = 0; k < 2; ++k) {
        for (int il = 0; il < nlep; ++il) {
            int     neta    = (il < 2 ? neta_el : neta_mu);
            double *etabins = (il < 2 ? etabins_el : etabins_mu);
            TH2F *FR_vtx = new TH2F(Form("%s_%s_vtx_%s",sels[is],leps[il],kinds[k]),"", nvtx, vtxbins, neta,etabins);
            TH2F *FR_jet = new TH2F(Form("%s_%s_jet_%s",sels[is],leps[il],kinds[k]),"", njet, jetbins, neta,etabins);
            TH2F *FR_bjet = new TH2F(Form("%s_%s_bjet_%s",sels[is],leps[il],kinds[k]),"", nbjet, bjetbins, neta,etabins);
        }
      }
    }

    TString baseCut = "LepGood%d_mcMatchId == 0 &&  minMllAFAS > 12 && LepGood%d_pt > 20 && ";
    baseCut += " (nLepGood == 2 && LepGood1_pdgId*LepGood2_pdgId > 0) && ";
    baseCut += "LepGood%d_innerHits*(abs(LepGood%d_pdgId) == 11) == 0 && "; // require to be zero if the lepton is an electron
    baseCut += "(LepGood%d_convVeto==0)*(abs(LepGood%d_pdgId) == 11) == 0 && ";
    baseCut += " (LepGood%d_tightCharge > (abs(LepGood%d_pdgId) == 11)) && ";
    //baseCutT += "(abs(LepGood%d_pdgId) == 11 || LepGood%d_tightId) && ";
   
    TString sample = "TTJets";
    const char *samples[2] = { "TTJets", "TTJetsSem" };
    for (int id = 0; id < 2; ++id) { 
        for (int is = 0; is < nsels; ++is) {
            for (int il = 0; il < nlep; ++il) {
                sample = TString(samples[id]);

                std::cout << "Processing " << sels[is] << " selection for " << leps[il] << " on " << sample << std::endl;
                fillFR(Form("%s_%s_jet",sels[is],leps[il]), "min(max(nJet25, 3),6)", baseCut + lcut[il], scut[is], sample, 2);
                fillFR(Form("%s_%s_bjet",sels[is],leps[il]), "min(max(nBJetMedium25, 1),3)", baseCut + lcut[il], scut[is], sample, 2);
            }
        }
    }

    fOut->Close();
}
