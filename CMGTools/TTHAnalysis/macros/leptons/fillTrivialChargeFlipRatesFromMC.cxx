TString gTreePath       = "/data/gpetrucc/8TeV/ttH/TREES_270213_HADD/%s/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root";
TString gFriendTreePath = "/data/gpetrucc/8TeV/ttH/TREES_270213_HADD/0_leptonMVA_v3/lepMVAFriend_%s.root";
void fillFR(TString hist, TString cut, TString pass, TString compName, int maxLep) {
    TDirectory *root = gDirectory;
    TFile *f = TFile::Open(Form(gTreePath.Data(),compName.Data()));
    TTree *tree = (TTree*) f->Get("ttHLepTreeProducerBase");
    tree->AddFriend("newMVA/t", Form(gFriendTreePath.Data(), compName.Data()));
    //tree->AddFriend("newMC/t", Form(gMCFriendTreePath.Data(), compName.Data()));
    root->cd();
    for (int i = 1; i <= maxLep; ++i) {
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

void fillTrivialChargeFlipRatesFromMC() {
#if 1
    const int npt = 3, neta = 2;
    //double ptbins[npt+1] = { 5.0, 7.5, 10.0, 12.5, 15.0, 20, 25.0, 30, 40, 60, 100.0 };
    double ptbins[npt+1] = { 5.0, 20.0, 50.0, 100.0 };
    double etabins[neta+1] = { 0.0, 1.479, 2.5 };
#endif

    TFile *fOut = TFile::Open("fakeRates_chargeFlip_TTLep_MC.root", "RECREATE");
    TH2F *FR_2lss_mu_den = new TH2F("QF_2lss_mu_den","",npt,ptbins,neta,etabins);
    TH2F *FR_2lss_mu_num = new TH2F("QF_2lss_mu_num","",npt,ptbins,neta,etabins);
    TH2F *FR_2lss_el_den = new TH2F("QF_2lss_el_den","",npt,ptbins,neta,etabins);
    TH2F *FR_2lss_el_num = new TH2F("QF_2lss_el_num","",npt,ptbins,neta,etabins);

    TString baseCut = "LepGood%d_mcMatchId > 0 && abs(LepGood%d_pdgId) == abs(GenLep%d_pdgId) && "; // require good match
    baseCut += "abs(GenLep1_pdgId + GenLep2_pdgId) == 2 && "; // e+mu
    baseCut += "LepGood%d_mvaNew >= -0.2 && ";
    baseCut += "LepGood%d_tightCharge && ";
    TString passCut = "LepGood%d_pdgId != GenLep%d_pdgId";

    fillFR("QF_2lss_el", baseCut + " abs(LepGood%d_pdgId) == 11", passCut, "TTLep", 2);
    fillFR("QF_2lss_mu", baseCut + " abs(LepGood%d_pdgId) == 13", passCut, "TTLep", 2);
    fOut->Close();
}
