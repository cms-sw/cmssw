//TString gTreePath = "/afs/cern.ch/user/g/gpetrucc/w/ttH/TREES_250513_FR/%s/ttHLepFRAnalyzer/ttHLepFRAnalyzer_tree.root";
//TString gTreePath = "/afs/cern.ch/user/g/gpetrucc/w/TREES_250513_FR_v2/%s/ttHLepFRAnalyzer/ttHLepFRAnalyzer_tree.root";
TString gTreePath = "/afs/cern.ch/user/g/gpetrucc/w/TREES_250513_FR_v4/%s/ttHLepFRAnalyzer/ttHLepFRAnalyzer_tree.root";


void fillFR(TString hist, TString cut, TString pass, TString compName) {
    TDirectory *root = gDirectory;
    TFile *f = TFile::Open(Form(gTreePath.Data(),compName.Data()));
    TTree *tree = (TTree*) f->Get("ttHLepFRAnalyzer");
    root->cd();
    tree->Draw(Form("abs(Probe_eta):min(Probe_pt,79.9)>>+%s_den", hist.Data()),
               cut.Data());
    tree->Draw(Form("abs(Probe_eta):min(Probe_pt,79.9)>>+%s_num", hist.Data()),
               Form("(%s) && (%s)", cut.Data(), pass.Data()));
    f->Close();
    TH2 *den = (TH2*) gROOT->FindObject(hist+"_den"); den->Sumw2(); den->Write();
    TH2 *num = (TH2*) gROOT->FindObject(hist+"_num"); num->Sumw2(); num->Write();
    TH2 *ratio = num->Clone(hist);
    ratio->Divide(num,den,1,1,"B");
    ratio->Write();
}

void fillTrivialFakeRatesFromFRTrees(int triggering=1) {
    gROOT->ProcessLine(".L ../../python/plotter/functions.cc+");
    gROOT->ProcessLine(".L ../../python/plotter/fakeRate.cc+");

    const int npt_mu = 8, npt_el = 7, neta_mu = 2, neta_el = 3;
    const int npt2_mu = 5, npt2_el = 3;
    double ptbins_mu[npt_mu+1] = { 5.0, 7.0, 8.5, 10, 15, 20, 25, 35, 80 };
    double ptbins_el[npt_el+1] = {        7, 8.5, 10, 15, 20, 25, 35, 80 };
    double ptbins2_mu[npt2_mu+1] = { 5.0, 8.5, 15, 25, 45, 80};
    double ptbins2_el[npt2_el+1] = {        7, 10, 20, 80 };
    double etabins_mu[neta_mu+1] = { 0.0, 1.5,   2.5 };
    double etabins_el[neta_el+1] = { 0.0, 0.8, 1.479, 2.5 };
    //double etabins_mu[neta+1] = { 0.0, 0.7, 1.5,   2.0,  2.5 };
    //double etabins_el[neta+1] = { 0.0, 0.7, 1.479, 2.0,  2.5 };

    TString fileName = "";
    switch(triggering) {
        case 0: fileName = "fakeRates_QCDMu_MC_NonTrig.root"; break;
        case 1: fileName = "fakeRates_QCDMu_MC.root"; break;
        case 2: fileName = "fakeRates_QCDMu_MC_SingleMu.root"; break;
    }
    TFile *fOut = TFile::Open(fileName, "RECREATE");
    const int  nsels = 3;
    const char *sels[nsels] = { "FR", "FRC", "FRH" };
    for (int is = 0; is < nsels; ++is) {
        TH2F *FR_mu_den = new TH2F(Form("%s_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_mu_num = new TH2F(Form("%s_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_el_den = new TH2F(Form("%s_el_den",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_el_num = new TH2F(Form("%s_el_num",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);

        TH2F *FR_loose_el_den = new TH2F(Form("%s_loose_el_den",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_loose_el_num = new TH2F(Form("%s_loose_el_num",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_loose_mu_den = new TH2F(Form("%s_loose_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_loose_mu_num = new TH2F(Form("%s_loose_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_tight_el_den = new TH2F(Form("%s_tight_el_den",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_tight_el_num = new TH2F(Form("%s_tight_el_num",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_tight_mu_den = new TH2F(Form("%s_tight_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_tight_mu_num = new TH2F(Form("%s_tight_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FRj_mu_den = new TH2F(Form("%sj_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FRj_mu_num = new TH2F(Form("%sj_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FRj_el_den = new TH2F(Form("%sj_el_den",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FRj_el_num = new TH2F(Form("%sj_el_num",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FRj_loose_el_den = new TH2F(Form("%sj_loose_el_den",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FRj_loose_el_num = new TH2F(Form("%sj_loose_el_num",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FRj_loose_mu_den = new TH2F(Form("%sj_loose_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FRj_loose_mu_num = new TH2F(Form("%sj_loose_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FRj_tight_el_den = new TH2F(Form("%sj_tight_el_den",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FRj_tight_el_num = new TH2F(Form("%sj_tight_el_num",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FRj_tight_mu_den = new TH2F(Form("%sj_tight_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FRj_tight_mu_num = new TH2F(Form("%sj_tight_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);

        TH2F *FR_loose2_mu_den = new TH2F(Form("%s_loose2_mu_den",sels[is]),"",npt2_mu,ptbins2_mu,neta_mu,etabins_mu);
        TH2F *FR_loose2_mu_num = new TH2F(Form("%s_loose2_mu_num",sels[is]),"",npt2_mu,ptbins2_mu,neta_mu,etabins_mu);
        TH2F *FR_loose2_el_den = new TH2F(Form("%s_loose2_el_den",sels[is]),"",npt2_el,ptbins2_el,neta_el,etabins_el);
        TH2F *FR_loose2_el_num = new TH2F(Form("%s_loose2_el_num",sels[is]),"",npt2_el,ptbins2_el,neta_el,etabins_el);
        TH2F *FR_tight2_mu_den = new TH2F(Form("%s_tight2_mu_den",sels[is]),"",npt2_mu,ptbins2_mu,neta_mu,etabins_mu);
        TH2F *FR_tight2_mu_num = new TH2F(Form("%s_tight2_mu_num",sels[is]),"",npt2_mu,ptbins2_mu,neta_mu,etabins_mu);
        TH2F *FR_tight2_el_den = new TH2F(Form("%s_tight2_el_den",sels[is]),"",npt2_el,ptbins2_el,neta_el,etabins_el);
        TH2F *FR_tight2_el_num = new TH2F(Form("%s_tight2_el_num",sels[is]),"",npt2_el,ptbins2_el,neta_el,etabins_el);

        TH2F *FR_tightSip4_mu_den = new TH2F(Form("%s_tightSip4_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_tightSip4_mu_num = new TH2F(Form("%s_tightSip4_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FRj_tightSip4_mu_den = new TH2F(Form("%sj_tightSip4_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FRj_tightSip4_mu_num = new TH2F(Form("%sj_tightSip4_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_tightSip42_mu_den = new TH2F(Form("%s_tightSip42_mu_den",sels[is]),"",npt2_mu,ptbins2_mu,neta_mu,etabins_mu);
        TH2F *FR_tightSip42_mu_num = new TH2F(Form("%s_tightSip42_mu_num",sels[is]),"",npt2_mu,ptbins2_mu,neta_mu,etabins_mu);

        TH2F *FR_tightSUS13_mu_den = new TH2F(Form("%s_tightSUS13_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_tightSUS13_mu_num = new TH2F(Form("%s_tightSUS13_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_tightSUS13_el_den = new TH2F(Form("%s_tightSUS13_el_den",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_tightSUS13_el_num = new TH2F(Form("%s_tightSUS13_el_num",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FRj_tightSUS13_mu_den = new TH2F(Form("%sj_tightSUS13_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FRj_tightSUS13_mu_num = new TH2F(Form("%sj_tightSUS13_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FRj_tightSUS13_el_den = new TH2F(Form("%sj_tightSUS13_el_den",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FRj_tightSUS13_el_num = new TH2F(Form("%sj_tightSUS13_el_num",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_tightSUS132_mu_den = new TH2F(Form("%s_tightSUS132_mu_den",sels[is]),"",npt2_mu,ptbins2_mu,neta_mu,etabins_mu);
        TH2F *FR_tightSUS132_mu_num = new TH2F(Form("%s_tightSUS132_mu_num",sels[is]),"",npt2_mu,ptbins2_mu,neta_mu,etabins_mu);
        TH2F *FR_tightSUS132_el_den = new TH2F(Form("%s_tightSUS132_el_den",sels[is]),"",npt2_el,ptbins2_el,neta_el,etabins_el);
        TH2F *FR_tightSUS132_el_num = new TH2F(Form("%s_tightSUS132_el_num",sels[is]),"",npt2_el,ptbins2_el,neta_el,etabins_el);
 
        TH2F *FR_tightSUS13C_mu_den = new TH2F(Form("%s_tightSUS13C_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_tightSUS13C_mu_num = new TH2F(Form("%s_tightSUS13C_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_tightSUS13C_el_den = new TH2F(Form("%s_tightSUS13C_el_den",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_tightSUS13C_el_num = new TH2F(Form("%s_tightSUS13C_el_num",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FRj_tightSUS13C_mu_den = new TH2F(Form("%sj_tightSUS13C_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FRj_tightSUS13C_mu_num = new TH2F(Form("%sj_tightSUS13C_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FRj_tightSUS13C_el_den = new TH2F(Form("%sj_tightSUS13C_el_den",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FRj_tightSUS13C_el_num = new TH2F(Form("%sj_tightSUS13C_el_num",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);

        TH2F *FR_tightSB_el_den = new TH2F(Form("%s_tightSB_el_den",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_tightSB_el_num = new TH2F(Form("%s_tightSB_el_num",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_tightSB_mu_den = new TH2F(Form("%s_tightSB_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_tightSB_mu_num = new TH2F(Form("%s_tightSB_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FRj_tightSB_el_den = new TH2F(Form("%sj_tightSB_el_den",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FRj_tightSB_el_num = new TH2F(Form("%sj_tightSB_el_num",sels[is]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FRj_tightSB_mu_den = new TH2F(Form("%sj_tightSB_mu_den",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FRj_tightSB_mu_num = new TH2F(Form("%sj_tightSB_mu_num",sels[is]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_tightSB2_mu_den = new TH2F(Form("%s_tightSB2_mu_den",sels[is]),"",npt2_mu,ptbins2_mu,neta_mu,etabins_mu);
        TH2F *FR_tightSB2_mu_num = new TH2F(Form("%s_tightSB2_mu_num",sels[is]),"",npt2_mu,ptbins2_mu,neta_mu,etabins_mu);
        TH2F *FR_tightSB2_el_den = new TH2F(Form("%s_tightSB2_el_den",sels[is]),"",npt2_el,ptbins2_el,neta_el,etabins_el);
        TH2F *FR_tightSB2_el_num = new TH2F(Form("%s_tightSB2_el_num",sels[is]),"",npt2_el,ptbins2_el,neta_el,etabins_el);

        const char *mvas[6] = { "05", "03", "00", "m03", "m05", "m07" };
        for (int imva = 0; imva < 6; ++imva) {
            TH2F *FR_tightMVA_mu_den = new TH2F(Form("%s_tightMVA%s_mu_den",sels[is],mvas[imva]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
            TH2F *FR_tightMVA_mu_num = new TH2F(Form("%s_tightMVA%s_mu_num",sels[is],mvas[imva]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
            TH2F *FRj_tightMVA_mu_den = new TH2F(Form("%sj_tightMVA%s_mu_den",sels[is],mvas[imva]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
            TH2F *FRj_tightMVA_mu_num = new TH2F(Form("%sj_tightMVA%s_mu_num",sels[is],mvas[imva]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
            TH2F *FR_tightMVA2_mu_den = new TH2F(Form("%s_tightMVA%s2_mu_den",sels[is],mvas[imva]),"",npt2_mu,ptbins2_mu,neta_mu,etabins_mu);
            TH2F *FR_tightMVA2_mu_num = new TH2F(Form("%s_tightMVA%s2_mu_num",sels[is],mvas[imva]),"",npt2_mu,ptbins2_mu,neta_mu,etabins_mu);
        }
    }
#if 0
    const int ntrig1mu = 5;
    const int trig1mu[ntrig1mu] = { 8, 12, 17, 24, 40 };
    for (int it = 0; it < ntrig1mu; ++it) {
        TH2F *FR_mu_den = new TH2F(Form("%s_TagMu%d_mu_den",sels[0],trig1mu[it]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_mu_num = new TH2F(Form("%s_TagMu%d_mu_num",sels[0],trig1mu[it]),"",npt_mu,ptbins_mu,neta_mu,etabins_mu);
        TH2F *FR_el_den = new TH2F(Form("%s_TagMu%d_el_den",sels[0],trig1mu[it]),"",npt_el,ptbins_el,neta_el,etabins_el);
        TH2F *FR_el_num = new TH2F(Form("%s_TagMu%d_el_num",sels[0],trig1mu[it]),"",npt_el,ptbins_el,neta_el,etabins_el);
    }
#endif

    TString baseCut = "Probe_mcMatchId == 0 && ";
    //baseCut += "Probe_mcMatchAny >= 2 && ";

    TString baseCutL = baseCut + "tagType == 13 && ";
    //baseCutL += "TagLepton_sip3d > 7 && ";
    //baseCutL += "Probe_pt/(TagLepton_pt*(1+TagLepton_relIso)) < 1 && abs(dphi_tp) > 2.0 && ";
    //baseCutL1l = baseCutL + "TagLepton_sip3d > 4 && Probe_pt/(TagLepton_pt*(1+TagLepton_relIso)) < 1.5 && abs(dphi_tp) > 1.0 && ";
    //baseCutL1b = baseCutL + "Probe_pt/(TagLepton_pt*(1+TagLepton_relIso)) < 1.0 && abs(dphi_tp) > 2.5 && ";
    baseCutL += "TagLepton_sip3d > 5 && Probe_pt/(TagLepton_pt*(1+TagLepton_relIso)) < 1.5 && abs(dphi_tp) > 1.0 && ";
    TString baseCutJ = baseCut + "tagType == 1 && ";
    baseCutJ += "Probe_pt/TagJet_pt < 1 && abs(dphi_tp) > 2.0 && TagJet_btagCSV > 0.679 && TagJet_pt > 40 && ";

    //TString baseCutJ2 = baseCut + "tagType == 1  && hasSecondB == 1 && (SecondJet_btagCSV > 0.679 || abs(SecondJet_mcFlavour) == 5) && deltaR(Probe_eta,Probe_phi,SecondJet_eta,SecondJet_phi) > 0.0 &&  ";
    //TString baseCutL2 = baseCut + "tagType == 13 && hasSecondB == 1 && (SecondJet_btagCSV > 0.679 || abs(SecondJet_mcFlavour) == 5) && ";
    //TString baseCutL2M  = baseCut + "tagType == 13 && hasSecondB == 1 && SecondJet_btagCSV > 0.679 && ";
    TString baseCutL2   = baseCut + "tagType == 13 && hasSecondB == 1 && (SecondJet_btagCSV > 0.274 || abs(SecondJet_mcFlavour) == 5) && ";


    if (triggering == 1) {
        baseCutL  += "(Trig_Pair_2Mu || Trig_Pair_MuEG) &&";
        baseCutL2 += "(Trig_Pair_2Mu || Trig_Pair_MuEG) &&";
        baseCutJ  += "(Trig_Probe_Mu5 || Trig_Probe_1ElT || Trig_Event_Mu40) &&";
        //baseCutJ2 += "(Trig_Probe_Mu5 || Trig_Probe_1ElT || Trig_Event_Mu40) &&";
    }

    TString tightC = ""; //"(Probe_tightCharge > (abs(Probe_pdgId) == 11)) && ";
    tightC += "Probe_innerHits*(abs(Probe_pdgId) == 11) == 0 && "; // require to be zero if the lepton is an electron
    tightC += "(Probe_convVeto==0)*(abs(Probe_pdgId) == 11) == 0 && "; // require to be zero if the lepton is an electron

    TString sip4cut  = "Probe_sip3d <  4 && ";
    TString sus13_el = "passEgammaTightMVA(Probe_pt,Probe_eta,Probe_tightId) && abs(Probe_dxy) < 0.0100  && abs(Probe_dz) < 0.1 && (abs(Probe_eta) < 1.4442 || abs(Probe_eta) > 1.566) && ";
    TString sus13_mu = "Probe_tightId && Probe_tightCharge > 0  && abs(Probe_dxy) < 0.0050 && abs(Probe_dz) < 0.1 && ";
#if 0
    fillFR("FR_loose_el", baseCutL +          "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_mva >= -0.30", "QCDMuPt15");
    fillFR("FR_loose_mu", baseCutL +          "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= -0.30", "QCDMuPt15");
    fillFR("FR_tight_el", baseCutL + tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_mva >=  0.70", "QCDMuPt15");
    fillFR("FR_tight_mu", baseCutL + tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >=  0.70", "QCDMuPt15");

    fillFR("FRj_loose_mu", baseCutJ +          "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= -0.30", "QCDMuPt15");
    fillFR("FRj_loose_el", baseCutJ +          "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_mva >= -0.30", "QCDElPt30To80");
    fillFR("FRj_loose_el", baseCutJ +          "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_mva >= -0.30", "QCDElPt80To170");
    fillFR("FRj_tight_mu", baseCutJ + tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >=  0.70", "QCDMuPt15");
    fillFR("FRj_tight_el", baseCutJ + tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_mva >=  0.70", "QCDElPt30To80");
    fillFR("FRj_tight_el", baseCutJ + tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_mva >=  0.70", "QCDElPt80To170");

    fillFR("FR_tight2_el", baseCutL2 + tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_mva >= 0.70", "QCDMuPt15");
    fillFR("FR_tight2_mu", baseCutL2 + tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= 0.70", "QCDMuPt15");
    fillFR("FR_loose2_mu", baseCutL2 + tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= -0.3", "QCDMuPt15");
    fillFR("FR_loose2_el", baseCutL2 + tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_mva >= -0.3", "QCDMuPt15");
#endif

#if 0
    fillFR("FR_tightSUS13_el",  baseCutL +  tightC + sus13_el + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_relIso < 0.1", "QCDMuPt15");
    fillFR("FR_tightSUS13_mu",  baseCutL +  tightC + sus13_mu + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_relIso < 0.1", "QCDMuPt15");
    fillFR("FRj_tightSUS13_el", baseCutJ +  tightC + sus13_el + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_relIso < 0.1", "QCDElPt30To80");
    fillFR("FRj_tightSUS13_el", baseCutJ +  tightC + sus13_el + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_relIso < 0.1", "QCDElPt80To170");
    fillFR("FRj_tightSUS13_mu", baseCutJ +  tightC + sus13_mu + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_relIso < 0.1", "QCDMuPt15");
    fillFR("FR_tightSUS132_el", baseCutL2 + tightC + sus13_el + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_relIso < 0.1", "QCDMuPt15");
    fillFR("FR_tightSUS132_mu", baseCutL2 + tightC + sus13_mu + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_relIso < 0.1", "QCDMuPt15");
#endif
#if 0
    fillFR("FR_tightSUS13C_el",  baseCutL +  tightC + sus13_el + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_chargedIso/Probe_pt < 0.05", "QCDMuPt15");
    fillFR("FR_tightSUS13C_mu",  baseCutL +  tightC + sus13_mu + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_chargedIso/Probe_pt < 0.05", "QCDMuPt15");
    fillFR("FRj_tightSUS13C_el", baseCutJ +  tightC + sus13_el + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_chargedIso/Probe_pt < 0.05", "QCDElPt30To80");
    fillFR("FRj_tightSUS13C_el", baseCutJ +  tightC + sus13_el + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_chargedIso/Probe_pt < 0.05", "QCDElPt80To170");
    fillFR("FRj_tightSUS13C_mu", baseCutJ +  tightC + sus13_mu + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_chargedIso/Probe_pt < 0.05", "QCDMuPt15");
#endif


#if 0
    fillFR("FR_tightSip4_mu",  baseCutL +  tightC + sip4cut + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= 0.70", "QCDMuPt15");
    fillFR("FRj_tightSip4_mu", baseCutJ +  tightC + sip4cut + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= 0.70", "QCDMuPt15");
    fillFR("FR_tightSip42_mu", baseCutL2 + tightC + sip4cut + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= 0.70", "QCDMuPt15");
#endif
#if 1
    fillFR("FR_tightMVA05_mu",   baseCutL +  tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= 0.50", "QCDMuPt15");
    fillFR("FRj_tightMVA05_mu",  baseCutJ +  tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= 0.50", "QCDMuPt15");
    fillFR("FR_tightMVA052_mu",  baseCutL2 + tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= 0.50", "QCDMuPt15");
    fillFR("FR_tightMVA03_mu",   baseCutL +  tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= 0.30", "QCDMuPt15");
    fillFR("FRj_tightMVA03_mu",  baseCutJ +  tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= 0.30", "QCDMuPt15");
    fillFR("FR_tightMVA032_mu",  baseCutL2 + tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= 0.30", "QCDMuPt15");
    fillFR("FR_tightMVA00_mu",   baseCutL +  tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= 0.00", "QCDMuPt15");
    fillFR("FRj_tightMVA00_mu",  baseCutJ +  tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= 0.00", "QCDMuPt15");
    fillFR("FR_tightMVA002_mu",  baseCutL2 + tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= 0.00", "QCDMuPt15");
    fillFR("FR_tightMVAm03_mu",  baseCutL +  tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= -0.3", "QCDMuPt15");
    fillFR("FRj_tightMVAm03_mu", baseCutJ +  tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= -0.3", "QCDMuPt15");
    fillFR("FR_tightMVAm032_mu", baseCutL2 + tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= -0.3", "QCDMuPt15");
    fillFR("FR_tightMVAm05_mu",  baseCutL +  tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= -0.5", "QCDMuPt15");
    fillFR("FRj_tightMVAm05_mu", baseCutJ +  tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= -0.5", "QCDMuPt15");
    fillFR("FR_tightMVAm052_mu", baseCutL2 + tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= -0.5", "QCDMuPt15");
    fillFR("FR_tightMVAm07_mu",  baseCutL +  tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= -0.7", "QCDMuPt15");
    fillFR("FRj_tightMVAm07_mu", baseCutJ +  tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= -0.7", "QCDMuPt15");
    fillFR("FR_tightMVAm072_mu", baseCutL2 + tightC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= -0.7", "QCDMuPt15");
#endif


#if 0
    TString tightSBC = tightC + "(Probe_mva > -0.7 && Probe_mva < 0.5 || Probe_mva > 0.7) && ";
    fillFR("FR_tightSB_el",  baseCutL + tightSBC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_mva >=  0.70", "QCDMuPt15");
    fillFR("FR_tightSB_mu",  baseCutL + tightSBC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >=  0.70", "QCDMuPt15");
    fillFR("FRj_tightSB_mu", baseCutJ + tightSBC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >=  0.70", "QCDMuPt15");
    fillFR("FRj_tightSB_el", baseCutJ + tightSBC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_mva >=  0.70", "QCDElPt30To80");
    fillFR("FRj_tightSB_el", baseCutJ + tightSBC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_mva >=  0.70", "QCDElPt80To170");
    fillFR("FR_tightSB2_el", baseCutL2 + tightSBC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 11", "Probe_mva >= 0.70", "QCDMuPt15");
    fillFR("FR_tightSB2_mu", baseCutL2 + tightSBC + "Probe_mcMatchId == 0 && abs(Probe_pdgId) == 13", "Probe_mva >= 0.70", "QCDMuPt15");
#endif


    fOut->Close();
}
