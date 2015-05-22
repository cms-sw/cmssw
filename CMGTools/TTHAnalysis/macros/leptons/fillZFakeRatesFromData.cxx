TString gTreePath = "/data/b/botta/TTHAnalysis/trees/TREES_250513_HADD/%s/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root";
void fillFR(TString hist, TString cut, TString pass, TString compName, int maxLep, double weight=1.0) {
    TDirectory *root = gDirectory;
    TFile *f = TFile::Open(Form(gTreePath.Data(),compName.Data()));
    TTree *tree = (TTree*) f->Get("ttHLepTreeProducerBase");
    root->cd();
    for (int i = 3; i <= maxLep; ++i) {
        TString mycut = cut;   mycut.ReplaceAll("%d", Form("%d",i));
        TString mypass = pass; mypass.ReplaceAll("%d", Form("%d",i));
        if (weight != 1.0)  mycut  = Form("%g*(%s)", weight, mycut.Data());
        tree->Draw(Form("abs(LepGood%d_eta):min(LepGood%d_pt,99.9)>>+%s_den", i,i, hist.Data()),
                   Form("%s", mycut.Data()));
        tree->Draw(Form("abs(LepGood%d_eta):min(LepGood%d_pt,99.9)>>+%s_num", i,i, hist.Data()),
                   Form("(%s)*(%s)", mycut.Data(), mypass.Data()));
    }
    f->Close();
    TH2 *den = (TH2*) gROOT->FindObject(hist+"_den"); den->Sumw2(); 
    TH2 *num = (TH2*) gROOT->FindObject(hist+"_num"); num->Sumw2();
}
void doEff(TString what,TString kind="FR") {
    TH2 *numD = (TH2*) gROOT->FindObject(kind+"_"+what+"_num");
    TH2 *denD = (TH2*) gROOT->FindObject(kind+"_"+what+"_den");
    TH2 *numM = (TH2*) gROOT->FindObject("PMC_"+kind+"_"+what+"_num");
    TH2 *denM = (TH2*) gROOT->FindObject("PMC_"+kind+"_"+what+"_den");
    if (denD == 0 || denD->GetEntries() == 0) return;
    TH2 *ratio = numD->Clone(kind+"_"+what);
    printf("Doing FR for %s\n", what.Data());
    for (int ix = 1, nx = ratio->GetNbinsX(), ny = ratio->GetNbinsY(); ix <= nx; ++ix) {
        for (int iy = 1; iy <= ny; ++iy) {
            double nD = numD->GetBinContent(ix,iy);
            double dD = denD->GetBinContent(ix,iy);
            double nM = numM->GetBinContent(ix,iy);
            double dM = denM->GetBinContent(ix,iy);
            double fr = (nD - nM)/(dD-dM);
            double pr = nM/dM;
            double frstat1 = sqrt(fr*(1-fr)/(dD-dM));
            double frstat2 = sqrt(pr*(1-pr)*nM)/(dD-dM);
            double frsyst  = 0.5*nM/(dD-dM);
            double frerr = sqrt(frstat1*frstat1 + frstat2*frstat2 + frsyst*frsyst);
            printf("  pt [%2.0f,%2.0f] eta [%.2f,%.2f]: nD %7.1f, dD %7.1f, nM %6.2f, dM %6.2f, f0 = %.3f, p = %.3f, f = %.3f +/- %.3f (stat1) +/- %.3f (stat2) +/- %.3f (syst) = %.3f +/- %.3f\n",
                numD->GetXaxis()->GetBinLowEdge(ix), 
                numD->GetXaxis()->GetBinUpEdge(ix), 
                numD->GetYaxis()->GetBinLowEdge(iy), 
                numD->GetYaxis()->GetBinUpEdge(iy), 
                nD,dD,nM,dM, nD/dD, pr, fr, frstat1, frstat2, frsyst, fr, frerr);
            ratio->SetBinContent(ix, iy, fr);
            ratio->SetBinError(ix, iy, frerr);
        }
    }
    numD->Write();
    numM->Write();
    denD->Write();
    denM->Write();
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

void fillZFakeRatesFromData(int withb=0) {
    gROOT->ProcessLine(".L ../../python/plotter/functions.cc+");
    gROOT->ProcessLine(".L ../../python/plotter/fakeRate.cc+");

    const int npt_mu = 4, npt_el = 4, neta_mu = 2, neta_el = 3;
    double ptbins_mu[npt_mu+1] = { 5.0, 7.0, 10, 15, 30 };
    double ptbins_el[npt_el+1] = {      7.0, 8.5, 10, 15, 30 };
    //double etabins_mu[neta+1] = { 0.0, 0.7, 1.5,   2.0,  2.5 };
    //double etabins_el[neta+1] = { 0.0, 0.7, 1.479, 2.0,  2.5 };
    double etabins_mu[neta_mu+1] = { 0.0, 1.5,   2.5 };
    double etabins_el[neta_el+1] = { 0.0, 0.8, 1.479, 2.5 };

    TFile *fOut = TFile::Open(withb ? "fakeRates_Zb_Data.root" : "fakeRates_Z_Data.root", "RECREATE");
    const int  nsels = 4;
    const char *sels[nsels] = { "FR", "PMC_FR", "FRC", "PMC_FRC" };
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
    baseCut += "minMllAFAS > 12 && abs(mZ1-91.2) < 10 && min(LepGood1_mva,LepGood2_mva) >= -0.3 && ";
    //if (withb) baseCut += "met*0.00397 + mhtJet25*0.00265 < 0.30 && ";
    //else       baseCut += "met*0.00397 + mhtJet25*0.00265 < 0.15 && ";
    baseCut += "met*0.00397 + mhtJet25*0.00265 < 0.3 && ";
    baseCut += "mtw_wz3l(LepGood1_pt,LepGood1_eta,LepGood1_phi,LepGood1_mass,LepGood2_pt,LepGood2_eta,LepGood2_phi,LepGood2_mass,LepGood3_pt,LepGood3_eta,LepGood3_phi,LepGood3_mass,mZ1,met,met_phi) < 40 && ";
    baseCut += "abs(mass_2(LepGood1_pt,LepGood1_eta,LepGood1_phi,LepGood1_mass,LepGood2_pt,LepGood2_eta,LepGood2_phi,LepGood2_mass)-mZ1) < 0.01 && ";
    //baseCut += "nBJetMedium25 == 1 && ";
    if (withb) baseCut += "nBJetMedium25 == 1 && ";

    TString baseCutT = baseCut; //;
    baseCutT += "LepGood%d_innerHits*(abs(LepGood%d_pdgId) == 11) == 0 && "; // require to be zero if the lepton is an electron
    baseCutT += "(LepGood%d_convVeto==0)*(abs(LepGood%d_pdgId) == 11) == 0 && ";
    TString baseCutTC  = baseCutT + " (LepGood%d_tightCharge > (abs(LepGood%d_pdgId) == 11)) && ";
    TString baseCutTCB = baseCutTC+ " (LepGood%d_sip3d < 4) && (abs(LepGood%d_pdgId) == 11 || LepGood%d_tightId) && (abs(LepGood%d_pdgId) == 13 || passEgammaTightMVA(LepGood%d_pt,LepGood%d_eta,LepGood%d_tightId)) && ";

    TString sample = "";
    const char *samples[10] = { "DoubleMuAB", "DoubleMuC", "DoubleMuD", "DoubleMuRec", "DoubleMuBadSIP",
                               "DoubleElectronAB", "DoubleElectronC", "DoubleElectronD", "DoubleElectronRec", "DoubleElectronBadSIP" };
    for (int id = 0; id < 10; ++id) { 
        sample = TString(samples[id]);

        std::cout << "Processing MVA selection on " << sample << std::endl;
        fillFR("FR_el", baseCutTC + "abs(LepGood%d_pdgId) == 11", "LepGood%d_mva >= -0.3", sample, 3);
        fillFR("FR_mu", baseCutTC + "abs(LepGood%d_pdgId) == 13", "LepGood%d_mva >= -0.3", sample, 3);
        fillFR("FR_loose_el", baseCut  + "abs(LepGood%d_pdgId) == 11", "LepGood%d_mva >= -0.3", sample, 3);
        fillFR("FR_loose_mu", baseCut  + "abs(LepGood%d_pdgId) == 13", "LepGood%d_mva >= -0.3", sample, 3);
        fillFR("FR_tight_el", baseCutTC + "abs(LepGood%d_pdgId) == 11", "LepGood%d_mva >= 0.70", sample, 3);
        fillFR("FR_tight_mu", baseCutTC + "abs(LepGood%d_pdgId) == 13", "LepGood%d_mva >= 0.70", sample, 3);
        fillFR("FRC_tight_el", baseCutTCB + "abs(LepGood%d_pdgId) == 11", "LepGood%d_relIso < 0.12", sample, 3);
        fillFR("FRC_tight_mu", baseCutTCB + "abs(LepGood%d_pdgId) == 13", "LepGood%d_relIso < 0.12", sample, 3);
    }


    const int nmc = 7;
    const char * mcs[nmc] = { "WZJets",  "ZZ2e2mu", "ZZ2e2tau", "ZZ2mu2tau", "ZZTo4e", "ZZTo4mu", "ZZTo4tau" };
    double      nevt[nmc] = { 2017979 ,   1277445,     823911,    823922,     1499093,   1499064,     824466 };
    double      xsbr[nmc] = { 1.057,        0.1767,    0.1767,    0.1767,     0.07691,   0.07691,    0.07691 }; 
    for (int id = 0; id < nmc; ++id) { 
        sample = TString(mcs[id]);
        double weight = 19.6 * 1000 * xsbr[id]/nevt[id];
        std::cout << "Processing MVA selection on " << sample << std::endl;
        fillFR("PMC_FR_el", baseCutTC + "abs(LepGood%d_pdgId) == 11", "LepGood%d_mva >= 0.70", sample, 3, weight);
        fillFR("PMC_FR_emu", baseCutTC + "abs(LepGood%d_pdgId) == 13", "LepGood%d_mva >= 0.70", sample, 3, weight);
        fillFR("PMC_FR_tight_el", baseCutTC + "abs(LepGood%d_pdgId) == 11", "LepGood%d_mva >= 0.70", sample, 3, weight);
        fillFR("PMC_FR_tight_mu", baseCutTC + "abs(LepGood%d_pdgId) == 13", "LepGood%d_mva >= 0.70", sample, 3, weight);
        fillFR("PMC_FR_loose_el", baseCut + "abs(LepGood%d_pdgId) == 11", "LepGood%d_mva >= -0.30", sample, 3, weight);
        fillFR("PMC_FR_loose_mu", baseCut + "abs(LepGood%d_pdgId) == 13", "LepGood%d_mva >= -0.30", sample, 3, weight);
        fillFR("PMC_FRC_tight_el", baseCutTCB + "abs(LepGood%d_pdgId) == 11", "LepGood%d_relIso < 0.12", sample, 3, weight);
        fillFR("PMC_FRC_tight_mu", baseCutTCB + "abs(LepGood%d_pdgId) == 13", "LepGood%d_relIso < 0.12", sample, 3, weight);
    }


    doEff("el");
    doEff("mu");
    doEff("tight_el");
    doEff("tight_mu");
    doEff("loose_el");
    doEff("loose_mu");
    doEff("tight_el", "FRC");
    doEff("tight_mu", "FRC");

    fOut->Close();
}
