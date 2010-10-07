static const double PI = 3.1415926535;
double deltaPhi(double phi1, double phi2){

    // in ORCA phi = [0,2pi], in TLorentzVector phi = [-pi,pi].
    // With the conversion below deltaPhi works ok despite the
    // 2*pi difference in phi definitions.
    if(phi1 < 0) phi1 += 2*PI;
    if(phi2 < 0) phi2 += 2*PI;

    double dphi = fabs(phi1-phi2);

    if(dphi > PI) dphi = 2*PI - dphi;
    return dphi;
}

enum  {TCAlgoUndetermined,
       TCAlgoMomentum,
       TCAlgoTrackProblem,
       TCAlgoMomentumECAL,
       TCAlgoCaloJet,
       TCAlgoHadronicJet};

void tcTauAnalysis(){

	float tau_pt_cut = 10;
	float tau_eta_cut = 2.1;

	int algoCounter[6] = {0,0,0,0,0,0};

//	TFile* inFile = TFile::Open("tctau.root");
	TFile* inFile = TFile::Open("tctau_newMap.root");
	TTree* tauTree = (TTree *) (inFile->Get("tauTree"));

	cout << "Entries " << tauTree->GetEntries() << endl;	

        float MCTau_pt,MCTau_eta,MCTau_phi;
        float PFTau_pt,PFTau_eta,PFTau_phi,PFTau_nProngs,PFTau_ltrackPt,PFTau_d_isol,PFTau_d_1,PFTau_d_2;
//        float CaloTau_pt,CaloTau_eta,CaloTau_phi,CaloTau_nProngs,CaloTau_ltrackPt,CaloTau_d_isol,CaloTau_d_1,CaloTau_d_2;
	float JPTTau_pt,JPTTau_eta,JPTTau_phi;
        float CaloTau_pt,CaloTau_eta,CaloTau_phi,CaloTau_nProngs,CaloTau_ltrackPt,CaloTau_d_isol,CaloTau_d_1,CaloTau_d_2,CaloTau_tcalgo;
	float CaloTau_pt_raw,CaloTau_eta_raw,CaloTau_phi_raw;

	tauTree->SetBranchAddress("MCTau_pt",&MCTau_pt);
        tauTree->SetBranchAddress("MCTau_eta",&MCTau_eta);
        tauTree->SetBranchAddress("MCTau_phi",&MCTau_phi);
/*
        tauTree->SetBranchAddress("CaloTau_pt",&CaloTau_pt);
        tauTree->SetBranchAddress("CaloTau_eta",&CaloTau_eta);
        tauTree->SetBranchAddress("CaloTau_phi",&CaloTau_phi);
        tauTree->SetBranchAddress("CaloTau_nProngs",&CaloTau_nProngs);
        tauTree->SetBranchAddress("CaloTau_ltrackPt",&CaloTau_ltrackPt);
        tauTree->SetBranchAddress("CaloTau_d_isol",&CaloTau_d_isol);
*/
	tauTree->SetBranchAddress("JPTTau_pt",&JPTTau_pt);
	tauTree->SetBranchAddress("JPTTau_eta",&JPTTau_eta);
	tauTree->SetBranchAddress("JPTTau_phi",&JPTTau_phi);
        tauTree->SetBranchAddress("CaloTau_pt",&CaloTau_pt);
        tauTree->SetBranchAddress("CaloTau_eta",&CaloTau_eta);
        tauTree->SetBranchAddress("CaloTau_phi",&CaloTau_phi);
        tauTree->SetBranchAddress("CaloTau_pt_raw",&CaloTau_pt_raw);
        tauTree->SetBranchAddress("CaloTau_eta_raw",&CaloTau_eta_raw);
        tauTree->SetBranchAddress("CaloTau_phi_raw",&CaloTau_phi_raw);
        tauTree->SetBranchAddress("CaloTau_nProngs",&CaloTau_nProngs);
        tauTree->SetBranchAddress("CaloTau_ltrackPt",&CaloTau_ltrackPt);
	tauTree->SetBranchAddress("CaloTau_tcalgo",&CaloTau_tcalgo);
	tauTree->SetBranchAddress("CaloTau_d_isol",&CaloTau_d_isol);
        tauTree->SetBranchAddress("PFTau_pt",&PFTau_pt);
        tauTree->SetBranchAddress("PFTau_eta",&PFTau_eta);
        tauTree->SetBranchAddress("PFTau_phi",&PFTau_phi);
        tauTree->SetBranchAddress("PFTau_nProngs",&PFTau_nProngs);
        tauTree->SetBranchAddress("PFTau_ltrackPt",&PFTau_ltrackPt);
        tauTree->SetBranchAddress("PFTau_d_isol",&PFTau_d_isol);


	int nMCTaus   = 0,
//	    nCaloTaus = 0,
	    nCaloTaus   = 0,
	    nPFTaus   = 0;
	int nCaloTausIn01Counter   = 0,
	    nPFTausIn01Counter   = 0;

	TH1F* h_CaloTau_dEt = new TH1F("h_CaloTau_dEt","",100,-50,50);
	TH1F* h_jptTau_dEt  = (TH1F*)h_CaloTau_dEt->Clone("h_JPTTau_dEt");
//	TH1F* h_CaloTau_dEt   = (TH1F*)h_CaloTau_dEt->Clone("h_CaloTau_dEt");
	TH1F* h_PFTau_dEt   = (TH1F*)h_CaloTau_dEt->Clone("h_PFTau_dEt");

        TH1F* h_CaloTau_dEtRaw = (TH1F*)h_CaloTau_dEt->Clone("h_CaloTau_dEtRaw");

	TH1F* h_PFTauRef_dEt = (TH1F*)h_CaloTau_dEt->Clone("h_PFTauRef_dEt");

	TH1F* h_CaloTau_dEta = new TH1F("h_CaloTau_dEta","",100,-0.5,0.5);
//	TH1F* h_CaloTau_dEta   = (TH1F*)h_CaloTau_dEta->Clone("h_CaloTau_dEta");
	TH1F* h_PFTau_dEta   = (TH1F*)h_CaloTau_dEta->Clone("h_PFTau_dEta");

        TH1F* h_CaloTau_dPhi = new TH1F("h_CaloTau_dPhi","",100,-0.5,0.5);
//	TH1F* h_CaloTau_dPhi = (TH1F*)h_CaloTau_dPhi->Clone("h_CaloTau_dPhi");
	TH1F* h_PFTau_dPhi = (TH1F*)h_CaloTau_dPhi->Clone("h_PFTau_dPhi");

        TH1F* h_CaloTau_dEtRatio = new TH1F("h_CaloTau_dEtRatio","",100,0,2);
//	TH1F* h_CaloTau_dEtRatio = (TH1F*)h_CaloTau_dEtRatio->Clone("h_CaloTau_dEtRatio");
	TH1F* h_JPTTau_dEtRatio = (TH1F*)h_CaloTau_dEtRatio->Clone("h_JPTTau_dEtRatio");
	TH1F* h_PFTau_dEtRatio = (TH1F*)h_CaloTau_dEtRatio->Clone("h_PFTau_dEtRatio");
        TH1F* h_CaloTau_dEtRawRatio = (TH1F*)h_CaloTau_dEtRatio->Clone("h_CaloTau_dEtRawRatio");

        TH1F* h_CaloTau_dEt_TCAlgoUndetermined = (TH1F*)h_CaloTau_dEtRatio->Clone("h_CaloTau_dEt_TCAlgoUndetermined");
        TH1F* h_CaloTau_dEt_TCAlgoMomentum     = (TH1F*)h_CaloTau_dEtRatio->Clone("h_CaloTau_dEt_TCAlgoMomentum");
        TH1F* h_CaloTau_dEt_TCAlgoTrackProblem = (TH1F*)h_CaloTau_dEtRatio->Clone("h_CaloTau_dEt_TCAlgoTrackProblem");
        TH1F* h_CaloTau_dEt_TCAlgoMomentumECAL = (TH1F*)h_CaloTau_dEtRatio->Clone("h_CaloTau_dEt_TCAlgoMomentumECAL");
        TH1F* h_CaloTau_dEt_TCAlgoCaloJet      = (TH1F*)h_CaloTau_dEtRatio->Clone("h_CaloTau_dEt_TCAlgoCaloJet");
        TH1F* h_CaloTau_dEt_TCAlgoHadronicJet  = (TH1F*)h_CaloTau_dEtRatio->Clone("h_CaloTau_dEt_TCAlgoHadronicJet");

	for(int i = 0; i < tauTree->GetEntries(); ++i){
		tauTree->GetEntry(i);

		if(MCTau_pt <= 0) continue; 
		nMCTaus++;
/*
		if(CaloTau_pt > tau_pt_cut && fabs(CaloTau_eta) < tau_eta_cut && CaloTau_d_isol != 0) {
			nCaloTaus++;
			double caloTauReso = (CaloTau_pt - MCTau_pt)/MCTau_pt;
			if(fabs(caloTauReso) < 0.1) nCaloTausIn01Counter++;

			h_CaloTau_dEt->Fill(CaloTau_pt - MCTau_pt);
			h_CaloTau_dEta->Fill(CaloTau_eta - MCTau_eta);
			h_CaloTau_dPhi->Fill(deltaPhi(CaloTau_phi,MCTau_phi));
			h_CaloTau_dEtRatio->Fill(CaloTau_pt/MCTau_pt);
		}
*/
		if(CaloTau_pt > tau_pt_cut && fabs(CaloTau_eta) < tau_eta_cut && CaloTau_d_isol != 0) {

			algoCounter[int(CaloTau_tcalgo)]++;

			nCaloTaus++;
                        double CaloTauReso = (CaloTau_pt - MCTau_pt)/MCTau_pt;
                        if(fabs(CaloTauReso) < 0.1) nCaloTausIn01Counter++;

			h_CaloTau_dEt->Fill(CaloTau_pt - MCTau_pt);
			h_CaloTau_dEta->Fill(CaloTau_eta - MCTau_eta);
			h_CaloTau_dPhi->Fill(deltaPhi(CaloTau_phi,MCTau_phi));
			h_CaloTau_dEtRatio->Fill(CaloTau_pt/MCTau_pt);

			h_CaloTau_dEtRawRatio->Fill(CaloTau_pt_raw/MCTau_pt);

			h_JPTTau_dEtRatio->Fill(JPTTau_pt/MCTau_pt);

                        if(CaloTau_tcalgo == TCAlgoUndetermined) h_CaloTau_dEt_TCAlgoUndetermined->Fill(CaloTau_pt/MCTau_pt);
                        if(CaloTau_tcalgo == TCAlgoMomentum)     h_CaloTau_dEt_TCAlgoMomentum->Fill(CaloTau_pt/MCTau_pt);
                        if(CaloTau_tcalgo == TCAlgoTrackProblem) h_CaloTau_dEt_TCAlgoTrackProblem->Fill(CaloTau_pt/MCTau_pt);
                        if(CaloTau_tcalgo == TCAlgoMomentumECAL) h_CaloTau_dEt_TCAlgoMomentumECAL->Fill(CaloTau_pt/MCTau_pt);
                        if(CaloTau_tcalgo == TCAlgoCaloJet)      h_CaloTau_dEt_TCAlgoCaloJet->Fill(CaloTau_pt/MCTau_pt);
			if(CaloTau_tcalgo == TCAlgoHadronicJet)  h_CaloTau_dEt_TCAlgoHadronicJet->Fill(CaloTau_pt/MCTau_pt);

		}
		if(PFTau_pt > tau_pt_cut && fabs(PFTau_eta) < tau_eta_cut && PFTau_d_isol != 0) {
			nPFTaus++;
                        double pfTauReso = (PFTau_pt - MCTau_pt)/MCTau_pt;
                        if(fabs(pfTauReso) < 0.1) nPFTausIn01Counter++;

			h_PFTau_dEt->Fill(PFTau_pt - MCTau_pt);
			h_PFTau_dEta->Fill(PFTau_eta - MCTau_eta);
			h_PFTau_dPhi->Fill(deltaPhi(PFTau_phi,MCTau_phi));
			h_PFTau_dEtRatio->Fill(PFTau_pt/MCTau_pt);
		}
//		if(PFTau_pt > tau_pt_cut && fabs(PFTau_eta) < tau_eta_cut && PFTau_d_isol != 0) {
//			h_PFTauRef_dEt->Fill(PFTau_pt - MCTau_pt);
//		}
	}

	cout << " MC taus           " << nMCTaus << endl;
	cout << " Isolated CaloTaus " << nCaloTaus << endl;
//	cout << " Isolated CaloTaus   " << nCaloTaus << endl;
	cout << " Isolated PFTaus   " << nPFTaus << endl;
	cout << endl;
/*
        enum  {TCAlgoUndetermined,
               TCAlgoBMomentum,
               TCAlgoTrackProblem,
               TCAlgoMomentumECAL,
               TCAlgoCaloJet,
               TCAlgoHadronicJet};
*/
	cout << "TCAlgoUndetermined " << algoCounter[TCAlgoUndetermined] << endl;
	cout << "TCAlgoMomentum     " << algoCounter[TCAlgoMomentum] << endl;
	cout << "TCAlgoTrackProblem " << algoCounter[TCAlgoTrackProblem] << endl;
	cout << "TCAlgoMomentumECAL " << algoCounter[TCAlgoMomentumECAL] << endl;
	cout << "TCAlgoCaloJet      " << algoCounter[TCAlgoCaloJet] << endl;
	cout << "TCAlgoHadronicJet  " << algoCounter[TCAlgoHadronicJet] << endl;
	int sum = 0;
	for(int iAlgo = 0; iAlgo < 6; ++ iAlgo) sum += algoCounter[iAlgo];
	cout << "Sum                " << sum << endl;

	
	//TH1F* h_CaloTauEff = (TH1F*) (inFile->Get("h_CaloTauEff"));
	//cout << " CaloTau algorithm efficiency " << h_CaloTauEff->GetBinContent(3) << endl;

        cout << endl;
        cout << " Fraction of jets in abs(dEt) < 0.1, reco::CaloTau                  " << double(nCaloTausIn01Counter)/nCaloTaus << endl;
        cout << " Fraction of jets in abs(dEt) < 0.1, reco::CaloTau+JPT+CaloTau        " << double(nCaloTausIn01Counter)/nCaloTaus << endl;
        cout << " Fraction of jets in abs(dEt) < 0.1, reco::PFTau                    " << double(nPFTausIn01Counter)/nPFTaus << endl;
        cout << endl;

//////////////////////////////////////////////////////////////////////////////////

	TCanvas* CaloTau_dEt = new TCanvas("CaloTau_dEt","",500,500);
	CaloTau_dEt->SetFillColor(0);
	CaloTau_dEt->SetFrameFillColor(0);
	CaloTau_dEt->SetLogy();
	CaloTau_dEt->cd();

	h_CaloTau_dEt->SetLineWidth(3);
	h_CaloTau_dEt->SetLineColor(2);
	h_CaloTau_dEt->SetLineStyle(2);
	h_CaloTau_dEt->SetStats(0);
	h_CaloTau_dEt->GetXaxis()->SetTitle("pt(RECO) - pt(MC) (GeV)");
	h_CaloTau_dEt->DrawClone();
/*
	h_CaloTau_dEt->SetLineWidth(3);
	h_CaloTau_dEt->DrawClone("same");
*/
	h_PFTau_dEt->SetLineWidth(3);
        h_PFTau_dEt->SetLineColor(3);
        h_PFTau_dEt->SetLineStyle(3);
        h_PFTau_dEt->DrawClone("same");

	CaloTau_dEt->Print("CaloTau_dEt.C"); 

//


        TCanvas* CaloTau_dEtRatio = new TCanvas("CaloTau_dEtRatio","",500,500);
        CaloTau_dEtRatio->SetFillColor(0);
        CaloTau_dEtRatio->SetFrameFillColor(0);
//        CaloTau_dEtRatio->SetLogy();
        CaloTau_dEtRatio->cd();

        h_CaloTau_dEtRatio->SetLineWidth(3);
        h_CaloTau_dEtRatio->SetLineColor(2);
        h_CaloTau_dEtRatio->SetLineStyle(2);
	h_CaloTau_dEtRatio->SetStats(0);
	h_CaloTau_dEtRatio->GetXaxis()->SetTitle("pt(RECO)/pt(MC)");
        h_CaloTau_dEtRatio->DrawClone();

	h_CaloTau_dEtRawRatio->SetLineWidth(3);
        h_CaloTau_dEtRawRatio->SetLineColor(6);
        h_CaloTau_dEtRawRatio->SetLineStyle(3);
        h_CaloTau_dEtRawRatio->DrawClone("same");
/*
        h_CaloTau_dEtRatio->SetLineWidth(3);
        h_CaloTau_dEtRatio->DrawClone("same");
*/
        h_JPTTau_dEtRatio->SetLineWidth(3);
	h_JPTTau_dEtRatio->SetLineColor(2);
        h_JPTTau_dEtRatio->DrawClone("same");

        h_PFTau_dEtRatio->SetLineWidth(4);
        h_PFTau_dEtRatio->SetLineColor(4);
        h_PFTau_dEtRatio->SetLineStyle(4);
        h_PFTau_dEtRatio->DrawClone("same");

	float CaloTau_dEtRatioFigureMax = h_PFTau_dEtRatio->GetMaximum();
	TLatex* tex = new TLatex(1.4,0.8*CaloTau_dEtRatioFigureMax,"CaloTau");
	tex->SetLineWidth(2);
	tex->DrawClone();	
        TLatex* tex = new TLatex(1.4,0.7*CaloTau_dEtRatioFigureMax,"CaloTau");
        tex->SetLineWidth(2);
        tex->DrawClone();
        TLatex* tex = new TLatex(1.4,0.6*CaloTau_dEtRatioFigureMax,"PFTau");
        tex->SetLineWidth(2);
        tex->DrawClone();
        TLatex* tex = new TLatex(1.4,0.5*CaloTau_dEtRatioFigureMax,"CaloTau(raw)");
        tex->SetLineWidth(2);
        tex->DrawClone();

	TLine *line = new TLine(1.1,0.82*CaloTau_dEtRatioFigureMax,1.3,0.82*CaloTau_dEtRatioFigureMax);
   	line->SetLineWidth(3);
   	line->DrawClone();

        TLine *line = new TLine(1.1,0.72*CaloTau_dEtRatioFigureMax,1.3,0.72*CaloTau_dEtRatioFigureMax);
        line->SetLineWidth(3);
	line->SetLineColor(2);
	line->SetLineStyle(2);
        line->DrawClone();

        TLine *line = new TLine(1.1,0.62*CaloTau_dEtRatioFigureMax,1.3,0.62*CaloTau_dEtRatioFigureMax);
        line->SetLineWidth(3);
	line->SetLineColor(4);
	line->SetLineStyle(3);
        line->DrawClone();

        TLine *line = new TLine(1.1,0.52*CaloTau_dEtRatioFigureMax,1.3,0.52*CaloTau_dEtRatioFigureMax);
        line->SetLineWidth(3);
        line->SetLineColor(6);
        line->SetLineStyle(3);
        line->DrawClone();

        CaloTau_dEtRatio->Print("CaloTau_dEtRatio.C");

////

        TCanvas* CaloTau_dEta = new TCanvas("CaloTau_dEta","",500,500);
        CaloTau_dEta->SetFillColor(0);
        CaloTau_dEta->SetFrameFillColor(0);
        CaloTau_dEta->SetLogy();
        CaloTau_dEta->cd();

        h_CaloTau_dEta->SetLineWidth(3);
        h_CaloTau_dEta->SetLineColor(2);
        h_CaloTau_dEta->SetLineStyle(2);
	h_CaloTau_dEta->SetStats(0);
        h_CaloTau_dEta->GetXaxis()->SetTitle("eta(RECO) - eta(MC)");
        h_CaloTau_dEta->DrawClone();
/*
        h_CaloTau_dEta->SetLineWidth(3);
        h_CaloTau_dEta->DrawClone("same");
*/
        h_PFTau_dEta->SetLineWidth(3);
        h_PFTau_dEta->SetLineColor(3);
        h_PFTau_dEta->SetLineStyle(3);
        h_PFTau_dEta->DrawClone("same");

        CaloTau_dEta->Print("CaloTau_dEta.C");

//

        TCanvas* CaloTau_dPhi = new TCanvas("CaloTau_dPhi","",500,500);
        CaloTau_dPhi->SetFillColor(0);
        CaloTau_dPhi->SetFrameFillColor(0);
        CaloTau_dPhi->SetLogy();
        CaloTau_dPhi->cd();

        h_CaloTau_dPhi->SetLineWidth(3);
        h_CaloTau_dPhi->SetLineColor(2);
        h_CaloTau_dPhi->SetLineStyle(2);
        h_CaloTau_dPhi->SetStats(0);
        h_CaloTau_dPhi->GetXaxis()->SetTitle("phi(RECO) - phi(MC)");
        h_CaloTau_dPhi->DrawClone();
/*
        h_CaloTau_dPhi->SetLineWidth(3);
        h_CaloTau_dPhi->DrawClone("same");
*/
        h_PFTau_dPhi->SetLineWidth(3);
        h_PFTau_dPhi->SetLineColor(3);
        h_PFTau_dPhi->SetLineStyle(3);
        h_PFTau_dPhi->DrawClone("same");

        CaloTau_dPhi->Print("CaloTau_dPhi.C");

//

        TCanvas* CaloTau_dEtNormalized = new TCanvas("CaloTau_dEtNormalized","",500,500);
        CaloTau_dEtNormalized->SetFillColor(0);
        CaloTau_dEtNormalized->SetFrameFillColor(0);
        CaloTau_dEtNormalized->cd();

        h_CaloTau_dEt->SetLineWidth(3);
        h_CaloTau_dEt->SetLineColor(2);
        h_CaloTau_dEt->SetLineStyle(2);
        h_CaloTau_dEt->SetStats(0);
        h_CaloTau_dEt->GetXaxis()->SetTitle("pt(RECO) - pt(MC) (GeV)");
	h_CaloTau_dEt->Scale(1/h_CaloTau_dEt->Integral());
        h_CaloTau_dEt->DrawClone();
/*
        h_CaloTau_dEt->SetLineWidth(3);
	h_CaloTau_dEt->Scale(1/h_CaloTau_dEt->Integral());
        h_CaloTau_dEt->DrawClone("same");
*/
        h_PFTau_dEt->SetLineWidth(3);
        h_PFTau_dEt->SetLineColor(3);
        h_PFTau_dEt->SetLineStyle(3);
	h_PFTau_dEt->Scale(1/h_PFTau_dEt->Integral());
        h_PFTau_dEt->DrawClone("same");

        CaloTau_dEtNormalized->Print("CaloTau_dEtNormalized.C");

	//

        TCanvas* CaloTau_dEtRatioNormalized = new TCanvas("CaloTau_dEtRatioNormalized","",500,500);
        CaloTau_dEtRatioNormalized->SetFillColor(0);
        CaloTau_dEtRatioNormalized->SetFrameFillColor(0);
        CaloTau_dEtRatioNormalized->cd();

	double CaloTau_dEtRatioScale = 1/h_CaloTau_dEtRatio->Integral();

        h_CaloTau_dEtRatio->SetLineWidth(3);
        h_CaloTau_dEtRatio->SetLineColor(2);
        h_CaloTau_dEtRatio->SetLineStyle(2);
        h_CaloTau_dEtRatio->SetStats(0);
        h_CaloTau_dEtRatio->GetXaxis()->SetTitle("pt(RECO)/pt(MC)");
        h_CaloTau_dEtRatio->Scale(CaloTau_dEtRatioScale);
        h_CaloTau_dEtRatio->DrawClone();
/*
        h_CaloTau_dEtRatio->SetLineWidth(3);
        h_CaloTau_dEtRatio->Scale(1/h_CaloTau_dEtRatio->Integral());
        h_CaloTau_dEtRatio->DrawClone("same");
*/
        h_PFTau_dEtRatio->SetLineWidth(3);
        h_PFTau_dEtRatio->SetLineColor(3);
        h_PFTau_dEtRatio->SetLineStyle(3);
        h_PFTau_dEtRatio->Scale(1/h_PFTau_dEtRatio->Integral());
        h_PFTau_dEtRatio->DrawClone("same");

        CaloTau_dEtRatioNormalized->Print("CaloTau_dEtRatioNormalized.C");

	//

	TCanvas* CaloTau_dEtRatioAlgosNormalized = new TCanvas("CaloTau_dEtRatioAlgosNormalized","",500,500);
        CaloTau_dEtRatioAlgosNormalized->SetFillColor(0);
        CaloTau_dEtRatioAlgosNormalized->SetFrameFillColor(0);
        CaloTau_dEtRatioAlgosNormalized->cd();
	CaloTau_dEtRatioAlgosNormalized->SetLogy();

        h_CaloTau_dEtRatio->SetLineWidth(3);
        h_CaloTau_dEtRatio->SetLineColor(1);
        h_CaloTau_dEtRatio->SetLineStyle(1);
        h_CaloTau_dEtRatio->SetStats(0);
        h_CaloTau_dEtRatio->GetXaxis()->SetTitle("pt(RECO)/pt(MC)");
//        h_CaloTau_dEtRatio->Scale(1/h_CaloTau_dEtRatio->Integral());
        h_CaloTau_dEtRatio->DrawClone();	

	h_CaloTau_dEt_TCAlgoUndetermined->SetLineWidth(1);
	h_CaloTau_dEt_TCAlgoUndetermined->SetLineColor(2);
	//h_CaloTau_dEt_TCAlgoUndetermined->SetLineStyle(2);
	h_CaloTau_dEt_TCAlgoUndetermined->Scale(CaloTau_dEtRatioScale);
	h_CaloTau_dEt_TCAlgoUndetermined->DrawClone("same");

        h_CaloTau_dEt_TCAlgoMomentum->SetLineWidth(1);
        h_CaloTau_dEt_TCAlgoMomentum->SetLineColor(3);
        //h_CaloTau_dEt_TCAlgoMomentum->SetLineStyle(2);
        h_CaloTau_dEt_TCAlgoMomentum->Scale(CaloTau_dEtRatioScale);
        h_CaloTau_dEt_TCAlgoMomentum->DrawClone("same");

        h_CaloTau_dEt_TCAlgoTrackProblem->SetLineWidth(1);
        h_CaloTau_dEt_TCAlgoTrackProblem->SetLineColor(4);
        //h_CaloTau_dEt_TCAlgoTrackProblem->SetLineStyle(2);
        h_CaloTau_dEt_TCAlgoTrackProblem->Scale(CaloTau_dEtRatioScale);
        h_CaloTau_dEt_TCAlgoTrackProblem->DrawClone("same");

        h_CaloTau_dEt_TCAlgoMomentumECAL->SetLineWidth(1);
        h_CaloTau_dEt_TCAlgoMomentumECAL->SetLineColor(6);
        //h_CaloTau_dEt_TCAlgoMomentumECAL->SetLineStyle(2);
        h_CaloTau_dEt_TCAlgoMomentumECAL->Scale(CaloTau_dEtRatioScale);
        h_CaloTau_dEt_TCAlgoMomentumECAL->DrawClone("same");

        h_CaloTau_dEt_TCAlgoCaloJet->SetLineWidth(1);
        h_CaloTau_dEt_TCAlgoCaloJet->SetLineColor(7);
        //h_CaloTau_dEt_TCAlgoCaloJet->SetLineStyle(2);
        h_CaloTau_dEt_TCAlgoCaloJet->Scale(CaloTau_dEtRatioScale);
        h_CaloTau_dEt_TCAlgoCaloJet->DrawClone("same");

        h_CaloTau_dEt_TCAlgoHadronicJet->SetLineWidth(1);
        h_CaloTau_dEt_TCAlgoHadronicJet->SetLineColor(8);
        //h_CaloTau_dEt_TCAlgoHadronicJet->SetLineStyle(2);
        h_CaloTau_dEt_TCAlgoHadronicJet->Scale(CaloTau_dEtRatioScale);
        h_CaloTau_dEt_TCAlgoHadronicJet->DrawClone("same");

	leg = new TLegend(0.55,0.72,0.94,0.92);
   	leg->SetHeader("CaloTau algo components");
	leg->AddEntry(h_CaloTau_dEtRatio,"Sum of all components","f");
   	leg->AddEntry(h_CaloTau_dEt_TCAlgoUndetermined,"TCAlgoUndetermined","f");
	leg->AddEntry(h_CaloTau_dEt_TCAlgoMomentum,"TCAlgoMomentum","f");
        leg->AddEntry(h_CaloTau_dEt_TCAlgoTrackProblem,"TCAlgoTrackProblem","f");
        leg->AddEntry(h_CaloTau_dEt_TCAlgoMomentumECAL,"TCAlgoMomentumECAL","f");
        leg->AddEntry(h_CaloTau_dEt_TCAlgoCaloJet,"TCAlgoCaloJet","f");
        leg->AddEntry(h_CaloTau_dEt_TCAlgoHadronicJet,"TCAlgoHadronicJet","f");
   	leg->Draw();

	CaloTau_dEtRatioAlgosNormalized->Print("CaloTau_dEtRatioAlgosNormalized.C");


        TCanvas* CaloTau_dEtRatioTCAlgoCaloJet = new TCanvas("CaloTau_dEtRatioTCAlgoCaloJet","",500,500);
        CaloTau_dEtRatioTCAlgoCaloJet->SetFillColor(0);
        CaloTau_dEtRatioTCAlgoCaloJet->SetFrameFillColor(0);
        CaloTau_dEtRatioTCAlgoCaloJet->cd();
        CaloTau_dEtRatioTCAlgoCaloJet->SetLogy();

        h_CaloTau_dEt_TCAlgoCaloJet->SetLineWidth(1);
        h_CaloTau_dEt_TCAlgoCaloJet->SetLineColor(7);
        //h_CaloTau_dEt_TCAlgoCaloJet->SetLineStyle(2);
        h_CaloTau_dEt_TCAlgoCaloJet->Scale(CaloTau_dEtRatioScale);
        h_CaloTau_dEt_TCAlgoCaloJet->DrawClone();

}
