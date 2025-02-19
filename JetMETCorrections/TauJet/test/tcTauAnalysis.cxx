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

	TFile* inFile = TFile::Open("tctau.root");
	TTree* tauTree = (TTree *) (inFile->Get("tauTree"));

	cout << "Entries " << tauTree->GetEntries() << endl;	

        float MCTau_pt,MCTau_eta,MCTau_phi;
        float PFTau_pt,PFTau_eta,PFTau_phi,PFTau_nProngs,PFTau_ltrackPt,PFTau_d_isol,PFTau_d_1,PFTau_d_2;
        float CaloTau_pt,CaloTau_eta,CaloTau_phi,CaloTau_nProngs,CaloTau_ltrackPt,CaloTau_d_isol,CaloTau_d_1,CaloTau_d_2;
	float JPTTau_pt,JPTTau_eta,JPTTau_phi;
        float TCTau_pt,TCTau_eta,TCTau_phi,TCTau_nProngs,TCTau_ltrackPt,TCTau_d_1,TCTau_d_2,TCTau_algo;
	float TCTau_pt_raw,TCTau_eta_raw,TCTau_phi_raw;

	tauTree->SetBranchAddress("MCTau_pt",&MCTau_pt);
        tauTree->SetBranchAddress("MCTau_eta",&MCTau_eta);
        tauTree->SetBranchAddress("MCTau_phi",&MCTau_phi);
        tauTree->SetBranchAddress("CaloTau_pt",&CaloTau_pt);
        tauTree->SetBranchAddress("CaloTau_eta",&CaloTau_eta);
        tauTree->SetBranchAddress("CaloTau_phi",&CaloTau_phi);
        tauTree->SetBranchAddress("CaloTau_nProngs",&CaloTau_nProngs);
        tauTree->SetBranchAddress("CaloTau_ltrackPt",&CaloTau_ltrackPt);
        tauTree->SetBranchAddress("CaloTau_d_isol",&CaloTau_d_isol);
	tauTree->SetBranchAddress("JPTTau_pt",&JPTTau_pt);
	tauTree->SetBranchAddress("JPTTau_eta",&JPTTau_eta);
	tauTree->SetBranchAddress("JPTTau_phi",&JPTTau_phi);
        tauTree->SetBranchAddress("TCTau_pt",&TCTau_pt);
        tauTree->SetBranchAddress("TCTau_eta",&TCTau_eta);
        tauTree->SetBranchAddress("TCTau_phi",&TCTau_phi);
        tauTree->SetBranchAddress("TCTau_pt_raw",&TCTau_pt_raw);
        tauTree->SetBranchAddress("TCTau_eta_raw",&TCTau_eta_raw);
        tauTree->SetBranchAddress("TCTau_phi_raw",&TCTau_phi_raw);
        tauTree->SetBranchAddress("TCTau_nProngs",&TCTau_nProngs);
        tauTree->SetBranchAddress("TCTau_ltrackPt",&TCTau_ltrackPt);
	tauTree->SetBranchAddress("TCTau_algo",&TCTau_algo);
        tauTree->SetBranchAddress("PFTau_pt",&PFTau_pt);
        tauTree->SetBranchAddress("PFTau_eta",&PFTau_eta);
        tauTree->SetBranchAddress("PFTau_phi",&PFTau_phi);
        tauTree->SetBranchAddress("PFTau_nProngs",&PFTau_nProngs);
        tauTree->SetBranchAddress("PFTau_ltrackPt",&PFTau_ltrackPt);
        tauTree->SetBranchAddress("PFTau_d_isol",&PFTau_d_isol);


	int nMCTaus   = 0,
	    nCaloTaus = 0,
	    nTCTaus   = 0,
	    nPFTaus   = 0;
	int nCaloTausIn01Counter = 0,
	    nTCTausIn01Counter   = 0,
	    nPFTausIn01Counter   = 0;

	TH1F* h_CaloTau_dEt = new TH1F("h_CaloTau_dEt","",100,-50,50);
	TH1F* h_jptTau_dEt  = (TH1F*)h_CaloTau_dEt->Clone("h_JPTTau_dEt");
	TH1F* h_TCTau_dEt   = (TH1F*)h_CaloTau_dEt->Clone("h_TCTau_dEt");
	TH1F* h_PFTau_dEt   = (TH1F*)h_CaloTau_dEt->Clone("h_PFTau_dEt");

        TH1F* h_TCTau_dEtRaw = (TH1F*)h_CaloTau_dEt->Clone("h_TCTau_dEtRaw");

	TH1F* h_PFTauRef_dEt = (TH1F*)h_CaloTau_dEt->Clone("h_PFTauRef_dEt");

	TH1F* h_CaloTau_dEta = new TH1F("h_CaloTau_dEta","",100,-0.5,0.5);
	TH1F* h_TCTau_dEta   = (TH1F*)h_CaloTau_dEta->Clone("h_TCTau_dEta");
	TH1F* h_PFTau_dEta   = (TH1F*)h_CaloTau_dEta->Clone("h_PFTau_dEta");

        TH1F* h_CaloTau_dPhi = new TH1F("h_CaloTau_dPhi","",100,-0.5,0.5);
	TH1F* h_TCTau_dPhi = (TH1F*)h_CaloTau_dPhi->Clone("h_TCTau_dPhi");
	TH1F* h_PFTau_dPhi = (TH1F*)h_CaloTau_dPhi->Clone("h_PFTau_dPhi");

        TH1F* h_CaloTau_dEtRatio = new TH1F("h_CaloTau_dEtRatio","",100,0,2);
	TH1F* h_TCTau_dEtRatio = (TH1F*)h_CaloTau_dEtRatio->Clone("h_TCTau_dEtRatio");
	TH1F* h_JPTTau_dEtRatio = (TH1F*)h_CaloTau_dEtRatio->Clone("h_JPTTau_dEtRatio");
	TH1F* h_PFTau_dEtRatio = (TH1F*)h_CaloTau_dEtRatio->Clone("h_PFTau_dEtRatio");
        TH1F* h_TCTau_dEtRawRatio = (TH1F*)h_CaloTau_dEtRatio->Clone("h_TCTau_dEtRawRatio");

        TH1F* h_TCTau_dEt_TCAlgoUndetermined = (TH1F*)h_CaloTau_dEtRatio->Clone("h_TCTau_dEt_TCAlgoUndetermined");
        TH1F* h_TCTau_dEt_TCAlgoMomentum     = (TH1F*)h_CaloTau_dEtRatio->Clone("h_TCTau_dEt_TCAlgoMomentum");
        TH1F* h_TCTau_dEt_TCAlgoTrackProblem = (TH1F*)h_CaloTau_dEtRatio->Clone("h_TCTau_dEt_TCAlgoTrackProblem");
        TH1F* h_TCTau_dEt_TCAlgoMomentumECAL = (TH1F*)h_CaloTau_dEtRatio->Clone("h_TCTau_dEt_TCAlgoMomentumECAL");
        TH1F* h_TCTau_dEt_TCAlgoCaloJet      = (TH1F*)h_CaloTau_dEtRatio->Clone("h_TCTau_dEt_TCAlgoCaloJet");
        TH1F* h_TCTau_dEt_TCAlgoHadronicJet  = (TH1F*)h_CaloTau_dEtRatio->Clone("h_TCTau_dEt_TCAlgoHadronicJet");

	for(int i = 0; i < tauTree->GetEntries(); ++i){
		tauTree->GetEntry(i);

		if(MCTau_pt <= 0) continue; 
		nMCTaus++;

		if(CaloTau_pt > tau_pt_cut && fabs(CaloTau_eta) < tau_eta_cut && CaloTau_d_isol != 0) {
			nCaloTaus++;
			double caloTauReso = (CaloTau_pt - MCTau_pt)/MCTau_pt;
			if(fabs(caloTauReso) < 0.1) nCaloTausIn01Counter++;

			h_CaloTau_dEt->Fill(CaloTau_pt - MCTau_pt);
			h_CaloTau_dEta->Fill(CaloTau_eta - MCTau_eta);
			h_CaloTau_dPhi->Fill(deltaPhi(CaloTau_phi,MCTau_phi));
			h_CaloTau_dEtRatio->Fill(CaloTau_pt/MCTau_pt);
		}
		if(TCTau_pt > tau_pt_cut && fabs(TCTau_eta) < tau_eta_cut && CaloTau_d_isol != 0) {

			algoCounter[int(TCTau_algo)]++;

			nTCTaus++;
                        double tcTauReso = (TCTau_pt - MCTau_pt)/MCTau_pt;
                        if(fabs(tcTauReso) < 0.1) nTCTausIn01Counter++;

			h_TCTau_dEt->Fill(TCTau_pt - MCTau_pt);
			h_TCTau_dEta->Fill(TCTau_eta - MCTau_eta);
			h_TCTau_dPhi->Fill(deltaPhi(TCTau_phi,MCTau_phi));
			h_TCTau_dEtRatio->Fill(TCTau_pt/MCTau_pt);

			h_TCTau_dEtRawRatio->Fill(TCTau_pt_raw/MCTau_pt);

			h_JPTTau_dEtRatio->Fill(JPTTau_pt/MCTau_pt);

                        if(TCTau_algo == TCAlgoUndetermined) h_TCTau_dEt_TCAlgoUndetermined->Fill(TCTau_pt/MCTau_pt);
                        if(TCTau_algo == TCAlgoMomentum)     h_TCTau_dEt_TCAlgoMomentum->Fill(TCTau_pt/MCTau_pt);
                        if(TCTau_algo == TCAlgoTrackProblem) h_TCTau_dEt_TCAlgoTrackProblem->Fill(TCTau_pt/MCTau_pt);
                        if(TCTau_algo == TCAlgoMomentumECAL) h_TCTau_dEt_TCAlgoMomentumECAL->Fill(TCTau_pt/MCTau_pt);
                        if(TCTau_algo == TCAlgoCaloJet)      h_TCTau_dEt_TCAlgoCaloJet->Fill(TCTau_pt/MCTau_pt);
			if(TCTau_algo == TCAlgoHadronicJet)  h_TCTau_dEt_TCAlgoHadronicJet->Fill(TCTau_pt/MCTau_pt);

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
	cout << " Isolated TCTaus   " << nTCTaus << endl;
	cout << " Isolated PFTaus   " << nPFTaus << endl;
	cout << endl;
/*
        enum  {TCAlgoUndetermined,
               TCAlgoMomentum,
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

	
	//TH1F* h_tcTauEff = (TH1F*) (inFile->Get("h_tcTauEff"));
	//cout << " TCTau algorithm efficiency " << h_tcTauEff->GetBinContent(3) << endl;

        cout << endl;
        cout << " Fraction of jets in abs(dEt) < 0.1, reco::CaloTau                  " << double(nCaloTausIn01Counter)/nCaloTaus << endl;
        cout << " Fraction of jets in abs(dEt) < 0.1, reco::CaloTau+JPT+TCTau        " << double(nTCTausIn01Counter)/nTCTaus << endl;
        cout << " Fraction of jets in abs(dEt) < 0.1, reco::PFTau                    " << double(nPFTausIn01Counter)/nPFTaus << endl;
        cout << endl;

//////////////////////////////////////////////////////////////////////////////////

	TCanvas* tctau_dEt = new TCanvas("tctau_dEt","",500,500);
	tctau_dEt->SetFillColor(0);
	tctau_dEt->SetFrameFillColor(0);
	tctau_dEt->SetLogy();
	tctau_dEt->cd();

	h_TCTau_dEt->SetLineWidth(3);
	h_TCTau_dEt->SetLineColor(2);
	h_TCTau_dEt->SetLineStyle(2);
	h_TCTau_dEt->SetStats(0);
	h_TCTau_dEt->GetXaxis()->SetTitle("pt(RECO) - pt(MC) (GeV)");
	h_TCTau_dEt->DrawClone();

	h_CaloTau_dEt->SetLineWidth(3);
	h_CaloTau_dEt->DrawClone("same");

	h_PFTau_dEt->SetLineWidth(3);
        h_PFTau_dEt->SetLineColor(3);
        h_PFTau_dEt->SetLineStyle(3);
        h_PFTau_dEt->DrawClone("same");

	tctau_dEt->Print("tctau_dEt.C"); 

//


        TCanvas* tctau_dEtRatio = new TCanvas("tctau_dEtRatio","",500,500);
        tctau_dEtRatio->SetFillColor(0);
        tctau_dEtRatio->SetFrameFillColor(0);
//        tctau_dEtRatio->SetLogy();
        tctau_dEtRatio->cd();

        h_TCTau_dEtRatio->SetLineWidth(3);
        h_TCTau_dEtRatio->SetLineColor(2);
        h_TCTau_dEtRatio->SetLineStyle(2);
	h_TCTau_dEtRatio->SetStats(0);
	h_TCTau_dEtRatio->GetXaxis()->SetTitle("pt(RECO)/pt(MC)");
        h_TCTau_dEtRatio->DrawClone();

	h_TCTau_dEtRawRatio->SetLineWidth(3);
        h_TCTau_dEtRawRatio->SetLineColor(6);
        h_TCTau_dEtRawRatio->SetLineStyle(3);
        h_TCTau_dEtRawRatio->DrawClone("same");

        h_CaloTau_dEtRatio->SetLineWidth(3);
        h_CaloTau_dEtRatio->DrawClone("same");

        h_JPTTau_dEtRatio->SetLineWidth(3);
	h_JPTTau_dEtRatio->SetLineColor(2);
        h_JPTTau_dEtRatio->DrawClone("same");

        h_PFTau_dEtRatio->SetLineWidth(4);
        h_PFTau_dEtRatio->SetLineColor(4);
        h_PFTau_dEtRatio->SetLineStyle(4);
        h_PFTau_dEtRatio->DrawClone("same");

	float tctau_dEtRatioFigureMax = h_PFTau_dEtRatio->GetMaximum();
	TLatex* tex = new TLatex(1.4,0.8*tctau_dEtRatioFigureMax,"CaloTau");
	tex->SetLineWidth(2);
	tex->DrawClone();	
        TLatex* tex = new TLatex(1.4,0.7*tctau_dEtRatioFigureMax,"TCTau");
        tex->SetLineWidth(2);
        tex->DrawClone();
        TLatex* tex = new TLatex(1.4,0.6*tctau_dEtRatioFigureMax,"PFTau");
        tex->SetLineWidth(2);
        tex->DrawClone();
        TLatex* tex = new TLatex(1.4,0.5*tctau_dEtRatioFigureMax,"TCTau(raw)");
        tex->SetLineWidth(2);
        tex->DrawClone();

	TLine *line = new TLine(1.1,0.82*tctau_dEtRatioFigureMax,1.3,0.82*tctau_dEtRatioFigureMax);
   	line->SetLineWidth(3);
   	line->DrawClone();

        TLine *line = new TLine(1.1,0.72*tctau_dEtRatioFigureMax,1.3,0.72*tctau_dEtRatioFigureMax);
        line->SetLineWidth(3);
	line->SetLineColor(2);
	line->SetLineStyle(2);
        line->DrawClone();

        TLine *line = new TLine(1.1,0.62*tctau_dEtRatioFigureMax,1.3,0.62*tctau_dEtRatioFigureMax);
        line->SetLineWidth(3);
	line->SetLineColor(4);
	line->SetLineStyle(3);
        line->DrawClone();

        TLine *line = new TLine(1.1,0.52*tctau_dEtRatioFigureMax,1.3,0.52*tctau_dEtRatioFigureMax);
        line->SetLineWidth(3);
        line->SetLineColor(6);
        line->SetLineStyle(3);
        line->DrawClone();

        tctau_dEtRatio->Print("tctau_dEtRatio.C");

////

        TCanvas* tctau_dEta = new TCanvas("tctau_dEta","",500,500);
        tctau_dEta->SetFillColor(0);
        tctau_dEta->SetFrameFillColor(0);
        tctau_dEta->SetLogy();
        tctau_dEta->cd();

        h_TCTau_dEta->SetLineWidth(3);
        h_TCTau_dEta->SetLineColor(2);
        h_TCTau_dEta->SetLineStyle(2);
	h_TCTau_dEta->SetStats(0);
        h_TCTau_dEta->GetXaxis()->SetTitle("eta(RECO) - eta(MC)");
        h_TCTau_dEta->DrawClone();

        h_CaloTau_dEta->SetLineWidth(3);
        h_CaloTau_dEta->DrawClone("same");

        h_PFTau_dEta->SetLineWidth(3);
        h_PFTau_dEta->SetLineColor(3);
        h_PFTau_dEta->SetLineStyle(3);
        h_PFTau_dEta->DrawClone("same");

        tctau_dEta->Print("tctau_dEta.C");

//

        TCanvas* tctau_dPhi = new TCanvas("tctau_dPhi","",500,500);
        tctau_dPhi->SetFillColor(0);
        tctau_dPhi->SetFrameFillColor(0);
        tctau_dPhi->SetLogy();
        tctau_dPhi->cd();

        h_TCTau_dPhi->SetLineWidth(3);
        h_TCTau_dPhi->SetLineColor(2);
        h_TCTau_dPhi->SetLineStyle(2);
        h_TCTau_dPhi->SetStats(0);
        h_TCTau_dPhi->GetXaxis()->SetTitle("phi(RECO) - phi(MC)");
        h_TCTau_dPhi->DrawClone();

        h_CaloTau_dPhi->SetLineWidth(3);
        h_CaloTau_dPhi->DrawClone("same");

        h_PFTau_dPhi->SetLineWidth(3);
        h_PFTau_dPhi->SetLineColor(3);
        h_PFTau_dPhi->SetLineStyle(3);
        h_PFTau_dPhi->DrawClone("same");

        tctau_dPhi->Print("tctau_dPhi.C");

//

        TCanvas* tctau_dEtNormalized = new TCanvas("tctau_dEtNormalized","",500,500);
        tctau_dEtNormalized->SetFillColor(0);
        tctau_dEtNormalized->SetFrameFillColor(0);
        tctau_dEtNormalized->cd();

        h_TCTau_dEt->SetLineWidth(3);
        h_TCTau_dEt->SetLineColor(2);
        h_TCTau_dEt->SetLineStyle(2);
        h_TCTau_dEt->SetStats(0);
        h_TCTau_dEt->GetXaxis()->SetTitle("pt(RECO) - pt(MC) (GeV)");
	h_TCTau_dEt->Scale(1/h_TCTau_dEt->Integral());
        h_TCTau_dEt->DrawClone();

        h_CaloTau_dEt->SetLineWidth(3);
	h_CaloTau_dEt->Scale(1/h_CaloTau_dEt->Integral());
        h_CaloTau_dEt->DrawClone("same");

        h_PFTau_dEt->SetLineWidth(3);
        h_PFTau_dEt->SetLineColor(3);
        h_PFTau_dEt->SetLineStyle(3);
	h_PFTau_dEt->Scale(1/h_PFTau_dEt->Integral());
        h_PFTau_dEt->DrawClone("same");

        tctau_dEtNormalized->Print("tctau_dEtNormalized.C");

	//

        TCanvas* tctau_dEtRatioNormalized = new TCanvas("tctau_dEtRatioNormalized","",500,500);
        tctau_dEtRatioNormalized->SetFillColor(0);
        tctau_dEtRatioNormalized->SetFrameFillColor(0);
        tctau_dEtRatioNormalized->cd();

	double TCTau_dEtRatioScale = 1/h_TCTau_dEtRatio->Integral();

        h_TCTau_dEtRatio->SetLineWidth(3);
        h_TCTau_dEtRatio->SetLineColor(2);
        h_TCTau_dEtRatio->SetLineStyle(2);
        h_TCTau_dEtRatio->SetStats(0);
        h_TCTau_dEtRatio->GetXaxis()->SetTitle("pt(RECO)/pt(MC)");
        h_TCTau_dEtRatio->Scale(TCTau_dEtRatioScale);
        h_TCTau_dEtRatio->DrawClone();

        h_CaloTau_dEtRatio->SetLineWidth(3);
        h_CaloTau_dEtRatio->Scale(1/h_CaloTau_dEtRatio->Integral());
        h_CaloTau_dEtRatio->DrawClone("same");

        h_PFTau_dEtRatio->SetLineWidth(3);
        h_PFTau_dEtRatio->SetLineColor(3);
        h_PFTau_dEtRatio->SetLineStyle(3);
        h_PFTau_dEtRatio->Scale(1/h_PFTau_dEtRatio->Integral());
        h_PFTau_dEtRatio->DrawClone("same");

        tctau_dEtRatioNormalized->Print("tctau_dEtRatioNormalized.C");

	//

	TCanvas* tctau_dEtRatioAlgosNormalized = new TCanvas("tctau_dEtRatioAlgosNormalized","",500,500);
        tctau_dEtRatioAlgosNormalized->SetFillColor(0);
        tctau_dEtRatioAlgosNormalized->SetFrameFillColor(0);
        tctau_dEtRatioAlgosNormalized->cd();
	tctau_dEtRatioAlgosNormalized->SetLogy();

        h_TCTau_dEtRatio->SetLineWidth(3);
        h_TCTau_dEtRatio->SetLineColor(1);
        h_TCTau_dEtRatio->SetLineStyle(1);
        h_TCTau_dEtRatio->SetStats(0);
        h_TCTau_dEtRatio->GetXaxis()->SetTitle("pt(RECO)/pt(MC)");
//        h_TCTau_dEtRatio->Scale(1/h_TCTau_dEtRatio->Integral());
        h_TCTau_dEtRatio->DrawClone();	

	h_TCTau_dEt_TCAlgoUndetermined->SetLineWidth(1);
	h_TCTau_dEt_TCAlgoUndetermined->SetLineColor(2);
	//h_TCTau_dEt_TCAlgoUndetermined->SetLineStyle(2);
	h_TCTau_dEt_TCAlgoUndetermined->Scale(TCTau_dEtRatioScale);
	h_TCTau_dEt_TCAlgoUndetermined->DrawClone("same");

        h_TCTau_dEt_TCAlgoMomentum->SetLineWidth(1);
        h_TCTau_dEt_TCAlgoMomentum->SetLineColor(3);
        //h_TCTau_dEt_TCAlgoMomentum->SetLineStyle(2);
        h_TCTau_dEt_TCAlgoMomentum->Scale(TCTau_dEtRatioScale);
        h_TCTau_dEt_TCAlgoMomentum->DrawClone("same");

        h_TCTau_dEt_TCAlgoTrackProblem->SetLineWidth(1);
        h_TCTau_dEt_TCAlgoTrackProblem->SetLineColor(4);
        //h_TCTau_dEt_TCAlgoTrackProblem->SetLineStyle(2);
        h_TCTau_dEt_TCAlgoTrackProblem->Scale(TCTau_dEtRatioScale);
        h_TCTau_dEt_TCAlgoTrackProblem->DrawClone("same");

        h_TCTau_dEt_TCAlgoMomentumECAL->SetLineWidth(1);
        h_TCTau_dEt_TCAlgoMomentumECAL->SetLineColor(6);
        //h_TCTau_dEt_TCAlgoMomentumECAL->SetLineStyle(2);
        h_TCTau_dEt_TCAlgoMomentumECAL->Scale(TCTau_dEtRatioScale);
        h_TCTau_dEt_TCAlgoMomentumECAL->DrawClone("same");

        h_TCTau_dEt_TCAlgoCaloJet->SetLineWidth(1);
        h_TCTau_dEt_TCAlgoCaloJet->SetLineColor(7);
        //h_TCTau_dEt_TCAlgoCaloJet->SetLineStyle(2);
        h_TCTau_dEt_TCAlgoCaloJet->Scale(TCTau_dEtRatioScale);
        h_TCTau_dEt_TCAlgoCaloJet->DrawClone("same");

        h_TCTau_dEt_TCAlgoHadronicJet->SetLineWidth(1);
        h_TCTau_dEt_TCAlgoHadronicJet->SetLineColor(8);
        //h_TCTau_dEt_TCAlgoHadronicJet->SetLineStyle(2);
        h_TCTau_dEt_TCAlgoHadronicJet->Scale(TCTau_dEtRatioScale);
        h_TCTau_dEt_TCAlgoHadronicJet->DrawClone("same");

	leg = new TLegend(0.55,0.72,0.94,0.92);
   	leg->SetHeader("TCTau algo components");
	leg->AddEntry(h_TCTau_dEtRatio,"Sum of all components","f");
   	leg->AddEntry(h_TCTau_dEt_TCAlgoUndetermined,"TCAlgoUndetermined","f");
	leg->AddEntry(h_TCTau_dEt_TCAlgoMomentum,"TCAlgoMomentum","f");
        leg->AddEntry(h_TCTau_dEt_TCAlgoTrackProblem,"TCAlgoTrackProblem","f");
        leg->AddEntry(h_TCTau_dEt_TCAlgoMomentumECAL,"TCAlgoMomentumECAL","f");
        leg->AddEntry(h_TCTau_dEt_TCAlgoCaloJet,"TCAlgoCaloJet","f");
        leg->AddEntry(h_TCTau_dEt_TCAlgoHadronicJet,"TCAlgoHadronicJet","f");
   	leg->Draw();

	tctau_dEtRatioAlgosNormalized->Print("tctau_dEtRatioAlgosNormalized.C");
}
