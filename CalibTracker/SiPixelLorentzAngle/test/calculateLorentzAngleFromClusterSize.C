double fitFunction(double *x, double *par)
{
	double a;
	if(x[0] < par[0]) a = par[1] + (x[0] - par[0]) * par[2]  + (x[0] - par[0]) *(x[0] - par[0]) * par[4] + (x[0] - par[0]) * (x[0] - par[0]) *(x[0] - par[0]) * par[6];
	else 		a = par[1] + (x[0] - par[0]) * par[3] + (x[0] - par[0]) * (x[0] - par[0]) * par[5] + (x[0] - par[0]) * (x[0] - par[0]) * (x[0] - par[0]) * par[7];
	return a;
}

double fitFunction1(double *x, double *par)
{
	double a;
	if(x[0] < par[0]) a = par[1] + (x[0] - par[0]) * par[2]  + (x[0] - par[0]) *(x[0] - par[0]) * par[4];
	else 		a = par[1] + (x[0] - par[0]) * par[3] + (x[0] - par[0]) * (x[0] - par[0]) * par[5];
	return a;
}

double fitFunction2(double *x, double *par)
{
	double a;
	if(x[0] < par[0]) a = par[1] + (x[0] - par[0]) * par[2];
	else 		a = par[1] + (x[0] - par[0]) * par[3];
	return a;
}

double fitFunction3(double *x, double *par)
{
	return par[1] + TMath::Sqrt(par[2] + par[3] * (x[0] - par[0]) * (x[0] - par[0]) );
}

Double_t fitf(Double_t *x,Double_t *par){	Double_t arg;  	if(x[0] < par[3]) {     	arg = par[1]*par[1] + par[2]*par[2]*(x[0]-par[3])*(x[0]-par[3]);  	} else {     	arg = par[1]*par[1] + par[4]*par[4]*(x[0]-par[3])*(x[0]-par[3]);  	}  	Double_t fitval = par[0]+sqrt(arg);  	return fitval;}
 
int calculateLorentzAngleFromClusterSize()
{
	
	int nhits = 0;
// 	setTDRStyle();
	cout << "hallo" << endl;
//	TFile *f = new TFile("/nfs/data5/wilke/TrackerPointing_ALL_V9/lorentzangle.root");
	TFile *f = new TFile("/data1/Users/wilke/LorentzAngle/CMSSW_2_2_3/lorentzangle_0T.root");
	f->cd();
// 	
	TF1 *f1 = new TF1("f1",fitFunction,80, 150, 8); 
	f1->SetParName(0,"p0");
	f1->SetParName(1,"p1");	
	f1->SetParName(2,"p2");
	f1->SetParName(3,"p3");
	f1->SetParName(4,"p4");
	f1->SetParName(5,"p5");
	f1->SetParameters(110,1,-0.01,0.01);
	
	TF1 *f2 = new TF1("f2",fitFunction1, -1., 1.0, 6); 
	f2->SetParName(0,"p0");
	f2->SetParName(1,"p1");	
	f2->SetParName(2,"p2");
	f2->SetParName(3,"p3");
	f2->SetParameters(-0.5, 1., -1.,1.0);
	
	TF1 *func = new TF1("func", fitf, -1., 1.0, 5);   	func->SetParameters(1.,0.1,1.6,-0.4,1.2);   	func->SetParNames("Offset","RMS Constant","SlopeL","cot(alpha)_min","SlopeR");
   	
   	TF1 *func_beta = new TF1("func_beta", fitf, -1., 1, 5);   	func_beta->SetParameters(1.,0.1,1.6,-0.,1.2);   	func_beta->SetParNames("Offset","RMS Constant","SlopeL","cot(beta)_min","SlopeR");   	
	int hist_drift_ = 200;
	int hist_depth_ = 50;
	double min_drift_ = -1000;
	double max_drift_ = 1000;
	double min_depth_ = -100;
	double max_depth_ = 400;
	double width_ = 0.0285;
	int anglebins = 90;	
	int anglebinscotan = 120;

	TH2F * h_sizex_alpha = new TH2F("h_sizex_alpha", "h_sizex_alpha", anglebins, 0, 180,10 , .5, 10.5);
	TH2F * h_sizex_alpha_cotan = new TH2F("h_sizex_cotanalpha", "h_sizex_cotanalpha", anglebinscotan, -3, 3,10 , .5, 10.5);
	TH2F * h_sizey_beta_cotan = new TH2F("h_sizey_cotanbeta", "h_sizey_cotanbeta", anglebinscotan, -3, 3,10 , .5, 10.5);
	TH2F * h_alpha_beta_cotan = new TH2F("h_cotanalpha_cotanbeta", "h_cotanalpha_cotanbeta", 100, -5, 5,100 , -5, 5);
	TH2F * h_alpha_beta = new TH2F("h_alpha_beta", "h_alpha_beta", anglebins, -180, 180,anglebins, -180, 180);
	TH1F * h_alpha_cotan = new TH1F("h_cotanalpha", "h_cotanalpha", anglebinscotan, -3, 3 );
	TH1F * h_beta_cotan = new TH1F("h_cotanbeta", "h_cotanbeta", anglebinscotan, -3, 3 );
	TH1F * h_alpha = new TH1F("h_alpha", "h_alpha", anglebins, -180, 180 );
	TH1F * h_beta = new TH1F("h_beta", "h_beta", anglebins, -180, 180 );
	TH1F * h_chi2 = new TH1F("h_chi2", "h_chi2", 200, 0, 10 );
	TH1F * h_charge = new TH1F("h_chi2", "h_chi2", 200, 0, 200 );
	
	int run_;
	int event_;
	int module_;
	int ladder_;
	int layer_;
	int isflipped_;
	float pt_;
	float eta_;
	float phi_;
	double chi2_;
	double ndof_;
	const int maxpix = 300;
	struct Pixinfo
	{
		int npix;
		float row[300];
		float col[300];
		float adc[300];
		float x[300];
		float y[300];
	} pixinfo_;
	
	struct Hit{
		float x;
		float y;
		double alpha;
		double beta;
		double gamma;
	}; 
	Hit simhit_, trackhit_;
	
	struct Clust {
		float x;
		float y;
		float charge;
		int size_x;
		int size_y;
		int maxPixelCol;
		int maxPixelRow;
		int minPixelCol;
		int minPixelRow;
	} clust_;
	
	struct Rechit {
		float x;
		float y;
	} rechit_;
	
	// fill the histrograms with the ntpl
	TTree * LATree = (TTree*)f->Get("SiPixelLorentzAngleTree_");
	int nentries = LATree->GetEntries();
	LATree->SetBranchAddress("run", &run_);
	LATree->SetBranchAddress("event", &event_);
	LATree->SetBranchAddress("module", &module_);
	LATree->SetBranchAddress("ladder", &ladder_);
	LATree->SetBranchAddress("layer", &layer_);
	LATree->SetBranchAddress("isflipped", &isflipped_);
	LATree->SetBranchAddress("pt", &pt_);
	LATree->SetBranchAddress("eta", &eta_);
	LATree->SetBranchAddress("phi", &phi_);
	LATree->SetBranchAddress("chi2", &chi2_);
	LATree->SetBranchAddress("ndof", &ndof_);
	LATree->SetBranchAddress("trackhit", &trackhit_);
	LATree->SetBranchAddress("simhit", &simhit_);
	LATree->SetBranchAddress("npix", &pixinfo_.npix);
	LATree->SetBranchAddress("rowpix", pixinfo_.row);
	LATree->SetBranchAddress("colpix", pixinfo_.col);
	LATree->SetBranchAddress("adc", pixinfo_.adc);
	LATree->SetBranchAddress("xpix", pixinfo_.x);
	LATree->SetBranchAddress("ypix", pixinfo_.y);
	LATree->SetBranchAddress("clust", &clust_);
	LATree->SetBranchAddress("rechit", &rechit_);
	
	cout << "Running over " << nentries << " hits" << endl;
	ofstream fAngles( "cotanangles.txt", ios::trunc );
	for(int ientrie = 0 ; ientrie < nentries ; ientrie++){
		LATree->GetEntry(ientrie);  
		bool large_pix = false;
		
		// is it a large pixel (needs to be excluded)
		for (int j = 0; j <  pixinfo_.npix; j++){
			int colpos = static_cast<int>(pixinfo_.col[j]);
			if (pixinfo_.row[j] == 0 || pixinfo_.row[j] == 79 || pixinfo_.row[j] == 80 || pixinfo_.row[j] == 159 || colpos % 52 == 0 || colpos % 52 == 51 ){
				large_pix = true;	
			}
		}
		
		// is it one of the problematic half ladders? (needs to be excluded)
		//if( (layer_ == 1 && (ladder_ == 5 || ladder_ == 6 || ladder_ == 15 || ladder_ == 16)) ||(layer_ == 2 && (ladder_ == 8 || ladder_ == 9 || ladder_ == 24 || ladder_ == 25)) ||(layer_ == 3 && (ladder_ ==11 || ladder_ == 12 || ladder_ == 33 || ladder_ == 34)) ) {
		//	continue;
	//s	}
		
		if(clust_.size_y < 2) continue;
		
//		double residual = TMath::Sqrt( (trackhit_.x - rechit_.x) * (trackhit_.x - rechit_.x) + (trackhit_.y - rechit_.y) * (trackhit_.y - rechit_.y) );
		//if(!isflipped_) continue;
		if(!large_pix){
			h_chi2->Fill((chi2_/ndof_));
			h_charge->Fill(clust_.charge);
		}
	
		if( (chi2_/ndof_) < 2 && !large_pix ){
			nhits++;
			if(trackhit_.alpha > 0){	
				h_sizex_alpha->Fill(trackhit_.alpha*180. / TMath::Pi(),clust_.size_x);
				h_sizex_alpha_cotan->Fill(TMath::Tan(TMath::Pi()/2. - trackhit_.alpha),clust_.size_x);
				h_sizey_beta_cotan->Fill(TMath::Tan(TMath::Pi()/2. - trackhit_.beta),clust_.size_y);
				h_alpha_cotan->Fill(TMath::Tan(TMath::Pi()/2. - trackhit_.alpha));
				h_beta_cotan->Fill(TMath::Tan(TMath::Pi()/2. - trackhit_.beta));
				fAngles << TMath::Tan(TMath::Pi()/2. - trackhit_.alpha) << "\t" << TMath::Tan(TMath::Pi()/2. - trackhit_.beta) << endl;
			}
			else{ 
				h_sizex_alpha->Fill( (trackhit_.alpha + TMath::Pi())*180. / TMath::Pi(),clust_.size_x);
				h_sizex_alpha_cotan->Fill(TMath::Tan(TMath::Pi()/2. - trackhit_.alpha),clust_.size_x);
				h_sizey_beta_cotan->Fill(TMath::Tan(TMath::Pi()/2. - trackhit_.beta),clust_.size_y);
				h_alpha_cotan->Fill(TMath::Tan(TMath::Pi()/2. - trackhit_.alpha));
				h_beta_cotan->Fill(TMath::Tan(TMath::Pi()/2. - trackhit_.beta));
				fAngles << TMath::Tan(TMath::Pi()/2. - trackhit_.alpha) << "\t" << TMath::Tan(TMath::Pi()/2. - trackhit_.beta) << endl;
			}
			h_alpha->Fill( trackhit_.alpha*180. / TMath::Pi());
			h_beta->Fill( trackhit_.beta*180. / TMath::Pi());
			h_alpha_beta->Fill( trackhit_.alpha*180. / TMath::Pi(), trackhit_.beta*180. / TMath::Pi());
			if(TMath::Tan(TMath::Pi()/2. - trackhit_.alpha) < -0.3 && TMath::Tan(TMath::Pi()/2. - trackhit_.alpha) > -0.5) h_alpha_beta_cotan->Fill(TMath::Tan(TMath::Pi()/2. - trackhit_.alpha), TMath::Tan(TMath::Pi()/2. - trackhit_.beta));
		} 
	}
	
	cout << TMath::Pi()/2. << endl;
	TH1F * h_mean = new TH1F("h_mean","h_mean", anglebins, 0, 180);
	TH1F * h_slice_ = new TH1F("h_slice","h_slice", 10, .5, 10.5);
	TH1F * h_mean_cotan_alpha = new TH1F("h_mean_cotan_alpha","h_mean_contan_alpha", anglebinscotan, -3, 3);
	TH1F * h_slice_cotan_alpha = new TH1F("h_slice_cotan_alpha","h_slice_cotan_alpha", 10, .5, 10.5);
	TH1F * h_mean_cotan_beta = new TH1F("h_mean_cotan_beta","h_mean_contan_beta", anglebinscotan, -3, 3);
	TH1F * h_slice_cotan_beta = new TH1F("h_slice_cotan_beta","h_slice_cotan_beta", 10, .5, 10.5);
	//loop over bins in depth (z-local-coordinate) (in order to fit slices)
	for( int i = 1; i <= anglebins; i++){
// 				findMean(i, (i_module + (i_layer - 1) * 8));
		h_slice_->Reset("ICE");
		
		// determine sigma and sigma^2 of the adc counts and average adc counts
			//loop over bins in drift width
		for( int j = 1; j<= 10; j++){
			h_slice_->SetBinContent(j,h_sizex_alpha->GetBinContent(i,j));
		} // end loop over bins in drift width
			
		double mean = h_slice_->GetMean(1); 
	  double error = h_slice_->GetMeanError(1);
			
		h_mean->SetBinContent(i, mean);
		h_mean->SetBinError(i, error);	
	}// end loop over bins in depth 
	
	for( int i = 1; i <= anglebinscotan; i++){
		h_slice_cotan_alpha->Reset("ICE");
		h_slice_cotan_beta->Reset("ICE");
		//loop over bins in drift width
		for( int j = 1; j<= 10; j++){
			h_slice_cotan_alpha->SetBinContent(j,h_sizex_alpha_cotan->GetBinContent(i,j));
			h_slice_cotan_beta->SetBinContent(j,h_sizey_beta_cotan->GetBinContent(i,j));
		} // end loop over bins in drift width
			
		double mean = h_slice_cotan_alpha->GetMean(1); 
		double error = h_slice_cotan_alpha->GetMeanError(1);
			
		h_mean_cotan_alpha->SetBinContent(i, mean);
		h_mean_cotan_alpha->SetBinError(i, error);	
		
		double mean_beta = h_slice_cotan_beta->GetMean(1); 
		double error_beta = h_slice_cotan_beta->GetMeanError(1);
			
		h_mean_cotan_beta->SetBinContent(i, mean_beta);
		h_mean_cotan_beta->SetBinError(i, error_beta);	
	}// end loop over bins in depth 
	
	
	gStyle->SetOptStat(0);
	gStyle->SetOptTitle(0);
	TCanvas * c1 = new TCanvas("c1", "c1", 1200, 600);
	c1->Divide(2,1);
	c1->cd(1);
	h_sizex_alpha->GetXaxis()->SetTitle("#alpha [^{o}]");
	h_sizex_alpha->GetYaxis()->SetTitle("cluster size [pixels]");
	h_sizex_alpha->Draw("colz");
	c1->cd(2);
	h_mean->GetXaxis()->SetTitle("#alpha [^{o}]");
	h_mean->GetYaxis()->SetTitle("average cluster size [pixels]");
	h_mean->Draw();
	h_mean->Fit(f1,"ERQ");
	
	TCanvas * c2 = new TCanvas("c2", "c2", 600, 600);
	//c2->Divide(2,1);
	//c2->cd(1);
	h_sizex_alpha_cotan->GetXaxis()->SetTitle("cotan(#alpha)");
	h_sizex_alpha_cotan->GetYaxis()->SetTitle("cluster size x[pixels]");
	h_sizex_alpha_cotan->Draw("col");
	//c2->cd(2);	
	h_mean_cotan_alpha->GetXaxis()->SetTitle("cotan(#alpha)");
	h_mean_cotan_alpha->GetYaxis()->SetTitle("transverse cluster size [pixels]");
	//h_mean_cotan_alpha->Draw();

	
	TCanvas * c3 = new TCanvas("c3", "c3", 600, 600);
	h_mean_cotan_alpha->Draw();
	h_mean_cotan_alpha->Fit(func,"ERQ");
	double correlation = h_alpha_beta_cotan->GetCorrelationFactor();
	cout << "corrlation between cotan(alpha) and cotan(beta): " << correlation << endl;
	cout << "number of hits used: " << nhits << endl;
	
	TCanvas * c8 = new TCanvas("c8", "c8", 600, 600);
	h_mean_cotan_beta->GetXaxis()->SetTitle("cotan(#beta)");
	h_mean_cotan_beta->GetYaxis()->SetTitle("longitudinal cluster size [pixels]");
	h_mean_cotan_beta->Draw();
	h_mean_cotan_beta->Fit(func_beta,"ERQ");
	
	TCanvas * c4 = new TCanvas("c4", "c4", 600, 600);
	h_alpha_cotan->Draw();
	
	TCanvas * c5 = new TCanvas("c5", "c5", 600, 600);
	h_beta_cotan->Draw();
	
	TCanvas * c6 = new TCanvas("c6", "c6", 600, 600);
	h_alpha->GetXaxis()->SetTitle("#alpha [^{o}]");
	h_alpha->GetYaxis()->SetTitle("number of hits");
	h_alpha->Draw();
	
	
	TCanvas * c7 = new TCanvas("c7", "c7", 600, 600);
	h_beta->GetXaxis()->SetTitle("#beta [^{o}]");
	h_beta->GetYaxis()->SetTitle("number of hits");
	h_beta->Draw();
	
	TCanvas * c9 = new TCanvas("c9", "c9", 600, 600);
	h_alpha_beta->GetXaxis()->SetTitle("#alpha [^{o}]");
	h_alpha_beta->GetYaxis()->SetTitle("#beta [^{o}]");
	h_alpha_beta->Draw("col");
	
	TCanvas * c10 = new TCanvas("c10", "c10", 600, 600);
	h_chi2->GetXaxis()->SetTitle("#chi^{2}/ndof");
	h_chi2->GetYaxis()->SetTitle("number of hits");
	h_chi2->Draw();
	
	TCanvas * c11 = new TCanvas("c11", "c11", 600, 600);
	h_charge->GetXaxis()->SetTitle("charge [1000 e^{-}]");
	h_charge->GetYaxis()->SetTitle("number of hits");
	h_charge->Draw();
	return 0;
}
