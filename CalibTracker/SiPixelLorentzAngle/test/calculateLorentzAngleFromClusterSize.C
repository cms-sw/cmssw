double fitFunction(double *x, double *par)
{
	double a;
	if(x[0] < par[0]) a = par[1] + (x[0] - par[0]) * par[2]  + (x[0] - par[0]) *(x[0] - par[0]) * par[4] + (x[0] - par[0]) * (x[0] - par[0]) *(x[0] - par[0]) * par[6];
	else 		a = par[1] + (x[0] - par[0]) * par[3] + (x[0] - par[0]) * (x[0] - par[0]) * par[5] + (x[0] - par[0]) * (x[0] - par[0]) * (x[0] - par[0]) * par[7];
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

int calculateLorentzAngleFromClusterSize()
{
// 	setTDRStyle();
	cout << "hallo" << endl;
	TFile *f = new TFile("/localscratch/w/wilke/crab/crab_0_081127_134601/res/lorentzangle.root");
	f->cd();
// 	
	TF1 *f1 = new TF1("f1",fitFunction,80, 150, 8); 
	f1->SetParName(0,"p0");
	f1->SetParName(1,"p1");	
	f1->SetParName(2,"p2");
	f1->SetParName(3,"p3");
	f1->SetParName(4,"p4");
	f1->SetParName(5,"p5");
	f1->SetParameters(114,1,-0.01,0.01);
	
	TF1 *f2 = new TF1("f2",fitFunction2, -1.5, 0.5, 4); 
	f2->SetParName(0,"p0");
	f2->SetParName(1,"p1");	
	f2->SetParName(2,"p2");
	f2->SetParName(3,"p3");
	f2->SetParameters(-0.4,1,-1,1);
	
	int hist_drift_ = 200;
	int hist_depth_ = 50;
	double min_drift_ = -1000;
	double max_drift_ = 1000;
	double min_depth_ = -100;
	double max_depth_ = 400;
	double width_ = 0.0285;
	int anglebins = 90;	
	int anglebinscotan = 180;

	TH2F * h_sizex_alpha = new TH2F("h_sizex_alpha", "h_sizex_alpha", anglebins, 0, 180,10 , .5, 10.5);
	TH2F * h_sizex_alpha_cotan = new TH2F("h_sizex_alpha", "h_sizex_alpha", anglebinscotan, -3, 3,10 , .5, 10.5);
	
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
	const int maxpix = 100;
	struct Pixinfo
	{
		int npix;
		float row[100];
		float col[100];
		float adc[100];
		float x[100];
		float y[100];
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
			int colpos = static_cast<int>(pixinfo_.col[j]-0.5);
			if (pixinfo_.row[j] == 0 || pixinfo_.row[j] == 79 || pixinfo_.row[j] == 80 || pixinfo_.row[j] == 159 || colpos % 52 == 0 || colpos % 52 == 51 ){
				large_pix = true;	
			}
		}
		
		// is it one of the problematic half ladders? (needs to be excluded)
		if( (layer_ == 1 && (ladder_ == 5 || ladder_ == 6 || ladder_ == 15 || ladder_ == 16)) ||(layer_ == 2 && (ladder_ == 8 || ladder_ == 9 || ladder_ == 24 || ladder_ == 25)) ||(layer_ == 3 && (ladder_ ==11 || ladder_ == 12 || ladder_ == 33 || ladder_ == 34)) ) {
			continue;
		}
		
		double residual = TMath::Sqrt( (trackhit_.x - rechit_.x) * (trackhit_.x - rechit_.x) + (trackhit_.y - rechit_.y) * (trackhit_.y - rechit_.y) );
		if( (chi2_/ndof_) < 10 && !large_pix){
			if(trackhit_.alpha > 0){	
				h_sizex_alpha->Fill(trackhit_.alpha*180. / TMath::Pi(),clust_.size_x);
				h_sizex_alpha_cotan->Fill(TMath::Tan(TMath::Pi()/2. - trackhit_.alpha),clust_.size_x);
				fAngles << TMath::Tan(TMath::Pi()/2. - trackhit_.alpha) << "\t" << TMath::Tan(TMath::Pi()/2. - trackhit_.beta) << endl;
			}
			else{ 
				h_sizex_alpha->Fill( (trackhit_.alpha + TMath::Pi())*180. / TMath::Pi(),clust_.size_x);
				h_sizex_alpha_cotan->Fill(TMath::Tan(TMath::Pi()/2. - trackhit_.alpha),clust_.size_x);
				fAngles << TMath::Tan(TMath::Pi()/2. - trackhit_.alpha) << "\t" << TMath::Tan(TMath::Pi()/2. - trackhit_.beta) << endl;
			}
			
		} 
	}
	
	cout << TMath::Pi()/2. << endl;
	TH1F * h_mean = new TH1F("h_mean","h_mean", anglebins, 0, 180);
	TH1F * h_slice_ = new TH1F("h_slice","h_slice", 10, .5, 10.5);
	TH1F * h_mean_cotan = new TH1F("h_mean_cotan","h_mean_contan", anglebinscotan, -3, 3);
	TH1F * h_slice_cotan_ = new TH1F("h_slice_cotan","h_slice_cotan", 10, .5, 10.5);
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
		h_slice_cotan_->Reset("ICE");
		
		//loop over bins in drift width
		for( int j = 1; j<= 10; j++){
			h_slice_cotan_->SetBinContent(j,h_sizex_alpha_cotan->GetBinContent(i,j));
		} // end loop over bins in drift width
			
		double mean = h_slice_cotan_->GetMean(1); 
		double error = h_slice_cotan_->GetMeanError(1);
			
		h_mean_cotan->SetBinContent(i, mean);
		h_mean_cotan->SetBinError(i, error);	
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
	
	TCanvas * c2 = new TCanvas("c2", "c2", 1200, 600);
	c2->Divide(2,1);
	c2->cd(1);
	h_sizex_alpha_cotan->GetXaxis()->SetTitle("cotan(#alpha)");
	h_sizex_alpha_cotan->GetYaxis()->SetTitle("cluster size [pixels]");
	h_sizex_alpha_cotan->Draw("colz");
	c2->cd(2);	
	h_mean_cotan->GetXaxis()->SetTitle("cotan(#alpha)");
	h_mean_cotan->GetYaxis()->SetTitle("average cluster size [pixels]");
	h_mean_cotan->Draw();
	h_mean_cotan->Fit(f2,"ERQ");
	
	return 0;
}
