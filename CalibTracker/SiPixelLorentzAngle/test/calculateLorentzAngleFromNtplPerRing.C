{
  gROOT->ProcessLine(".L ./style-CMSTDR.C");
  gROOT->ProcessLine("setTDRStyle()");
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(10);
  gStyle->SetPalette(1,0);
	
  TFile *f = new TFile("./merged_output_run191277.root");


  f->cd();
	
  TF1 *f1 = new TF1("f1","[0] + [1]*x",50., 235.); 
  f1->SetParName(0,"p0");
  f1->SetParName(1,"p1");
  f1->SetParameter(0,0);
  f1->SetParameter(1,0.4);
  f1->SetLineColor(2);
  f1->SetLineWidth(3.5);
	
  int hist_drift_ = 200;
  int hist_depth_ = 50;
  double min_drift_ = -1000;
  double max_drift_ = 1000;
  double min_depth_ = -100;
  double max_depth_ = 400;
  double width_ = 0.0285;

  ofstream fAngles("anglesFromNtpl.txt", ios::trunc); 
  // 	ofstream fLorentzFit( "lorentzFit.txt", ios::trunc );
  // 	fLorentzFit.precision( 4 );
  // 	fLorentzFit << "module" << "\t" << "layer" << "\t" << "offset" << "\t" << "error" << "\t" << "slope" << "\t" << "error" << "\t" "rel.err" << "\t" "pull" << "\t" << "chi2" << "\t" << "prob" << endl;
  //   TH2F * h_drift_depth_adc = new TH2F("h_drift_depth_adc", "h_drift_depth_adc",hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
  //   TH2F * h_drift_depth_adc2 = new TH2F("h_drift_depth_adc2","h_drift_depth_adc2",hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
  //   TH2F * h_drift_depth_noadc = new TH2F("h_drift_depth_noadc","h_drift_depth_noadc;drift in #mum;depth in #mum",hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);

  //   TH1F *h_pt = new TH1F("h_pt","h_pt",100,0,5);
  //   TH1F *h_ndof = new TH1F("h_ndof","h_ndof",101,0,100);
  //   TH1I *h_trackQuality = new TH1I("h_trackQuality","h_trackQuality",6,0,6);
  //   TH1I *h_nHitsPerTrack = new TH1I("h_nHitsPerTrack","h_nHitsPerTrack",100,0,100);
	
  int run_;
  int event_;
  int module_;
  int ladder_;
  int layer_;
  int isflipped_;
  float pt_;
  float p_;
  float eta_;
  float phi_;
  double chi2_;
  double ndof_;
  int trackQuality_;
  int nHitsPerTrack_;
  int isHighPurity_;
  int maxpix = 8000;
  struct Pixinfo
  {
    int npix;
    float row[maxpix];
    float col[maxpix];
    float adc[maxpix];
    float x[maxpix];
    float y[maxpix];
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
  LATree->SetBranchAddress("p", &p_);//M
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
  LATree->SetBranchAddress("clust", &clust_); // charge is given in 1000 e
  LATree->SetBranchAddress("rechit", &rechit_);
  LATree->SetBranchAddress("trackQuality", &trackQuality_);
  LATree->SetBranchAddress("isHighPurity", &isHighPurity_);
  LATree->SetBranchAddress("nHitsPerTrack", &nHitsPerTrack_);


  ofstream fFit;
  fFit.open("fitPerRing-withAngles.txt");
  cout.precision(4);
  fFit<<"module"<<"\t"<<"layer"<<"\t"<<"offset"<<"\t"<<"error"<<"\t"<<"slope"<<"\t"<<"error"<<"\t"<<"rel.error"<<"\t"<<"chi2"<<"\t"<<"prob"<<endl;
 
	
  cout << "Running over " << nentries << " hits" << endl;

  //cuts
  float pt_cut = 3.0;
  float clusterSizeY_cut = 4.0;
  float residual_cut = 0.005;
  float normChi2_cut = 2.0;
  float clusterCharge_cut = 120.0;//120
  float trackQuality_cut = 2.0;
  int highPurity_cut = 1;


  TH1F *h_alpha_pZ = new TH1F("h_alpha_pZ","h_alpha_pZ",100,-4,4);
  TH1F *h_beta_pZ = new TH1F("h_beta_pZ","h_beta_pZ",100,-4,4);
  TH1F *h_gamma_pZ = new TH1F("h_gamma_pZ","h_gamma_pZ",100,-4,4);
  TH1F *h_alpha_nZ = new TH1F("h_alpha_nZ","h_alpha_nZ",100,-4,4);
  TH1F *h_beta_nZ = new TH1F("h_beta_nZ","h_beta_nZ",100,-4,4);
  TH1F *h_gamma_nZ = new TH1F("h_gamma_nZ","h_gamma_nZ",100,-4,4);


  //loop over modules and layers to fit the lorentz angle
  for( int i_layer = 1; i_layer<=3; i_layer++){
    for(int i_module = 1; i_module<=8; i_module++){

      TH2F * h_drift_depth_adc = new TH2F("h_drift_depth_adc", "h_drift_depth_adc",hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
      TH2F * h_drift_depth_adc2 = new TH2F("h_drift_depth_adc2","h_drift_depth_adc2",hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
      TH2F * h_drift_depth_noadc = new TH2F("h_drift_depth_noadc","h_drift_depth_noadc;drift in #mum;depth in #mum",hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);

      TH1F *h_pt = new TH1F("h_pt","h_pt",100,0,5);
      TH1F *h_ndof = new TH1F("h_ndof","h_ndof",101,0,100);
      TH1I *h_trackQuality = new TH1I("h_trackQuality","h_trackQuality",6,0,6);
      TH1I *h_nHitsPerTrack = new TH1I("h_nHitsPerTrack","h_nHitsPerTrack",100,0,100);

      char name[128];
      sprintf(name,"h_alpha_layer%i_module%i",i_layer,i_module);
      TH1F *h_alpha = new TH1F(name,name,100,-4,4);
      sprintf(name,"h_beta_layer%i_module%i",i_layer,i_module);
      TH1F *h_beta = new TH1F(name,name,100,-4,4);
      sprintf(name,"h_beta_layer%i_module%i",i_layer,i_module);
      TH1F *h_gamma = new TH1F(name,name,100,-4,4);

	
      for(int ientrie = 0 ; ientrie<nentries; ientrie++){
      //      for(int ientrie = 0 ; ientrie<1000000; ientrie++){
	LATree->GetEntry(ientrie);
	if(layer_!=i_layer || module_!=i_module)continue;  
	//if(!isflipped_)continue;
	//cout<<"ladder: "<<ladder_<<endl;
	//if(ladder_<11)continue;
	
	/*
	//half-shell 2:
	if(layer_ == 1){
	  if((  (ladder_ >=1 && ladder_ <=5) || (ladder_>=16 && ladder_<=20)  ))continue;
	}
	if(layer_ == 2){
	  if((  (ladder_ >=1 && ladder_ <=8) || (ladder_>=25 && ladder_<=32)  ))continue;	  
	}
	if(layer_ == 3){
	  if((  (ladder_ >=1 && ladder_ <=11) || (ladder_>=34 && ladder_<=44)  ))continue;	  	  
	  }
	*/

	h_trackQuality->Fill(trackQuality_);
	bool large_pix = false;
	for (int j = 0; j <  pixinfo_.npix; j++){
	  int colpos = static_cast<int>(pixinfo_.col[j]);
	  if (pixinfo_.row[j] == 0 || pixinfo_.row[j] == 79 || pixinfo_.row[j] == 80 || pixinfo_.row[j] == 159 || colpos % 52 == 0 || colpos % 52 == 51 ){
	    large_pix = true;	
	  }
	}

	//cuts by M
	//if(ndof_<10)continue;
	//if(ndof_==0)continue;//because of some crashes
	if(isHighPurity_ !=1) continue;
	if(pt_< pt_cut)continue;
	
	double residual = TMath::Sqrt( (trackhit_.x - rechit_.x) * (trackhit_.x - rechit_.x) + (trackhit_.y - rechit_.y) * (trackhit_.y - rechit_.y) );
	if( (clust_.size_y >= clusterSizeY_cut) && (chi2_/ndof_) < normChi2_cut && !large_pix && residual < residual_cut && clust_.charge < clusterCharge_cut){
    
	  //fAnles->open();
	  //write in file (cotan(alpha) cotan(beta) p)
	  //cout<<trackhit_.x<<"\t"<<rechit_.x<<endl;
	  //cout<<"pt: "<<pt_<<endl;
	  //cout<<"p: "<<p_<<endl;
	  //fill histos with angles
	  
	  if(i_module<=4){
	  h_alpha_pZ->Fill(trackhit_.alpha);
	  h_beta_pZ->Fill(trackhit_.beta);
	  h_gamma_pZ->Fill(trackhit_.gamma);
	  }
	  if(i_module>=5 && i_module<=8){
	  h_alpha_nZ->Fill(trackhit_.alpha);
	  h_beta_nZ->Fill(trackhit_.beta);
	  h_gamma_nZ->Fill(trackhit_.gamma);
	  }
	  
	  h_alpha->Fill(trackhit_.alpha);
	  h_beta->Fill(trackhit_.beta);
	  h_gamma->Fill(trackhit_.gamma);
	  fAngles << TMath::Tan(TMath::Pi()/2. - trackhit_.alpha) << "\t" << TMath::Tan(TMath::Pi()/2. - trackhit_.beta) << "\t"<<p_ <<endl;
	  h_pt->Fill(pt_);
	  h_ndof->Fill(ndof_);
	  h_nHitsPerTrack->Fill(nHitsPerTrack_);
	  //h_trackQuality->Fill(trackQuality_);
	  //fAngles.close();
	  for (int j = 0; j <  pixinfo_.npix; j++){
	    float dx = (pixinfo_.x[j]  - (trackhit_.x - width_/2. / TMath::Tan(trackhit_.alpha))) * 10000.;
	    float dy = (pixinfo_.y[j]  - (trackhit_.y - width_/2. / TMath::Tan(trackhit_.beta))) * 10000.;
	    float depth = dy * tan(trackhit_.beta);
	    float drift = dx - dy * tan(trackhit_.gamma);
	    h_drift_depth_adc->Fill(drift, depth, pixinfo_.adc[j]);
	    h_drift_depth_adc2->Fill(drift, depth, pixinfo_.adc[j]*pixinfo_.adc[j]);
	    h_drift_depth_noadc->Fill(drift, depth);
					
	  }
	} 
      }
	
      TH1F * h_mean = new TH1F("h_mean","h_mean;depth in #mum;drift in #mum", hist_depth_, min_depth_, max_depth_);
      TH1F * h_drift_depth_adc_slice_ = new TH1F("h_slice","h_slice", hist_drift_, min_drift_, max_drift_);
      //loop over bins in depth (z-local-coordinate) (in order to fit slices)
      for( int i = 1; i <= hist_depth_; i++){
	// 				findMean(i, (i_module + (i_layer - 1) * 8));
	double npix = 0;

	h_drift_depth_adc_slice_->Reset("ICE");
		
	// determine sigma and sigma^2 of the adc counts and average adc counts
	//loop over bins in drift width
	for( int j = 1; j<= hist_drift_; j++){
	  if(h_drift_depth_noadc->GetBinContent(j, i) >= 1){
	    double adc_error2 = (h_drift_depth_adc2->GetBinContent(j,i) - h_drift_depth_adc->GetBinContent(j,i)*h_drift_depth_adc->GetBinContent(j, i) / h_drift_depth_noadc->GetBinContent(j, i)) /  h_drift_depth_noadc->GetBinContent(j, i);
	    h_drift_depth_adc_slice_->SetBinContent(j, h_drift_depth_adc->GetBinContent(j,i));
	    h_drift_depth_adc_slice_->SetBinError(j, sqrt(adc_error2));
	    npix += h_drift_depth_noadc->GetBinContent(j,i);	
	  }else{
	    h_drift_depth_adc_slice_->SetBinContent(j, h_drift_depth_adc->GetBinContent(j,i));
	    h_drift_depth_adc_slice_->SetBinError(j, 0);
	  }
	} // end loop over bins in drift width
			
	double mean = h_drift_depth_adc_slice_->GetMean(1); 
	double error = 0;
	if(npix != 0){
	  error = h_drift_depth_adc_slice_->GetRMS(1) / sqrt(npix);
	}
			
	h_mean->SetBinContent(i, mean);
	h_mean->SetBinError(i, error);	
      }// end loop over bins in depth 
	
      TCanvas * c1 = new TCanvas("c1", "c1", 1200, 600);
      c1->Divide(2,1);
      c1->cd(1);
      // 	h_drift_depth_noadc->
      h_drift_depth_noadc->Draw("colz");
      // 	c1->cd(2);
      // 	h_drift_depth_adc->Draw();
      c1->cd(2);
      h_mean->Draw();
      // 	c1->cd(4);
	
      h_mean->Fit(f1,"ERQ");
      double p0 = f1->GetParameter(0);
      double e0 = f1->GetParError(0);
      double p1 = f1->GetParameter(1);
      double e1 = f1->GetParError(1);
      double chi2 = f1->GetChisquare();
      double prob = f1->GetProb();	

      /*
      c1->SaveAs("c1.eps");

      
      TCanvas * c2 = new TCanvas("c2", "c2", 800, 600);
      h_pt->Draw();
      c2->SaveAs("c2.eps");

      TCanvas * c3 = new TCanvas("c3", "3", 800, 600);
      h_ndof->Draw();
      c3->SaveAs("c3.eps");

      TCanvas * c4 = new TCanvas("c4", "c4", 800, 600);
      h_nHitsPerTrack->Draw();
      c4->SaveAs("c4.eps");

      TCanvas * c5 = new TCanvas("c5", "c5", 800, 600);
      h_trackQuality->Draw();
      c5->SaveAs("c5.eps");
	
      char name[128];
      sprintf(name,"alpha_layer%i_module%i.eps",i_layer, i_module);
      TCanvas * c6 = new TCanvas("c6","c6",800, 600);
      h_alpha->Draw();
      c6->SaveAs(name);

      sprintf(name,"beta_layer%i_module%i.eps",i_layer, i_module);
      TCanvas * c7 = new TCanvas("c7","c7",800, 600);
      h_beta->Draw();
      c7->SaveAs(name);

      sprintf(name,"gamma_layer%i_module%i.eps",i_layer, i_module);
      TCanvas * c8 = new TCanvas("c8","c8",800, 600);
      h_gamma->Draw();
      c8->SaveAs(name);
      

      */


      // 	delete h_mean;
      // 	delete h_drift_depth_adc_slice_;
      cout<<"layer "<<i_layer<<"   module "<<i_module<<endl;
      cout << "offset" << "\t" << "error" << "\t" << "slope" << "\t" << "error" << "\t" "rel.err" << "\t" << "chi2" << "\t" << "prob" << endl;
      cout  << p0 << "\t" << e0 << "\t" << p1 << "\t" << e1 << "\t" << e1 / p1 *100. << "\t" << chi2 << "\t" << prob << endl;
      fFit<<i_module<<"\t"<<i_layer<<"\t"<<p0<<"\t"<<e0<<"\t"<<p1<<"\t"<<e1<<"\t"<<e1/p1 *100.<<"\t"<<chi2<<"\t"<<prob<<endl;

    }//end of loop on modules
  }//end of loop on layers

  /*  
  TCanvas * c9 = new TCanvas("c9","c9",800,600);
  h_alpha_pZ->Draw();
  c9->SaveAs("c9.eps");

  TCanvas * c10 = new TCanvas("c10","c10",800,600);
  h_beta_pZ->Draw();
  c10->SaveAs("c10.eps");

  TCanvas * c11 = new TCanvas("c11","c11",800,600);
  h_gamma_pZ->Draw();
  c11->SaveAs("c11.eps");

  TCanvas * c12 = new TCanvas("c12","c12",800,600);
  h_alpha_nZ->Draw();
  c12->SaveAs("c12.eps");

  TCanvas * c13 = new TCanvas("c13","c13",800,600);
  h_beta_nZ->Draw();
  c13->SaveAs("c13.eps");

  TCanvas * c14 = new TCanvas("c14","c14",800,600);
  h_gamma_nZ->Draw();
  c14->SaveAs("c14.eps");
  
  */



  fFit.close();//close of output file



	
}




