{
  gROOT->ProcessLine(".L ./style-CMSTDR.C");
  gROOT->ProcessLine("setTDRStyle()");
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(10);
  gStyle->SetPalette(1,0);

  TFile *f = new TFile("./merged_output_run191277.root");


  f->cd();
	
  TF1 *f1 = new TF1("f1","[0] + [1]*x",50., 235.); 
  f1->SetParName(0,"offset");
  f1->SetParName(1,"tan#theta_{LA}");
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

  ofstream fAngles("Run2012A_modules5to8_layer3.txt", ios::trunc); 
  // 	ofstream fLorentzFit( "lorentzFit.txt", ios::trunc );
  // 	fLorentzFit.precision( 4 );
  // 	fLorentzFit << "module" << "\t" << "layer" << "\t" << "offset" << "\t" << "error" << "\t" << "slope" << "\t" << "error" << "\t" "rel.err" << "\t" "pull" << "\t" << "chi2" << "\t" << "prob" << endl;
  TH2F * h_drift_depth_adc = new TH2F("h_drift_depth_adc", "h_drift_depth_adc",hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
  TH2F * h_drift_depth_adc2 = new TH2F("h_drift_depth_adc2","h_drift_depth_adc2",hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
  TH2F * h_drift_depth_noadc = new TH2F("h_drift_depth_noadc",";drift [#mum];depth [#mum]",hist_drift_ , min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);

  TH1F *h_pt = new TH1F("h_pt","h_pt",100,0,5);
  TH1F *h_ndof = new TH1F("h_ndof","h_ndof",101,0,100);
  TH1I *h_trackQuality = new TH1I("h_trackQuality","h_trackQuality",6,0,6);
  TH1I *h_nHitsPerTrack = new TH1I("h_nHitsPerTrack","h_nHitsPerTrack",100,0,100);
	
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
	
  cout << "Running over " << nentries << " hits" << endl;

  //cuts
  float pt_cut = 3.0;
  float clusterSizeY_cut = 4.0;
  float residual_cut = 0.005;
  float normChi2_cut = 2.0;
  float clusterCharge_cut = 120.0;
  float trackQuality_cut = 2.0;
  int highPurity_cut = 1;
	
  for(int ientrie = 0 ; ientrie < nentries; ientrie++){
  //  for(int ientrie = 0 ; ientrie < 20000000; ientrie++){
    LATree->GetEntry(ientrie);


    if(module_ <=4) continue;
    if(layer_ !=3) continue;
    //if(run_ != 175990)continue;
    //if(!(module_==1 && layer_==1)) continue;
    //cout<<"module = "<<module_<<endl;
    //cout<<"layer = "<<layer_<<endl;


  
    h_trackQuality->Fill(trackQuality_);
    bool large_pix = false;
    for (int j = 0; j <  pixinfo_.npix; j++){
      int colpos = static_cast<int>(pixinfo_.col[j]);
      if (pixinfo_.row[j] == 0 || pixinfo_.row[j] == 79 || pixinfo_.row[j] == 80 || pixinfo_.row[j] == 159 || colpos % 52 == 0 || colpos % 52 == 51 ){
	large_pix = true;	
      }
    }

    //if(ndof_<10)continue;
    //if(ndof_==0)continue;//because of some crashes
    //if(trackQuality_<trackQuality_cut)continue;
    if(isHighPurity_ !=1) continue;
    if(pt_< pt_cut)continue;
    double residual = TMath::Sqrt( (trackhit_.x - rechit_.x) * (trackhit_.x - rechit_.x) + (trackhit_.y - rechit_.y) * (trackhit_.y - rechit_.y) );

    if( (clust_.size_y >= clusterSizeY_cut) && (chi2_/ndof_) < normChi2_cut && !large_pix && residual < residual_cut && clust_.charge < clusterCharge_cut){
    
      //fAnles->open();
      //write in file (cotan(alpha) cotan(beta) p)
      //cout<<trackhit_.x<<"\t"<<rechit_.x<<endl;
      //cout<<"pt: "<<pt_<<endl;
      //cout<<"p: "<<p_<<endl;
      fAngles << TMath::Tan(TMath::Pi()/2. - trackhit_.alpha) << "\t" << TMath::Tan(TMath::Pi()/2. - trackhit_.beta) << "\t"<<p_ <<"\t"<<isflipped_<<endl;
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
	
  TH1F * h_mean = new TH1F("h_mean",";depth [#mum];drift [#mum]", hist_depth_, min_depth_, max_depth_);
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
	
  TCanvas * c1 = new TCanvas("c1", "c1", 800, 600);
  //c1->Divide(2,1);
  //c1->cd(1);
  // 	h_drift_depth_noadc->
  h_drift_depth_noadc->Draw("colz");
  // 	c1->cd(2);
  // 	h_drift_depth_adc->Draw();
  //c1->cd(2);
  c1->SaveAs("c1.eps");

  TCanvas * c11 = new TCanvas("c11", "c11", 800, 600);

  h_mean->Draw();
  // 	c1->cd(4);
	
  h_mean->Fit(f1,"ERQ");
  double p0 = f1->GetParameter(0);
  double e0 = f1->GetParError(0);
  double p1 = f1->GetParameter(1);
  double e1 = f1->GetParError(1);
  double chi2 = f1->GetChisquare();
  double prob = f1->GetProb();	

  c11->SaveAs("c11.eps");

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
	
  // 	delete h_mean;
  // 	delete h_drift_depth_adc_slice_;
  cout << "offset" << "\t" << "error" << "\t" << "slope" << "\t" << "error" << "\t" "rel.err" << "\t" << "chi2" << "\t" << "prob" << endl;
  cout  << p0 << "\t" << e0 << "\t" << p1 << "\t" << e1 << "\t" << e1 / p1 *100. << "\t" << chi2 << "\t" << prob << endl;
	
}
