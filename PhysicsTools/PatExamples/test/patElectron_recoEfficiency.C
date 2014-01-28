void patElectron_recoEfficiency()
{
  // define proper canvas style
  setNiceStyle();
  gStyle->SetOptStat(0);

  // open file
  TFile* file = new TFile("analyzePatElectron.root");

  // get reco histograms
  TH1F* recPt_  = file->Get("analyzePatElectron/pt");
  TH1F* recEta_ = file->Get("analyzePatElectron/eta");
  TH1F* recPhi_ = file->Get("analyzePatElectron/phi");

  // get gene histograms
  TH1F* genPt_  = file->Get("analyzePatElectron/genPt");
  TH1F* genEta_ = file->Get("analyzePatElectron/genEta");
  TH1F* genPhi_ = file->Get("analyzePatElectron/genPhi");

  // draw canvas with electron kinematics
  TCanvas* canv0 = new TCanvas("canv0", "electron kinemtics", 600, 300);
  canv0->Divide(2,1);
  canv0->cd(1);
  TH1F* kinPt_=recPt_->Clone();
  kinPt_->SetFillStyle(3005.);
  kinPt_->SetFillColor(4.);
  setHistStyle(kinPt_);
  kinPt_ ->DrawCopy();

  canv0->cd(2);
  TH1F* kinEta_=recEta_->Clone();
  kinEta_->SetFillStyle(3005.);
  kinEta_->SetFillColor(4.);
  setHistStyle(kinEta_);
  kinEta_->DrawCopy();

  // draw canvas with reco efficiencies
  TCanvas* canv1 = new TCanvas("canv1", "electron reco efficiency", 600, 600);
  canv1->Divide(2,2);
  if(correlatedError(recPt_,  genPt_ )==0){
    canv1->cd(1);
    canv1->GetPad(1)->SetGridx(1);
    canv1->GetPad(1)->SetGridy(1);
    recPt_ ->SetMaximum(1.3);
    setHistStyle(recPt_);
    recPt_ ->DrawCopy();
  }
  if(correlatedError(recEta_, genEta_)==0){
    canv1->cd(2);
    canv1->GetPad(2)->SetGridx(1);
    canv1->GetPad(2)->SetGridy(1);
    recEta_ ->SetMaximum(1.3);
    setHistStyle(recEta_);
    recEta_->DrawCopy();
  }
  if(correlatedError(recPhi_, genPhi_)==0){
    canv1->cd(3);
    canv1->GetPad(3)->SetGridx(1);
    canv1->GetPad(3)->SetGridy(1);
    recPhi_ ->SetMaximum(1.3);
    setHistStyle(recPhi_);
    recPhi_->DrawCopy();
  }
}

int correlatedError(TH1F* nominator, TH1F* denominator)
{
  // --------------------------------------------------
  // get error of correlated ratio for two histograms
  // for gaussian distributed quantities the formular 
  //
  //  * de=e*Sqrt((dr/r)^2+(1-2e)*(dv/d)^2)
  //
  // turns automatically into 
  // 
  //  * de=Sqrt(e*(1-e)/r)
  // --------------------------------------------------
  if(nominator->GetNbinsX()!=denominator->GetNbinsX()){
    // these histogramsd do not correspond to each other
    return -1;
  }

  // loop over the denominator bins
  for(Int_t i=0; i<denominator->GetNbinsX(); ++i){
    float dval =  nominator->GetBinError(i+1);
    float val  =  nominator->GetBinContent(i+1);
    float dref =  denominator->GetBinError(i+1);
    float ref  =  denominator->GetBinContent(i+1);

    float err;
    if(val<=0){ 
      // val should never be smaller than 0
      err=0; continue;
    }
    if(ref==0){
      // ref should never be 0
      err=0; continue;
    }

    if(val/ref<1)
      err=(val/ref)*TMath::Sqrt(TMath::Abs((dref*dref)/(ref*ref)+(1.-2.*(val/ref))*(dval*dval)/(val*val)));
    else
      err=(ref/val)*TMath::Sqrt(TMath::Abs((dval*dval)/(val*val)+(1.-2.*(ref/val))*(dref*dref)/(ref*ref)));
    
    // set proper values and errors
    nominator->SetBinContent(i+1, val/ref);
    nominator->SetBinError(i+1, err);
  }
  return 0;
}

void setAxisStyle(TH1* hist) {
  // --------------------------------------------------
  // define proper axsis style for a given histogram
  // --------------------------------------------------
	hist->GetXaxis()->SetTitleSize( 0.06);
	hist->GetXaxis()->SetTitleColor( 1);
	hist->GetXaxis()->SetTitleOffset( 0.8);
	hist->GetXaxis()->SetTitleFont( 62);
	hist->GetXaxis()->SetLabelSize( 0.05);
	hist->GetXaxis()->SetLabelFont( 62);
	hist->GetXaxis()->CenterTitle();
	hist->GetXaxis()->SetNdivisions( 505);

	hist->GetYaxis()->SetTitleSize( 0.07);
	hist->GetYaxis()->SetTitleColor( 1);
	hist->GetYaxis()->SetTitleOffset( 0.5);
	hist->GetYaxis()->SetTitleFont( 62);
	hist->GetYaxis()->SetLabelSize( 0.05);
	hist->GetYaxis()->SetLabelFont( 62);
}

void setHistStyle(TH1F* hist)
{
  // --------------------------------------------------
  // define proper histogram style
  // --------------------------------------------------
  setAxisStyle(hist);
  hist->GetXaxis()->SetTitle(hist->GetTitle());
  hist->SetTitle();
  hist->SetLineColor(4.);
  hist->SetLineWidth(3.);
  hist->SetMarkerSize(0.75);
  hist->SetMarkerColor(4.);
  hist->SetMarkerStyle(20.);
}

void setNiceStyle() 
{
  // --------------------------------------------------
  // define proper canvas style
  // --------------------------------------------------
  TStyle *MyStyle = new TStyle ("MyStyle", "My style for nicer plots");
  
  Float_t xoff = MyStyle->GetLabelOffset("X"),
          yoff = MyStyle->GetLabelOffset("Y"),
          zoff = MyStyle->GetLabelOffset("Z");

  MyStyle->SetCanvasBorderMode ( 0 );
  MyStyle->SetPadBorderMode    ( 0 );
  MyStyle->SetPadColor         ( 0 );
  MyStyle->SetCanvasColor      ( 0 );
  MyStyle->SetTitleColor       ( 0 );
  MyStyle->SetStatColor        ( 0 );
  MyStyle->SetTitleBorderSize  ( 0 );
  MyStyle->SetTitleFillColor   ( 0 );
  MyStyle->SetTitleH        ( 0.07 );
  MyStyle->SetTitleW        ( 1.00 );
  MyStyle->SetTitleFont     (  132 );

  MyStyle->SetLabelOffset (1.5*xoff, "X");
  MyStyle->SetLabelOffset (1.5*yoff, "Y");
  MyStyle->SetLabelOffset (1.5*zoff, "Z");

  MyStyle->SetTitleOffset (0.9,      "X");
  MyStyle->SetTitleOffset (0.9,      "Y");
  MyStyle->SetTitleOffset (0.9,      "Z");

  MyStyle->SetTitleSize   (0.045,    "X");
  MyStyle->SetTitleSize   (0.045,    "Y");
  MyStyle->SetTitleSize   (0.045,    "Z");

  MyStyle->SetLabelFont   (132,      "X");
  MyStyle->SetLabelFont   (132,      "Y");
  MyStyle->SetLabelFont   (132,      "Z");

  MyStyle->SetPalette(1);

  MyStyle->cd();
}
