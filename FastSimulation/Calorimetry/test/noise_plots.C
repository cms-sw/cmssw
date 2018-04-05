void noise_plots(char* newFile="DQM_fastsim.root",char* refFile="DQM_fullsim.root") {

  gROOT ->Reset();
  gROOT ->SetBatch();

  //=========  settings ====================
  gROOT->SetStyle("Plain");
  gStyle->SetPadGridX(kTRUE);
  gStyle->SetPadGridY(kTRUE);
  gStyle->SetPadRightMargin(0.07);
  gStyle->SetPadLeftMargin(0.13);
  char* refLabel("reference histogram");
  char* newLabel("new histogram");

  delete gROOT->GetListOfFiles()->FindObject(refFile);
  delete gROOT->GetListOfFiles()->FindObject(newFile); 
  
  TText* te = new TText();
  TFile * sfile = new TFile(newFile);
  TDirectory * sdir=gDirectory;
  TFile * rfile = new TFile(refFile);
  TDirectory * rdir=gDirectory;

  if(sfile->GetDirectory("DQMData/Run 1/HcalRecHitsV")) sfile->cd("DQMData/Run 1/HcalRecHitsV/Run summary/HcalRecHitTask");
  else if(sfile->GetDirectory("DQMData/HcalRecHitsV/HcalRecHitTask"))sfile->cd("DQMData/HcalRecHitsV/HcalRecHitTask");
  sdir=gDirectory;
  TList *sl= sdir->GetListOfKeys();

  if(rfile->GetDirectory("DQMData/Run 1/HcalRecHitsV")) rfile->cd("DQMData/Run 1/HcalRecHitsV/Run summary/HcalRecHitTask");
  else if(rfile->GetDirectory("DQMData/HcalRecHitsV/HcalRecHitTask"))rfile->cd("DQMData/HcalRecHitsV/HcalRecHitTask");
  rdir=gDirectory;
  TList *rl= rdir->GetListOfKeys();

  TCanvas *canvas;

  TH1F *sh1,*rh1;
  TH1F *sh2,*rh2;
  TH1F *sh3,*rh3;
  TH1F *sh4,*rh4;
  TH1F *sh5,*rh5;
  

  rdir->GetObject("e_hb",rh1);
  sdir->GetObject("e_hb",sh1);
  rdir->GetObject("e_he",rh2);
  sdir->GetObject("e_he",sh2);
  rdir->GetObject("e_hfl",rh3);
  sdir->GetObject("e_hfl",sh3);
  rdir->GetObject("e_hfs",rh4);
  sdir->GetObject("e_hfs",sh4);
  rdir->GetObject("e_ho",rh5);
  sdir->GetObject("e_ho",sh5);

  rh1->GetXaxis()->SetTitle("E");
  rh1->GetYaxis()->SetTitleSize(0.05);
  rh1->GetYaxis()->SetTitleOffset(1.2);
  //  rh1->GetYaxis()->SetRangeUser(0.4,1.);
  //  sh1->GetYaxis()->SetRangeUser(0.4,1.);
  //  rh1->Rebin(5);
  //  sh1->Rebin(5);
  NormalizeHistograms(rh1,sh1);
  double maxH = max(rh1,sh1)*1.1;
  rh1->GetYaxis()->SetRangeUser(0.,maxH);
  sh1->GetYaxis()->SetRangeUser(0.,maxH);

  rh2->GetXaxis()->SetTitle("E");
  rh2->GetYaxis()->SetTitleSize(0.05);
  rh2->GetYaxis()->SetTitleOffset(1.2);
  //  rh2->GetYaxis()->SetRangeUser(0.4,1.);
  //  sh2->GetYaxis()->SetRangeUser(0.4,1.);
  //  rh2->Rebin(5);
  //  sh2->Rebin(5);
  NormalizeHistograms(rh2,sh2);
  double maxH = max(rh2,sh2)*1.1;
  rh2->GetYaxis()->SetRangeUser(0.,maxH);
  sh2->GetYaxis()->SetRangeUser(0.,maxH);

  rh3->GetXaxis()->SetTitle("E");
  rh3->GetYaxis()->SetTitleSize(0.05);
  rh3->GetYaxis()->SetTitleOffset(1.2);
  NormalizeHistograms(rh3,sh3);
  double maxH = max(rh3,sh3)*1.1;
  rh3->GetYaxis()->SetRangeUser(0.,maxH);
  sh3->GetYaxis()->SetRangeUser(0.,maxH);

  rh4->GetXaxis()->SetTitle("E");
  rh4->GetYaxis()->SetTitleSize(0.05);
  rh4->GetYaxis()->SetTitleOffset(1.2);
  NormalizeHistograms(rh4,sh4);
  double maxH = max(rh4,sh4)*1.1;
  rh4->GetYaxis()->SetRangeUser(0.,maxH);
  sh4->GetYaxis()->SetRangeUser(0.,maxH);

  rh5->GetXaxis()->SetTitle("E");
  rh5->GetYaxis()->SetTitleSize(0.05);
  rh5->GetYaxis()->SetTitleOffset(1.2);
  NormalizeHistograms(rh5,sh5);
  double maxH = max(rh5,sh5)*1.1;
  rh5->GetYaxis()->SetRangeUser(0.,maxH);
  sh5->GetYaxis()->SetRangeUser(0.,maxH);

  canvas = new TCanvas("Noise","Noise",1000,1400);


  TH1F * r[5]={rh1,rh2,rh3,rh4,rh5};
  TH1F * s[5]={sh1,sh2,sh3,sh4,sh5};
  plotBuilding(canvas,s, r,5,
	       te,"UU",-1);//, 1, true, 0);

  canvas->cd();
  l = new TLegend(0.50,0.14,0.90,0.19);
  l->SetTextSize(0.016);
  l->SetLineColor(1);
  l->SetLineWidth(1);
  l->SetLineStyle(1);
  l->SetFillColor(0);
  l->SetBorderSize(3);
  l->AddEntry(rh1,refLabel,"LPF");
  l->AddEntry(sh1,newLabel,"LPF");
  l->Draw();
  TString namepdf = "hcal_noise.pdf";
  canvas->Print(namepdf);   
  delete l;

  delete canvas;

}

void plotBuilding(TCanvas *canvas, TH1F **s, TH1F **r, int n,TText* te,
		  char * option, double startingY, double startingX = .1,bool fit = false, unsigned int logx=0){
  canvas->Divide(2,(n+1)/2); //this should work also for odd n
  for(int i=0; i<n;i++){
    s[i]->SetMarkerStyle(20);
    r[i]->SetMarkerStyle(21);
    s[i]->SetMarkerColor(2);
    r[i]->SetMarkerColor(4);
    s[i]->SetMarkerSize(0.7);
    r[i]->SetMarkerSize(0.7);
    s[i]->SetLineColor(1);
    r[i]->SetLineColor(1);
    s[i]->SetLineWidth(1);
    r[i]->SetLineWidth(1);

    TPad *pad=canvas->cd(i+1);
    setStats(s[i],r[i], -1, 0, fit);
    if((logx>>i)&1)pad->SetLogx();
    r[i]->Draw("lpf");
    s[i]->Draw("sameslpf");
  }

}

void setStats(TH1* s,TH1* r, double startingY, double startingX = .1,bool fit){
  if (startingY<0){
    s->SetStats(0);
    r->SetStats(0);
  } else {
    //gStyle->SetOptStat(1001);
    s->SetStats(1);
    r->SetStats(1);

    if (fit){
      s->Fit("gaus");
      TF1* f1 = (TF1*) s->GetListOfFunctions()->FindObject("gaus");
      f1->SetLineColor(2);
      f1->SetLineWidth(1);
    }
    s->Draw();
    gPad->Update(); 
    TPaveStats* st1 = (TPaveStats*) s->GetListOfFunctions()->FindObject("stats");
    if (fit) {st1->SetOptFit(0010);    st1->SetOptStat(1001);}
    st1->SetX1NDC(startingX);
    st1->SetX2NDC(startingX+0.30);
    st1->SetY1NDC(startingY+0.20);
    st1->SetY2NDC(startingY+0.35);
    st1->SetTextColor(2);
    if (fit) {
      r->Fit("gaus");
      TF1* f2 = (TF1*) r->GetListOfFunctions()->FindObject("gaus");
      f2->SetLineColor(4);
      f2->SetLineWidth(1);    
    }
    r->Draw();
    gPad->Update(); 
    TPaveStats* st2 = (TPaveStats*) r->GetListOfFunctions()->FindObject("stats");
    if (fit) {st2->SetOptFit(0010);    st2->SetOptStat(1001);}
    st2->SetX1NDC(startingX);
    st2->SetX2NDC(startingX+0.30);
    st2->SetY1NDC(startingY);
    st2->SetY2NDC(startingY+0.15);
    st2->SetTextColor(4);
  }
}

void NormalizeHistograms(TH1F* h1, TH1F* h2)
{
  if (h1==0 || h2==0) return;
  float scale1 = -9999.9;
  float scale2 = -9999.9;

  if ( h1->Integral() != 0 && h2->Integral() != 0 ){
    scale1 = 1.0/(float)h1->Integral();
    scale2 = 1.0/(float)h2->Integral();
    
    //h1->Sumw2();
    //h2->Sumw2();
    h1->Scale(scale1);
    h2->Scale(scale2);
 
  }
}

double max(TH1F* h1, TH1F* h2)
{
  double a = h1->GetMaximum();
  double b = h2->GetMaximum();
  return (a > b) ? a : b;
}
