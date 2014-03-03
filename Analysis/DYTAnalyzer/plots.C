{
  gStyle->SetOptStat("emruo");
  gStyle->SetOptFit(1);
  TCanvas *c = new TCanvas("c","",1200,600);
  c->Divide(2,1);

  int rebinPT = 10;

  TFile *f = new TFile("output.root","OLD");
  TH1F *DYT_pt = (TH1F*) f->Get("demo/DYT_pt");
  TH1F *TuneP_pt = (TH1F*) f->Get("demo/TuneP_pt");

  char Ylabel[100];
  sprintf(Ylabel, "Events / %i GeV/c", rebinPT);
  TuneP_pt->Rebin(rebinPT);
  TuneP_pt->SetTitle("TuneP");
  TuneP_pt->GetXaxis()->SetTitle("p_{T} [GeV/c]");
  TuneP_pt->GetXaxis()->SetTitleOffset(1.4);
  TuneP_pt->GetYaxis()->SetTitle(Ylabel);
  TuneP_pt->GetYaxis()->SetTitleOffset(1.6);
  DYT_pt->Rebin(rebinPT);
  DYT_pt->SetTitle("DYT");
  DYT_pt->GetXaxis()->SetTitle("p_{T} [GeV/c]");
  DYT_pt->GetXaxis()->SetTitleOffset(1.4);
  DYT_pt->GetYaxis()->SetTitle(Ylabel);
  DYT_pt->GetYaxis()->SetTitleOffset(1.6);


  c->cd(1);
  TuneP_pt->Fit("gaus","R","",750,1200);
  TuneP_pt->Draw();
  c->cd(2);
  DYT_pt->Fit("gaus","R","",750,1200);
  DYT_pt->Draw();
}
