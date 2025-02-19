


void Plot(const char* rootfile1, 
	  const char* rootfile2, 
	  const char* legend1, 
	  const char* legend2, 
	  int nrebin ) {


  gROOT->Reset();
  TFile *f = new TFile(rootfile1);
  f->cd();
  
  TH1F* mcpf = (TH1F*)f->Get("h_deltaETvisible_MCPF");
  TH1F* mceht = (TH1F*)f->Get("h_deltaETvisible_MCEHT");

  if(!mcpf || !mceht) {
    gDirectory->ls();
  }

  mcpf->SetStats(1); 
  mcpf->GetXaxis()->SetTitle("#DeltaE_{T} (GeV)");
  mcpf->SetTitle("Tau Benchmark, E_{T}(rec)-E_{T}(true)");
  mcpf->SetLineColor(2);
  mcpf->SetLineWidth(2);
  mcpf->Draw();
  mceht->Draw("same");
  mceht->SetLineWidth(2);
  
  TFile *f_f = new TFile(rootfile2);
  f_f->cd();
  
  TH1F* mcpf_f = (TH1F*) f_f->Get("h_deltaETvisible_MCPF");
  TH1F* mceht_f = (TH1F*)f_f->Get("h_deltaETvisible_MCEHT");
  
  mcpf_f->Scale( mcpf->GetEntries()/ mcpf_f->GetEntries());
  mceht_f->Scale( mceht->GetEntries()/ mceht_f->GetEntries());
  mcpf_f->SetLineColor(4);
  mceht_f->SetLineColor(4);
  
  mcpf_f->Draw("same");
  mceht_f->Draw("same");
  
  Double_t x_1=0.55; Double_t y_1 = 0.61;
  Double_t x_2=0.98; Double_t y_2 = 0.82;
  
  TLegend *leg = new TLegend(x_1,y_1,x_2,y_2);
  leg->SetFillColor(0);

  string lpf1 =  "PFlow Jets ";
  string lpf2 = lpf1;
  string leht1 = "Calo Jets  ";
  string leht2 = leht1;
  
  lpf1 += legend1;
  lpf2 += legend2;
  leht1 += legend1;
  leht2 += legend2;

    

  leg->AddEntry(mcpf,   lpf1.c_str() ,"l");
  leg->AddEntry(mceht,  leht1.c_str() ,"l");
  leg->AddEntry(mcpf_f, lpf2.c_str() ,"l");
  leg->AddEntry(mceht_f,leht2.c_str() ,"l");
  leg->Draw();
  
  
  mcpf->Rebin(nrebin);
  mceht->Rebin(nrebin);
  mcpf_f->Rebin(nrebin);
  mceht_f->Rebin(nrebin);
    
  //c1->Print("tauBenchmark.gif");
  
}
