
void makeTauPlot(TTree *tree, 
		 const char* label,
		 int relative = 0);

void makeTauPlotChain(const char* pattern, 
		      const char* label, 
		      int relative = 0 ) {

  gSystem->Load("libChain");
  
  Chain chain("Eff", pattern);

  makeTauPlot( &chain, label, relative );
}


void makeTauPlotTree(const char* filename, 
		     const char* label, 
		     int relative = 0,
		     int hmax = 600. ) {

  TChain chain("Eff");
  chain.Add( filename );
  

  makeTauPlot( &chain, label, relative, hmax );
}



void makeTauPlot(TTree *tree, 
		 const char* label,
		 int relative,
		 int hmax) {

  
  


  int nentries = 999999;
  
//   string root = label; root += ".root";
//   TFile f(root.c_str(), "recreate");
  
  string pfhname = label;
  pfhname += "_PF";
  string  pfvar = "(jetsPF_.et - jetsMC_.et)>>";
  if(relative==1)   
    pfvar = "(jetsPF_.et - jetsMC_.et)/jetsMC_.et>>";
  else if (relative==2)
    pfvar = "(jetsPF_.et - jetsMC_.et)/jetsPF_.et>>";
  
  pfvar += pfhname;
  
  
  int nbins = 100;
  double min = -1; 
  double max = 1;
  string title = "Tau Benchmark, (E_{T}(rec)-E_{T}(true))/E_{T}(true); #DeltaE_{T}/E_{T}";
  if(!relative) {
    double absmax = 50;
    min = -absmax;
    max = absmax;
    title = "Tau Benchmark, E_{T}(rec)-E_{T}(true); #DeltaE_{T} (GeV)";
  }
  
  TH1F* h_deltaETvisible_MCPF = new TH1F(pfhname.c_str(),title.c_str(),
					 nbins, min , max);
  
  h_deltaETvisible_MCPF->SetMaximum(hmax);
  h_deltaETvisible_MCPF->SetStats(1); 
  h_deltaETvisible_MCPF->SetLineWidth(2);

  string ehthname = label;
  ehthname += "_EHT";
  string ehtvar = "(jetsEHT_.et - jetsMC_.et)>>";
  if(relative==1)   
    ehtvar = "(jetsEHT_.et - jetsMC_.et)/jetsMC_.et>>";
  else if (relative==2)
    ehtvar = "(jetsEHT_.et - jetsMC_.et)/jetsEHT_.et>>";
  ehtvar += ehthname;

  TH1F* h_deltaETvisible_MCEHT 
    = (TH1F*) h_deltaETvisible_MCPF->Clone(ehthname.c_str());
  h_deltaETvisible_MCEHT->SetLineColor(2);
  //jetsMC_.et<100
  tree->Draw(pfvar.c_str(), 
	     "", "", nentries);
  tree->Draw(ehtvar.c_str(), 
	     "", "", nentries);

  cout<<pfvar<<endl;
  cout<<ehtvar<<endl;
//   tree->Draw("(jetsPF_.et - jetsMC_.et)/jetsPF_.et>>h_deltaETvisible_MCPF");
//   tree->Draw("(jetsEHT_.et - jetsMC_.et)/jetsEHT_.et>>h_deltaETvisible_MCEHT");
  h_deltaETvisible_MCPF->Draw();
  h_deltaETvisible_MCEHT->Draw("same");
  
  Double_t x_1=0.60; Double_t y_1 = 0.60;
  Double_t x_2=0.85; Double_t y_2 = 0.70;
  
  TLegend *leg = new TLegend(x_1,y_1,x_2,y_2,NULL,"brNDC");
  leg->SetTextSize(0.035);
  leg->SetFillStyle(0);
  leg->SetFillColor(0);
  leg->SetTextFont(52);
  leg->SetTextAlign(32);
  
  leg->AddEntry(h_deltaETvisible_MCPF,"Particle Flow Jets","l");
  leg->AddEntry(h_deltaETvisible_MCEHT,"caloTower Jets","l");
  leg->Draw();
  
  string eps = label; eps += ".png";
  c1->Print(eps.c_str());
  //c1->Print("tauBenchmark.gif");
//   gApplication->Terminate();

//   f.Write();
//   f.Close();
}
