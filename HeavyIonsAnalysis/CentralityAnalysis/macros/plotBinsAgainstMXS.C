

static const int maxtables = 25;

void plotBinsAgainstMXS(){

   int nTables = 16;
   int nBins = 18;

   CentralityBins* tables[maxtables];
   TGraph * graphs[40];

   TFile* inf = new TFile("tables.root","read");
   TFile* outf = new TFile("ErrorMap.root","recreate");

   for(int i  = 0 ; i < nTables; ++i){
      tables[i] = (CentralityBins*)inf->Get(Form("HFhits20_MXS%d_Hydjet4TeV_MC_3XY_V21_v0",i)); 
   }

   for(int j = 0; j< nBins; ++j){
      graphs[j] = new TGraph(nTables);
      graphs[j]->SetName(Form("Bin%d",j));
      graphs[j]->SetMarkerColor(nBins-j);
      graphs[j]->SetMarkerSize(1.);
      graphs[j]->SetMarkerStyle(20);
      
      for(int i  = 0 ; i < nTables; ++i){
	 double reference = tables[0]->NpartMeanOfBin(j);
	 if(reference <= 0) reference = 0.001;
	 graphs[j]->SetPoint(i,((double)i)/100,tables[i]->NpartMeanOfBin(j)/reference);
      }
   }

   graphs[0]->SetMarkerColor(49);

   TGraph* g1 = new TGraph(nBins);
   g1->SetName("g1");
   g1->SetMarkerSize(1.);
   g1->SetMarkerStyle(20);
   
   TH2D* hPad1 = new TH2D("hPad1",";Fraction of Missing XS;[N_{part}]/[N_{part}(100%)]",100,-0.02,0.2,450,0,7);
   TH2D* hPad2 = new TH2D("hPad2",";bin;#Delta N_{part}/N_{part} per unit efficiency",nBins+1,-0.5,nBins+0.5,450,0,35);

   TCanvas* c1 = new TCanvas("c1","c1",400,400);
   TLegend* leg3 = new TLegend(0.2,0.65,0.55,0.92,NULL,"brNDC");
   leg3->SetFillColor(0);
   leg3->SetTextSize(0.06);
   leg3->SetBorderSize(0);

   hPad1->Draw();
   for(int j = 0; j< nBins; ++j){

      graphs[j]->Fit("pol1");
      if(j > 3) graphs[j]->Draw("p");
      TF1* f = graphs[j]->GetFunction("pol1");
      f->SetLineColor(graphs[j]->GetMarkerColor());
      if(j % 5 == 0) leg3->AddEntry(graphs[j],Form("Bin %d",j),"pl");

      double slope = f->GetParameter(1);
      g1->SetPoint(j,j,slope);
   }

   leg3->Draw();
   c1->Print("NpartMean_vs_MXS.gif");

   TCanvas* c2 = new TCanvas("c2","c2",400,400);
   hPad2->Draw();
   g1->Draw("p");

   c2->Print("delta_vs_bins.gif");


   hPad2->Write();
   g1->Write();
   outf->Write();


}


