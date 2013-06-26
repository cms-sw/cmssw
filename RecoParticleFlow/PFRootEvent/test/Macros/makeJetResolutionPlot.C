#include <vector>



class Results {
public:
  double sigma;
  double rms;
  double mean;
  double arithmav;
};

void GetParameters( TH1* h, Results& results) {

  h->Fit( "gaus","Q0");

  TF1* gaus = h->GetFunction( "gaus" );

  results.sigma = gaus->GetParameter(2);
  results.mean = gaus->GetParameter(1);
  results.rms = h->GetRMS();
  results.arithmav = h->GetMean();
}


void PlotGraphs( TGraph* gra, TGraph* grb, 
		 const char* hname,
		 const char* title, 
		 const char* xtitle,
		 const char* ytitle,
		 float xmin,float xmax,float ymin,float ymax) {

  TH2F *h = new TH2F(hname,"", 
		     10, xmin, xmax, 10, ymin, ymax );
  h->SetTitle( title );
  h->SetXTitle( xtitle );
  h->SetYTitle( ytitle );
  h->SetStats(0);
  h->Draw();

  grb->SetMarkerColor(4);						
  grb->SetLineColor(4);						  
  grb->SetLineWidth(2);						  
  grb->SetLineStyle(2);
  grb->Draw("C*");


  gra->SetMarkerStyle(25);						
  gra->SetMarkerColor(2);						
  gra->SetLineColor(2);						  
  gra->SetLineWidth(2);						  
  gra->Draw("CP");
  
  h->GetYaxis()->SetTitleSize(0.06);
  h->GetYaxis()->SetTitleOffset(1.4);
  h->GetXaxis()->SetTitleSize(0.06);
  h->GetYaxis()->SetLabelSize(0.045);
  h->GetXaxis()->SetLabelSize(0.045);

  gPad->SetGridx();
  gPad->SetGridy();
  gPad->SetBottomMargin(0.14);
  gPad->SetLeftMargin(0.2);
  gPad->SetRightMargin(0.05);


}

void Resolution2() {
  
  gROOT->Reset();
  TFile *f = new TFile("pfjetBenchmark.root");
  if(f->IsZombie() ) return;

  vector< string > hists;

  vector<float> pts;

  hists.push_back( "BRPt20_40") ;
  pts.push_back(30);
  hists.push_back( "BRPt40_60");
  pts.push_back(50);
  hists.push_back( "BRPt60_80");
  pts.push_back(70);
  hists.push_back( "BRPt80_100");
  pts.push_back(90);
  hists.push_back( "BRPt100_150");
  pts.push_back(125);
  hists.push_back( "BRPt150_200");
  pts.push_back(175);
  hists.push_back( "BRPt200_250");
  pts.push_back(225);
  hists.push_back( "BRPt250_300");
  pts.push_back(275);
  hists.push_back( "BRPt300_400");
  pts.push_back(350);
  hists.push_back( "BRPt400_500");
  pts.push_back(450);
  hists.push_back( "BRPt500_750");
  pts.push_back(600);

  vector<float> sigmas;
  vector<float> rmss;
  vector<float> means;
  vector<float> arithmavs;

  int n=0;
  for( unsigned i=0; i<hists.size(); ++i) {

    TH1* h = (TH1*) f->Get( hists[i].c_str() );
    if( !h ) {
      cerr<<h->GetName()<<" does not exist"<<endl;
      continue;
    }

    Results results;
    GetParameters( h, results);

    sigmas.push_back( results.sigma );
    rmss.push_back( results.rms );
    means.push_back( results.mean );
    arithmavs.push_back( results.arithmav );
    
    ++n;
  }
  
  TGraph* gr1  = new TGraph ( n, &pts[0], &sigmas[0] );
  TGraph* gr12 = new TGraph ( n, &pts[0], &rmss[0] );

  TGraph* gr2  = new TGraph ( n, &pts[0], &means[0] );
  TGraph* gr22 = new TGraph ( n, &pts[0], &arithmavs[0] );

  TCanvas *c = new TCanvas("c","",1000, 600);
  c->Divide(2,1);
  
  c->cd(1);
  PlotGraphs( gr1, gr12, "width",
	      "", 
	      "p_{T} (GeV/c)", 
	      "Width #Delta p_{T} / p_{T}",
	      0, 700, 0.05, 0.2);

  c->cd(2);
  PlotGraphs( gr2, gr22, "mean",
	      "", 
	      "p_{T} (GeV/c)", 
	      "Mean #Delta p_{T} / p_{T}",
	      0, 700, -0.2, 0.01);

  c->cd(2);
  TLegend *leg=new TLegend(0.25,0.3,0.9,0.5);
  leg->AddEntry( gr12, "Arithmetic Estimate", "lp");
  leg->AddEntry( gr1, "Gaussian Fit", "lp");
  leg->SetTextSize(0.05);
  leg->Draw();

  c->cd();
  
  
}
