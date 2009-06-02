
void drawLM()
{
  TString gname_0;

  gname_0 = "SuperModule_";
  for( int ii=1; ii<=36; ii++ )
    {
      TString gname = gname_0;
      gname += ii;
      
      TGraph* g_ = (TGraph*) gROOT->FindObject(gname);
      g_->SetLineWidth(2);
      g_->Draw("LSame");
    }
  gname_0 = "MonitoringRegion_";
  for( int ii=11; ii<=82; ii++ )
    {
      TString gname = gname_0;
      gname += ii;
      
      TGraph* g_ = (TGraph*) gROOT->FindObject(gname);
      g_->SetLineWidth(1);
      g_->Draw("LSame");
    }
}

void drawLM_loc()
{
  TString gname_0;

  gname_0 = "SuperModule";
  TString gname = gname_0;
  
  TGraph* g_ = (TGraph*) gROOT->FindObject(gname);
  g_->SetLineWidth(2);
  g_->Draw("LSame");

  gname_0 = "Side_";
  for( int ii=0; ii<2; ii++ )
    {
      TString gname = gname_0;
      gname += ii;
      
      TGraph* g_ = (TGraph*) gROOT->FindObject(gname);
      g_->SetLineWidth(1);
      g_->Draw("LSame");
    }
}

void drawEB( TString ext, TString title )
{
  TString hname = "eb_";
  hname += ext;
  TH2* h = (TH2*) gROOT->FindObject( hname );
  TString tname = "EB Geometry - ";
  tname += title;
  h->SetTitle(tname);
  h->GetXaxis()->CenterTitle();
  h->GetXaxis()->SetTitle("ieta");
  h->GetYaxis()->CenterTitle();
  h->GetYaxis()->SetTitle("iphi");
  TString cname = "canv_";
  cname += ext;
  TCanvas* canv = new TCanvas( cname, tname, 10, 10, 500, 800 );
  h->SetTitle(title);
  h->Draw("COLZ");
  drawLM();
}

void drawEB_loc( TString ext, TString title )
{
  TString hname = "eb_loc_";
  hname += ext;
  TH2* h = (TH2*) gROOT->FindObject( hname );
  TString tname = "EB Local Geometry - ";
  tname += title;
  h->SetTitle(tname);
  h->GetXaxis()->CenterTitle();
  h->GetXaxis()->SetTitle("ix");
  h->GetYaxis()->CenterTitle();
  h->GetYaxis()->SetTitle("iy");
  TString cname = "canv_loc_";
  cname += ext;
  TCanvas* canv = new TCanvas( cname, tname, 10, 10, 1000, 500 );
  h->SetTitle(title);
  h->Draw("COLZ");
  drawLM_loc();
}

void ebgeom()
{
  //  gROOT->GetStyle("GHM")->SetOptStat(0);
  //  gROOT->GetStyle("GHM")->SetPalette(1);
  gStyle->SetOptStat(0);
  gStyle->SetPalette(1);

  TString fname = "ebgeom.root";

  TFile* file = TFile::Open(fname);

  drawEB_loc( "side",  "Laser Monitoring Sides" );
  drawEB_loc( "lmmod", "Laser Monitoring Modules" );
  drawEB_loc( "tt",    "Trigger Towers" );
  drawEB_loc( "hv",    "HV Channels" );
  drawEB_loc( "lv",    "LV Channels" );
  drawEB_loc( "cr",    "Crystals" );
  drawEB_loc( "elecr", "Electronic Channels" );

  drawEB( "sm",        "Super Modules" );
  drawEB( "lmmod",     "Laser Monitoring Modules" );
  drawEB( "lmr",       "Laser Monitoring Regions" );
  drawEB( "tt",        "Trigger Towers" );
  drawEB( "cr_in_sm",  "Crystals in Super Modules" );

}
