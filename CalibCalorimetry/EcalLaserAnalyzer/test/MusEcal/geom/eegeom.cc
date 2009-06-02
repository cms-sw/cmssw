
void drawLM( int isect=0 )
{
  TString gname_0;

  int kk=0;
  int lmr_(0); 
  int sect_=isect;
  int lmm_(0);
  if( sect_==0 )
    {
      gname_0 = "LMModule_";
      for( int xside=1; xside<=2; xside++ )
	{
	  for( int ii=1; ii<=38; ii++ )
	    {
	      TString gname = gname_0;
	      gname += ii;
	      gname += "_";
	      gname += xside;
	      
	      TGraph* g_ = (TGraph*) gROOT->FindObject(gname);
	      if( g_==0 ) continue;
	      g_->SetLineWidth(1);
	      g_->Draw("LSame");
	    }
	}
    }
  else
    {
      if( sect_>9 ) 
	{
	  kk=1;
	}
    }
  gname_0 = "LMRegion_";
  for( int ii=1; ii<=20; ii++ )
    {
      lmr_ = 72+ii;
      if( sect_!=0 )
	{
	  if( sect_==1 && ii!=4 ) continue;
	  if( sect_==2 && ii!=5 ) continue;
	  if( sect_==3 && ii!=6 ) continue;
	  if( sect_==4 && ii!=7 ) continue;
	  if( sect_==5 && ( ii!=8 && ii!=9 ) ) continue;
	  if( sect_==6 && ii!=10 ) continue;
	  if( sect_==7 && ii!=1 ) continue;
	  if( sect_==8 && ii!=2 ) continue;
	  if( sect_==9 && ii!=3 ) continue;
	  if( sect_==10 && ii!=14 ) continue;
	  if( sect_==11 && ii!=15 ) continue;
	  if( sect_==12 && ii!=16 ) continue;
	  if( sect_==13 && ii!=17 ) continue;
	  if( sect_==14 && ( ii!=18 && ii!=19 ) ) continue;
	  if( sect_==15 && ii!=20 ) continue;
	  if( sect_==16 && ii!=11 ) continue;
	  if( sect_==17 && ii!=12 ) continue;
	  if( sect_==18 && ii!=13 ) continue;
	}
      TString gname = gname_0;
      gname += lmr_;
           
      TGraph* g_ = (TGraph*) gROOT->FindObject(gname);
      if( g_==0 ) 
	{
	  cout << "Warning TGraph not found ! " << gname << endl;
	  continue;
	}

      g_->SetLineWidth(2);
      g_->Draw("LSame");
    }
}

void drawEE( TString ext, TString title )
{
  TString hname = "ee_";
  hname += ext;
  TH2* h = (TH2*) gROOT->FindObject( hname );
  TString tname = "EE Geometry - ";
  tname += title;
  h->SetTitle(tname);
  h->GetXaxis()->CenterTitle();
  h->GetXaxis()->SetTitle("ix");
  h->GetYaxis()->CenterTitle();
  h->GetYaxis()->SetTitle("iy");
  TString cname = "canv_";
  cname += ext;
  TCanvas* canv = new TCanvas( cname, tname, 10, 10, 500, 1000 );
  h->SetTitle(title);
  h->Draw("COLZ");
  drawLM();
}

void drawEE_loc( int isect, TString ext, TString title )
{
  TString hname = "ee_S"; hname += isect; hname += "_";
  hname += ext;
  TH2* h = (TH2*) gROOT->FindObject( hname );
  TString tname = "EE Local Geometry - ";
  tname += title;
  h->SetTitle(tname);
  h->GetXaxis()->CenterTitle();
  h->GetXaxis()->SetTitle("ix");
  h->GetYaxis()->CenterTitle();
  h->GetYaxis()->SetTitle("iy");
  TString cname = "canv_loc_S"; cname += isect;
  cname += ext;
  TCanvas* canv = new TCanvas( cname, tname, 10, 10, 500, 500 );
  h->SetTitle(title);
  h->Draw("COLZ");
  drawLM( isect );
}

void eegeom()
{
  gROOT->GetStyle("GHM")->SetOptStat(0);

  TString fname = "eegeom.root";

  TFile* file = TFile::Open(fname);

  for( int isect=1; isect<=18; isect++ )
    {
      drawEE_loc( isect, "sc", TString("Super Crystals for Sector ")+isect );
      drawEE_loc( isect, "lmmod", TString("Laser monitoring Modules for Sector ")+isect );
      drawEE_loc( isect, "cr", TString("Crystals for Sector ")+isect );
    }

  drawEE( "lmr",         "Laser Monitoring Region" );
  drawEE( "sect",        "Sectors" );
  drawEE( "sc",          "Super Crystals" );
  drawEE( "cr_in_sc",    "Crystals in Super Crystals" );
  drawEE( "lmmod",       "Laser Monitoring Modules" );
  drawEE( "cr",          "Crystals" );
  drawEE( "quad",        "Quadrants" );

}
