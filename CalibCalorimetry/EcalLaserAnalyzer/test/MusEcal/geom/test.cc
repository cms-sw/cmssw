
void drawLM( int isect=0 )
{
  TString gname_0;

  if( isect==0 )
    {
      gname_0 = "LMModule_";
      for( int xside=1; xside<=2; xside++ )
	{
	  for( int ii=1; ii<=19; ii++ )
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
  gname_0 = "LMRegion_";
  for( int ii=1; ii<=10; ii++ )
    {
      if( isect!=0 )
	{
	  if( isect==1 && ii!=4 ) continue;
	  if( isect==2 && ii!=5 ) continue;
	  if( isect==3 && ii!=6 ) continue;
	  if( isect==4 && ii!=7 ) continue;
	  if( isect==5 && ( ii!=8 && ii!=9 ) ) continue;
	  if( isect==6 && ii!=10 ) continue;
	  if( isect==7 && ii!=1 ) continue;
	  if( isect==8 && ii!=2 ) continue;
	  if( isect==9 && ii!=3 ) continue;
	}
      TString gname = gname_0;
      gname += ii;
           
      TGraph* g_ = (TGraph*) gROOT->FindObject(gname);

      g_->SetLineWidth(2);
      g_->Draw("LSame");
    }
}

void drawEE( TString ext, TString title )
{
  TString hname = "eem_";
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
  TCanvas* canv = new TCanvas( cname, tname, 10, 10, 500, 500 );
  h->SetTitle(title);
  h->Draw("COLZ");
  drawLM();
}

void drawEE_loc( int isect, TString ext, TString title )
{
  TString hname = "eem_S"; hname += isect; hname += "_";
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

void test()
{
  gROOT->GetStyle("GHM")->SetOptStat(0);

  TString fname = "/home/gautier/cms/store/MusEcal/EndCap/CRAFT1/EE+2/LMF_EE+2_0_BlueLaser_Run99999_LB0001_TS5220223587952099328.root";

  TFile* file = TFile::Open(fname);

  TString hname = "LMF_LASER_BLUE_PRIM_DAT__MEAN";
  TH2* h = (TH2*) gROOT->FindObject( hname );
  TString tname = "EE Geometry - ";
  //  tname += title;
  h->SetTitle(tname);
  h->GetXaxis()->CenterTitle();
  h->GetXaxis()->SetTitle("ix");
  h->GetYaxis()->CenterTitle();
  h->GetYaxis()->SetTitle("iy");
  TString cname = "canv_test";
  TCanvas* canv = new TCanvas( cname, tname, 10, 10, 500, 500 );
  //  h->SetTitle(title);
  h->DrawClone("COLZ");

  // file->Close();
  
  file = TFile::Open("eegeom_1.root");
  
  drawLM();
}
