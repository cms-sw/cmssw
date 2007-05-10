#define analyse_residuals_cxx
#include "analyse_residuals.h"
#include <TH1F.h>
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
#include <TF1.h>
#include <TMath.h>

double ng(Double_t* xx, Double_t* par)
{
  Double_t x = xx[0];
  Double_t norm = par[0];
  Double_t mean = par[1];
  Double_t sigma = par[2];

  return norm * 1.0/sqrt(2.0*TMath::Pi())/sigma * exp(-0.5*(x-mean)*(x-mean)/sigma/sigma);
}

const double a_min = 1.37078;
const double a_max = 1.77078;
const double a_bin = 0.10000;

// MAGIC NUMBERS: beta ranges depending on y cluster size
double ys_bl[6];
double ys_bh[6];

TH1F* h_yres_npix_alpha_beta[6][3][10]; 
TH1F* h_xres_npix_beta_alpha[3][4][10];
TH1F* h_xres_npix_beta[3][4];

TH1F* h_forward_yres_npix_beta[2][10]; 
TH1F* h_forward_yres_npix[2]; 

TH1F* h_forward_xres_npix_alpha[2][10]; 
TH1F* h_forward_xres_npix[2]; 

void analyse_residuals::Loop()
{
  if (fChain == 0) return;
  
  Long64_t nentries = fChain->GetEntries();
  cout << "nentries = " << nentries << endl;

  char hname[100];

  ys_bl[0] = 0.00;
  ys_bh[0] = 0.60;
  ys_bl[1] = 0.10; 
  ys_bh[1] = 0.90;
  ys_bl[2] = 0.60; 
  ys_bh[2] = 1.05;
  
  ys_bl[3] = 0.90; 
  ys_bh[3] = 1.15;
  ys_bl[4] = 1.05; 
  ys_bh[4] = 1.22;
  ys_bl[5] = 1.15; 
  ys_bh[5] = 1.41;
  
  // barrel histograms --------------------------------------------------------------
  for (int i=0; i<6; ++i) // loop over size_y
    for (int j=0; j<3; ++j) // loop over alpha bins
      for (int k=0; k<10; ++k) // loop over beta bins
	{
	  sprintf(hname, "h_yres_npix_alpha_beta_%i_%i_%i", i, j, k );
	  h_yres_npix_alpha_beta[i][j][k]= new TH1F(hname, hname, 100, -0.02, 0.02);
	}
  
  for (int i=0; i<3; ++i) // loop over size_x
    for (int j=0; j<4; ++j) // loop over beta bins
      for (int k=0; k<10; ++k) // loop over alpha bins
	{
	  sprintf(hname, "h_xres_npix_beta_alpha_%i_%i_%i", i, j, k );
	  h_xres_npix_beta_alpha[i][j][k]= new TH1F(hname, hname, 100, -0.01, 0.01);
	}

  for (int i=0; i<3; ++i) // loop over size_x
    for (int j=0; j<4; ++j) // loop over beta bins
      {
	sprintf(hname, "h_xres_npix_beta_%i_%i", i, j );
	h_xres_npix_beta[i][j]= new TH1F(hname, hname, 10, a_min, a_max);
      }

  // forward histograms --------------------------------------------------------------
  for (int i=0; i<2; ++i) // loop ove sizey bins
    for (int j=0; j<10; ++j) // loop over beta bins
      {
      sprintf(hname, "h_forward_yres_npix_beta_%i_%i", i, j );
      h_forward_yres_npix_beta[i][j] = new TH1F(hname, hname, 100, -0.01, 0.01);
    }

  for (int i=0; i<2; ++i) // loop ove sizey bins
    {
      sprintf(hname, "h_forward_yres_npix_%i", i );
      h_forward_yres_npix[i] = new TH1F(hname, hname, 10, 0.3, 0.4);
    }

  for (int i=0; i<2; ++i) // loop ove sizey bins
    for (int j=0; j<10; ++j) // loop over beta bins
      {
	sprintf(hname, "h_forward_xres_npix_alpha_%i_%i", i, j );
	h_forward_xres_npix_alpha[i][j] = new TH1F(hname, hname, 100, -0.01, 0.01);
      }

  sprintf(hname, "h_forward_xres_npix_%i", 0 );
  h_forward_xres_npix[0] = new TH1F(hname, hname, 10, 0.15, 0.3);
  sprintf(hname, "h_forward_xres_npix_%i", 1 );
  h_forward_xres_npix[1] = new TH1F(hname, hname, 10, 0.15, 0.5);

  for (Long64_t jentry=0; jentry<nentries; jentry++) 
    {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      fChain->GetEntry(jentry);
      
      double alpha_rad = fabs(alpha)/180.0*TMath::Pi();
      double beta_rad  = fabs(beta) /180.0*TMath::Pi();
      double betap_rad = fabs( TMath::Pi()/2.0 - beta_rad );
      double alphap_rad = fabs( TMath::Pi()/2.0 - alpha_rad );
 
      if ( subdetId == 1 )
	{
	  // y residuals----------------------------------------------------------------
	  int sizey = nypix;
	  
	  if ( !( sizey == 1 && bigy == 1 ) ) // skip ( sizey == 1 && bigy == 1 ) clusters; the associated error is pitch_y/sqrt(12.0) 
	    {
	      if ( sizey > 6 ) sizey = 6;
	      
	      int ind_sizey = sizey - 1;
	      int ind_alpha = -9999;
	      int ind_beta  = -9999; 
	      
	      if ( sizey > 0 && sizey < 4 )
		{
		  if      ( a_min           <= alpha_rad && alpha_rad <= a_min+    a_bin ) 
		    {
		      ind_alpha = 0;
		      if      ( betap_rad <= ys_bl[sizey-1] ) ind_beta = 0;
		      else if ( betap_rad >= ys_bh[sizey-1] ) ind_beta = 2;
		      else if ( betap_rad >  ys_bl[sizey-1] && 
				betap_rad <  ys_bh[sizey-1] ) ind_beta = 1; 
		      else cout << " Wrong betap_rad = " << betap_rad << endl << endl;
		      
		    }
		  else if ( a_min+    a_bin <  alpha_rad && alpha_rad <= a_min+2.0*a_bin ) 
		    {
		      ind_alpha = 1;
		      if      ( betap_rad <= ys_bl[sizey-1] ) ind_beta = 0;
		      else if ( betap_rad >= ys_bh[sizey-1] ) ind_beta = 9;
		      else if ( betap_rad >  ys_bl[sizey-1] && 
				betap_rad <  ys_bh[sizey-1] ) 
			{
			  double binw = ( ys_bh[sizey-1] - ys_bl[sizey-1] ) / 8.0;
			  ind_beta = 1 + (int)( ( betap_rad - ys_bl[sizey-1] ) / binw );
			}
		      else cout << " Wrong betap_rad = " << betap_rad << endl << endl;
		    }
		  else if ( a_min+2.0*a_bin <  alpha_rad && alpha_rad <= a_max           ) 
		    {
		      ind_alpha = 2;
		      if      ( betap_rad <= ys_bl[sizey-1] ) ind_beta = 0;
		      else if ( betap_rad >= ys_bh[sizey-1] ) ind_beta = 2;
		      else if ( betap_rad >  ys_bl[sizey-1] && 
				betap_rad <  ys_bh[sizey-1] ) ind_beta = 1; 
		      else cout << " Wrong betap_rad = " << betap_rad << endl << endl;
		    }
		  else 
		    cout << " Wrong alpha_rad = " << alpha_rad << endl << endl;
		}
	      else if ( sizey >= 4 )
		{
		  ind_alpha = 0;
		  
		  if ( sizey == 4 || sizey == 6 )
		    {
		      if      ( betap_rad <= ys_bl[sizey-1] ) ind_beta = 0;
		      else if ( betap_rad >= ys_bh[sizey-1] ) ind_beta = 9;
		      else if ( betap_rad >  ys_bl[sizey-1] && 
				betap_rad <  ys_bh[sizey-1] ) 
			{
			  double binw = ( ys_bh[sizey-1] - ys_bl[sizey-1] ) / 8.0;
			  ind_beta = 1 + (int)( ( betap_rad - ys_bl[sizey-1] ) / binw );
			}
		      else cout << " Wrong betap_rad = " << betap_rad << endl << endl;
		    }
		  else if ( sizey == 5 )
		    {
		      if      ( betap_rad <= ys_bl[sizey-1] ) ind_beta = 0;
		      else if ( betap_rad >= ys_bh[sizey-1] ) ind_beta = 4;
		      else if ( betap_rad >  ys_bl[sizey-1] && 
				betap_rad <  ys_bh[sizey-1] ) 
			{
			  double binw = ( ys_bh[sizey-1] - ys_bl[sizey-1] ) / 3.0;
			  ind_beta = 1 + (int)( ( betap_rad - ys_bl[sizey-1] ) / binw );
			}
		      else cout << " Wrong betap_rad = " << betap_rad << endl << endl;
		    }
		  else 
		    cout << " Wrong sizey = " << sizey << endl << endl;
		}
	      else cout << " Wrong sizey = " << sizey << endl << endl;
	      
	      h_yres_npix_alpha_beta[ind_sizey][ind_alpha][ind_beta]->Fill( rechitresy ); 
	      
	    } // if ( !( sizey == 1 && bigy == 1 ) )
	  
	  // x residuals----------------------------------------------------------------
	  int sizex = nxpix;
	  if ( !( sizex == 1 && bigx == 1 ) ) // skip ( sizex == 1 && bigx == 1 ) clusters; the associated error is pitch_x/sqrt(12.0) 
	    {
	      if ( sizex > 3 ) sizex = 3;
	      
	      int ind_sizex = sizex - 1;
	      int ind_beta  = -9999;
	      int ind_alpha = -9999;
	      
	      if ( sizex == 1 )
		ind_beta = 0;
	      else 
		{
		  if      (                     betap_rad <= 0.7 ) ind_beta = 0;
		  else if ( 0.7 <  betap_rad && betap_rad <= 1.0 ) ind_beta = 1;
		  else if ( 1.0 <  betap_rad && betap_rad <= 1.2 ) ind_beta = 2;
		  else if ( 1.2 <= betap_rad                     ) ind_beta = 3;
		  else cout << " Wrong betap_rad = " << betap_rad << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << endl;
		}
	      
	      ind_alpha = (int) ( ( alpha_rad - a_min ) / ( ( a_max - a_min ) / 10.0 ) );  
	      
	      h_xres_npix_beta_alpha[ind_sizex][ind_beta][ind_alpha]->Fill( rechitresx ); 
	      
	    } //  if ( !( sizex == 1 && bigx == 1 ) )
	  
	} // if ( subdetId == 1 )
      else if ( subdetId == 2 )
	{
	  // y residuals----------------------------------------------------------------
	  int sizey = nypix;
	  
	  if ( !( sizey == 1 && bigy == 1 ) ) // skip ( sizex == y && bigx == y ) clusters; the associated error is pitch_y/sqrt(12.0) 
	    {
	      if ( sizey > 2 ) sizey = 2;
	      
	      int ind_sizey = sizey - 1;
	      int ind_beta  = -9999; 
	      
	      if ( betap_rad < 0.3 ) betap_rad = 0.3;
	      if ( betap_rad > 0.4 ) betap_rad = 0.4;
	      
	      ind_beta = (int) ( ( betap_rad - 0.3 ) / ( ( 0.4 - 0.3 ) / 10.0 ) );  
	      
	      h_forward_yres_npix_beta[ind_sizey][ind_beta]->Fill( rechitresy ); 
	    } // if ( !( sizey == 1 && bigy == 1 ) )
	  
	  // x residuals----------------------------------------------------------------
	  int sizex = nxpix;
	  
	  if ( !( sizex == 1 && bigx == 1 ) ) // skip ( sizex == 1 && bigx == 1 ) clusters; the associated error is pitch_x/sqrt(12.0) 
	    {
	      if ( sizex > 2 ) sizex = 2;
	      
	      int ind_sizex = sizex - 1;
	      int ind_alpha  = -9999; 
	      
	      if ( sizex == 1 )
		{
		  if ( alphap_rad < 0.15 ) alphap_rad = 0.15;
		  if ( alphap_rad > 0.30 ) alphap_rad = 0.30;
		  ind_alpha = (int) ( ( alphap_rad - 0.15 ) / ( ( 0.3 - 0.15 ) / 10.0 ) );  
		}
	      if ( sizex > 1 )
		{
		  if ( alphap_rad < 0.15 ) alphap_rad = 0.15;
		  if ( alphap_rad > 0.50 ) alphap_rad = 0.50;
		  ind_alpha = (int) ( ( alphap_rad - 0.15 ) / ( ( 0.5 - 0.15 ) / 10.0 ) );  
		}
	      
	      h_forward_xres_npix_alpha[ind_sizex][ind_alpha]->Fill( rechitresx ); 
	    } // if ( !( sizex == 1 && bigx == 1 ) )
	  
	} // else if ( subdetId == 2 )
      else
	cout << " Wrong Detector ID !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << endl;
      
    } //  for (Long64_t jentry=0; jentry<nentries; jentry++) 
  
  TF1* myfunc = new TF1("myfunc", ng, -0.01, 0.01, 3);
  myfunc->SetParNames("norm", "mean", "sigma");
  myfunc->SetParameter(0, 100.0);
  myfunc->SetParameter(1, 0.0);
  myfunc->SetParameter(2, 0.0020);
  myfunc->SetLineColor(kRed);
  
  int k = -99999; // sizey index
  int i = -99999; // alpha index

  Double_t sigma  = -99999.9;
  Double_t ssigma = -99999.9;

  bool do_yb = true;
  bool do_xb = true;
  bool do_yf = true;
  bool do_xf = true;

  if ( do_yb )
    {
      k = 0;
      TCanvas* c0 = new TCanvas("c0", "c0", 1200, 400);
      c0->Divide(3,1);
      i = 0;
      for (int j=0; j<3; ++j)
	{
	  c0->cd(j + 1);
	  h_yres_npix_alpha_beta[k][i][j]->Draw();
	  
	  double n = h_yres_npix_alpha_beta[k][i][j]->GetEntries();
	  double rms = h_yres_npix_alpha_beta[k][i][j]->GetRMS();
	  double binw = h_yres_npix_alpha_beta[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0.0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_yres_npix_alpha_beta[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<3; ++j)
      
      TCanvas* c1 = new TCanvas("c1", "c1", 1200, 500);
      c1->Divide(5,2);
      i = 1;
      for (int j=0; j<10; ++j)
	{
	  Double_t sigma = -99999.9;
	  c1->cd(j + 1);
	  h_yres_npix_alpha_beta[k][i][j]->Draw();
	  
	  double n = h_yres_npix_alpha_beta[k][i][j]->GetEntries();
	  double rms = h_yres_npix_alpha_beta[k][i][j]->GetRMS();
	  double binw = h_yres_npix_alpha_beta[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_yres_npix_alpha_beta[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<10; ++j)
      
      TCanvas* c2 = new TCanvas("c2", "c2", 1200, 400);
      c2->Divide(3,1);
      i = 2;
      for (int j=0; j<3; ++j)
	{
	  Double_t sigma = -99999.9;
	  c2->cd(j + 1);
	  h_yres_npix_alpha_beta[k][i][j]->Draw();
	  
	  double n = h_yres_npix_alpha_beta[k][i][j]->GetEntries();
	  double rms = h_yres_npix_alpha_beta[k][i][j]->GetRMS();
	  double binw = h_yres_npix_alpha_beta[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0.0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_yres_npix_alpha_beta[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<3; ++j)
      
      k = 1;
      TCanvas* c3 = new TCanvas("c3", "c3", 1200, 400);
      c3->Divide(3,1);
      i = 0;
      for (int j=0; j<3; ++j)
	{
	  Double_t sigma = -99999.9;
	  c3->cd(j + 1);
	  h_yres_npix_alpha_beta[k][i][j]->Draw();
	  
	  double n = h_yres_npix_alpha_beta[k][i][j]->GetEntries();
	  double rms = h_yres_npix_alpha_beta[k][i][j]->GetRMS();
	  double binw = h_yres_npix_alpha_beta[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0.0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_yres_npix_alpha_beta[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<3; ++j)
      
      TCanvas* c4 = new TCanvas("c4", "c4", 1200, 500);
      c4->Divide(5,2);
      i = 1;
      for (int j=0; j<10; ++j)
	{
	  Double_t sigma = -99999.9;
	  c4->cd(j + 1);
	  h_yres_npix_alpha_beta[k][i][j]->Draw();
	  
	  double n = h_yres_npix_alpha_beta[k][i][j]->GetEntries();
	  double rms = h_yres_npix_alpha_beta[k][i][j]->GetRMS();
	  double binw = h_yres_npix_alpha_beta[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_yres_npix_alpha_beta[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<10; ++j)
      
      TCanvas* c5 = new TCanvas("c5", "c5", 1200, 400);
      c5->Divide(3,1);
      i = 2;
      for (int j=0; j<3; ++j)
	{
	  Double_t sigma = -99999.9;
	  c5->cd(j + 1);
	  h_yres_npix_alpha_beta[k][i][j]->Draw();
	  
	  double n = h_yres_npix_alpha_beta[k][i][j]->GetEntries();
	  double rms = h_yres_npix_alpha_beta[k][i][j]->GetRMS();
	  double binw = h_yres_npix_alpha_beta[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0.0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_yres_npix_alpha_beta[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<3; ++j)
      
      k = 2;
      TCanvas* c6 = new TCanvas("c6", "c6", 1200, 400);
      c6->Divide(3,1);
      i = 0;
      for (int j=0; j<3; ++j)
	{
	  Double_t sigma = -99999.9;
	  c6->cd(j + 1);
	  h_yres_npix_alpha_beta[k][i][j]->Draw();
	  
	  double n = h_yres_npix_alpha_beta[k][i][j]->GetEntries();
	  double rms = h_yres_npix_alpha_beta[k][i][j]->GetRMS();
	  double binw = h_yres_npix_alpha_beta[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0.0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_yres_npix_alpha_beta[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<3; ++j)
      
      TCanvas* c7 = new TCanvas("c7", "c7", 1200, 500);
      c7->Divide(5,2);
      i = 1;
      for (int j=0; j<10; ++j)
	{
	  Double_t sigma = -99999.9;
	  c7->cd(j + 1);
	  h_yres_npix_alpha_beta[k][i][j]->Draw();
      
	  double n = h_yres_npix_alpha_beta[k][i][j]->GetEntries();
	  double rms = h_yres_npix_alpha_beta[k][i][j]->GetRMS();
	  double binw = h_yres_npix_alpha_beta[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_yres_npix_alpha_beta[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<10; ++j)
      
      TCanvas* c8 = new TCanvas("c8", "c8", 1200, 400);
      c8->Divide(3,1);
      i = 2;
      for (int j=0; j<3; ++j)
	{
	  Double_t sigma = -99999.9;
	  c8->cd(j + 1);
	  h_yres_npix_alpha_beta[k][i][j]->Draw();
	  
	  double n = h_yres_npix_alpha_beta[k][i][j]->GetEntries();
	  double rms = h_yres_npix_alpha_beta[k][i][j]->GetRMS();
	  double binw = h_yres_npix_alpha_beta[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0.0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_yres_npix_alpha_beta[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<3; ++j)
      
      k = 3;
      TCanvas* c9 = new TCanvas("c9", "c9", 1200, 500);
      c9->Divide(5,2);
      i = 0;
      for (int j=0; j<10; ++j)
	{
	  Double_t sigma = -99999.9;
	  c9->cd(j + 1);
	  h_yres_npix_alpha_beta[k][i][j]->Draw();
	  
	  double n = h_yres_npix_alpha_beta[k][i][j]->GetEntries();
	  double rms = h_yres_npix_alpha_beta[k][i][j]->GetRMS();
	  double binw = h_yres_npix_alpha_beta[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_yres_npix_alpha_beta[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<10; ++j)
      
      k = 4;
      TCanvas* c10 = new TCanvas("c10", "c10", 1200, 300);
      c10->Divide(5,1);
      i = 0;
      for (int j=0; j<5; ++j)
	{
	  Double_t sigma = -99999.9;
	  c10->cd(j + 1);
	  h_yres_npix_alpha_beta[k][i][j]->Draw();
	  
	  double n = h_yres_npix_alpha_beta[k][i][j]->GetEntries();
	  double rms = h_yres_npix_alpha_beta[k][i][j]->GetRMS();
	  double binw = h_yres_npix_alpha_beta[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_yres_npix_alpha_beta[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<10; ++j)
      
      k = 5;
      TCanvas* c11 = new TCanvas("c11", "c11", 1200, 500);
      c11->Divide(5,2);
      i = 0;
      for (int j=0; j<10; ++j)
	{
	  Double_t sigma = -99999.9;
	  c11->cd(j + 1);
	  h_yres_npix_alpha_beta[k][i][j]->Draw();
	  
	  double n = h_yres_npix_alpha_beta[k][i][j]->GetEntries();
	  double rms = h_yres_npix_alpha_beta[k][i][j]->GetRMS();
	  double binw = h_yres_npix_alpha_beta[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_yres_npix_alpha_beta[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<10; ++j)
  

      c0->SaveAs("yb_npix1_alpha1_beta_res.eps");
      c1->SaveAs("yb_npix1_alpha2_beta_res.eps");
      c2->SaveAs("yb_npix1_alpha3_beta_res.eps");
      c3->SaveAs("yb_npix2_alpha1_beta_res.eps");
      c4->SaveAs("yb_npix2_alpha2_beta_res.eps");
      c5->SaveAs("yb_npix2_alpha3_beta_res.eps");
      c6->SaveAs("yb_npix3_alpha1_beta_res.eps");
      c7->SaveAs("yb_npix3_alpha2_beta_res.eps");
      c8->SaveAs("yb_npix3_alpha3_beta_res.eps");
      c9->SaveAs("yb_npix4_alpha0_beta_res.eps");
      c10->SaveAs("yb_npix5_alpha0_beta_res.eps");
      c11->SaveAs("yb_npix6_alpha0_beta_res.eps");


    } // if ( do_yb )




  if ( do_xb )
    {
      k = 0;
      TCanvas* c12 = new TCanvas("c12", "c12", 1200, 500);
      c12->Divide(5,2);
      i = 0;
      for (int j=0; j<10; ++j)
	{
	  Double_t sigma = -99999.9;
	  c12->cd(j + 1);
	  h_xres_npix_beta_alpha[k][i][j]->Draw();
	  
	  double n = h_xres_npix_beta_alpha[k][i][j]->GetEntries();
	  double rms = h_xres_npix_beta_alpha[k][i][j]->GetRMS();
	  double binw = h_xres_npix_beta_alpha[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0.0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_xres_npix_beta_alpha[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	      ssigma = myfunc->GetParError(2);
	      h_xres_npix_beta[k][i]->SetBinContent(j+1, sigma);
	      h_xres_npix_beta[k][i]->SetBinError(j+1, ssigma);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<3; ++j)
      
      k = 1;
      TCanvas* c13 = new TCanvas("c13", "c13", 1200, 500);
      c13->Divide(5,2);
      i = 0;
      for (int j=0; j<10; ++j)
	{
	  Double_t sigma = -99999.9;
	  c13->cd(j + 1);
	  h_xres_npix_beta_alpha[k][i][j]->Draw();
	  
	  double n = h_xres_npix_beta_alpha[k][i][j]->GetEntries();
	  double rms = h_xres_npix_beta_alpha[k][i][j]->GetRMS();
	  double binw = h_xres_npix_beta_alpha[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0.0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_xres_npix_beta_alpha[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	      ssigma = myfunc->GetParError(2);
	      h_xres_npix_beta[k][i]->SetBinContent(j+1, sigma);
	      h_xres_npix_beta[k][i]->SetBinError(j+1, ssigma);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<3; ++j)
      
      TCanvas* c14 = new TCanvas("c14", "c14", 1200, 500);
      c14->Divide(5,2);
      i = 1;
      for (int j=0; j<10; ++j)
	{
	  Double_t sigma = -99999.9;
	  c14->cd(j + 1);
	  h_xres_npix_beta_alpha[k][i][j]->Draw();
	  
	  double n = h_xres_npix_beta_alpha[k][i][j]->GetEntries();
	  double rms = h_xres_npix_beta_alpha[k][i][j]->GetRMS();
	  double binw = h_xres_npix_beta_alpha[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_xres_npix_beta_alpha[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	      ssigma = myfunc->GetParError(2);
	      h_xres_npix_beta[k][i]->SetBinContent(j+1, sigma);
	      h_xres_npix_beta[k][i]->SetBinError(j+1, ssigma);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<10; ++j)
      
      TCanvas* c15 = new TCanvas("c15", "c15", 1200, 500);
      c15->Divide(5,2);
      i = 2;
      for (int j=0; j<10; ++j)
	{
	  Double_t sigma = -99999.9;
	  c15->cd(j + 1);
	  h_xres_npix_beta_alpha[k][i][j]->Draw();
	  
	  double n = h_xres_npix_beta_alpha[k][i][j]->GetEntries();
	  double rms = h_xres_npix_beta_alpha[k][i][j]->GetRMS();
	  double binw = h_xres_npix_beta_alpha[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0.0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_xres_npix_beta_alpha[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	      ssigma = myfunc->GetParError(2);
	      h_xres_npix_beta[k][i]->SetBinContent(j+1, sigma);
	      h_xres_npix_beta[k][i]->SetBinError(j+1, ssigma);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<3; ++j)
      
      TCanvas* c16 = new TCanvas("c16", "c16", 1200, 500);
      c16->Divide(5,2);
      i = 3;
      for (int j=0; j<10; ++j)
	{
	  Double_t sigma = -99999.9;
	  c16->cd(j + 1);
	  h_xres_npix_beta_alpha[k][i][j]->Draw();
	  
	  double n = h_xres_npix_beta_alpha[k][i][j]->GetEntries();
	  double rms = h_xres_npix_beta_alpha[k][i][j]->GetRMS();
	  double binw = h_xres_npix_beta_alpha[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0.0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_xres_npix_beta_alpha[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	      ssigma = myfunc->GetParError(2);
	      h_xres_npix_beta[k][i]->SetBinContent(j+1, sigma);
	      h_xres_npix_beta[k][i]->SetBinError(j+1, ssigma);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<3; ++j)


      k = 2;
      TCanvas* c17 = new TCanvas("c17", "c17", 1200, 500);
      c17->Divide(5,2);
      i = 0;
      for (int j=0; j<10; ++j)
	{
	  Double_t sigma = -99999.9;
	  c17->cd(j + 1);
	  h_xres_npix_beta_alpha[k][i][j]->Draw();
	  
	  double n = h_xres_npix_beta_alpha[k][i][j]->GetEntries();
	  double rms = h_xres_npix_beta_alpha[k][i][j]->GetRMS();
	  double binw = h_xres_npix_beta_alpha[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0.0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_xres_npix_beta_alpha[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	      ssigma = myfunc->GetParError(2);
	      h_xres_npix_beta[k][i]->SetBinContent(j+1, sigma);
	      h_xres_npix_beta[k][i]->SetBinError(j+1, ssigma);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<3; ++j)
      
      TCanvas* c18 = new TCanvas("c18", "c18", 1200, 500);
      c18->Divide(5,2);
      i = 1;
      for (int j=0; j<10; ++j)
	{
	  Double_t sigma = -99999.9;
	  c18->cd(j + 1);
	  h_xres_npix_beta_alpha[k][i][j]->Draw();
      
	  double n = h_xres_npix_beta_alpha[k][i][j]->GetEntries();
	  double rms = h_xres_npix_beta_alpha[k][i][j]->GetRMS();
	  double binw = h_xres_npix_beta_alpha[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_xres_npix_beta_alpha[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	      ssigma = myfunc->GetParError(2);
	      h_xres_npix_beta[k][i]->SetBinContent(j+1, sigma);
	      h_xres_npix_beta[k][i]->SetBinError(j+1, ssigma);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<10; ++j)
      
      TCanvas* c19 = new TCanvas("c19", "c19", 1200, 500);
      c19->Divide(5,2);
      i = 2;
      for (int j=0; j<10; ++j)
	{
	  Double_t sigma = -99999.9;
	  c19->cd(j + 1);
	  h_xres_npix_beta_alpha[k][i][j]->Draw();
	  
	  double n = h_xres_npix_beta_alpha[k][i][j]->GetEntries();
	  double rms = h_xres_npix_beta_alpha[k][i][j]->GetRMS();
	  double binw = h_xres_npix_beta_alpha[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0.0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_xres_npix_beta_alpha[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	      ssigma = myfunc->GetParError(2);
	      h_xres_npix_beta[k][i]->SetBinContent(j+1, sigma);
	      h_xres_npix_beta[k][i]->SetBinError(j+1, ssigma);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<3; ++j)
          
      TCanvas* c20 = new TCanvas("c20", "c20", 1200, 500);
      c20->Divide(5,2);
      i = 3;
      for (int j=0; j<10; ++j)
	{
	  Double_t sigma = -99999.9;
	  c20->cd(j + 1);
	  h_xres_npix_beta_alpha[k][i][j]->Draw();
	  
	  double n = h_xres_npix_beta_alpha[k][i][j]->GetEntries();
	  double rms = h_xres_npix_beta_alpha[k][i][j]->GetRMS();
	  double binw = h_xres_npix_beta_alpha[k][i][j]->GetBinWidth(1);
	  
	  if ( n != 0.0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms);
	      
	      h_xres_npix_beta_alpha[k][i][j]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	      ssigma = myfunc->GetParError(2);
	      h_xres_npix_beta[k][i]->SetBinContent(j+1, sigma);
	      h_xres_npix_beta[k][i]->SetBinError(j+1, ssigma);
	    }
	  cout << "ind_sizey = " << k+1 << ", ind_alpha = " << i+1 << ", ind_beta = " << j+1 << ", sigma = " << sigma << endl; 
	  
	} //  for (int j=0; j<10; ++j)

      TCanvas* c21 = new TCanvas("c21", "c21", 1200, 500);
      c21->Divide(5,2);
      c21->cd(1);
      h_xres_npix_beta[0][0]->Draw();
      for (int i=2; i<6; ++i)
	{
	  c21->cd(i);
	  h_xres_npix_beta[1][i-2]->Draw();
	  
	}
      for (int i=7; i<11; ++i)
	{
	  c21->cd(i);
	  h_xres_npix_beta[2][i-7]->Draw();
	  
	}
	

      c12->SaveAs("xb_npix1_beta0_alpha_res.eps");
      c13->SaveAs("xb_npix2_beta1_alpha_res.eps");
      c14->SaveAs("xb_npix2_beta2_alpha_res.eps");
      c15->SaveAs("xb_npix2_beta3_alpha_res.eps");
      c16->SaveAs("xb_npix2_beta4_alpha_res.eps");
      c17->SaveAs("xb_npix3_beta1_alpha_res.eps");
      c18->SaveAs("xb_npix3_beta2_alpha_res.eps");
      c19->SaveAs("xb_npix3_beta3_alpha_res.eps");
      c20->SaveAs("xb_npix3_beta4_alpha_res.eps");
      c21->SaveAs("xb_summary_res.eps");

    } //  if ( do_xb )



  if ( do_yf )
    {
      TCanvas* c22 = new TCanvas("c22", "c22", 1200, 500);
      c22->Divide(5,2);
      for (int i=0; i<10; ++i)
	{
	  c22->cd(i+1);
	  h_forward_yres_npix_beta[0][i]->Draw();
  
	  double n = h_forward_yres_npix_beta[0][i]->GetEntries();
	  double rms = h_forward_yres_npix_beta[0][i]->GetRMS();
	  double binw = h_forward_yres_npix_beta[0][i]->GetBinWidth(1);
	  
	  if ( n != 0.0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms/2.0);
	      
	      h_forward_yres_npix_beta[0][i]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	      ssigma = myfunc->GetParError(2);
	      h_forward_yres_npix[0]->SetBinContent(i+1, sigma);
	      h_forward_yres_npix[0]->SetBinError(i+1, ssigma);
	    }
	}

      TCanvas* c23 = new TCanvas("c23", "c23", 1200, 500);
      c23->Divide(5,2);
      for (int i=0; i<10; ++i)
	{
	  c23->cd(i+1);
	  h_forward_yres_npix_beta[1][i]->Draw();
	 
	  double n = h_forward_yres_npix_beta[1][i]->GetEntries();
	  double rms = h_forward_yres_npix_beta[1][i]->GetRMS();
	  double binw = h_forward_yres_npix_beta[1][i]->GetBinWidth(1);
	  
	  if ( n != 0.0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms/2.0);
	      
	      h_forward_yres_npix_beta[1][i]->Fit("myfunc", "QR");
	      
	      sigma = myfunc->GetParameter(2);
	      ssigma = myfunc->GetParError(2);
	      h_forward_yres_npix[1]->SetBinContent(i+1, sigma);
	      h_forward_yres_npix[1]->SetBinError(i+1, ssigma);
	    }
	}

      
      TCanvas* c24 = new TCanvas("c24", "c24", 800, 400);
      c24->Divide(2,1);
      c24->cd(1);
      h_forward_yres_npix[0]->Draw();
      c24->cd(2);
      h_forward_yres_npix[1]->Draw();

      
      c22->SaveAs("yf_npix1_beta_res.eps");
      c23->SaveAs("yf_npix2_beta_res.eps");
      c24->SaveAs("yf_summary_res.eps");

    } // if ( do_yf )
  

  if ( do_xf )
    {
      TCanvas* c25 = new TCanvas("c25", "c25", 1200, 500);
      c25->Divide(5,2);
      for (int i=0; i<10; ++i)
	{
	  c25->cd(i+1);
	  h_forward_xres_npix_alpha[0][i]->Draw();
  
	  double n = h_forward_xres_npix_alpha[0][i]->GetEntries();
	  double rms = h_forward_xres_npix_alpha[0][i]->GetRMS();
	  double binw = h_forward_xres_npix_alpha[0][i]->GetBinWidth(1);
	  
	  if ( n != 0.0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms/2.0);
	      
	      h_forward_xres_npix_alpha[0][i]->Fit("myfunc", "QR");
	      sigma = myfunc->GetParameter(2);
	      ssigma = myfunc->GetParError(2);
	      h_forward_xres_npix[0]->SetBinContent(i+1, sigma);
	      h_forward_xres_npix[0]->SetBinError(i+1, ssigma);
	    }
	}

      TCanvas* c26 = new TCanvas("c26", "c26", 1200, 500);
      c26->Divide(5,2);
      for (int i=0; i<10; ++i)
	{
	  c26->cd(i+1);
	  h_forward_xres_npix_alpha[1][i]->Draw();
	 
	  double n = h_forward_xres_npix_alpha[1][i]->GetEntries();
	  double rms = h_forward_xres_npix_alpha[1][i]->GetRMS();
	  double binw = h_forward_xres_npix_alpha[1][i]->GetBinWidth(1);
	  
	  if ( n != 0.0 )
	    {
	      double norm = n*binw;
	      
	      myfunc->SetParameter(0, norm);
	      myfunc->SetParameter(1, 0.0);
	      myfunc->SetParameter(2, rms/2.0);
	      
	      h_forward_xres_npix_alpha[1][i]->Fit("myfunc", "QR");
	      
	      sigma = myfunc->GetParameter(2);
	      ssigma = myfunc->GetParError(2);
	      h_forward_xres_npix[1]->SetBinContent(i+1, sigma);
	      h_forward_xres_npix[1]->SetBinError(i+1, ssigma);
	    }
	}

      
      TCanvas* c27 = new TCanvas("c27", "c27", 800, 400);
      c27->Divide(2,1);
      c27->cd(1);
      h_forward_xres_npix[0]->Draw();
      c27->cd(2);
      h_forward_xres_npix[1]->Draw();

      c25->SaveAs("xf_npix1_alpha_res.eps");
      c26->SaveAs("xf_npix2_alpha_res.eps");
      c27->SaveAs("xf_summary_res.eps");

    } // if ( do_yf )






}
