#define analyse_residuals_cxx
#include "analyse_residuals.h"
#include <TH1F.h>
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
#include <TF1.h>
#include <TMath.h>
#include "TFile.h"

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

double aa_min[3] = {1.50, 1.45, 1.40};
double aa_max[3] = {1.75, 1.70, 1.65};

double ys_bl[6];
double ys_bh[6];

TCanvas* can_y_barrel_sizey_alpha[6][4];
TCanvas* can_y_barrel_sizey[6];
TH1F* h_yres_npix_alpha_beta[6][4][10]; 
TH1F* h_yres_npix_alpha[6][4]; 
TH1F* h_yres_npix_alpha_rms[6][4]; 

TCanvas* can_x_barrel_sizex_beta[3][4];
TCanvas* can_x_barrel_sizex[3];
TH1F* h_xres_npix_beta_alpha[3][4][10];
TH1F* h_xres_npix_beta[3][4];
TH1F* h_xres_npix_beta_rms[3][4];


/*
TCanvas* can_x_barrel_sizex_beta_flipy[3][4];
TCanvas* can_x_barrel_sizex_flipy[3];
TH1F* h_xres_npix_beta_alpha_flipy[3][4][10];
TH1F* h_xres_npix_beta_flipy[3][4];
TH1F* h_xres_npix_beta_rms_flipy[3][4];

TCanvas* can_x_barrel_sizex_beta_flipn[3][4];
TCanvas* can_x_barrel_sizex_flipn[3];
TH1F* h_xres_npix_beta_alpha_flipn[3][4][10];
TH1F* h_xres_npix_beta_flipn[3][4];
TH1F* h_xres_npix_beta_rms_flipn[3][4];
*/


TCanvas* can_y_forward_sizey[2];
TCanvas* can_y_forward;
TH1F* h_forward_yres_npix_beta[2][10]; 
TH1F* h_forward_yres_npix[2]; 
TH1F* h_forward_yres_npix_rms[2]; 

TCanvas* can_x_forward_sizex[2];
TCanvas* can_x_forward;
TH1F* h_forward_xres_npix_alpha[2][10]; 
TH1F* h_forward_xres_npix[2];
TH1F* h_forward_xres_npix_rms[2];

const char* fname;

void analyse_residuals::Loop()
{
  if (fChain == 0) return;
 
  Long64_t nentries = fChain->GetEntries();
  cout << "nentries = " << nentries << endl;

  bool do_residuals = true;
  bool do_plots = true;
 
  bool do_yb = true;
  bool do_xb = true;
  bool do_yf = true;
  bool do_xf = true;

  bool do_fix = true;

  if ( do_residuals )
    fname = "residuals.dat";
  else
    fname = "pulls.dat";
  
  printf("%s \n", fname);
  FILE* datfile; 
  if ( (datfile = fopen(fname, "w"))==NULL )
    {      cout << "could not open the output file." << endl;
      exit(-1);
    }

  char hname[100];

  ys_bl[0] = 0.05;
  ys_bh[0] = 0.50;

  ys_bl[1] = 0.15; 
  ys_bh[1] = 0.90;

  ys_bl[2] = 0.70; 
  ys_bh[2] = 1.05;

  ys_bl[3] = 0.95; 
  ys_bh[3] = 1.15;

  ys_bl[4] = 1.15; 
  ys_bh[4] = 1.20;

  ys_bl[5] = 1.20; 
  ys_bh[5] = 1.40;

  if ( do_yb )
    {
      for (int i=0; i<6; ++i)
        {
          for (int j=0; j<4; ++j)
            {
              sprintf(hname, "can_y_barrel_sizey_alpha_%i_%i", i, j);
              can_y_barrel_sizey_alpha[i][j] = new TCanvas(hname, hname, 1200, 500);
              can_y_barrel_sizey_alpha[i][j]->Divide(5,2);
            }
         
          sprintf(hname, "can_y_barrel_sizey_alpha_%i", i);
          can_y_barrel_sizey[i] = new TCanvas(hname, hname, 1200, 350);
          can_y_barrel_sizey[i]->Divide(4);
        }
    }
  
  if ( do_xb )
    {
      for (int i=0; i<3; ++i)
        {
          for (int j=0; j<4; ++j)
            {
              sprintf(hname, "can_x_barrel_sizex_beta_%i_%i", i, j);
              can_x_barrel_sizex_beta[i][j] = new TCanvas(hname, hname, 1200, 500);
              can_x_barrel_sizex_beta[i][j]->Divide(5,2);
            }
          
          sprintf(hname, "can_x_barrel_sizex_beta_%i", i);
          can_x_barrel_sizex[i] = new TCanvas(hname, hname, 1200, 350);
          can_x_barrel_sizex[i]->Divide(4);
        }
    }
  
  if ( do_yf )
    {
      for (int i=0; i<2; ++i)
        {
          sprintf(hname, "can_y_forward_sizey_%i", i);
          can_y_forward_sizey[i] = new TCanvas(hname, hname, 1200, 500);
          can_y_forward_sizey[i]->Divide(5,2);
        } 
      
      sprintf(hname, "can_y_forward");
      can_y_forward = new TCanvas(hname, hname, 800, 400);
      can_y_forward->Divide(2);
    }

  if ( do_xf )
    {
      for (int i=0; i<2; ++i)
        {
          sprintf(hname, "can_x_forward_sizex_%i", i);
          can_x_forward_sizex[i] = new TCanvas(hname, hname, 1200, 500);
          can_x_forward_sizex[i]->Divide(5,2);
        } 
      
      sprintf(hname, "can_x_forward");
      can_x_forward = new TCanvas(hname, hname, 800, 400);
      can_x_forward->Divide(2);
    }

  // barrel histograms --------------------------------------------------------------
  for (int i=0; i<6; ++i) // loop over size_y
    for (int j=0; j<4; ++j) // loop over alpha bins
      for (int k=0; k<10; ++k) // loop over be5a bins
        {
          sprintf(hname, "h_yres_npix_alpha_beta_%i_%i_%i", i, j, k );
          if ( do_residuals )
            h_yres_npix_alpha_beta[i][j][k]= new TH1F(hname, hname, 100, -0.02, 0.02);
          else // do pulls
            h_yres_npix_alpha_beta[i][j][k]= new TH1F(hname, hname, 100, -10.0, 10.0);
        }

  for (int i=0; i<6; ++i) // loop over size_y
    for (int j=0; j<4; ++j) // loop over alpha bins
      {
        sprintf(hname, "h_yres_npix_alpha_%i_%i", i, j );
        h_yres_npix_alpha[i][j] = new TH1F(hname, hname, 10, ys_bl[i], ys_bh[i]);
        
        sprintf(hname, "h_yres_npix_alpha_rms_%i_%i", i, j );
        h_yres_npix_alpha_rms[i][j] = new TH1F(hname, hname, 10, ys_bl[i], ys_bh[i]);
        h_yres_npix_alpha_rms[i][j]->SetLineColor(kRed);
        if ( do_residuals )
          {
            h_yres_npix_alpha_rms[i][j]->SetMinimum(0.0);
            h_yres_npix_alpha_rms[i][j]->SetMaximum(0.0060);
          }
        else
          {
            h_yres_npix_alpha_rms[i][j]->SetMinimum(0.0);
            h_yres_npix_alpha_rms[i][j]->SetMaximum(2.0);
          }
      }
 
  for (int i=0; i<3; ++i) // loop over size_x
    for (int j=0; j<4; ++j) // loop over beta bins
      for (int k=0; k<10; ++k) // loop over alpha bins
        {
          sprintf(hname, "h_xres_npix_beta_alpha_%i_%i_%i", i, j, k );
          if ( do_residuals )
            h_xres_npix_beta_alpha[i][j][k] = new TH1F(hname, hname, 100, -0.01, 0.01);
          else // do pulls 
            h_xres_npix_beta_alpha[i][j][k] = new TH1F(hname, hname, 100, -10.0, 10.0);
          
          /*
          sprintf(hname, "h_xres_npix_beta_alpha_%i_%i_%i_flipy", i, j, k );
          if ( do_residuals )
            h_xres_npix_beta_alpha_flipy[i][j][k] = new TH1F(hname, hname, 100, -0.01, 0.01);
          else // do pulls 
            h_xres_npix_beta_alpha_flipy[i][j][k] = new TH1F(hname, hname, 100, -10.0, 10.0);
        
          sprintf(hname, "h_xres_npix_beta_alpha_%i_%i_%i_flipn", i, j, k );
          if ( do_residuals )
            h_xres_npix_beta_alpha_flipn[i][j][k] = new TH1F(hname, hname, 100, -0.01, 0.01);
          else // do pulls 
            h_xres_npix_beta_alpha_flipn[i][j][k] = new TH1F(hname, hname, 100, -10.0, 10.0); 
          */

        }

  for (int i=0; i<3; ++i) // loop over size_x
    for (int j=0; j<4; ++j) // loop over beta bins
      {
        sprintf(hname, "h_xres_npix_beta_%i_%i", i, j );
        h_xres_npix_beta[i][j]= new TH1F(hname, hname, 10, a_min, a_max);
      
        sprintf(hname, "h_xres_npix_beta_rms_%i_%i", i, j );
        h_xres_npix_beta_rms[i][j]= new TH1F(hname, hname, 10, a_min, a_max);
        h_xres_npix_beta_rms[i][j]->SetLineColor(kRed);
        if ( do_residuals )
          {
            h_xres_npix_beta_rms[i][j]->SetMinimum(0.0);
            h_xres_npix_beta_rms[i][j]->SetMaximum(0.0060);
          }
        else
          {
            h_xres_npix_beta_rms[i][j]->SetMinimum(0.0);
            h_xres_npix_beta_rms[i][j]->SetMaximum(2.0);
          }

        /*
        sprintf(hname, "h_xres_npix_beta_%i_%i_flipy", i, j );
        h_xres_npix_beta_flipy[i][j]= new TH1F(hname, hname, 10, a_min, a_max);
      
        sprintf(hname, "h_xres_npix_beta_rms_%i_%i_flipy", i, j );
        h_xres_npix_beta_rms_flipy[i][j]= new TH1F(hname, hname, 10, a_min, a_max);
        h_xres_npix_beta_rms_flipy[i][j]->SetLineColor(kRed);
        if ( do_residuals )
          {
            h_xres_npix_beta_rms_flipy[i][j]->SetMinimum(0.0);
            h_xres_npix_beta_rms_flipy[i][j]->SetMaximum(0.0060);
          }
        else
          {
            h_xres_npix_beta_rms_flipy[i][j]->SetMinimum(0.0);
            h_xres_npix_beta_rms_flipy[i][j]->SetMaximum(2.0);
          }
        
        sprintf(hname, "h_xres_npix_beta_%i_%i_flipn", i, j );
        h_xres_npix_beta_flipn[i][j]= new TH1F(hname, hname, 10, a_min, a_max);
      
        sprintf(hname, "h_xres_npix_beta_rms_%i_%i_flipn", i, j );
        h_xres_npix_beta_rms_flipn[i][j]= new TH1F(hname, hname, 10, a_min, a_max);
        h_xres_npix_beta_rms_flipn[i][j]->SetLineColor(kRed);
        if ( do_residuals )
          {
            h_xres_npix_beta_rms_flipn[i][j]->SetMinimum(0.0);
            h_xres_npix_beta_rms_flipn[i][j]->SetMaximum(0.0060);
          }
        else
          {
            h_xres_npix_beta_rms_flipn[i][j]->SetMinimum(0.0);
            h_xres_npix_beta_rms_flipn[i][j]->SetMaximum(2.0);
          }
        */

      }

  // forward histograms --------------------------------------------------------------
  for (int i=0; i<2; ++i) // loop ove sizey bins
    for (int j=0; j<10; ++j) // loop over beta bins
      {
      sprintf(hname, "h_forward_yres_npix_beta_%i_%i", i, j );
      if ( do_residuals )
        h_forward_yres_npix_beta[i][j] = new TH1F(hname, hname, 100, -0.01, 0.01);
      else // do pulls
        h_forward_yres_npix_beta[i][j] = new TH1F(hname, hname, 100, -10.0, 10.0);
      }

  for (int i=0; i<2; ++i) // loop ove sizey bins
    {
      sprintf(hname, "h_forward_yres_npix_%i", i );
      h_forward_yres_npix[i] = new TH1F(hname, hname, 10, 0.3, 0.4);
    
      sprintf(hname, "h_forward_yres_npix_rms_%i", i );
      h_forward_yres_npix_rms[i] = new TH1F(hname, hname, 10, 0.3, 0.4);
      h_forward_yres_npix_rms[i]->SetLineColor(kRed);
      if ( do_residuals )
        {
          h_forward_yres_npix_rms[i]->SetMinimum(0.0);
          h_forward_yres_npix_rms[i]->SetMaximum(0.0040);
        }
      else
        {
          h_forward_yres_npix_rms[i]->SetMinimum(0.0);
          h_forward_yres_npix_rms[i]->SetMaximum(2.0);
        }
    }

  for (int i=0; i<2; ++i) // loop ove sizex bins
    for (int j=0; j<10; ++j) // loop over alpha bins
      {
        sprintf(hname, "h_forward_xres_npix_alpha_%i_%i", i, j );
        if ( do_residuals )
          h_forward_xres_npix_alpha[i][j] = new TH1F(hname, hname, 100, -0.01, 0.01);
        else // do pulls
          h_forward_xres_npix_alpha[i][j] = new TH1F(hname, hname, 100, -10.0, 10.0);
      }
  
  sprintf(hname, "h_forward_xres_npix_%i", 0 );
  h_forward_xres_npix[0] = new TH1F(hname, hname, 10, 0.15, 0.3);
  sprintf(hname, "h_forward_xres_npix_%i", 1 );
  h_forward_xres_npix[1] = new TH1F(hname, hname, 10, 0.15, 0.5);

  sprintf(hname, "h_forward_xres_npix_rms_%i", 0 );
  h_forward_xres_npix_rms[0] = new TH1F(hname, hname, 10, 0.15, 0.3);
  h_forward_xres_npix_rms[0]->SetLineColor(kRed);
  sprintf(hname, "h_forward_xres_npix_rms_%i", 1 );
  h_forward_xres_npix_rms[1] = new TH1F(hname, hname, 10, 0.15, 0.5);
  h_forward_xres_npix_rms[1]->SetLineColor(kRed);

  if ( do_residuals )
    {
      h_forward_xres_npix_rms[0]->SetMinimum(0.0);
      h_forward_xres_npix_rms[0]->SetMaximum(0.0040);
      h_forward_xres_npix_rms[1]->SetMinimum(0.0);
      h_forward_xres_npix_rms[1]->SetMaximum(0.0040);
    }
  else
    {
      h_forward_xres_npix_rms[0]->SetMinimum(0.0);
      h_forward_xres_npix_rms[0]->SetMaximum(2.0);
      h_forward_xres_npix_rms[1]->SetMinimum(0.0);
      h_forward_xres_npix_rms[1]->SetMaximum(2.0);
    }

  for (Long64_t jentry=0; jentry<nentries; jentry++) 
    {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      fChain->GetEntry(jentry);
      
      if ( !do_residuals )
        {
          rechitresx = rechitpullx;
          rechitresy = rechitpully;
        }
      
      if ( do_fix )
        {
          alpha = trk_alpha;
          beta  = trk_beta ;
        }

      double alpha_rad = fabs(alpha)/180.0*TMath::Pi();
      double beta_rad  = fabs(beta) /180.0*TMath::Pi();
      double betap_rad = fabs( TMath::Pi()/2.0 - beta_rad );
      double alphap_rad = fabs( TMath::Pi()/2.0 - alpha_rad );
 
      if ( subdetId == 1 )
        {
          // y residuals----------------------------------------------------------------
          int sizey = nypix;
          
          // skip ( sizey == 1 && bigy == 1 ) clusters; the associated error is pitch_y/sqrt(12.0) 
          if ( !( sizey == 1 && bigy == 1  ) ) 
            {
              if ( sizey > 6 ) sizey = 6;
              
              int ind_sizey = sizey - 1;
              int ind_alpha = -9999;
              int ind_beta  = -9999; 
              
              if      ( alpha_rad <= a_min ) ind_alpha = 0;
              else if ( alpha_rad >= a_max ) ind_alpha = 3;
              else if ( alpha_rad > a_min && 
                        alpha_rad < a_max ) 
                {
                  double binw = ( a_max - a_min ) / 4.0;
                  ind_alpha = (int)( ( alpha_rad - a_min ) / binw );
                }               
              else cout << " Wrong alpha_rad = " << alpha_rad << endl << endl;

              if      ( betap_rad <= ys_bl[sizey-1] ) ind_beta = 0;
              else if ( betap_rad >= ys_bh[sizey-1] ) ind_beta = 9;
              else if ( betap_rad >  ys_bl[sizey-1] && 
                        betap_rad <  ys_bh[sizey-1] ) 
                {
                  double binw = ( ys_bh[sizey-1] - ys_bl[sizey-1] ) / 8.0;
                  ind_beta = 1 + (int)( ( betap_rad - ys_bl[sizey-1] ) / binw );
                }               
              else cout << " Wrong betap_rad = " << betap_rad << endl << endl;
                              
              h_yres_npix_alpha_beta[ind_sizey][ind_alpha][ind_beta]->Fill( rechitresy ); 
              
            } // if ( !( sizey == 1 && bigy == 1 ) )
          
          // x residuals----------------------------------------------------------------
          int sizex = nxpix;
          // skip ( sizex == 1 && bigx == 1 ) clusters; the associated error is pitch_x/sqrt(12.0) 
          if ( !( sizex == 1 && bigx == 1 ) ) 
            {
              if ( sizex > 3 ) sizex = 3;
              
              int ind_sizex = sizex - 1;
              int ind_beta  = -9999;
              int ind_alpha = -9999;
              
              if      (                     betap_rad <= 0.7 ) ind_beta = 0;
              else if ( 0.7 <  betap_rad && betap_rad <= 1.0 ) ind_beta = 1;
              else if ( 1.0 <  betap_rad && betap_rad <= 1.2 ) ind_beta = 2;
              else if ( 1.2 <= betap_rad                     ) ind_beta = 3;
              else cout << " Wrong betap_rad = " << betap_rad << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << endl;
                      
              if      ( alpha_rad <= aa_min[ind_sizex] ) ind_alpha = 0;
              else if ( alpha_rad >= aa_max[ind_sizex] ) ind_alpha = 9;
              else
                ind_alpha = (int) ( ( alpha_rad - aa_min[ind_sizex] ) / ( ( aa_max[ind_sizex] - aa_min[ind_sizex] ) / 10.0 ) );  
              
              h_xres_npix_beta_alpha[ind_sizex][ind_beta][ind_alpha]->Fill( rechitresx ); 
              
              /*
              if ( flipped )
                h_xres_npix_beta_alpha_flipy[ind_sizex][ind_beta][ind_alpha]->Fill( rechitresx );
              else
                h_xres_npix_beta_alpha_flipn[ind_sizex][ind_beta][ind_alpha]->Fill( rechitresx );
              */

 
            } //  if ( !( sizex == 1 && bigx == 1 ) )
          
        } // if ( subdetId == 1 )
      else if ( subdetId == 2 )
        {
          // forward y residuals----------------------------------------------------------------
          int sizey = nypix;
          // skip ( sizex == y && bigx == y ) clusters; the associated error is pitch_y/sqrt(12.0)
          if ( !( sizey == 1 && bigy == 1 ) )  
            {
              if ( sizey > 2 ) sizey = 2;
              
              int ind_sizey = sizey - 1;
              int ind_beta  = -9999; 
              
              if      ( betap_rad < 0.3 ) ind_beta = 0;
              else if ( betap_rad > 0.4 ) ind_beta = 9;
              else 
                ind_beta = (int) ( ( betap_rad - 0.3 ) / ( ( 0.4 - 0.3 ) / 10.0 ) );  
              
              h_forward_yres_npix_beta[ind_sizey][ind_beta]->Fill( rechitresy ); 
            
            } // if ( !( sizey == 1 && bigy == 1 ) )
          
          // forward x residuals----------------------------------------------------------------
          int sizex = nxpix;
          // skip ( sizex == 1 && bigx == 1 ) clusters; the associated error is pitch_x/sqrt(12.0)
          if ( !( sizex == 1 && bigx == 1 ) )  
            {
              if ( sizex > 2 ) sizex = 2;
              
              int ind_sizex = sizex - 1;
              int ind_alpha  = -9999; 
              
              if ( sizex == 1 )
                {
                  if      ( alphap_rad < 0.15 ) ind_alpha = 0;
                  else if ( alphap_rad > 0.30 ) ind_alpha = 9;
                  else 
                    ind_alpha = (int) ( ( alphap_rad - 0.15 ) / ( ( 0.3 - 0.15 ) / 10.0 ) );  
                }
              if ( sizex > 1 )
                {
                  if      ( alphap_rad < 0.15 ) ind_alpha = 0;
                  else if ( alphap_rad > 0.50 ) ind_alpha = 9;
                  else 
                    ind_alpha = (int) ( ( alphap_rad - 0.15 ) / ( ( 0.5 - 0.15 ) / 10.0 ) );  
                }
            
              h_forward_xres_npix_alpha[ind_sizex][ind_alpha]->Fill( rechitresx ); 
            
            } // if ( !( sizex == 1 && bigx == 1 ) )
          
        } // else if ( subdetId == 2 )
      else
        cout << " Wrong Detector ID !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << endl;
      
    } //  for (Long64_t jentry=0; jentry<nentries; jentry++) 
  
  float low = -999.9;
  float high = -999.9;
  if ( do_residuals )
    {
      low = -0.02;
      high = 0.02;
    }
  else
    {
      low = -10.0;
      high = 10.0;
    }
  TF1* myfunc = new TF1("myfunc", ng, low, high, 3);
  myfunc->SetParNames("norm", "mean", "sigma");
  myfunc->SetParameter(0, 100.0);
  myfunc->FixParameter(1, 0.0);
  myfunc->SetParameter(2, 0.00020);
  myfunc->SetLineColor(kRed);
  
  Double_t sigma  = -99999.9;
  Double_t ssigma = -99999.9;

  // ----------------------------------------- barrel Y ------------------------------------------------------------

  if ( do_yb )
    {
      for (int k=0; k<6; ++k)
        {
          for (int i=0; i<4; ++i)
            for (int j=0; j<10; ++j)
              {
                can_y_barrel_sizey_alpha[k][i]->cd(j + 1);
                h_yres_npix_alpha_beta[k][i][j]->Draw();
                
                double n = h_yres_npix_alpha_beta[k][i][j]->GetEntries();
                double rms = h_yres_npix_alpha_beta[k][i][j]->GetRMS();
                double binw = h_yres_npix_alpha_beta[k][i][j]->GetBinWidth(1);
              
                if ( n != 0.0 )
                  {
                    double norm = n*binw;
                    
                    myfunc->SetParameter(0, norm);
                    myfunc->FixParameter(1, 0.0);
                    myfunc->SetParameter(2, rms/3.0);
                    
                    
                    //if ( do_residuals && k==0 && j<3 )
                    if ( k==0 && j<3 )
                      h_yres_npix_alpha_beta[k][i][j]->Fit("myfunc", "LQR");
                    else
                      h_yres_npix_alpha_beta[k][i][j]->Fit("myfunc", "QR");
                    
                    sigma = myfunc->GetParameter(2);
                    ssigma = myfunc->GetParError(2);
                    h_yres_npix_alpha[k][i]->SetBinContent(j+1, sigma);
                    h_yres_npix_alpha[k][i]->SetBinError(j+1, ssigma);
                  
                    h_yres_npix_alpha_rms[k][i]->SetBinContent(j+1, rms);
                  }

		if ( sigma < 0.0 )
		  {
		    sigma = rms;
		    cout << "Bad error, check fit convergence !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
		  }

                fprintf( datfile,
                         "%d %d %d %d %f %f \n", 
                         1, k, i, j, sigma, rms );
                
              } //  for (int j=0; j<3; ++j)
         
          for (int i=0; i<4; ++i)
            {
              if ( !do_residuals )
                {
                  h_yres_npix_alpha[k][i]->SetMinimum(0.0);
                  h_yres_npix_alpha[k][i]->SetMinimum(2.0);
                }
              can_y_barrel_sizey[k]->cd(i+1);
              h_yres_npix_alpha_rms[k][i]->Draw("");
              h_yres_npix_alpha[k][i]->Draw("same");
            }
        }

      if ( do_plots )
        {
          for (int i=0; i<6; ++i)
            {
              for (int j=0; j<4; ++j)
                {
                  if ( do_residuals )
                    sprintf(hname, "res_y_barrel_sizey_alpha_%i_%i.eps", i, j);
                  else
                    sprintf(hname, "pull_y_barrel_sizey_alpha_%i_%i.eps", i, j);
                
                  can_y_barrel_sizey_alpha[i][j]->SaveAs(hname);
                }

              if ( do_residuals )
                sprintf(hname, "res_y_barrel_sizey_%i.eps", i);
              else
                sprintf(hname, "pull_y_barrel_sizey_%i.eps", i);
              
              can_y_barrel_sizey[i]->SaveAs(hname);
            }
        }
    } // if ( do_yb )


  // ------------------------------------------- barrel X --------------------------------------------------------


  if ( do_xb )
    {
      for (int k=0; k<3; ++k)
        {
          for (int i=0; i<4; ++i)
            for (int j=0; j<10; ++j)
              {
                Double_t sigma = -99999.9;
                can_x_barrel_sizex_beta[k][i]->cd(j + 1);
                h_xres_npix_beta_alpha[k][i][j]->Draw();
                
                double n = h_xres_npix_beta_alpha[k][i][j]->GetEntries();
                double rms = h_xres_npix_beta_alpha[k][i][j]->GetRMS();
                double binw = h_xres_npix_beta_alpha[k][i][j]->GetBinWidth(1);
                
                if ( n != 0.0 )
                  {
                    double norm = n*binw;
                    
                    myfunc->SetParameter(0, norm);
                    myfunc->FixParameter(1, 0.0);
                    myfunc->SetParameter(2, rms/3.0);
                    
                    h_xres_npix_beta_alpha[k][i][j]->Fit("myfunc", "QR");
                    sigma = myfunc->GetParameter(2);
                    ssigma = myfunc->GetParError(2);
                    h_xres_npix_beta[k][i]->SetBinContent(j+1, sigma);
                    h_xres_npix_beta[k][i]->SetBinError(j+1, ssigma);

                    h_xres_npix_beta_rms[k][i]->SetBinContent(j+1, rms);
                  }

		if ( sigma < 0.0 )
		  {
		    sigma = rms;
		    cout << "Bad error, check fit convergence !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
		  }
		
                fprintf( datfile,
                         "%d %d %d %d %f %f \n",
                         2, k, i, j, sigma, rms );
                
              } //  for (int j=0; j<3; ++j)
          
          for (int i=0; i<4; ++i)
            {
              if ( !do_residuals )
                {
                  h_xres_npix_beta[k][i]->SetMinimum(0.0);
                  h_xres_npix_beta[k][i]->SetMinimum(2.0);
                }
              can_x_barrel_sizex[k]->cd(i+1);
              h_xres_npix_beta_rms[k][i]->Draw("");
              h_xres_npix_beta[k][i]->Draw("same");
            }
        }

      if ( do_plots )
        {
          for (int i=0; i<3; ++i)
            {
              for (int j=0; j<4; ++j)
                {
                  if ( do_residuals )
                    sprintf(hname, "res_x_barrel_sizex_beta_%i_%i.eps", i, j);
                  else
                    sprintf(hname, "pull_x_barrel_sizex_beta_%i_%i.eps", i, j);

                  can_x_barrel_sizex_beta[i][j]->SaveAs(hname);
                }
              
              if ( do_residuals )
                sprintf(hname, "res_x_barrel_sizex_%i.eps", i);
              else
                sprintf(hname, "pull_x_barrel_sizex_%i.eps", i);

              can_x_barrel_sizex[i]->SaveAs(hname);
            }
        }
      
    } //  if ( do_xb )


  // ---------------------------------------------- forward Y -------------------------------------------------

  if ( do_yf )
    {
      for ( int k=0; k<2; ++k )
        for (int i=0; i<10; ++i)
          {
            can_y_forward_sizey[k]->cd(i+1);
            h_forward_yres_npix_beta[k][i]->Draw();
            
            double n = h_forward_yres_npix_beta[k][i]->GetEntries();
            double rms = h_forward_yres_npix_beta[k][i]->GetRMS();
            double binw = h_forward_yres_npix_beta[k][i]->GetBinWidth(1);
            
            if ( n != 0.0 )
              {
                double norm = n*binw;
                
                myfunc->SetParameter(0, norm);
                myfunc->FixParameter(1, 0.0);
                myfunc->SetParameter(2, rms/3.0);
                
                h_forward_yres_npix_beta[k][i]->Fit("myfunc", "QR");
                sigma = myfunc->GetParameter(2);
                ssigma = myfunc->GetParError(2);
                h_forward_yres_npix[k]->SetBinContent(i+1, sigma);
                h_forward_yres_npix[k]->SetBinError(i+1, ssigma);
                
                h_forward_yres_npix_rms[k]->SetBinContent(i+1, rms);
              }

	    if ( sigma < 0.0 )
	      {
		sigma = rms;
		cout << "Bad error, check fit convergence !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
	      }
	    
            fprintf( datfile,
                     "%d %d %d %d %f %f \n",
                     3, 0, k, i, sigma, rms );
            
          }   
      
      for (int i=0; i<2; ++i)
        {
          if ( !do_residuals )
            {
              h_forward_yres_npix[i]->SetMinimum(0.0);
              h_forward_yres_npix[i]->SetMinimum(2.0);
            }
          can_y_forward->cd(i+1);
          h_forward_yres_npix_rms[i]->Draw("");
          h_forward_yres_npix[i]->Draw("same");
        }
      
      if ( do_plots )
        {
          for (int i=0; i<2; ++i)
            {
              if ( do_residuals )
                sprintf(hname, "res_y_forward_sizey_%i.eps", i);
              else
                sprintf(hname, "pull_y_forward_sizey_%i.eps", i);

              can_y_forward_sizey[i]->SaveAs(hname);
            }
          
          if ( do_residuals )
            sprintf(hname, "res_y_forward.eps");
          else
            sprintf(hname, "pull_y_forward.eps");

          can_y_forward->SaveAs(hname);
        }
    } // if ( do_yf )
  

  // ----------------------------------------------------- forward X -----------------------------------------

  if ( do_xf )
    {
      for (int k=0; k<2; ++k)
        for (int i=0; i<10; ++i)
          {
            can_x_forward_sizex[k]->cd(i+1);
            h_forward_xres_npix_alpha[k][i]->Draw();
            
            double n = h_forward_xres_npix_alpha[k][i]->GetEntries();
            double rms = h_forward_xres_npix_alpha[k][i]->GetRMS();
            double binw = h_forward_xres_npix_alpha[k][i]->GetBinWidth(1);
            
            if ( n != 0.0 )
              {
                double norm = n*binw;
                
                myfunc->SetParameter(0, norm);
                myfunc->FixParameter(1, 0.0);
                myfunc->SetParameter(2, rms/3.0);
                
                h_forward_xres_npix_alpha[k][i]->Fit("myfunc", "QR");
                sigma = myfunc->GetParameter(2);
                ssigma = myfunc->GetParError(2);
                h_forward_xres_npix[k]->SetBinContent(i+1, sigma);
                h_forward_xres_npix[k]->SetBinError(i+1, ssigma);
        
                h_forward_xres_npix_rms[k]->SetBinContent(i+1, rms);
                
              }

	    if ( sigma < 0.0 )
	      {
		sigma = rms;
		cout << "Bad error, check fit convergence !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
	      }
	    
            fprintf( datfile,
                     "%d %d %d %d %f %f \n",
                     4, 0, k, i, sigma, rms );
            
          }
      
      for (int i=0; i<2; ++i)
        {
          if ( !do_residuals )
            {
              h_forward_xres_npix[i]->SetMinimum(0.0);
              h_forward_xres_npix[i]->SetMinimum(2.0);
            }
          can_x_forward->cd(i+1);
          h_forward_xres_npix_rms[i]->Draw("");
          h_forward_xres_npix[i]->Draw("same");
        }
      
      if ( do_plots )
        {
          for (int i=0; i<2; ++i)
            {
              if ( do_residuals )
                sprintf(hname, "res_x_forward_sizex_%i.eps", i);
              else
                sprintf(hname, "pull_x_forward_sizex_%i.eps", i);

              can_x_forward_sizex[i]->SaveAs(hname);
            }

          if ( do_residuals )
            sprintf(hname, "res_x_forward.eps");
          else
            sprintf(hname, "pull_x_forward.eps");

          can_x_forward->SaveAs(hname);
        }

    } // if ( do_yf )

}
