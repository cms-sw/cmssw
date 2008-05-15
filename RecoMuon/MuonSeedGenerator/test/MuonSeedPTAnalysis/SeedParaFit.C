#include <vector>
#include <stdio.h>
#include <TFile.h>
#include <iostream>
#include <fstream>
#include <string>
// define the fitting function
Double_t fitf(Double_t *x, Double_t *par) {
         Double_t fitval =  par[0]
                         + (par[1]*x[0])
                         + (par[2]*x[0]*x[0]) 
                         + (par[3]*x[0]*x[0]*x[0]); 
         return fitval;
}

void SeedParaFit(int type, int s1, int b1, int b2, int np1){

  /*
   * Get the profile histogram for Pt*dphi vs eta for segment pair case
   * Slice each eta bin and do the gaussian fit
   * plot the mean and the sigma against the middle of the Y
   *
   * Author:  Shih-Chuan Kao  --- UCR
   *
   *  type  : seed type, 1: CSC(CSC), 2:Overlap(OL), 3:DT(DT), 4:DT_Single(SMB) 5:CSC_Sigle(SME)
   *  s1    : station combination ex: st1 and st2 => 12
   *  b1,b2 : range, in term of bin number
   *  np1   : number of fitting parameters p0 + p1*x + p2*x^2 + .... => np1 = 3 
   *
   */

 TFile *file = TFile::Open("para_mp5-200.root");

 float  nsigmas = 1.5;  
 TString suffixps = ".jpg";
 TString hfolder = "h_pt5to200";

 // name the file title by detector type
 char det[6];
 char dphi_case[20];
 TString dphi_type ;
 TString det_id ;
 if( type == 1 ) {

   sprintf(det, "CSC_%d", s1);
   det_id = det;
   dphi_type = det_id+"_eta_rdphiPt";
   if (s1 == 11) {
      sprintf(det,"CSC_0%d",1);
      det_id = det;
      dphi_type = det_id+"_eta_rdphiPt";
   }
   if (s1 == 12 && b1 > 58) {
      sprintf(det,"CSC_0%d",2);
      det_id = det;
      dphi_type = "CSC_12_eta_rdphiPt";
   }
   if (s1 == 13 && b1 > 58) {
      sprintf(det,"CSC_0%d",3);
      det_id = det;
      dphi_type = "CSC_13_eta_rdphiPt";
   } 
 }
 if( type == 2 ) { 
   sprintf(det, "OL_%d", s1);
   det_id = det;
   dphi_type = det_id+"_eta_rdphiPtA";
 }
 if( type == 3 ) { 
   sprintf(det, "DT_%d", s1);
   det_id = det;
   dphi_type = det_id+"_eta_rdphiPt";
 }
 if( type == 4 ) {
   sprintf(det, "SMB_%d", s1);
   det_id = det;
   dphi_type = det_id+"_eta_rdphiPt1";
 }
 if( type == 5 ) { 
   sprintf(det, "SME_%d", s1);
   det_id = det;
   dphi_type = det_id+"_eta_rdphiPt1";
 }

 cout <<" File name = "<<dphi_type<<endl;

 // Name the plots 
 TString plot01 = det_id+"_eta_pTxdphi"+suffixps;
 TString plot02 = det_id+"_eta_RelError"+suffixps;
 TString plot03 = det_id+"_eta_pTxdphi_scalar"+suffixps;
 TString plot04 = det_id+"_eta_pTxdphi_test"+suffixps;
 TString plot05 = det_id+"_eta_RelError_test"+suffixps;


// ********************************************************************
// Pointers to histograms
// ********************************************************************

 if (type ==1 ) {
    heta_dphiPt  = (TH2F *) file->Get("CSC_All/"+dphi_type);
 } 
 if (type ==2 ) {
    heta_dphiPt  = (TH2F *) file->Get("OL_All/"+dphi_type);
 }
 if (type ==3 ) {
    heta_dphiPt  = (TH2F *) file->Get("DT_All/"+dphi_type);
 }
 if (type ==4 ) {
    heta_dphiPt  = (TH2F *) file->Get("MB_All/"+dphi_type);
 }
 if (type ==5 ) {
    heta_dphiPt  = (TH2F *) file->Get("ME_All/"+dphi_type);
 }

// ********************************************************************
// create a folder to store all histograms
// ********************************************************************

  gSystem->mkdir(hfolder);
  gSystem->cd(hfolder);
  FILE *dfile = fopen("ptSeedParameterization.cfi","a");
  FILE *logfile = fopen("parafitting.log","a");

// *****************************************************************
// main program -- for pt : 5 ~ 200 GeV
// *****************************************************************


 // *****************************************
 // ***** 1 hdeta vs. Pt*dphi   Low eta *****
 // *****************************************

 Double_t r  = 0.;
 Double_t f1 = 0.;
 Double_t f2 = 0.;
 Double_t ini =0.;
 Double_t fnl =2.505;
 if (type == 1 || type == 5) {
    ini = 0.995;
    fnl = 2.505;
    r = 1.005 + (b1-1.0)*0.01;
    f1= 1.0   + (b1-1.0)*0.01;
    f2= 1.01  + (b2-1.0)*0.01;
 } 
 if (type ==3 || type ==4) {
    ini = -0.005;
    fnl = 1.105;
    r  = 0.005 + (b1-1.0)*0.01;
    f1 = 0.0   + (b1-1.0)*0.01;
    f2 = 0.01  + (b2-1.0)*0.01;
 }
 if (type ==2 ) {
    ini = 0.795;
    fnl = 1.805;
    r  = 0.805 + (b1-1.0)*0.01;
    f1 = 0.8   + (b1-1.0)*0.01;
    f2 = 0.81  + (b2-1.0)*0.01;
 }

 vector<double> yv1;
 vector<double> xv1;
 vector<double> yv1e;
 vector<double> xv1e;

 vector<double> rErr1v;

 yv1.clear();
 xv1.clear();
 yv1e.clear();
 xv1e.clear();
 rErr1v.clear();


 // ***************************************
 //   Main Loop for full eta range 
 // ***************************************  
 fprintf (logfile,"            ");
 fprintf (logfile,"p0        p1        p2        chi2        rms       nu \n");
 for (int i=b1; i<b2; i++) {

     heta_dphiPt->ProjectionY("heta_dphiPt_pjy",i,i);

     double amp = heta_dphiPt_pjy->Integral() / 2.0 ;
     double ave = heta_dphiPt_pjy->GetMean(1);
     double rms = heta_dphiPt_pjy->GetRMS(1);
     heta_dphiPt_pjy->Fit("gaus","N0RQ","",-1.5,1.5);
     gaus->SetParameters(amp,ave,rms);
     heta_dphiPt_pjy->Fit("gaus","N0RQ","",-1.5,1.5);
     double par0 = gaus->GetParameter(0);  // hieght
     double par1 = gaus->GetParameter(1);  // mean value
     double par2 = gaus->GetParameter(2);  // sigma
     double chi2 = gaus->GetChisquare();   // chi2
     double ndf  = gaus->GetNDF();         // ndf
     
     double L1 = par1 - (nsigmas * par2);
     double H1 = par1 + (nsigmas * par2);

     heta_dphiPt_pjy->Fit("gaus","N0RQ","",L1,H1);
     par0 = gaus->GetParameter(0);
     par1 = gaus->GetParameter(1);
     par2 = gaus->GetParameter(2);
     
     double  nu = heta_dphiPt_pjy->Integral();

     // exclude bad fitting!!
     //bool badfit1 =  ( par2 > (2.0*rms) ) ? true : false ;
     if (ndf ==0) {
         ndf =1.0 ;
     }
     bool badfit1 =  ( (chi2/ndf) > 4. ) ? true : false ;
     bool badfit2 =  ( par0 > amp ) ? true : false ;
     bool badfit3 =  ( fabs(par1 - ave) > rms ) ? true : false ;
 
     if ( badfit1 ) {
        fprintf (logfile," Bad1={ %-6.6f, %-6.6f, %-6.6f, %-6.6f, %-6.6f, %-6.6f} \n" 
                                  ,par0,par1,par2,chi2,rms,nu );
        par2 =0.0;
        par1 =0.0;
     }
     if ( badfit2 ) {
        fprintf (logfile," Bad2={ %-6.6f, %-6.6f, %-6.6f, %-6.6f, %-6.6f, %-6.6f} \n" 
                                  ,par0,par1,par2,chi2,rms,nu );
        par2 =0.0;
        par1 =0.0;
     }
     if ( badfit3 ) {
        fprintf (logfile," Bad3={ %-6.6f, %-6.6f, %-6.6f, %-6.6f, %-6.6f, %-6.6f} \n" 
                                  ,par0,par1,par2,chi2,rms,nu );
        par2 =0.0;
        par1 =0.0;
     }
     
     if ( (nu < 15.) || (fabs(par1) > 1.5) ) {
        fprintf (logfile," Bad4={ %-6.6f, %-6.6f, %-6.6f, %-6.6f, %-6.6f, %-6.6f} \n" 
                                  ,par0,par1,par2,chi2,rms,nu );
        par2 =0.0;
        par1 =0.0;
     }

     if ( (par1 != 0.0) && (par2 != 0.0) ){
        yv1.push_back(par1);
        yv1e.push_back(par2);
        xv1.push_back(r);
        xv1e.push_back(0.0);
        rErr1v.push_back(par2/par1);
     }
     r=r+0.01;

 }

 cout<<" sliced and fit done!!! "<<endl; 

 // *************************************************
 //   Fill the array for histogram from vector !!!
 // *************************************************

 const Int_t sz = xv1.size();

 Double_t ya1[sz]={0.0};
 Double_t xa1[sz]={0.0};
 Double_t ya1e[sz]={0.0};
 Double_t xa1e[sz]={0.0};
 Double_t rErr1a[sz]={0.0};
 
 for(int j = 0; j < sz ; j++) {
    ya1[j]=yv1[j];
    xa1[j]=xv1[j];
    ya1e[j]=yv1e[j];
    xa1e[j]=xv1e[j];
    rErr1a[j]=rErr1v[j];
 }
 
 // preliminary fitting - rejecting bad points

 gStyle->SetOptStat(kFALSE);
 gStyle->SetOptFit(0111);  
 c4 = new TCanvas("c4");
 c4->SetFillColor(10);
 c4->SetGrid(); 
 c4->GetFrame()->SetFillColor(21);
 c4->GetFrame()->SetBorderSize(12);
 c4->cd();

 //eta_dphiPt_pre = new TGraphErrors(sz,xa1,ya1,xa1e,ya1e);
 eta_dphiPt_pre = new TGraph(sz,xa1,ya1);
 eta_dphiPt_pre->SetTitle(" Test Fitting ");
 eta_dphiPt_pre->SetMarkerColor(4);
 eta_dphiPt_pre->SetMarkerStyle(21);
 eta_dphiPt_pre->GetXaxis()->SetTitle(" #eta  ");
 eta_dphiPt_pre->GetYaxis()->SetTitle(" pT*#Delta#phi  ");

 cout<<" parameterization !!!!"<<endl; 

 int np2 = np1 -1;
 if (np1 ==1 ) {
    np2 = np1;
 }
 TF1 *func0 = new TF1("fitf",fitf,f1,f2,np2);
 eta_dphiPt_pre->Fit("fitf","R","",f1,f2);

 double t0 = func0->GetParameter(0);
 double t1 = func0->GetParameter(1);
 double t2 = func0->GetParameter(2);

 eta_dphiPt_pre->Draw("AP");
 c4->Update();
 c4->Print(plot04);

 gStyle->SetOptStat(kFALSE);
 gStyle->SetOptFit(0111);  
 c5 = new TCanvas("c5");
 c5->SetFillColor(10);
 c5->SetGrid(); 
 c5->GetFrame()->SetFillColor(21);
 c5->GetFrame()->SetBorderSize(12);
 c5->cd();

 eta_rErr_pre = new TGraph(sz,xa1,rErr1a);
 eta_rErr_pre->SetTitle(" Test Fitting ");
 eta_rErr_pre ->SetMarkerColor(4);
 eta_rErr_pre ->SetMarkerStyle(21);
 eta_rErr_pre ->GetXaxis()->SetTitle(" #eta  ");
 eta_rErr_pre ->GetYaxis()->SetTitle(" #sigma/#mu of pT*#Delta#phi  ");

 TF1 *func1 = new TF1("fitf",fitf,f1,f2,np2);
 eta_rErr_pre->Fit("fitf","R","",f1,f2);
 double u0 = func1->GetParameter(0);
 double u1 = func1->GetParameter(1);
 double u2 = func1->GetParameter(2);

 eta_rErr_pre->Draw("AP");
 c5->Update();
 c5->Print(plot05);

 // calculate the sigma of normal distribution w.r.t. the fitting
 double dv1 = 0.0;
 double dv2 = 0.0;
 for(int j = 0; j < sz; j++) {
    double fity = t0 + (t1*xa1[j]) + (t2*xa1[j]*xa1[j]);
    dv1 += (ya1[j] - fity)*(ya1[j] - fity) ;
    double fite = u0 + (u1*xa1[j]) + (u2*xa1[j]*xa1[j]);
    dv2 += (rErr1a[j] - fite)*(rErr1a[j] - fite) ;
 }

 double sigma1 = sqrt( dv1/((sz*1.0)-1.0) );
 double sigma2 = sqrt( dv2/((sz*1.0)-1.0) );

 // Refitting in order to get better parameters
 // Chauvenet's Criterion to reject data for fitting
 vector<double> xv2;
 vector<double> yv2;
 vector<double> xv2e;
 vector<double> yv2e;
 vector<double> rErr2v;
 yv2.clear();
 xv2.clear();
 yv2e.clear();
 xv2e.clear();
 rErr2v.clear();

 xv2.push_back(ini);
 xv2e.push_back(0.0);
 yv2.push_back(0.0);
 yv2e.push_back(0.0); 
 rErr2v.push_back(0.0);

 for(int j = 0; j < sz; j++) {
    double fity = t0 + (t1*xa1[j]) + (t2*xa1[j]*xa1[j]);
    double width = fabs(ya1[j] - fity) ;

    double fite = u0 + (u1*xa1[j]) + (u2*xa1[j]*xa1[j]);
    double widthe = fabs(rErr1a[j] - fite) ;

    /// gaussian probability for y
    double p_gaus = 0.0;
    double k = 0.0;
    for (int i=0; i != 10000; i++ ) {
        k += (width*0.0001) ;
        double n1 = 1.0/ (sigma1*sqrt(2.0*3.14159)) ;
        double x2 = (-1.0*k*k)/(2.0*sigma1*sigma1) ;
        double gaus1 = n1*exp(x2);
        p_gaus += (gaus1*width*0.0001);
    }
    /// expected number outside the width of the distribution
    double nj = (1.0-(p_gaus*2.0))*(sz*1.0);

    /// gaussian probability for e
    double p_gaus2 = 0.0;
    double L = 0.0;
    for (int i=0; i != 10000; i++ ) {
        L += (widthe*0.0001) ;
        double n1 = 1.0/ (sigma2*sqrt(2.0*3.14159)) ;
        double x2 = (-1.0*L*L)/(2.0*sigma2*sigma2) ;
        double gaus2 = n1*exp(x2);
        p_gaus2 += (gaus2*widthe*0.0001);
    }
    double nj2 = (1.0-(p_gaus2*2.0))*(sz*1.0);

    if ( (nj > 0.99)&&(nj2 > 0.99) ) {
       xv2.push_back(xa1[j]); 
       yv2.push_back(ya1[j]);
       xv2e.push_back(xa1e[j]); 
       yv2e.push_back(ya1e[j]);
       rErr2v.push_back(rErr1a[j]);
    }else {
          fprintf (logfile," out=> %f, %f,  |  %f, %f } \n" 
                                 ,nj ,p_gaus*2., nj2, p_gaus2*2.  );
          cout <<j<<" n1= "<<nj<<" w= "<<width/sigma1<<" g= "<< p_gaus*2.0 ;
          cout <<" n2= "<<nj2<<" w= "<<widthe/sigma2<<" g= "<< p_gaus2*2.0 <<endl;
    }

 }

 xv2.push_back(fnl);
 xv2e.push_back(0.0);
 yv2.push_back(0.0);
 yv2e.push_back(0.0); 
 rErr2v.push_back(0.0);


 gStyle->SetOptStat(kTRUE);
 gStyle->SetOptFit(0111);  
 c1 = new TCanvas("c1");
 c1->SetFillColor(10);
 c1->SetGrid(); 
 c1->GetFrame()->SetFillColor(21);
 c1->GetFrame()->SetBorderSize(12);
 c1->cd();

 const Int_t sz2 = xv2.size();

 Double_t ya2[sz2]={0.0};
 Double_t xa2[sz2]={0.0};
 Double_t ya2e[sz2]={0.0};
 Double_t xa2e[sz2]={0.0};
 Double_t rErr2a[sz2]={0.0};

 for(int j = 0; j < xv2.size(); j++) {
    ya2[j]=yv2[j];
    xa2[j]=xv2[j];
    ya2e[j]=yv2e[j];
    xa2e[j]=xv2e[j];
    rErr2a[j]=rErr2v[j];
 }

 double qq[2] = {0.0,0.0} ;
 if ( (xv1.size() < 10) || (sz2 < 5) ) {
     heta_dphiPt->ProjectionY("heta_dphiPt_pjy",b1,b2);
     heta_dphiPt_pjy->Fit("gaus","N0RQ","",-1.5,1.5);
     double par0 = gaus->GetParameter(0);  // hieght
     double par1 = gaus->GetParameter(1);  // mean value
     double par2 = gaus->GetParameter(2);  // sigma
     double L1 = par1 - (nsigmas * par2);
     double H1 = par1 + (nsigmas * par2);
     heta_dphiPt_pjy->Fit("gaus","N0RQ","",L1,H1);
     par1 = gaus->GetParameter(1);  // mean value
     par2 = gaus->GetParameter(2);  // sigma
     qq[0] = par1 ;
     qq[1] = par2/par1 ;
     /*
     q1 = 0.0;
     q2 = 0.0;
     q3 = 0.0;
     k1 = 0.0;
     k2 = 0.0;
     k3 = 0.0;
     */
 }

 eta_dphiPt_prf = new TGraphErrors(sz2,xa2,ya2,xa2e,ya2e);
 eta_dphiPt_prf->SetTitle(det_id+" eta vs. pTdphi");
 eta_dphiPt_prf->SetMarkerColor(4);
 eta_dphiPt_prf->SetMarkerStyle(21);
 eta_dphiPt_prf->GetXaxis()->SetTitle(" #eta  ");
 eta_dphiPt_prf->GetYaxis()->SetTitle(" pT*#Delta#phi  ");

 TF1 *func = new TF1("fitf",fitf,f1,f2,np1);

 if ( (xv1.size() < 10) || (sz2 < 5) ) {
    func->FixParameter(0,qq[0]);
    func->SetParameter(1,0.);
    func->SetParameter(2,0.);
    //eta_dphiPt_prf->Draw("AP");
    //func->Draw("same");
 } 
 //else {
   eta_dphiPt_prf->Fit("fitf","R","",f1,f2);
   eta_dphiPt_prf->Draw("AP");
 //}

 //eta_dphiPt_prf->Print();
 double q0 = func->GetParameter(0);
 double q1 = func->GetParameter(1);
 double q2 = func->GetParameter(2);
 double q3 = func->GetParameter(3);

 c1->Update();
 c1->Print(plot01);

 // ***********************************
 // the relative error parameterizaton
 // ***********************************

 gStyle->SetOptStat(kTRUE);
 gStyle->SetOptFit(0111);  
 c2 = new TCanvas("c2");
 c2->SetFillColor(10);
 c2->SetGrid(); 
 c2->GetFrame()->SetFillColor(21);
 c2->GetFrame()->SetBorderSize(12);
 c2->cd();

 eta_rErr = new TGraph(sz2,xa2,rErr2a);
 eta_rErr ->SetTitle(det_id+" Relative Error vs eta");
 eta_rErr ->SetMarkerColor(4);
 eta_rErr ->SetMarkerStyle(21);
 eta_rErr ->GetXaxis()->SetTitle(" #eta  ");
 eta_rErr ->GetYaxis()->SetTitle(" #sigma/#mu of pT*#Delta#phi  ");

 TF1 *func2 = new TF1("fitf",fitf,f1,f2,np1);
 if ( (xv1.size() < 10) || (sz2 < 5) ) {
    func2->FixParameter(0,qq[1]);
    func2->SetParameter(1,0.);
    func2->SetParameter(2,0.);
    //eta_rErr->Draw("AP");
    //func2->Draw("same");
 } 
 //else {
   eta_rErr->Fit("fitf","R","",f1,f2);
   eta_rErr->Draw("AP");
 //}
 double k0 = func2->GetParameter(0);
 double k1 = func2->GetParameter(1);
 double k2 = func2->GetParameter(2);
 double k3 = func2->GetParameter(3);

 //eta_rErr->Print();
 c2->Update();
 c2->Print(plot02);


 //fprintf (dfile,"%d %f %f %f %f %f %f %f %f\n"
 //               ,s1,f1,f2,q0,q1,q2,k0,k1,k2);
 fprintf (dfile," vdouble %s = { %-6.6f, %-6.6f, %-6.6f, %-6.6f, %-6.6f, %-6.6f } \n"
                        , det   ,q0,q1,q2,k0,k1,k2);


// ********************************************************************
// Draw the origin scalar plots
// ********************************************************************

 gStyle->SetOptStat(kTRUE);
 TCanvas *c3 = new TCanvas("c3","");
 c3->SetFillColor(10);
 c3->SetFillColor(10);
 heta_dphiPt->SetTitle("pT x #Delta#phi 2D histogram");
 heta_dphiPt->Draw();
 heta_dphiPt->GetXaxis()->SetTitle(" #eta  ");
 heta_dphiPt->GetYaxis()->SetTitle("pT x #Delta#phi  ");
 c3->Update();
 c3->Print(plot03);

 fclose(dfile);
 cout<<" Finished  !! "<<endl;
 fclose(logfile);
 file->Close();
 gROOT->Reset();

 //gROOT->ProcessLine(".q");
 
}
