#include "SeedParaFit.h"

SeedParaFit::SeedParaFit() {
  /*
   * Get the profile histogram for Pt*dphi vs eta for segment pair case
   * Slice each eta bin and do the gaussian fit
   * plot the mean and the sigma against the middle of the Y
   *
   * Author:  Shih-Chuan Kao  --- UCR
   *
   */

  // Name of root file
  fName = "mix31x_v3.root";
  // folder for all plots
  dname = "Pt31x_v3";
  // plot formate
  suffixps = ".gif";
  // fitting range
  nsigmas = 1.5;
  // debug for each slice and fitting => fitting for each point
  debug = false;  
  // show the pre-fitting of PtxdPhi and Rel_Error
  debugAll = false ;  
  // bin size for eta, unit bin size is 0.01
  // bin range => CSC: 0.9 - 2.7 ,  DT: 0.0 - 1.2 , OL: 0.7 - 1.3
  // # of bin        : 180            : 120           : 60
  // possible rbin value = 1, 2, 3, 4, 5, 10 ;
  rbin = 3 ;
  bsz = 0.01*rbin ;
 
  ptFunc = new SeedPtFunction();

  gSystem->mkdir(dname);
  dfile   = fopen(dname+"/ptSeedParameterization_cfi.py","a");
  if (debug) logfile = fopen(dname+"/parafitting.log","a");


}

SeedParaFit::~SeedParaFit() {

  if ( debug ) delete cv;
  delete c1;
  delete c2;
  delete c3;
  //delete c4;

  delete ptFunc ;

  fclose(dfile);
  if (debug) fclose(logfile);

}

void SeedParaFit::ParaFit(int type, int s1, double h1, double h2, int np1){
//void SeedParaFit::ParaFit(int type, int s1, int b1, int b2, int np1){
 
  /*
   *  type  : seed type, 1: CSC(CSC), 2:Overlap(OL), 3:DT(DT), 4:DT_Single(SMB) 5:CSC_Sigle(SME)
   *  s1    : station combination ex: st1 and st2 => 12
   *  h1,h2 : eta range
   *  np1   : number of fitting parameters p0 + p1*x + p2*x^2 + .... => np1 = 3 
   *
   */

  // name the file title by detector type
  if( type == 1 ) {
    sprintf(det, "CSC_%d", s1);
    det_id = det;
    dphi_type = det_id+"_eta_rdphiPt";
    if (s1 == 11) {
       sprintf(det,"CSC_0%d",1);
       det_id = det;
       dphi_type = det_id+"_eta_rdphiPt";
    }
    if (s1 == 12 && h1 > 1.7) {
       sprintf(det,"CSC_0%d",2);
       det_id = det;
       dphi_type = "CSC_12_eta_rdphiPt";
    }
    if (s1 == 13 && h1 > 1.75) {
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
    dphi_type = det_id+"_eta_rdphiPt";
  }
  if( type == 5 ) { 
    sprintf(det, "SME_%d", s1);
    det_id = det;
    dphi_type = det_id+"_eta_rdphiPt";
  }

  // Name the plots 
  plot01 = dname+"/"+det_id+"_eta_pTxdphi"+suffixps;
  plot02 = dname+"/"+det_id+"_eta_RelError"+suffixps;
  plot03 = dname+"/"+det_id+"_eta_pTxdphi_scalar"+suffixps;
  plot04 = dname+"/"+det_id+"_prefit_test"+suffixps;

// ********************************************************************
// Pointers to histograms
// ********************************************************************
  
  file = TFile::Open(fName);
  if (type ==1 )  heta_dphiPt  = (TH2F *) file->Get("CSC_All/"+dphi_type);
  if (type ==2 )  heta_dphiPt  = (TH2F *) file->Get("OL_All/"+dphi_type);
  if (type ==3 )  heta_dphiPt  = (TH2F *) file->Get("DT_All/"+dphi_type);
  if (type ==4 )  heta_dphiPt  = (TH2F *) file->Get("MB_All/"+dphi_type);
  if (type ==5 )  heta_dphiPt  = (TH2F *) file->Get("ME_All/"+dphi_type);
  heta_dphiPt->RebinX(rbin); 


// ********************************************************************
// create a folder to store all histograms
// ********************************************************************

  //gSystem->cd(dname);

  if ( debug ) gSystem->mkdir("BugInfo");

 // *****************************************
 // ***** 1 hdeta vs. Pt*dphi   Low eta *****
 // *****************************************

 Double_t r  = 0.;
 Double_t ini =0.;
 Double_t fnl =2.5;
 Double_t f1 = 0.;
 Double_t f2 = 0.;
 int b1 = 0;
 int b2 = 0;

 if (type == 1 || type == 5) {
    b1 = (h1 - 0.9)/bsz ;
    b2 = (h2 - 0.9)/bsz ;
    ini = 0.9 - bsz/2.;
    fnl = 2.5 + bsz/2. ;
    r = 0.9   + (b1-1.0)*bsz + (bsz/2.);
    f1= 0.9   + (b1-1.0)*bsz;
    f2= 0.9   + (b2*bsz);
 } 
 if (type ==3 || type ==4) {
    b1 = (h1 - 0.)/bsz ;
    b2 = (h2 - 0.)/bsz ;
    if (b1 == 0) b1 = 1 ;
    ini = 0.  - bsz/2. ;
    fnl = 1.2 + bsz/2;
    r  = 0.0   + (b1-1.0)*bsz + (bsz/2.);
    f1 = 0.0   + (b1-1.0)*bsz;
    f2 = 0.0   + (b2*bsz);
 }
 if (type ==2 ) {
    b1 = (h1 - 0.7)/bsz ;
    b2 = (h2 - 0.7)/bsz ;
    ini = 0.7 - bsz/2.;
    fnl = 1.3 + bsz/2.;
    r  = 0.7   + (b1-1.0)*bsz + (bsz/2.);
    f1 = 0.7   + (b1-1.0)*bsz;
    f2 = 0.7   + (b2*bsz);
    cout<<" b1:"<<b1<<"   b2:"<<b2 <<endl;;
    cout<<" f1:"<<f1<<"  f2:"<<f2 <<endl;
 }

 vector<double> yv1;
 vector<double> xv1;
 vector<double> yv1e;

 yv1.clear();
 xv1.clear();
 yv1e.clear();

 // ***************************************
 //   Main Loop for full eta range 
 // ***************************************  
 if (debug) fprintf (logfile,"            ");
 if (debug) fprintf (logfile,"DetId     p0        p1        p2        chi2        rms       nu \n");
 double par0 = 0.;
 double par1 = 0.;
 double par2 = 0.;
 double chi2 = 0.;
 double  ndf = 0.;

 TF1 *gausf = new TF1("gausf", SeedPtFunction::fgaus ,-2., 2., 3  );

 if (debug) {
    cv = new TCanvas("cv");
    gStyle->SetOptStat(kTRUE);
    gStyle->SetOptFit(0111);  
 }

 for (int i=b1; i<b2; i++) {

     if ( debug ) {
        sprintf(dbplot, "dbug_%d.gif",i);
        plot_id = dbplot;
     }
 
     TH1D* heta_dphiPt_pjy = heta_dphiPt->ProjectionY("heta_dphiPt_pjy",i,i);
     if ( debug )  heta_dphiPt_pjy->Draw();
     double  nu = heta_dphiPt_pjy->Integral();
     double ave = heta_dphiPt_pjy->GetMean();
     double rms = heta_dphiPt_pjy->GetRMS();

     double L1 = ave - (3. * rms);
     double H1 = ave + (3. * rms);

     //cout<<" fk1 nu:"<< nu <<" ave:"<< ave <<"  rms:"<< rms <<endl;
     if ( nu >= 10  ) {
        //gausf->SetParameter(0, amp );
        gausf->SetParameter(1, ave );
        gausf->SetParameter(2, rms );
        gausf->SetParLimits(1, L1, H1 );
        gausf->SetParLimits(2, 0., 5. );
        if (!debug) heta_dphiPt_pjy->Fit("gausf", "N0RQL", "", L1, H1 );
        if ( debug) {
                    gausf->SetLineStyle(2);
                    gausf->SetLineColor(2);
                    heta_dphiPt_pjy->Fit("gausf", "RQL", "sames", L1, H1 );
                    cv->Update();
                    cv->Print("BugInfo/"+plot_id);
        }
	par0 = gausf->GetParameter(0);  // hieght
        if ( par0 < 1. || par0 > nu*0.8 ) heta_dphiPt_pjy->Fit("gausf","N0Q" );
	par1 = gausf->GetParameter(1);  // mean value
	par2 = gausf->GetParameter(2);  // sigma

        L1 = par1 - (nsigmas * par2);
        H1 = par1 + (nsigmas * par2);

        //gausf->SetParameter(0, par0 );
        gausf->SetParameter(1, par1 );
        gausf->SetParameter(2, par2 );
	heta_dphiPt_pjy->Fit("gausf","N0RQ","", L1, H1);
        if ( debug) {
                    gausf->SetLineStyle(1);
                    gausf->SetLineColor(4);
                    gausf->Draw("sames");
        }
	par0 = gausf->GetParameter(0);  // hieght
	par1 = gausf->GetParameter(1);  // mean value
	par2 = gausf->GetParameter(2);  // sigma
	chi2 = gausf->GetChisquare();   // chi2
	ndf  = gausf->GetNDF();         // ndf
     }
    
     if (debug) {
        cv->Update();
        cv->Print("BugInfo/"+plot_id);
     }
     // exclude bad fitting!!
     if (ndf == 0)  ndf = 1.0 ;
     bool badfit1 =  ( (chi2/ndf) > 10. ) ? true : false ;
     bool badfit2 =  ( fabs(par1 - ave) > rms ) ? true : false ;
 
     if ( badfit1 ) {
        if (debug) fprintf (logfile," Bad1={ %s, %-6.6f, %-6.6f, %-6.6f, %-6.6f, %-6.6f, %-6.6f} \n" 
                                ,det,   par0,   par1,   par2,   chi2,    rms,    nu );
        par2 =0.0;
        par1 =0.0;
     }
     if ( badfit2 ) {
        if (debug) fprintf (logfile," Bad2={ %s, %-6.6f, %-6.6f, %-6.6f, %-6.6f, %-6.6f, %-6.6f} \n" 
                                ,det,   par0,   par1,   par2,   chi2,    rms,    nu );
        par2 =0.0;
        par1 =0.0;
     }
     
     if ( (nu < 15.) || (fabs(par1) > 1.5) ) {
        if (debug) fprintf (logfile," Bad3={ %s, %-6.6f, %-6.6f, %-6.6f, %-6.6f, %-6.6f, %-6.6f} \n" 
                                ,det,   par0,   par1,   par2,   chi2,    rms,    nu );
        par2 =0.0;
        par1 =0.0;
     }

     if ( (par1 != 0.0) && (par2 > 0.0001) ){
        if ( (par2/par1) < 100. && (par2/par1) > 0.01) {
           yv1.push_back(par1);
	   yv1e.push_back(par2);
	   xv1.push_back(r);
        }
     }
     r = r + bsz;

     delete heta_dphiPt_pjy;
 }
 cout<<" sliced and fit done!!! "<<endl; 

 // *************************************************
 //   Fill the array for histogram from vector !!!
 // *************************************************

 const Int_t sz = xv1.size();
 Double_t ya1[sz], xa1[sz], ya1e[sz] ;

 for(int j = 0; j < sz ; j++) {
    ya1[j]=yv1[j];
    xa1[j]=xv1[j];
    ya1e[j]=yv1e[j];
 }
 
 // preliminary fitting - rejecting bad points
 // Chauvenet's Criterion to reject data from fitting
 int np2 = ( np1 == 1) ? np1 : np1 -1 ;
 TF1 *func0 = new TF1("func0", SeedPtFunction::linear, f1, f2, np2 );
 vector<int> goodYs = preFitter( func0, "func0", f1, f2, sz, xa1, ya1 );
 const Int_t sz2 = ( sz >=3 && goodYs.size() > 1 ) ? goodYs.size() + 2  : sz+2; 
 cout<<" sz2 size "<<sz2<<endl; 
 Double_t ya2[sz2] , xa2[sz2], ya2e[sz2],  xa2e[sz2] ;

 xa2[0]= ini;
 ya2[0]= 0.;
 ya2e[0]= 0.;
 for(size_t j = 0; j < goodYs.size() ; j++) {
    int k = goodYs[j] ;
    xa2[j+1]  = xa1[k]  ; 
    ya2[j+1]  = ya1[k]  ;
    ya2e[j+1] = ya1e[k] ;
    xa2e[j+1] = 0.0 ; 
 }
 xa2[sz2-1] = fnl;
 ya2[sz2-1]  = 0.;
 ya2e[sz2-1] = 0.;
 xa2e[sz2-1] = 0.; 

 if ( sz2 < 5 ) {
    for (int j=1; j< sz2-1; j++ ) {  
        xa2[j]  = xa1[j-1]  ; 
        ya2[j]  = ya1[j-1]  ;
        ya2e[j] = ya1e[j-1] ;
        xa2e[j] = 0.0 ; 
    }
 }
 

 gStyle->SetOptStat(kTRUE);
 gStyle->SetOptFit(0111);  
 c1 = new TCanvas("c1");
 c1->SetFillColor(10);
 c1->SetGrid(); 
 c1->GetFrame()->SetFillColor(21);
 c1->GetFrame()->SetBorderSize(12);
 c1->cd();

 double qq[2] = { 0., 0. } ;
 if ( sz2 < 5 ) {
     for (int j=1; j< sz2-1; j++ ) { 
         qq[0] += ya2[j];
         qq[1] += ya2e[j]*ya2e[j] ;
     }
     qq[0] = qq[0]/(sz2-2) ;
     //qq[1] = sqrt( qq[1] ) /qq[0] ;
     qq[1] = sqrt( qq[1] /(sz2-2) ) ;
 }

 TGraphErrors* eta_dphiPt_prf = new TGraphErrors(sz2,xa2,ya2,xa2e,ya2e);
 eta_dphiPt_prf->SetTitle(det_id+"  #eta vs. pT*#Delta#phi");
 eta_dphiPt_prf->SetMarkerColor(4);
 eta_dphiPt_prf->SetMarkerStyle(21);
 eta_dphiPt_prf->GetXaxis()->SetTitle(" #eta  ");
 eta_dphiPt_prf->GetYaxis()->SetTitle(" pT*#Delta#phi  ");

 TF1 *func = new TF1("func", SeedPtFunction::linear, f1, f2, np1 );

 if ( sz2 < 5 ) {
    func->FixParameter(0,qq[0]);
    func->FixParameter(1, 0.);
    func->FixParameter(2, 0.);
 } 
 eta_dphiPt_prf->Draw("AP");
 eta_dphiPt_prf->Fit("func","RQ","sames",f1,f2);

 double q0 = func->GetParameter(0);
 double q1 = func->GetParameter(1);
 double q2 = func->GetParameter(2);

 c1->Update();
 c1->Print(plot01);

 // ***********************************
 // the relative error parameterizaton
 // ***********************************

 Double_t rErr2a[sz2] ;
 for ( int i=0; i< sz2; i++ ) {
     //rErr2a[i] = ya2e[i] / func->Eval( xa2[i] ) ;
     rErr2a[i] = ya2e[i] ;
 }

 int np3 = ( np2 == 1) ? np2 : np2 -1 ;
 TF1 *func1 = new TF1("func1", SeedPtFunction::linear, f1, f2, np3 );
 vector<int> goodErrs = preFitter( func1, "func1", f1, f2, sz2, xa2, rErr2a );

 const Int_t sz3 = ( sz2 >= 5 && goodErrs.size() > 1 ) ? goodErrs.size() + 2 : sz2;
 Double_t rErr3a[sz3] ;
 Double_t xa3[sz3]    ;

 xa3[0] = ini ;
 rErr3a[0] = 0.0 ;
 for (int j=1; j< sz3-1 ; j++ ) {  
     int k = goodErrs[j-1] ;
     xa3[j]    = xa2[k] ;
     rErr3a[j] = rErr2a[k] ;
 }
 xa3[sz3-1] = fnl ;
 rErr3a[sz3-1] = 0.0 ;

 if ( sz3 < 5 ) {
    for (int j=1; j< sz2-1; j++ ) {  
        xa3[j]    = xa2[j]  ; 
        rErr3a[j] = rErr2a[j] ;
    }
 }

 // get better fitting 
 gStyle->SetOptStat(kTRUE);
 gStyle->SetOptFit(0111);  
 c2 = new TCanvas("c2");
 c2->SetFillColor(10);
 c2->SetGrid(); 
 c2->GetFrame()->SetFillColor(21);
 c2->GetFrame()->SetBorderSize(12);
 c2->cd();

 TGraph* eta_rErr = new TGraph(sz3,xa3,rErr3a);
 eta_rErr ->SetTitle(det_id+" Relative Error vs eta");
 eta_rErr ->SetMarkerColor(4);
 eta_rErr ->SetMarkerStyle(21);
 eta_rErr ->GetXaxis()->SetTitle(" #eta  ");
 eta_rErr ->GetYaxis()->SetTitle(" #sigma of pT*#Delta#phi  ");

 int np4 = ( b2-b1 < 4) ? 1 : np2 ;
 TF1 *func2 = new TF1("func2", SeedPtFunction::linear, f1, f2, np4 );
 eta_rErr->Draw("AP");
 eta_rErr->Fit("func2","RQ","sames",f1,f2);
 
 if ( sz2 < 5 ) {
    func2->FixParameter(0,qq[1]);
    func2->FixParameter(1, 0.);
    func2->FixParameter(2, 0.);
 } 
 
 double k0 = func2->GetParameter(0);
 double k1 = func2->GetParameter(1);
 double k2 = func2->GetParameter(2);

 c2->Update();
 c2->Print(plot02);

 fprintf (dfile," %s = cms.vdouble( %-6.3f, %-6.3f, %-6.3f, %-6.3f, %-6.3f, %-6.3f ), \n",
                 det,                   q0,     q1,     q2,     k0,     k1,     k2 );


// ********************************************************************
// Draw the origin scalar plots
// ********************************************************************

 gStyle->SetOptStat("ne");
 //gStyle->SetPalette(10);
 TCanvas *c3 = new TCanvas("c3","");
 c3->SetFillColor(10);
 c3->SetFillColor(10);
 heta_dphiPt->SetTitle("pT x #Delta#phi 2D histogram");
 heta_dphiPt->RebinY(2);
 heta_dphiPt->Draw("COLZ");
 heta_dphiPt->GetXaxis()->SetTitle(" #eta  ");
 heta_dphiPt->GetYaxis()->SetTitle("pT x #Delta#phi  ");
 c3->Update();
 c3->Print(plot03);

  cout<<" Finished  !! "<<endl;

  file->Close();

  delete c1;
  delete c2;
  delete c3;
  delete gausf;
  delete func;
  delete func0;
  delete func1;
  delete func2;
  delete eta_dphiPt_prf;
  delete eta_rErr;

}

vector<int> SeedParaFit::preFitter( TF1* ffunc, TString falias, double h1, double h2, int asize, Double_t* xa, Double_t* ya ){

    if ( debugAll ) {
       gStyle->SetOptStat(kFALSE);
       gStyle->SetOptFit(0111);  
       c4 = new TCanvas("c4");
       c4->SetFillColor(10);
       c4->SetGrid(); 
       c4->GetFrame()->SetFillColor(21);
       c4->GetFrame()->SetBorderSize(12);
       c4->cd();
    }

    vector<int> goodone;
    goodone.clear();
    if ( asize < 3 ) return goodone ;

    TGraph* preh = new TGraph(asize,xa,ya);
    preh->SetTitle(" Test Fitting ");
    preh->SetMarkerColor(4);
    preh->SetMarkerStyle(21);
    preh->GetXaxis()->SetTitle(" #eta  ");

    preh->Fit( falias ,"N0RQ","",h1,h2 );

    // get the sigma of the pre-fitting
    double dv = 0.0;
    for (int j = 0; j < asize; j++) {
       double fite = ffunc->Eval( xa[j] ) ;
       dv += ( ya[j] - fite )*( ya[j] - fite ) ;
    }
    double sigma = sqrt( dv/( asize - 1.) );

    for (int j = 0; j < asize; j++) {

        double fitV = ffunc->Eval( xa[j] ) ;
        double width = fabs( ya[j] - fitV ) ;

        /// gaussian probability for number of data expected
        bool reject = ptFunc->DataRejection( sigma , width, asize );

        if ( !reject ) {
           goodone.push_back( j ); 
        }
    }

    if ( debugAll ) {
       preh->Draw("AP");
       ffunc->Draw("sames");
       c4->Update();
       c4->Print(plot04);
    }
  
    delete preh;
    if ( debugAll ) delete c4;

    return goodone;
}


void SeedParaFit::PrintTitle(){

  fprintf (dfile,"import FWCore.ParameterSet.Config as cms \n");
  fprintf (dfile,"ptSeedParameterization = cms.PSet( \n");
  fprintf (dfile," \n");
}

void SeedParaFit::PrintEnd(){

  fprintf (dfile," \n");
  fprintf (dfile,") ");
}
