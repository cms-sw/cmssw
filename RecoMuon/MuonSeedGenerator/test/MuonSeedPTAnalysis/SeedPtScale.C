#include "SeedPtScale.h"
//ClassImp(SeedPtScale);

SeedPtScale::SeedPtScale() {

  dname = "SeedPtScale3" ;
  gSystem->mkdir(dname);
  suffixps = ".gif";

  debug = false;
  // bin size for eta, unit bin size is 0.01
  // bin range => CSC: 0.9 - 2.7 ,  DT: 0.0 - 1.2 , OL: 0.7 - 1.3
  // # of bin        : 180            : 120           : 60
  // possible rbin value = 1, 2, 3, 4, 5, 10 ;
  xbsz = 0.01 ;
  //ybsz = 0.0001;

}

SeedPtScale::~SeedPtScale() {

  delete c1;
  delete c2;
  delete c6;
  if (debug) {
     delete c3;
     delete c4;
     delete c5;
     delete cv;
  }

  delete plot01;
  delete plot02;
  delete plot06;
  if (debug) {
     delete plot03;
     delete plot04;
     delete plot05;
  }

}

void SeedPtScale::PtScale( int type, int st, double h1, double h2, int idx, int np){
//void SeedPtScale::PtScale( int type, int st, int b1, int b2, int idx, int np){

  /*
   * Get the profile histogram for Pt*dphi vs pt or 1/phi 
   *
   * Author:  Shih-Chuan Kao  --- UCR
   *
   * type = 1:CSC,  2:Overlap,  3:DT  4:DTSingle   5:CSCSingle
   * st   = 12 => station1 + station2 ; for type 4 & 5 => 12:ME12 or MB12 
   * h1,h2= eta range
   * b1   = lst bin for lower fitting bound
   * b2   = 2nd bin for upper fitting bound
   * idx  = index for different eta range
   *
   */

  int b1 = 0;
  int b2 = 0;
  if (type == 1 || type == 5) {
     b1 = (h1 - 0.9)/xbsz ;
     b2 = (h2 - 0.9)/xbsz ;
  }
  if (type ==3 || type == 4) {
     b1 = (h1 - 0.)/xbsz ;
     b2 = (h2 - 0.)/xbsz ;
     if (b1 == 0) b1 = 1 ;
  }
  if (type ==2 ) {
     b1 = (h1 - 0.7)/xbsz ;
     b2 = (h2 - 0.7)/xbsz ;
  }
  cout<<" Bin1:"<< b1 <<"  Bin2:"<<b2 <<endl;

// ********************************************************************
// create a folder to store all histograms
// ********************************************************************

  float nsigmas = 1.5;
  char det[8];
  char plotname[9];

  if (type == 1)   { 
     if (st == 01) {
       sprintf(det,"CSC_01");
       sprintf(plotname,"CSC_01_%d",idx);
     } else {
       sprintf(det,"CSC_%d",st);
       sprintf(plotname,"CSC_%d_%d",st,idx);
     }
     Dir = "CSC_All";
     det_id = det;
  }
  if (type == 2)   {
     sprintf(det,"OL_%d",st);
     sprintf(plotname,"OL_%d",st);
     Dir = "OL_All";
     det_id = det;     
  }
  if (type == 3)   {
     sprintf(det,"DT_%d",st);
     sprintf(plotname,"DT_%d_%d",st,idx);
     Dir = "DT_All";
     det_id = det;     
  }
  if (type == 4)   {
     sprintf(det,"SMB_%d",st);
     sprintf(plotname,"SMB_%d",st);
     Dir = "MB_All";
     det_id = det;     
  }
  if (type == 5)   {
     sprintf(det,"SME_%d",st);
     sprintf(plotname,"SME_%d",st);
     Dir = "ME_All";
     det_id = det;     
  }

  pname = plotname;
 
  cout<<" Set Path" <<endl;
  TString thePath  = Dir+"/"+det_id+"_eta_rdphiPt";
  TString thePath1 = Dir+"/"+det_id+"_eta_rdphi";
  if (type ==2 ) {
      thePath  = Dir+"/"+det_id+"_eta_rdphiPtA";
      thePath1 = Dir+"/"+det_id+"_eta_rdphiA";
  }
  
  plot01 = pname+"_pt_ptxdPhi"+suffixps;
  plot02 = pname+"_pt_odphi"+suffixps;
  plot03 = pname+"_pt_dphi"+suffixps;
  plot04 = pname+"_corr"+suffixps;
  plot05 = pname+"_normal"+suffixps;
  plot06 = pname+"_odphi_ptxdPhi"+suffixps;

  int ptarr[] ={10,20,50,100,150,200,350,500};
  vector<int> ptlist( ptarr, ptarr+8 );
  /*
  vector<int> ptlist;
  for (int ii = 0; ii < 7; ii++) {
      ptlist.push_back( ptarr[ii] );
  }*/
  const Int_t sz = ptlist.size();

  if ( debug ) gSystem->mkdir("BugInfo");

// *****************************************************************
// main program -- looping pt from 10 -> 1000
// *****************************************************************
  
   float L1,H1 = 0.0;
   float mean,rms,ent =0.0;
   bool fitfail = false ;

   Double_t mean1[sz], sigma1[sz] ;
   Double_t pt[sz],     ptErr[sz] ;
   Double_t dphi[sz], dphiErr[sz] ;
   Double_t dphi_real[sz], dphiErr_real[sz] ;
   Double_t ophi[sz], ophiErr[sz] ;
   Double_t ratio[sz],   corr[sz] ;   

   //SeedPtFunction* fptr = new SeedPtFunction() ; 
   //TF1 *gausf = new TF1("gausf", SeedFitFunc, &SeedPtFunction::fgaus, -1., 3., 3, "SeedPtFunction", "fgaus" );
   TF1 *gausf = new TF1("gausf", SeedPtFunction::fgaus, -1., 3., 3  );

   if (debug) {
      cv = new TCanvas("cv");
      gStyle->SetOptStat(kTRUE);
      gStyle->SetOptFit(0111);
   }

   for ( int i=0; i<sz; i++) {

       if ( debug ) {
          sprintf(dbplot, "dbug_%d.gif",i);
          plot_id = dbplot;
          cv->cd();
       }

       char filename[18];
       sprintf(filename,"para%d.root",ptlist[i]);
       TString fname = filename;
       pt[i] = ptlist[i] ;

       // pt*dphi 
       TFile *file      = TFile::Open(fname);
       TH2F* hdphiPt    = (TH2F *) file->Get(thePath);
       TH1D* hdphiPt_py = hdphiPt->ProjectionY("hdphiPt_py",b1,b2);
       if ( debug ) hdphiPt_py->Draw();

       mean = hdphiPt_py->GetMean();
       rms  = hdphiPt_py->GetRMS();
       ent  = hdphiPt_py->GetEntries();
       L1 = mean - (1.*rms);
       H1 = mean + (1.*rms);
       fitfail = false ;
       
       gausf->SetParLimits(0, 1., 0.8*ent);
       gausf->SetParLimits(1, mean-rms, mean+rms);
       gausf->SetParLimits(2, 0.2*rms, 2.*rms);
       gausf->SetParameter(1,    mean);
       gausf->SetParameter(2,     rms);

       if ( !debug ) hdphiPt_py->Fit( "gausf" ,"N0RQC","", L1, H1 );
       if (  debug ) deBugger(gausf, "gausf", hdphiPt_py, L1, H1 , 2 );
       fitfail = BadFitting(gausf, hdphiPt_py) ;
       if ( fitfail ) hdphiPt_py->Fit("gausf","N0Q" );
       mean1[i]  = gausf->GetParameter(1); 
       sigma1[i] = gausf->GetParameter(2);
       L1 = mean1[i] - (nsigmas * sigma1[i]);
       H1 = mean1[i] + (nsigmas * sigma1[i]);

       if ( !debug ) hdphiPt_py->Fit( "gausf" ,"N0RQC","",L1, H1);
       if (  debug ) deBugger(gausf, "gausf", hdphiPt_py, L1, H1 , 4 );
       mean1[i]  = gausf->GetParameter(1);
       sigma1[i] = gausf->GetParameter(2);

       fitfail = false;

       // dphi information , no fit just use the mean and rms
       TH2F* hdphi   = (TH2F *) file->Get(thePath1);
       TH1D* dphi_py = hdphi->ProjectionY("dphi_py",b1,b2);

       // calculated dphi
       dphi[i]    = mean1[i]/pt[i];
       dphiErr[i] = dphi_py->GetRMS();
       // real dphi
       dphi_real[i]    = 1000*dphi_py->GetMean();
       dphiErr_real[i] = 1000*dphi_py->GetRMS();
       ophi[i]    = 1./dphi[i] ;
       if ( i > 0 ) {
          if ( ophi[i] < ophi[i-1] ) {
              dphi[i] = mean1[i-1] / pt[i] ;
              ophi[i]    = 1./dphi[i] ;
          }  
       }
       ophiErr[i] = dphiErr[i] / (dphi[i]*dphi[i]);

       cout<<" pt:"<< pt[i] <<" mean = "<< mean1[i] <<" df :"<< dphi[i] <<endl;

       delete hdphiPt ;
       delete hdphiPt_py ;
       delete hdphi ;
       delete dphi_py ;
       file->Close(); 
   }

 // *************************************************
 //   Fill the array for histogram from vector !!!
 // *************************************************

 gSystem->cd(dname);
 FILE *dfile = fopen("MuonSeedPtScale.py","a");
 FILE *dfile2 = fopen("MuonSeeddPhiScale.py","a");
 
 // 1. pt vs pt*dphi
 gStyle->SetOptStat(kFALSE);
 gStyle->SetOptFit(0111);  
 c1 = new TCanvas("c1");
 c1->SetFillColor(10);
 c1->SetGrid(); 
 c1->GetFrame()->SetFillColor(21);
 c1->GetFrame()->SetBorderSize(12);
 c1->cd();

 TGraphErrors* pt_dphiPt = new TGraphErrors(sz,pt,mean1,ptErr,sigma1);
 pt_dphiPt->SetMarkerColor(4);
 pt_dphiPt->SetMarkerStyle(21);
 pt_dphiPt->SetTitle(plot01);
 pt_dphiPt->GetXaxis()->SetTitle(" pt ");
 pt_dphiPt->GetYaxis()->SetTitle(" pT*#Delta#phi  ");

 TF1 *func0 = new TF1("func0", SeedPtFunction::fitf, 5, 1100, np );
 pt_dphiPt->Fit( func0 ,"R","",5,1100);

 double q0 = func0->GetParameter(0);
 double q1 = func0->GetParameter(1);
 double q2 = func0->GetParameter(2);
 double t1 = q1/q0;
 double t2 = q2/q0;

 //fprintf (dfile,"  %s_%d_scale = cms.vdouble( %f, %f, %f, %f, %f ), \n"
 //                  ,det ,idx                 ,q0, q1, q2, t1, t2 );
 fprintf (dfile,"  %s_%d_scale = cms.vdouble(  %f, %f ), \n",
                   det, idx,                   t1, t2 );
 
 pt_dphiPt->Draw("AP");
 c1->Update();
 c1->Print(plot01);

 // get the ratio and correction  
 for (int j=0; j<sz; j++) {
     double theX = 1./ (10. + ophi[j]) ;
     ratio[j] = 1. + (t1 * theX) + ( t2 * theX * theX ) ;
     corr[j] = 1./ratio[j] ;
 }	 
 			     

 if ( debug ) {
    // 2. pt vs 1/dphi
    gStyle->SetOptStat(kFALSE);
    gStyle->SetOptFit(0111);  
    c2 = new TCanvas("c2");
    c2->SetFillColor(10);
    c2->SetGrid(); 
    c2->GetFrame()->SetFillColor(21);
    c2->GetFrame()->SetBorderSize(12);
    c2->cd();

    TGraph* ophi_Pt = new TGraph(sz, pt, ophi);
    ophi_Pt->SetMarkerColor(4);
    ophi_Pt->SetMarkerStyle(21);
    ophi_Pt->SetTitle(plot02);
    ophi_Pt->GetXaxis()->SetTitle(" pt ");
    ophi_Pt->GetYaxis()->SetTitle(" 1/*#Delta#phi  ");

    TF1 *func2 = new TF1("func2", SeedPtFunction::linear, 5, 1000, 2 );
    ophi_Pt->Fit( func2 ,"R","",0,1000);

    ophi_Pt->Draw("AP");
    c2->Update();
    c2->Print(plot02);


    // 4. pt vs correction factor
    gStyle->SetOptStat(kFALSE);
    gStyle->SetOptFit(0111);  
    c4 = new TCanvas("c4");
    c4->SetFillColor(10);
    c4->SetGrid(); 
    c4->GetFrame()->SetFillColor(21);
    c4->GetFrame()->SetBorderSize(12);
    c4->cd();

    //TGraph* cor_ophi = new TGraph(sz,ophi,corr);
    TGraph* cor_pt = new TGraph(sz,pt,corr);
    cor_pt->SetMarkerColor(4);
    cor_pt->SetMarkerStyle(21);
    cor_pt->SetTitle(plot04);
    cor_pt->GetXaxis()->SetTitle(" Pt  ");
    cor_pt->GetYaxis()->SetTitle(" correction factor ");

    cor_pt->Draw("AP");
    c4->Update();
    c4->Print(plot04);

    //5. 1/dphi vs normalized factor
    gStyle->SetOptStat(kFALSE);
    gStyle->SetOptFit(0111);  
    c5 = new TCanvas("c5");
    c5->SetFillColor(10);
    c5->SetGrid(); 
    c5->GetFrame()->SetFillColor(21);
    c5->GetFrame()->SetBorderSize(12);
    c5->cd();

    TGraph* nom_ophi = new TGraph(sz,ophi,ratio);
    nom_ophi->SetMarkerColor(4);
    nom_ophi->SetMarkerStyle(21);
    nom_ophi->SetTitle(plot05);
    nom_ophi->GetXaxis()->SetTitle(" 1/#Delta#phi  ");
    nom_ophi->GetYaxis()->SetTitle(" normalized factor ");

    nom_ophi->Draw("AP");
    c5->Update();
    c5->Print(plot05);

    delete nom_ophi;
    delete cor_pt;
    delete ophi_Pt;
    delete func2;
 }

 // 3. pt vs dphi
 gStyle->SetOptStat(kFALSE);
 gStyle->SetOptFit(0111);  
 c3 = new TCanvas("c3");
 c3->SetFillColor(10);
 c3->SetGrid(); 
 c3->GetFrame()->SetFillColor(21);
 c3->GetFrame()->SetBorderSize(12);
 c3->cd();

 TGraphErrors* dphi_Pt = new TGraphErrors(sz,pt,dphi_real,ptErr,dphiErr_real);
 dphi_Pt->SetMarkerColor(4);
 dphi_Pt->SetMarkerStyle(21);
 dphi_Pt->SetTitle(plot03);
 dphi_Pt->GetXaxis()->SetTitle(" pt ");
 dphi_Pt->GetYaxis()->SetTitle(" #Delta#phi (mrad) ");

 dphi_Pt->Draw("AP");
 c3->Update();
 c3->Print(plot03);

 // 6. ptxdPhi vs 1/dphi
 gStyle->SetOptStat(kFALSE);
 gStyle->SetOptFit(0111);  
 c6 = new TCanvas("c6");
 c6->SetFillColor(10);
 c6->SetGrid(); 
 c6->GetFrame()->SetFillColor(21);
 c6->GetFrame()->SetBorderSize(12);
 c6->cd();

 // get the 1/dphi from ptxdPhi fitting 
 Double_t odf_fit[sz] ;
 for (int j=0; j<sz; j++) {
     double theX = 1./ (10. + pt[j]) ;
     double BFit = q0 + (q1 * theX) + ( q2 * theX * theX ) ;
     odf_fit[j] =  pt[j] / BFit ;
 }
	 
 TGraphErrors* ophi_dphiPt = new TGraphErrors(sz, odf_fit, mean1, ptErr, sigma1);
 ophi_dphiPt->SetMarkerColor(4);
 ophi_dphiPt->SetMarkerStyle(21);
 ophi_dphiPt->SetTitle(plot02);
 ophi_dphiPt->GetXaxis()->SetTitle(" 1/#Delta#phi ");
 ophi_dphiPt->GetYaxis()->SetTitle(" ptx*#Delta#phi  ");

 TF1 *func6 = new TF1("func6", SeedPtFunction::fitf, 5, 25000, np );
 if ( type < 4 ) { 
    func6->SetParLimits(1, -10., -0.0001);
    func6->SetParameter(1, q1);
 }
 ophi_dphiPt->Fit( func6 ,"R","", 0, 25000);

 double r0 = func6->GetParameter(0);
 double r1 = func6->GetParameter(1);
 double r2 = func6->GetParameter(2);
 double s1 = r1/r0;
 double s2 = r2/r0;

 fprintf (dfile2,"  %s_%d_scale = cms.vdouble( %f, %f, %f, %f, %f ), \n"
                  ,det ,idx                   ,r0, r1, r2, s1, s2 );

 ophi_dphiPt->Draw("AP");
 c6->Update();
 c6->Print(plot06);

 fclose(dfile);
 fclose(dfile2);

 gSystem->cd("../");


 delete gausf;
 delete func0;
 delete func6;

 delete dphi_Pt;
 delete pt_dphiPt;
 delete ophi_dphiPt;

 //gROOT->Reset();
 //gROOT->ProcessLine(".q"); 

}

void SeedPtScale::deBugger( TF1* fitfunc, TString funcName ,TH1D* histo, double L2, double H2, int cr ) {

      fitfunc->SetLineStyle(cr);
      fitfunc->SetLineColor(cr);
      double D2 = H2 - L2 ;

      histo->SetAxisRange(L2-D2,H2+D2,"X");
      histo->Fit( funcName , "RQ", "sames", L2, H2 );
     
      cv->Update();
      cv->Print("BugInfo/"+plot_id);
}

bool SeedPtScale::BadFitting( TF1* fitfunc, TH1D* histo ) {

    double p0 = fitfunc->GetParameter(0);
    double p1 = fitfunc->GetParameter(1);
    double p2 = fitfunc->GetParameter(2);
    double mean_h = histo->GetMean(); 
    double rms_h  = histo->GetRMS(); 
    double ent_h  = histo->GetEntries(); 

    bool badfit = false ;
    if ( p0 > 0.9* ent_h )               badfit = true; 
    if ( p0 < 1. && ent_h > 10. )        badfit = true; 
    if ( p1 < 0. )                       badfit = true;
    if ( fabs(p1 - mean_h) >  rms_h*3. ) badfit = true;
    if ( p2 < 0. )                       badfit = true;
    if ( p2 > rms_h*3. )                 badfit = true;

    if (badfit) cout<<" Fit Fails !!!" <<endl;
    return badfit;
}

