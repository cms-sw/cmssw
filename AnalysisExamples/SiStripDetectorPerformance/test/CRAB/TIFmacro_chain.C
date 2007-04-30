//#include "langaus.C"
// Include here the langaus.C file
/-----------------------------------------------------------------------
//
//	Convoluted Landau and Gaussian Fitting Function
//         (using ROOT's Landau and Gauss functions)
//
//  Based on a Fortran code by R.Fruehwirth (fruhwirth@hephy.oeaw.ac.at)
//  Adapted for C++/ROOT by H.Pernegger (Heinz.Pernegger@cern.ch) and
//   Markus Friedl (Markus.Friedl@cern.ch)
//
//  to execute this example, do:
//  root > .x langaus.C
// or
//  root > .x langaus.C++
//
//-----------------------------------------------------------------------

#include "TH1.h"
#include "TF1.h"
#include "TROOT.h"
#include "TStyle.h"

Double_t langaufun(Double_t *, Double_t *);

int nFitNum = 0;

Double_t langaufun(Double_t *x, Double_t *par) {

   //Fit parameters:
   //par[0]=Width (scale) parameter of Landau density
   //par[1]=Most Probable (MP, location) parameter of Landau density
   //par[2]=Total area (integral -inf to inf, normalization constant)
   //par[3]=Width (sigma) of convoluted Gaussian function
   //
   //In the Landau distribution (represented by the CERNLIB approximation), 
   //the maximum is located at x=-0.22278298 with the location parameter=0.
   //This shift is corrected within this function, so that the actual
   //maximum is identical to the MP parameter.

      // Numeric constants
      Double_t invsq2pi = 0.3989422804014;   // (2 pi)^(-1/2)
      Double_t mpshift  = -0.22278298;       // Landau maximum location

      // Control constants
      Double_t np = 100.0;      // number of convolution steps
      Double_t sc =   5.0;      // convolution extends to +-sc Gaussian sigmas

      // Variables
      Double_t xx;
      Double_t mpc;
      Double_t fland;
      Double_t sum = 0.0;
      Double_t xlow,xupp;
      Double_t step;
      Double_t i;


      // MP shift correction
      mpc = par[1] - mpshift * par[0]; 

      // Range of convolution integral
      xlow = x[0] - sc * par[3];
      xupp = x[0] + sc * par[3];

      step = (xupp-xlow) / np;

      // Convolution integral of Landau and Gaussian by sum
      for(i=1.0; i<=np/2; i++) {
         xx = xlow + (i-.5) * step;
         fland = TMath::Landau(xx,mpc,par[0]) / par[0];
         sum += fland * TMath::Gaus(x[0],xx,par[3]);

         xx = xupp - (i-.5) * step;
         fland = TMath::Landau(xx,mpc,par[0]) / par[0];
         sum += fland * TMath::Gaus(x[0],xx,par[3]);
      }

      return (par[2] * step * sum * invsq2pi / par[3]);
}



TF1 *langaufit(TH1F *his, Double_t *fitrange, Double_t *startvalues, Double_t *parlimitslo, Double_t *parlimitshi, Double_t *fitparams, Double_t *fiterrors, Double_t *ChiSqr, Int_t *NDF)
{
   // Once again, here are the Landau * Gaussian parameters:
   //   par[0]=Width (scale) parameter of Landau density
   //   par[1]=Most Probable (MP, location) parameter of Landau density
   //   par[2]=Total area (integral -inf to inf, normalization constant)
   //   par[3]=Width (sigma) of convoluted Gaussian function
   //
   // Variables for langaufit call:
   //   his             histogram to fit
   //   fitrange[2]     lo and hi boundaries of fit range
   //   startvalues[4]  reasonable start values for the fit
   //   parlimitslo[4]  lower parameter limits
   //   parlimitshi[4]  upper parameter limits
   //   fitparams[4]    returns the final fit parameters
   //   fiterrors[4]    returns the final fit errors
   //   ChiSqr          returns the chi square
   //   NDF             returns ndf

   Int_t i;
   Char_t FunName[100];

   sprintf(FunName,"Fitfcn_%s%d",his->GetName(), nFitNum++);

   TF1 *ffitold = (TF1*)gROOT->GetListOfFunctions()->FindObject(FunName);
   if (ffitold) delete ffitold;

   TF1 *ffit = new TF1(FunName,langaufun,fitrange[0],fitrange[1],4);
   ffit->SetParameters(startvalues);
   ffit->SetParNames("Width","MP","Area","GSigma");
   
   for (i=0; i<4; i++) {
      ffit->SetParLimits(i, parlimitslo[i], parlimitshi[i]);
   }

   his->Fit(FunName,"RB0");   // fit within specified range, use ParLimits, do not plot

   ffit->GetParameters(fitparams);    // obtain fit parameters
   for (i=0; i<4; i++) {
      fiterrors[i] = ffit->GetParError(i);     // obtain fit parameter errors
   }
   ChiSqr[0] = ffit->GetChisquare();  // obtain chi^2
   NDF[0] = ffit->GetNDF();           // obtain ndf

   return (ffit);              // return fit function

}


Int_t langaupro(Double_t *params, Double_t &maxx, Double_t &FWHM) {

   // Seaches for the location (x value) at the maximum of the 
   // Landau-Gaussian convolute and its full width at half-maximum.
   //
   // The search is probably not very efficient, but it's a first try.

   Double_t p,x,fy,fxr,fxl;
   Double_t step;
   Double_t l,lold;
   Int_t i = 0;
   Int_t MAXCALLS = 10000;


   // Search for maximum

   p = params[1] - 0.1 * params[0];
   step = 0.05 * params[0];
   lold = -2.0;
   l    = -1.0;


   while ( (l != lold) && (i < MAXCALLS) ) {
      i++;

      lold = l;
      x = p + step;
      l = langaufun(&x,params);
 
      if (l < lold)
         step = -step/10;
 
      p += step;
   }

   if (i == MAXCALLS)
      return (-1);

   maxx = x;

   fy = l/2;


   // Search for right x location of fy

   p = maxx + params[0];
   step = params[0];
   lold = -2.0;
   l    = -1e300;
   i    = 0;


   while ( (l != lold) && (i < MAXCALLS) ) {
      i++;

      lold = l;
      x = p + step;
      l = TMath::Abs(langaufun(&x,params) - fy);
 
      if (l > lold)
         step = -step/10;
 
      p += step;
   }

   if (i == MAXCALLS)
      return (-2);

   fxr = x;


   // Search for left x location of fy

   p = maxx - 0.5 * params[0];
   step = -params[0];
   lold = -2.0;
   l    = -1e300;
   i    = 0;

   while ( (l != lold) && (i < MAXCALLS) ) {
      i++;

      lold = l;
      x = p + step;
      l = TMath::Abs(langaufun(&x,params) - fy);
 
      if (l > lold)
         step = -step/10;
 
      p += step;
   }

   if (i == MAXCALLS)
      return (-3);


   fxl = x;

   FWHM = fxr - fxl;
   return (0);
}

void langaus( TH1F *poHist) {
  // Fill Histogram
  printf("Fitting...\n");

  // Setting fit range and start values
  Double_t fr[2];
  Double_t sv[4], pllo[4], plhi[4], fp[4], fpe[4];
  fr[0]=0.3*poHist->GetMean();
  fr[1]=3.0*poHist->GetMean();

  pllo[0]=5.0; pllo[1]=30.0; pllo[2]=1.0; pllo[3]=10.0;
  plhi[0]=25.0; plhi[1]=200.0; plhi[2]=1000000.0; plhi[3]=50.0;
  sv[0]=17.9; sv[1]=100.0; sv[2]=50000.0; sv[3]=42.1;

  Double_t chisqr;
  Int_t    ndf;
  TF1 *fitsnr = langaufit(poHist,fr,sv,pllo,plhi,fp,fpe,&chisqr,&ndf);

  Double_t SNRPeak, SNRFWHM;
  langaupro(fp,SNRPeak,SNRFWHM);

  printf("Fitting done\nPlotting results...\n");

  poHist->Draw( "pe");
  fitsnr->Draw("lsame");
}

void langausN( TH1F *poHist) {
  // Fill Histogram
  printf("Fitting...\n");

  // Setting fit range and start values
  Double_t fr[2];
  Double_t sv[4], pllo[4], plhi[4], fp[4], fpe[4];
  fr[0]=0.3*poHist->GetMean();
  fr[1]=3.0*poHist->GetMean();

  pllo[0]=1.0; pllo[1]=4.0; pllo[2]=0.2; pllo[3]=2.0;
  plhi[0]=30.0; plhi[1]=50.0; plhi[2]=200000.0; plhi[3]=10.0;
  sv[0]=15.0; sv[1]=30.0; sv[2]=10000.0; sv[3]=8.0;

  Double_t chisqr;
  Int_t    ndf;
  TF1 *fitsnr = langaufit(poHist,fr,sv,pllo,plhi,fp,fpe,&chisqr,&ndf);

  Double_t SNRPeak, SNRFWHM;
  langaupro(fp,SNRPeak,SNRFWHM);

  printf("Fitting done\nPlotting results...\n");

  poHist->Draw( "pe");
  fitsnr->Draw("lsame");
}
///////////////////////////////////////////
///////////////////////////////////////////

/////////////////
// Begin macro //
/////////////////


// TFile	    *poFileIn;
TCanvas     *poCanvas;
TPaveLabel  *title;

TTree	    *poTrackTree;
TTree	    *poLATree;
TTree	    *poTrackNum;

char cMaxPadsX = 2;
char cMaxPadsY = 2;
char cCurPad   = 0;
bool bCanvasSaved = false;
std::string oPageTitle = "";

void createPagePre() {
  if( 0 == cCurPad) {
    poCanvas->Clear();
    poCanvas->Divide( cMaxPadsX, cMaxPadsY);
    title->SetLabel( oPageTitle.c_str());
    title->Draw();
  }
  
  ++cCurPad;
  poCanvas->cd( cCurPad);

  bCanvasSaved = false;
}

void createPagePost( const char *pcFILE_OUT) {
  poCanvas->Update();

  if( cMaxPadsX * cMaxPadsY == cCurPad) {
    poCanvas->Print( pcFILE_OUT);
    bCanvasSaved = true;
    // Divide Canvas
    cCurPad = 0;
  }
}

void insertPageBreak( const char *pcFILE_OUT) {
  cCurPad = cMaxPadsX * cMaxPadsY;
  createPagePost( pcFILE_OUT);
}

void setPadsLayout( const char *pcFILE_OUT,
		    const char cMAX_PADSX,
		    const char cMAX_PADSY) {
  insertPageBreak( pcFILE_OUT);
  cMaxPadsX = cMAX_PADSX;
  cMaxPadsY = cMAX_PADSY;
  cCurPad = 0;
}

void setPageTitle( const char *pcPAGE_TITLE) {
  oPageTitle = pcPAGE_TITLE;
}

void createPage( TTree *poTree,
		 const char *pcFILE_OUT,
		 const std::string &roFUNC,
		 const bool cREBIN = false,
		 const bool overflow = false) {

  if (overflow) {
    gStyle->SetOptStat("emro");
  }
  createPagePre();
  poTree->Draw( roFUNC.c_str());
  createPagePost( pcFILE_OUT);
}

void createPage( TTree *poTree,
		 const char *pcFILE_OUT,
		 const std::string &roFUNC,
		 const std::string &roARGS,
		 const char cREBIN = 0) {

  createPagePre();
  if( cREBIN) {
    poTree->Draw( roFUNC.c_str(), roARGS.c_str());
    TH1F *htemp = dynamic_cast<TH1F *>( gPad->GetPrimitive("htemp"));
    TH1F *hnew =  dynamic_cast<TH1F*>( htemp->Rebin(3,"hnew"));
    hnew->Draw();
  } else {
    poTree->Draw( roFUNC.c_str(), roARGS.c_str());
  }
  createPagePost( pcFILE_OUT);
}

void createPage( TTree *poTree,
		 const char *pcFILE_OUT,
		 const std::string &roFUNC,
		 const std::string &roARGS,
		 const int ymax,
		 const char * hname) {

  createPagePre();

  poTree->Draw( roFUNC.c_str(), roARGS.c_str());
  TH1F *htemp = dynamic_cast<TH1F *>( gPad->GetPrimitive(hname) );
  htemp->SetMaximum(ymax);
  htemp->Draw();
  createPagePost( pcFILE_OUT);
}

// -------------------- 28/1/2007 --------------------
// New createPage function to allow profile histograms
// ---------------------------------------------------
void createPage( TTree *poTree,
		 const char *pcFILE_OUT,
		 const std::string &roFUNC,
		 const std::string &roARGS,
		 const std::string &roOPTS,
		 const char cREBIN = 0) {

  // show also overflows
  gStyle->SetOptStat("emro");

  createPagePre();
  if( cREBIN) {
    poTree->Draw( roFUNC.c_str(), roARGS.c_str(), roOPTS.c_str());
    TH1F *htemp = dynamic_cast<TH1F *>( gPad->GetPrimitive("htemp"));
    TH1F *hnew =  dynamic_cast<TH1F*>( htemp->Rebin(3,"hnew"));
    hnew->Draw();
  } else {
    poTree->Draw( roFUNC.c_str(), roARGS.c_str(), roOPTS.c_str());
  }
  createPagePost( pcFILE_OUT);
}

void closeFile( char *pcFILE_OUT) {
  if( !bCanvasSaved) {
    poCanvas->Print( pcFILE_OUT);
  }

  {
    std::string oFile( pcFILE_OUT);
    oFile += "]";
    poCanvas->Print( oFile.c_str());
  }

}

// void TIFmacro_chain( const char *pcFILE_IN,
// void TIFmacro_chain( const char *pcFILE_OUT = "out.ps",
// 	       bool TIB_ON = false,
// 	       bool TOB_ON = true,
// 	       bool TID_ON = false,
// 	       bool TEC_ON = false) {

// Use put the files in the macro and then select only the kind of detector
void TIFmacro_chain( const char *pcFILE_IN,
		     const char *jobList,
		     const char *pcFILE_OUT,
		     bool TIB_ON = false,
		     bool TOB_ON = false,
		     bool TID_ON = false,
		     bool TEC_ON = false ) {

  std::cout << "file in = " << pcFILE_IN << std::endl;

  SetStyle();
  //  poFileIn  = new TRFIOFile( pcFILE_IN);

  poCanvas = new TCanvas();
  poCanvas->Draw();

  title = new TPaveLabel( 0.01, 0.01, 0.9, 0.04, oPageTitle.c_str());
  title->SetFillColor(0);
  title->SetTextColor(1);
  title->SetTextFont(52);
  title->SetBorderSize(0);
  title->Draw();

  // poTrackTree = dynamic_cast<TTree *>( poFileIn->Get( "TrackTree"));
  // poLATree    = dynamic_cast<TTree *>( poFileIn->Get( "TIFNtupleMakerTree"));
  // poTrackNum = dynamic_cast<TTree *>( poFileIn->Get( "TrackNum"));

  TChain * tkchain = new TChain("TrackTree");
  TChain * LAchain = new TChain("TIFNtupleMakerTree");
  TChain * TNchain = new TChain("TrackNum");

  TString FileName;

  char * ptr = std::strtok(jobList,"-");
  std::cout << "ptr = " << ptr << std::endl;
  while ( ptr != NULL ) {
    std::cout << "Chaining file: " << pcFILE_IN << ptr << std::endl;
//    FileName = *pcFILE_IN + "_" + *ptr + ".root";
    FileName=pcFILE_IN;
    FileName+="_";
    FileName+=ptr;
    FileName+=".root";
    std::cout << "FileName = " << FileName << std::endl;
    tkchain->Add( FileName );
    LAchain->Add( FileName );
    TNchain->Add( FileName );
    ptr = std::strtok(NULL,"-");
  }

  {
    std::string oFile( pcFILE_OUT);
    oFile += "[";
    poCanvas->Print( oFile.c_str());
  }

  setPageTitle( "Tracks number");
  createPage( TNchain, pcFILE_OUT, "numberoftracks>>tknum(10,0,10)", false, true);
  insertPageBreak( pcFILE_OUT);

  setPageTitle( "Track: pt, eta, phi, chi2");
  createPage( tkchain, pcFILE_OUT, "pt>>hpt(50, 0, 50)");
  createPage( tkchain, pcFILE_OUT, "eta>>heta(100, -2, 2)");
  createPage( tkchain, pcFILE_OUT, "phi>>hphi(100, -3.14, 3.14)");
  createPage( tkchain, pcFILE_OUT, "chi2>>hchi2(100,0,100)", false, true);
  //   insertPageBreak( pcFILE_OUT);

  setPageTitle("Tracks: hits per track");
  createPage( tkchain, pcFILE_OUT, "hitspertrack>>hhitpertk(12,0,12)");	
  createPage( tkchain, pcFILE_OUT, "hitspertrack:eta", "", "prof");
  createPage( tkchain, pcFILE_OUT, "hitspertrack:phi", "", "prof");
  createPage( LAchain, pcFILE_OUT, "tk_id>>htk_id(5,0,5)",false, true);
  //   insertPageBreak( pcFILE_OUT);

  if ( TIB_ON ) {
    setPageTitle("Global positions");
    createPage( LAchain, pcFILE_OUT, "globalPositionY:globalPositionX>>hyx(120,-60,60,120,-60,60)", "tk_id > 0");
    createPage( LAchain, pcFILE_OUT, "globalPositionX:globalPositionZ", "tk_id > 0");
    createPage( LAchain, pcFILE_OUT, "globalPositionY:globalPositionZ", "tk_id > 0");
    insertPageBreak( pcFILE_OUT);
  }
  if ( TOB_ON ) {
    setPageTitle("Global positions");
    createPage( LAchain, pcFILE_OUT, "globalPositionY:globalPositionX>>hyx(240,-120,120,240,-120,120)", "tk_id > 0");
    createPage( LAchain, pcFILE_OUT, "globalPositionX:globalPositionZ", "tk_id > 0");
    createPage( LAchain, pcFILE_OUT, "globalPositionY:globalPositionZ", "tk_id > 0");
    insertPageBreak( pcFILE_OUT);
  }

  setPageTitle( "Number of Clusters" );
  createPage( TNchain, pcFILE_OUT, "numberofclusters>>hnumcluster(100,0,100)" );
  createPage( TNchain, pcFILE_OUT, "numberofclusters>>hnumclusterofftk(100,0,100)", "numberoftracks == 0" );
  createPage( TNchain, pcFILE_OUT, "numberofclusters>>hnumclusteronetk(100,0,100)", "numberoftracks == 1" );
  insertPageBreak( pcFILE_OUT);

  setPageTitle( "Number of Clusters for multitrack events" );
  createPage( TNchain, pcFILE_OUT, "numberofclusters>>hnumclustertwotk(100,0,100)", "numberoftracks == 2" );
  createPage( TNchain, pcFILE_OUT, "numberofclusters>>hnumclusterthreetk(100,0,100)", "numberoftracks == 3" );
  createPage( TNchain, pcFILE_OUT, "numberofclusters>>hnumclustermoretk(100,0,100)", "numberoftracks > 3" );
  insertPageBreak( pcFILE_OUT);

  // cluster leaves: angle, tk_id, bwfw, charge, chi2, clusterchg, clusterchgl,
  // clusterchgr, clustereta, clustermaxchg, clusternoise, clusterpos, eta, event, 
  // extint, hitspertrack, layer, localmagfield, module, momentum, monostereo, 
  // ndof, normchi2, phi, pt, rod, run, sign, size, stereocorrection, string, type, 
  // bTriggerDT, bTriggerCSC, BTriggerRBC1, bTriggerRBC2, bTRiggerRPC, eventcounter
  // wheel

  // hists: # of clusters per layer, # of track clusters / total # of clusters
  // # of hits per track, phi, cluster size vs angle, angle per layer, cluster charge,
  // cluster charge vs angle
 
  setPageTitle("track phi for layers");
  createPage( LAchain, pcFILE_OUT, "phi", "pt<100 && monostereo == 0 && layer==2 && tk_id>0");
  createPage( LAchain, pcFILE_OUT, "phi", "pt<100 && monostereo == 0 && layer==3 && tk_id>0");
  createPage( LAchain, pcFILE_OUT, "phi", "pt<100 && monostereo == 0 && layer==3 && tk_id>0");
  createPage( LAchain, pcFILE_OUT, "phi", "pt<100 && monostereo == 0 && layer==4 && tk_id>0");

  setPageTitle("track eta for layers");
  createPage( LAchain, pcFILE_OUT, "eta", "pt<100 && monostereo == 0 && layer==2 && tk_id>0");
  createPage( LAchain, pcFILE_OUT, "eta", "pt<100 && monostereo == 0 && layer==3 && tk_id>0");
  createPage( LAchain, pcFILE_OUT, "eta", "pt<100 && monostereo == 0 && layer==3 && tk_id>0");
  createPage( LAchain, pcFILE_OUT, "eta", "pt<100 && monostereo == 0 && layer==4 && tk_id>0");
  
  // Keep this for the rods.
  // The rods were: layer 2 < 0; layer 3 <0 and >0; layer 4 >0.
  // ----------------------------------------------------------
  //  createPage( LAchain, pcFILE_OUT, "chi2", "pt<100 && monostereo == 0 && layer==2 && tk_id>0 && chi2<50");
  // ----------------------------------------------------------

  // Setup to have 2 pads along x and 3 along y.
  setPadsLayout(pcFILE_OUT, 2, 3);

  // type == 3 for TIB and 5 for TOB
  // -------------------------------

  // TIB
  // ---
  if (TIB_ON) {
    //     setPageTitle("hitspertrack for layers");
    //     createPage( LAchain, pcFILE_OUT, "hitspertrack>>TIBhptkl1(12,0,12)", "type == 3 && monostereo == 0 && layer==1 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "hitspertrack>>TIBhptkl2(12,0,12)", "type == 3 && monostereo == 0 && layer==2 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "hitspertrack>>TIBhptkl3(12,0,12)", "type == 3 && monostereo == 0 && layer==3 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "hitspertrack>>TIBhptkl4(12,0,12)", "type == 3 && monostereo == 0 && layer==4 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "hitspertrack>>TIBhptkl1S(12,0,12)", "type == 3 && monostereo == 1 && layer==1 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "hitspertrack>>TIBhptkl2S(12,0,12)", "type == 3 && monostereo == 1 && layer==2 && tk_id>0");

    setPageTitle("TIB chi2 for layers");
    createPage( LAchain, pcFILE_OUT, "chi2>>TIBchi2l1(100,0,200)", "type == 3 && layer==1 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "chi2>>TIBchi2l2(100,0,200)", "type == 3 && layer==2 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "chi2>>TIBchi2l3(100,0,200)", "type == 3 && layer==3 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "chi2>>TIBchi2l4(100,0,200)", "type == 3 && layer==4 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "chi2>>TIBchi2l1S(100,0,200)", "type == 3 && layer==1 && monostereo == 1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "chi2>>TIBchi2l2S(100,0,200)", "type == 3 && layer==2 && monostereo == 1 && tk_id>0");
  
    setPageTitle("track pt vs eta for layers");
    createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 3 && layer==1 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 3 && layer==2 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 3 && layer==3 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 3 && layer==4 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 3 && layer==1 && monostereo == 1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 3 && layer==2 && monostereo == 1 && tk_id>0");

    //     setPageTitle("cluster position for all clusters");
    //     createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterposl1(200,0,800)", "type == 3 && layer==1 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterposl2(200,0,800)", "type == 3 && layer==2 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterposl3(200,0,800)", "type == 3 && layer==3 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterposl4(200,0,800)", "type == 3 && layer==4 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterposl1S(200,0,800)", "type == 3 && layer==1 && monostereo == 1", 300, "TIBclusterposl1S");
    //     createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterposl2S(200,0,800)", "type == 3 && layer==2 && monostereo == 1");

    setPageTitle("cluster position for track clusters");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterpostkl1(200,0,800)", "type == 3 && layer==1 && monostereo == 0 && tk_id>=1");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterpostkl2(200,0,800)", "type == 3 && layer==2 && monostereo == 0 && tk_id>=1");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterpostkl3(200,0,800)", "type == 3 && layer==3 && monostereo == 0 && tk_id>=1");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterpostkl4(200,0,800)", "type == 3 && layer==4 && monostereo == 0 && tk_id>=1");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterpostkl1S(200,0,800)", "type == 3 && layer==1 && monostereo == 1 && tk_id>=1");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterpostkl2S(200,0,800)", "type == 3 && layer==2 && monostereo == 1 && tk_id>=1");

    setPageTitle("cluster position for off track clusters");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterposofftkl1(200,0,800)", "type == 3 && layer==1 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterposofftkl2(200,0,800)", "type == 3 && layer==2 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterposofftkl3(200,0,800)", "type == 3 && layer==3 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterposofftkl4(200,0,800)", "type == 3 && layer==4 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterposofftkl1S(200,0,800)", "type == 3 && layer==1 && monostereo == 1 && tk_id==0", 300, "TIBclusterposofftkl1S");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIBclusterposofftkl2S(200,0,800)", "type == 3 && layer==2 && monostereo == 1 && tk_id==0");

    //   setPageTitle("cluster charge vs position for all clusters");
    //   createPage( LAchain, pcFILE_OUT, "clusterchg:clusterpos", "monostereo == 0 && layer==2 && clusterchg<500");
    //   createPage( LAchain, pcFILE_OUT, "clusterchg:clusterpos", "monostereo == 0 && layer==3 && clusterchg<500");
    //   createPage( LAchain, pcFILE_OUT, "clusterchg:clusterpos", "monostereo == 0 && layer==3 && clusterchg<500");
    //   createPage( LAchain, pcFILE_OUT, "clusterchg:clusterpos", "monostereo == 0 && layer==4 && clusterchg<500");
  
    //   setPageTitle("cluster charge vs position for track clusters");
    //   createPage( LAchain, pcFILE_OUT, "clusterchg:clusterpos", "monostereo == 0 && layer==2 && tk_id>0 && clusterchg<500");
    //   createPage( LAchain, pcFILE_OUT, "clusterchg:clusterpos", "monostereo == 0 && layer==3 && tk_id>0 && clusterchg<500");
    //   createPage( LAchain, pcFILE_OUT, "clusterchg:clusterpos", "monostereo == 0 && layer==3 && tk_id>0 && clusterchg<500");
    //   createPage( LAchain, pcFILE_OUT, "clusterchg:clusterpos", "monostereo == 0 && layer==4 && tk_id>0 && clusterchg<500");

    //     setPageTitle("cluster noise vs position for all clusters");
    //     createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==1 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==2 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==3 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==4 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==1 && monostereo == 1");
    //     createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==2 && monostereo == 1");

    setPageTitle("cluster noise vs position for track clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==1 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==2 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==3 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==4 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==1 && monostereo == 1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==2 && monostereo == 1 && tk_id>0");

    setPageTitle("cluster noise vs position for off track clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==1 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==2 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==3 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==4 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==1 && monostereo == 1 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 3 && layer==2 && monostereo == 1 && tk_id==0");

    //   setPageTitle("cluster charge vs chi2 for track clusters");
    //   createPage( LAchain, pcFILE_OUT, "clusterchg:chi2", "monostereo == 0 && layer==2 && tk_id>0 && clusterchg<500 && chi2<50");
    //   createPage( LAchain, pcFILE_OUT, "clusterchg:chi2", "monostereo == 0 && layer==3 && tk_id>0 && clusterchg<500 && chi2<50");
    //   createPage( LAchain, pcFILE_OUT, "clusterchg:chi2", "monostereo == 0 && layer==3 && tk_id>0 && clusterchg<500 && chi2<50");
    //   createPage( LAchain, pcFILE_OUT, "clusterchg:chi2", "monostereo == 0 && layer==4 && tk_id>0 && clusterchg<500 && chi2<50");

    //     setPageTitle("cluster size for all clusters");
    //     createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfacL1(20,0,20)", "type == 3 && layer==1 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfacL2(20,0,20)", "type == 3 && layer==2 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfacL3(20,0,20)", "type == 3 && layer==3 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfacL4(20,0,20)", "type == 3 && layer==4 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfacL1S(20,0,20)", "type == 3 && layer==1 && monostereo == 1", 300, "TIBhcsfacL1S");
    //     createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfacL2S(20,0,20)", "type == 3 && layer==2 && monostereo == 1");

    setPageTitle("cluster size for track clusters");
    createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfacbL1(20,0,20)", "type == 3 && layer==1 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfacbL2(20,0,20)", "type == 3 && layer==2 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfacbL3(20,0,20)", "type == 3 && layer==3 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfacbL4(20,0,20)", "type == 3 && layer==4 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfacbL1S(20,0,20)", "type == 3 && layer==1 && monostereo == 1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfacbL2S(20,0,20)", "type == 3 && layer==2 && monostereo == 1 && tk_id>0");

    setPageTitle("cluster size for off track clusters");
    createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfoffacbL1(20,0,20)", "type == 3 && layer==1 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfoffacbL2(20,0,20)", "type == 3 && layer==2 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfoffacbL3(20,0,20)", "type == 3 && layer==3 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfoffacbL4(20,0,20)", "type == 3 && layer==4 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfoffacbL1S(20,0,20)", "type == 3 && layer==1 && monostereo == 1 && tk_id==0", 300, "TIBhcsfoffacbL1S");
    createPage( LAchain, pcFILE_OUT, "size>>TIBhcsfoffacbL2S(20,0,20)", "type == 3 && layer==2 && monostereo == 1 && tk_id==0");

    //     setPageTitle("cluster charge-to-noise for all clusters");
    //     createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgl1(150,0,150)", "type == 3 && layer==1 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgl2(150,0,150)", "type == 3 && layer==2 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgl3(150,0,150)", "type == 3 && layer==3 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgl4(150,0,150)", "type == 3 && layer==4 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgl1S(150,0,150)", "type == 3 && layer==1 && monostereo == 1", 850, "TIBhcluoverchgl1S");
    //     createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgl2S(150,0,150)", "type == 3 && layer==2 && monostereo == 1");

    setPageTitle("cluster charge-to-noise for track clusters");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgtkl1(150,0,150)", "type == 3 && layer==1 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgtkl2(150,0,150)", "type == 3 && layer==2 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgtkl3(150,0,150)", "type == 3 && layer==3 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgtkl4(150,0,150)", "type == 3 && layer==4 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgtkl1S(150,0,150)", "type == 3 && layer==1 && monostereo == 1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgtkl2S(150,0,150)", "type == 3 && layer==2 && monostereo == 1 && tk_id>0");

    setPageTitle("cluster charge-to-noise for off track clusters");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgofftkl1(150,0,150)", "type == 3 && layer==1 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgofftkl2(150,0,150)", "type == 3 && layer==2 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgofftkl3(150,0,150)", "type == 3 && layer==3 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgofftkl4(150,0,150)", "type == 3 && layer==4 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgofftkl1S(150,0,150)", "type == 3 && layer==1 && monostereo == 1 && tk_id==0", 280, "TIBhcluoverchgofftkl1S");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>TIBhcluoverchgofftkl2S(150,0,150)", "type == 3 && layer==2 && monostereo == 1 && tk_id==0");

    setPageTitle("track theta");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 3 && layer==1 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 3 && layer==2 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 3 && layer==3 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 3 && layer==4 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 3 && layer==1 && monostereo == 1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 3 && layer==2 && monostereo == 1 && tk_id>0");

    //   setPageTitle("cluster noise vs size number for all clusters");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:size", "monostereo == 0 && layer==2 && size<10");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:size", "monostereo == 0 && layer==3 && size<10");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:size", "monostereo == 0 && layer==3 && size<10");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:size", "monostereo == 0 && layer==4 && size<10");

    //   setPageTitle("cluster noise vs size number for track clusters");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:size", "monostereo == 0 && layer==2 && tk_id>0 && size<10");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:size", "monostereo == 0 && layer==3 && tk_id>0 && size<10");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:size", "monostereo == 0 && layer==3 && tk_id>0 && size<10");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:size", "monostereo == 0 && layer==4 && tk_id>0 && size<10");
  
    //   setPageTitle("module number");
    //   createPage( LAchain, pcFILE_OUT, "module-369000000", "monostereo == 0 && layer==2 && tk_id>0");
    //   createPage( LAchain, pcFILE_OUT, "module-369000000", "monostereo == 0 && layer==3 && tk_id>0");
    //   createPage( LAchain, pcFILE_OUT, "module-436000000", "monostereo == 0 && layer==3 && tk_id>0");
    //   createPage( LAchain, pcFILE_OUT, "module-436000000", "monostereo == 0 && layer==4 && tk_id>0");

    //   setPageTitle("cluster charge vs module number");
    //   createPage( LAchain, pcFILE_OUT, "clusterchg:(module-369000000)", "clusterchg < 500 && monostereo == 0 && layer==2 && tk_id>0");
    //   createPage( LAchain, pcFILE_OUT, "clusterchg:(module-369000000)", "clusterchg < 500 && monostereo == 0 && layer==3 && tk_id>0");
    //   createPage( LAchain, pcFILE_OUT, "clusterchg:(module-436000000)", "clusterchg < 500 && monostereo == 0 && layer==3 && tk_id>0");
    //   createPage( LAchain, pcFILE_OUT, "clusterchg:(module-436000000)", "clusterchg < 500 && monostereo == 0 && layer==4 && tk_id>0");

    //   setPageTitle("cluster noise vs module number for all clusters");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:(module-369000000)", "monostereo == 0 && layer==2");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:(module-369000000)", "monostereo == 0 && layer==3");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:(module-436000000)", "monostereo == 0 && layer==3");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:(module-436000000)", "monostereo == 0 && layer==4");

    //   setPageTitle("cluster noise vs module number for track clusters");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:(module-369000000)", "monostereo == 0 && layer==2 && tk_id>0");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:(module-369000000)", "monostereo == 0 && layer==3 && tk_id>0");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:(module-436000000)", "monostereo == 0 && layer==3 && tk_id>0");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:(module-436000000)", "monostereo == 0 && layer==4 && tk_id>0");

    //   setPageTitle("cluster noise vs run number for all clusters");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && layer==2");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && layer==3");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && layer==3");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && layer==4");

    setPageTitle("cluster noise vs run number for track clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 3 && layer==1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 3 && layer==2 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 3 && layer==3 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 3 && layer==4 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 1 && type == 3 && layer==1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 1 && type == 3 && layer==2 && tk_id>0");

    setPageTitle("cluster eta for all clusters");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 3 && layer==1 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 3 && layer==2 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 3 && layer==3 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 3 && layer==4 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 1 && type == 3 && layer==1 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 1 && type == 3 && layer==2 && clustereta>0");

    setPageTitle("cluster eta for track clusters");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 3 && layer==1 && tk_id>0 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 3 && layer==2 && tk_id>0 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 3 && layer==3 && tk_id>0 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 3 && layer==4 && tk_id>0 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 1 && type == 3 && layer==1 && tk_id>0 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 1 && type == 3 && layer==2 && tk_id>0 && clustereta>0");

    //   setPageTitle("cluster noise for all clusters");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 0 && type == 3 && layer==1", 1);
    //   createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 0 && type == 3 && layer==2", 1);
    //   createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 0 && type == 3 && layer==3", 1);
    //   createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 0 && type == 3 && layer==4", 1);
    //   createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 1 && type == 3 && layer==1", 1);
    //   createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 1 && type == 3 && layer==2", 1);

    //   setPageTitle("cluster noise for track clusters");
    //   createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 0 && type == 3 && layer==1 && tk_id>0", 1);
    //   createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 0 && type == 3 && layer==2 && tk_id>0", 1);
    //   createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 0 && type == 3 && layer==3 && tk_id>0", 1);
    //   createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 0 && type == 3 && layer==4 && tk_id>0", 1);
    //   createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 1 && type == 3 && layer==1 && tk_id>0", 1);
    //   createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 1 && type == 3 && layer==2 && tk_id>0", 1);

    setPageTitle("cluster noise for all clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoisel1(50,0,5)", "monostereo == 0 && type == 3 && layer==1");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoisel2(50,0,5)", "monostereo == 0 && type == 3 && layer==2");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoisel3(50,0,5)", "monostereo == 0 && type == 3 && layer==3");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoisel4(50,0,5)", "monostereo == 0 && type == 3 && layer==4");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoisel1S(50,0,5)", "monostereo == 1 && type == 3 && layer==1", 6000, "TIBhclnoisel1S");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoisel2S(50,0,5)", "monostereo == 1 && type == 3 && layer==2");

    setPageTitle("cluster noise for track clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoisetkl1(50,0,5)", "monostereo == 0 && type == 3 && layer==1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoisetkl2(50,0,5)", "monostereo == 0 && type == 3 && layer==2 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoisetkl3(50,0,5)", "monostereo == 0 && type == 3 && layer==3 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoisetkl4(50,0,5)", "monostereo == 0 && type == 3 && layer==4 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoisetkl1S(50,0,5)", "monostereo == 1 && type == 3 && layer==1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoisetkl2S(50,0,5)", "monostereo == 1 && type == 3 && layer==2 && tk_id>0");

    setPageTitle("cluster noise for off track clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoiseofftkl1(50,0,5)", "monostereo == 0 && type == 3 && layer==1 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoiseofftkl2(50,0,5)", "monostereo == 0 && type == 3 && layer==2 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoiseofftkl3(50,0,5)", "monostereo == 0 && type == 3 && layer==3 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoiseofftkl4(50,0,5)", "monostereo == 0 && type == 3 && layer==4 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoiseofftkl1S(50,0,5)", "monostereo == 1 && type == 3 && layer==1 && tk_id==0", 2400, "TIBhclnoiseofftkl1S");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TIBhclnoiseofftkl2S(50,0,5)", "monostereo == 1 && type == 3 && layer==2 && tk_id==0");

    // Fits
    setPageTitle("cluster charge for all clusters");
    fitCharge( pcFILE_OUT, "TIB L1", "clusterchg", "monostereo == 0 && type == 3 && layer==1", LAchain);
    fitCharge( pcFILE_OUT, "TIB L2", "clusterchg", "monostereo == 0 && type == 3 && layer==2", LAchain);
    fitCharge( pcFILE_OUT, "TIB L3", "clusterchg", "monostereo == 0 && type == 3 && layer==3", LAchain);
    fitCharge( pcFILE_OUT, "TIB L4", "clusterchg", "monostereo == 0 && type == 3 && layer==4", LAchain);
    fitCharge( pcFILE_OUT, "TIB L1S", "clusterchg", "monostereo == 1 && type == 3 && layer==1", LAchain, 2200);
    fitCharge( pcFILE_OUT, "TIB L2S", "clusterchg", "monostereo == 1 && type == 3 && layer==2", LAchain);

    setPageTitle("cluster charge for track clusters");
    fitCharge( pcFILE_OUT, "TIB L1", "clusterchg", "monostereo == 0 && type == 3 && layer==1 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L2", "clusterchg", "monostereo == 0 && type == 3 && layer==2 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L3", "clusterchg", "monostereo == 0 && type == 3 && layer==3 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L4", "clusterchg", "monostereo == 0 && type == 3 && layer==4 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L1S", "clusterchg", "monostereo == 1 && type == 3 && layer==1 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L2S", "clusterchg", "monostereo == 1 && type == 3 && layer==2 && tk_id>0", LAchain);

    setPageTitle("cluster charge for off track clusters");
    fitCharge( pcFILE_OUT, "TIB L1", "clusterchg", "monostereo == 0 && type == 3 && layer==1 && tk_id == 0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L2", "clusterchg", "monostereo == 0 && type == 3 && layer==2 && tk_id == 0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L3", "clusterchg", "monostereo == 0 && type == 3 && layer==3 && tk_id == 0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L4", "clusterchg", "monostereo == 0 && type == 3 && layer==4 && tk_id == 0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L1S", "clusterchg", "monostereo == 1 && type == 3 && layer==1 && tk_id == 0", LAchain, 700);
    fitCharge( pcFILE_OUT, "TIB L2S", "clusterchg", "monostereo == 1 && type == 3 && layer==2 && tk_id == 0", LAchain);

    setPageTitle("cluster charge-to-noise for all clusters");
    fitChargeN( pcFILE_OUT, "TIB L1", "clusterchg/clusternoise", "monostereo == 0 && type == 3 && layer==1", LAchain);
    fitChargeN( pcFILE_OUT, "TIB L2", "clusterchg/clusternoise", "monostereo == 0 && type == 3 && layer==2", LAchain);
    fitChargeN( pcFILE_OUT, "TIB L3", "clusterchg/clusternoise", "monostereo == 0 && type == 3 && layer==3", LAchain);
    fitChargeN( pcFILE_OUT, "TIB L4", "clusterchg/clusternoise", "monostereo == 0 && type == 3 && layer==4", LAchain);
    fitChargeN( pcFILE_OUT, "TIB L1S", "clusterchg/clusternoise", "monostereo == 1 && type == 3 && layer==1", LAchain, 1700);
    fitChargeN( pcFILE_OUT, "TIB L2S", "clusterchg/clusternoise", "monostereo == 1 && type == 3 && layer==2", LAchain);

    setPageTitle("cluster charge-to-noise for track clusters");
    fitChargeN( pcFILE_OUT, "TIB L1", "clusterchg/clusternoise", "monostereo == 0 && type == 3 && layer==1 && tk_id>0", LAchain);
    fitChargeN( pcFILE_OUT, "TIB L2", "clusterchg/clusternoise", "monostereo == 0 && type == 3 && layer==2 && tk_id>0", LAchain);
    fitChargeN( pcFILE_OUT, "TIB L3", "clusterchg/clusternoise", "monostereo == 0 && type == 3 && layer==3 && tk_id>0", LAchain);
    fitChargeN( pcFILE_OUT, "TIB L4", "clusterchg/clusternoise", "monostereo == 0 && type == 3 && layer==4 && tk_id>0", LAchain);
    fitChargeN( pcFILE_OUT, "TIB L1S", "clusterchg/clusternoise", "monostereo == 1 && type == 3 && layer==1 && tk_id>0", LAchain);
    fitChargeN( pcFILE_OUT, "TIB L2S", "clusterchg/clusternoise", "monostereo == 1 && type == 3 && layer==2 && tk_id>0", LAchain);

    setPageTitle("cluster charge-to-noise for off track clusters");
    fitChargeN( pcFILE_OUT, "TIB L1", "clusterchg/clusternoise", "monostereo == 0 && type == 3 && layer==1 && tk_id==0", LAchain);
    fitChargeN( pcFILE_OUT, "TIB L2", "clusterchg/clusternoise", "monostereo == 0 && type == 3 && layer==2 && tk_id==0", LAchain);
    fitChargeN( pcFILE_OUT, "TIB L3", "clusterchg/clusternoise", "monostereo == 0 && type == 3 && layer==3 && tk_id==0", LAchain);
    fitChargeN( pcFILE_OUT, "TIB L4", "clusterchg/clusternoise", "monostereo == 0 && type == 3 && layer==4 && tk_id==0", LAchain);
    fitChargeN( pcFILE_OUT, "TIB L1S", "clusterchg/clusternoise", "monostereo == 1 && type == 3 && layer==1 && tk_id==0", LAchain, 700);
    fitChargeN( pcFILE_OUT, "TIB L2S", "clusterchg/clusternoise", "monostereo == 1 && type == 3 && layer==2 && tk_id==0", LAchain);

    // Old way of correcting
    ////////////////////////
    // fitCharge( pcFILE_OUT, "TIB L1", "clusterchg * cos( 1.0 * (angle-int(angle/90)*180) / 180 * 3.14)", "monostereo == 0 && type == 3 && layer==2 && tk_id>0", LAchain);
    ////////////////////////

    // Normalized clusterCharge values
    fitCharge( pcFILE_OUT, "TIB L1", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 3 && layer==2 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L2", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 3 && layer==2 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L3", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 3 && layer==3 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L4", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 3 && layer==4 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L1S", "clusterchg * fabs( cos( angle ) )", "monostereo == 1 && type == 3 && layer==1 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L2S", "clusterchg * fabs( cos( angle ) )", "monostereo == 1 && type == 3 && layer==2 && tk_id>0", LAchain);

    setPageTitle("cluster charge-to-noise for track clusters (normalized)");
    fitCharge( pcFILE_OUT, "TIB L1", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 3 && layer==1 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L2", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 3 && layer==2 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L3", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 3 && layer==3 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L4", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 3 && layer==4 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L1S", "clusterchg * fabs( cos( angle ) )", "monostereo == 1 && type == 3 && layer==1 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TIB L2S", "clusterchg * fabs( cos( angle ) )", "monostereo == 1 && type == 3 && layer==2 && tk_id>0", LAchain);
  }

  if (TOB_ON) {
    // TOB
    // ---
    // Setup to have 2 pads along x and 4 along y.
    setPadsLayout(pcFILE_OUT, 2, 4);

    //     setPageTitle("hitspertrack for layers");
    //     createPage( LAchain, pcFILE_OUT, "hitspertrack>>TOBhptkl1(14,0,14)", "type == 5 && monostereo == 0 && layer==1 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "hitspertrack>>TOBhptkl2(14,0,14)", "type == 5 && monostereo == 0 && layer==2 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "hitspertrack>>TOBhptkl3(14,0,14)", "type == 5 && monostereo == 0 && layer==3 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "hitspertrack>>TOBhptkl4(14,0,14)", "type == 5 && monostereo == 0 && layer==4 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "hitspertrack>>TOBhptkl5(14,0,14)", "type == 5 && monostereo == 0 && layer==5 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "hitspertrack>>TOBhptkl6(14,0,14)", "type == 5 && monostereo == 0 && layer==6 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "hitspertrack>>TOBhptkl1S(14,0,14)", "type == 5 && monostereo == 1 && layer==1 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "hitspertrack>>TOBhptkl2S(14,0,14)", "type == 5 && monostereo == 1 && layer==2 && tk_id>0");

    //     setPageTitle("TOB chi2 for layers");
    //     createPage( LAchain, pcFILE_OUT, "chi2>>TOBchi2l1(80,0,80)", "type == 5 && layer==1 && monostereo == 0 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "chi2>>TOBchi2l2(80,0,80)", "type == 5 && layer==2 && monostereo == 0 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "chi2>>TOBchi2l3(80,0,80)", "type == 5 && layer==3 && monostereo == 0 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "chi2>>TOBchi2l4(80,0,80)", "type == 5 && layer==4 && monostereo == 0 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "chi2>>TOBchi2l5(80,0,80)", "type == 5 && layer==5 && monostereo == 0 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "chi2>>TOBchi2l6(80,0,80)", "type == 5 && layer==6 && monostereo == 0 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "chi2>>TOBchi2l1S(80,0,80)", "type == 5 && layer==1 && monostereo == 1 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "chi2>>TOBchi2l2S(80,0,80)", "type == 5 && layer==2 && monostereo == 1 && tk_id>0");
  
    //     setPageTitle("track pt vs eta for layers");
    //     createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 5 && layer==1 && monostereo == 0 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 5 && layer==2 && monostereo == 0 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 5 && layer==3 && monostereo == 0 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 5 && layer==4 && monostereo == 0 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 5 && layer==5 && monostereo == 0 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 5 && layer==6 && monostereo == 0 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 5 && layer==1 && monostereo == 1 && tk_id>0");
    //     createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 5 && layer==2 && monostereo == 1 && tk_id>0");

    setPageTitle("cluster position for all clusters");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterposl1(200,0,800)", "type == 5 && layer==1 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterposl2(200,0,800)", "type == 5 && layer==2 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterposl3(200,0,800)", "type == 5 && layer==3 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterposl4(200,0,800)", "type == 5 && layer==4 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterposl5(200,0,800)", "type == 5 && layer==5 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterposl6(200,0,800)", "type == 5 && layer==6 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterposl1S(200,0,800)", "type == 5 && layer==1 && monostereo == 1");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterposl2S(200,0,800)", "type == 5 && layer==2 && monostereo == 1");

    setPageTitle("cluster position for track clusters");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterpostkl1(200,0,800)", "type == 5 && layer==1 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterpostkl2(200,0,800)", "type == 5 && layer==2 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterpostkl3(200,0,800)", "type == 5 && layer==3 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterpostkl4(200,0,800)", "type == 5 && layer==4 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterpostkl5(200,0,800)", "type == 5 && layer==5 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterpostkl6(200,0,800)", "type == 5 && layer==6 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterpostkl1S(200,0,800)", "type == 5 && layer==1 && monostereo == 1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterpostkl2S(200,0,800)", "type == 5 && layer==2 && monostereo == 1 && tk_id>0");

    setPageTitle("cluster position for off track clusters");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterposofftkl1(200,0,800)", "type == 5 && layer==1 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterposofftkl2(200,0,800)", "type == 5 && layer==2 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterposofftkl3(200,0,800)", "type == 5 && layer==3 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterposofftkl4(200,0,800)", "type == 5 && layer==4 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterposofftkl5(200,0,800)", "type == 5 && layer==5 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterposofftkl6(200,0,800)", "type == 5 && layer==6 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterposofftkl1S(200,0,800)", "type == 5 && layer==1 && monostereo == 1 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TOBclusterposofftkl2S(200,0,800)", "type == 5 && layer==2 && monostereo == 1 && tk_id==0");

    setPageTitle("cluster noise vs position for all clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==1 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==2 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==3 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==4 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==5 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==6 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==1 && monostereo == 1");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==2 && monostereo == 1");

    setPageTitle("cluster noise vs position for track clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==1 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==2 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==3 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==4 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==5 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==6 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==1 && monostereo == 1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==2 && monostereo == 1 && tk_id>0");

    setPageTitle("cluster noise vs position for off track clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==1 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==2 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==3 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==4 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==5 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==6 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==1 && monostereo == 1 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 5 && layer==2 && monostereo == 1 && tk_id==0");

    setPageTitle("cluster size for all clusters");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfaccL1(20,0,20)", "type == 5 && layer==1 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfaccL2(20,0,20)", "type == 5 && layer==2 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfaccL3(20,0,20)", "type == 5 && layer==3 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfaccL4(20,0,20)", "type == 5 && layer==4 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfaccL5(20,0,20)", "type == 5 && layer==5 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfaccL6(20,0,20)", "type == 5 && layer==6 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfaccL1S(20,0,20)", "type == 5 && layer==1 && monostereo == 1");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfaccL2S(20,0,20)", "type == 5 && layer==2 && monostereo == 1");

    setPageTitle("cluster size for track clusters");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacdL1(20,0,20)", "type == 5 && layer==1 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacdL2(20,0,20)", "type == 5 && layer==2 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacdL3(20,0,20)", "type == 5 && layer==3 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacdL4(20,0,20)", "type == 5 && layer==4 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacdL5(20,0,20)", "type == 5 && layer==5 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacdL6(20,0,20)", "type == 5 && layer==6 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacdL1S(20,0,20)", "type == 5 && layer==1 && monostereo == 1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacdL2S(20,0,20)", "type == 5 && layer==2 && monostereo == 1 && tk_id>0");

    setPageTitle("cluster size for off track clusters");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfoffacdL1(20,0,20)", "type == 5 && layer==1 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfoffacdL2(20,0,20)", "type == 5 && layer==2 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfoffacdL3(20,0,20)", "type == 5 && layer==3 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfoffacdL4(20,0,20)", "type == 5 && layer==4 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfoffacdL5(20,0,20)", "type == 5 && layer==5 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfoffacdL6(20,0,20)", "type == 5 && layer==6 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfoffacdL1S(20,0,20)", "type == 5 && layer==1 && monostereo == 1 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfoffacdL2S(20,0,20)", "type == 5 && layer==2 && monostereo == 1 && tk_id==0");

    //     setPageTitle("cluster charge-to-noise for all clusters");
    //     createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoiseL1(100,0,300)", "type == 5 && layer==1 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoiseL2(100,0,300)", "type == 5 && layer==2 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoiseL3(100,0,300)", "type == 5 && layer==3 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoiseL4(100,0,300)", "type == 5 && layer==4 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoiseL5(100,0,300)", "type == 5 && layer==5 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoiseL6(100,0,300)", "type == 5 && layer==6 && monostereo == 0");
    //     createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoiseL1S(100,0,300)", "type == 5 && layer==1 && monostereo == 1");
    //     createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoiseL2S(100,0,300)", "type == 5 && layer==2 && monostereo == 1");

    setPageTitle("cluster charge-to-noise for track clusters");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoisetkL1(100,0,300)", "type == 5 && layer==1 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoisetkL2(100,0,300)", "type == 5 && layer==2 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoisetkL3(100,0,300)", "type == 5 && layer==3 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoisetkL4(100,0,300)", "type == 5 && layer==4 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoisetkL5(100,0,300)", "type == 5 && layer==5 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoisetkL6(100,0,300)", "type == 5 && layer==6 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoisetkL1S(100,0,300)", "type == 5 && layer==1 && monostereo == 1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoisetkL2S(100,0,300)", "type == 5 && layer==2 && monostereo == 1 && tk_id>0");

    setPageTitle("cluster charge-to-noise for off track clusters");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoiseofftkL1(100,0,300)", "type == 5 && layer==1 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoiseofftkL2(100,0,300)", "type == 5 && layer==2 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoiseofftkL3(100,0,300)", "type == 5 && layer==3 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoiseofftkL4(100,0,300)", "type == 5 && layer==4 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoiseofftkL5(100,0,300)", "type == 5 && layer==5 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoiseofftkL6(100,0,300)", "type == 5 && layer==6 && monostereo == 0 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoiseofftkL1S(100,0,300)", "type == 5 && layer==1 && monostereo == 1 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise>>hchgvsnoiseofftkL2S(100,0,300)", "type == 5 && layer==2 && monostereo == 1 && tk_id==0");

    setPageTitle("track theta");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 5 && layer==1 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 5 && layer==2 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 5 && layer==3 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 5 && layer==4 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 5 && layer==5 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 5 && layer==6 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 5 && layer==1 && monostereo == 1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 5 && layer==2 && monostereo == 1 && tk_id>0");

    setPageTitle("cluster noise vs run number for track clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 5 && layer==1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 5 && layer==2 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 5 && layer==3 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 5 && layer==4 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 5 && layer==5 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 5 && layer==6 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 1 && type == 5 && layer==1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 1 && type == 5 && layer==2 && tk_id>0");

    setPageTitle("cluster noise vs run number for off track clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 5 && layer==1 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 5 && layer==2 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 5 && layer==3 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 5 && layer==4 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 5 && layer==5 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 5 && layer==6 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 1 && type == 5 && layer==1 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 1 && type == 5 && layer==2 && tk_id==0");

    setPageTitle("cluster eta for all clusters");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 5 && layer==1 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 5 && layer==2 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 5 && layer==3 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 5 && layer==4 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 5 && layer==5 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 5 && layer==6 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 1 && type == 5 && layer==1 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 1 && type == 5 && layer==2 && clustereta>0");

    setPageTitle("cluster eta for track clusters");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 5 && layer==1 && tk_id>0 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 5 && layer==2 && tk_id>0 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 5 && layer==3 && tk_id>0 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 5 && layer==4 && tk_id>0 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 5 && layer==5 && tk_id>0 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 5 && layer==6 && tk_id>0 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 1 && type == 5 && layer==1 && tk_id>0 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 1 && type == 5 && layer==2 && tk_id>0 && clustereta>0");

    //     setPageTitle("cluster noise for all clusters");
    //     createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoisel1(20,0,5)", "monostereo == 0 && type == 5 && layer==1");
    //     createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoisel2(20,0,5)", "monostereo == 0 && type == 5 && layer==2");
    //     createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoisel3(20,0,5)", "monostereo == 0 && type == 5 && layer==3");
    //     createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoisel4(20,0,5)", "monostereo == 0 && type == 5 && layer==4");
    //     createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoisel5(20,0,5)", "monostereo == 0 && type == 5 && layer==5");
    //     createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoisel6(20,0,5)", "monostereo == 0 && type == 5 && layer==6");
    //     createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoisel1S(20,0,5)", "monostereo == 1 && type == 5 && layer==1");
    //     createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoisel2S(20,0,5)", "monostereo == 1 && type == 5 && layer==2");

    setPageTitle("cluster noise for track clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoisetkl1(20,0,10)", "monostereo == 0 && type == 5 && layer==1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoisetkl2(20,0,10)", "monostereo == 0 && type == 5 && layer==2 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoisetkl3(20,0,10)", "monostereo == 0 && type == 5 && layer==3 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoisetkl4(20,0,10)", "monostereo == 0 && type == 5 && layer==4 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoisetkl5(20,0,10)", "monostereo == 0 && type == 5 && layer==5 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoisetkl6(20,0,10)", "monostereo == 0 && type == 5 && layer==6 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoisetkl1S(20,0,10)", "monostereo == 1 && type == 5 && layer==1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoisetkl2S(20,0,10)", "monostereo == 1 && type == 5 && layer==2 && tk_id>0");

    setPageTitle("cluster noise for off track clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoiseofftkl1(20,0,10)", "monostereo == 0 && type == 5 && layer==1 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoiseofftkl2(20,0,10)", "monostereo == 0 && type == 5 && layer==2 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoiseofftkl3(20,0,10)", "monostereo == 0 && type == 5 && layer==3 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoiseofftkl4(20,0,10)", "monostereo == 0 && type == 5 && layer==4 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoiseofftkl5(20,0,10)", "monostereo == 0 && type == 5 && layer==5 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoiseofftkl6(20,0,10)", "monostereo == 0 && type == 5 && layer==6 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoiseofftkl1S(20,0,10)", "monostereo == 1 && type == 5 && layer==1 && tk_id==0");
    createPage( LAchain, pcFILE_OUT, "clusternoise>>TOBhclnoiseofftkl2S(20,0,10)", "monostereo == 1 && type == 5 && layer==2 && tk_id==0");

    // Fits
    //     setPageTitle("cluster charge for all clusters");
    //     fitCharge( pcFILE_OUT, "TOB L1", "clusterchg", "monostereo == 0 && type == 5 && layer==1", LAchain);
    //     fitCharge( pcFILE_OUT, "TOB L2", "clusterchg", "monostereo == 0 && type == 5 && layer==2", LAchain);
    //     fitCharge( pcFILE_OUT, "TOB L3", "clusterchg", "monostereo == 0 && type == 5 && layer==3", LAchain);
    //     fitCharge( pcFILE_OUT, "TOB L4", "clusterchg", "monostereo == 0 && type == 5 && layer==4", LAchain);
    //     fitCharge( pcFILE_OUT, "TOB L5", "clusterchg", "monostereo == 0 && type == 5 && layer==5", LAchain);
    //     fitCharge( pcFILE_OUT, "TOB L6", "clusterchg", "monostereo == 0 && type == 5 && layer==6", LAchain);
    //     fitCharge( pcFILE_OUT, "TOB L1S", "clusterchg", "monostereo == 1 && type == 5 && layer==1", LAchain);
    //     fitCharge( pcFILE_OUT, "TOB L2S", "clusterchg", "monostereo == 1 && type == 5 && layer==2", LAchain);

    setPageTitle("cluster charge for track clusters");
    fitCharge( pcFILE_OUT, "TOB L1", "clusterchg", "monostereo == 0 && type == 5 && layer==1 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L2", "clusterchg", "monostereo == 0 && type == 5 && layer==2 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L3", "clusterchg", "monostereo == 0 && type == 5 && layer==3 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L4", "clusterchg", "monostereo == 0 && type == 5 && layer==4 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L5", "clusterchg", "monostereo == 0 && type == 5 && layer==5 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L6", "clusterchg", "monostereo == 0 && type == 5 && layer==6 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L1S", "clusterchg", "monostereo == 1 && type == 5 && layer==1 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L2S", "clusterchg", "monostereo == 1 && type == 5 && layer==2 && tk_id>0", LAchain);

    setPageTitle("cluster charge for off track clusters");
    fitCharge( pcFILE_OUT, "TOB L1", "clusterchg", "monostereo == 0 && type == 5 && layer==1 && tk_id==0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L2", "clusterchg", "monostereo == 0 && type == 5 && layer==2 && tk_id==0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L3", "clusterchg", "monostereo == 0 && type == 5 && layer==3 && tk_id==0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L4", "clusterchg", "monostereo == 0 && type == 5 && layer==4 && tk_id==0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L5", "clusterchg", "monostereo == 0 && type == 5 && layer==5 && tk_id==0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L6", "clusterchg", "monostereo == 0 && type == 5 && layer==6 && tk_id==0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L1S", "clusterchg", "monostereo == 1 && type == 5 && layer==1 && tk_id==0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L2S", "clusterchg", "monostereo == 1 && type == 5 && layer==2 && tk_id==0", LAchain);

    //     setPageTitle("cluster charge-to-noise for all clusters");
    //     fitChargeN( pcFILE_OUT, "TOB L1", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==1", LAchain);
    //     fitChargeN( pcFILE_OUT, "TOB L2", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==2", LAchain);
    //     fitChargeN( pcFILE_OUT, "TOB L3", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==3", LAchain);
    //     fitChargeN( pcFILE_OUT, "TOB L4", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==4", LAchain);
    //     fitChargeN( pcFILE_OUT, "TOB L5", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==5", LAchain);
    //     fitChargeN( pcFILE_OUT, "TOB L6", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==6", LAchain);
    //     fitChargeN( pcFILE_OUT, "TOB L1S", "clusterchg/clusternoise", "monostereo == 1 && type == 5 && layer==1", LAchain);
    //     fitChargeN( pcFILE_OUT, "TOB L2S", "clusterchg/clusternoise", "monostereo == 1 && type == 5 && layer==2", LAchain);

    setPageTitle("cluster charge-to-noise for track clusters");
    fitChargeN( pcFILE_OUT, "TOB L1", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==1 && tk_id>0", LAchain);
    fitChargeN( pcFILE_OUT, "TOB L2", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==2 && tk_id>0", LAchain);
    fitChargeN( pcFILE_OUT, "TOB L3", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==3 && tk_id>0", LAchain);
    fitChargeN( pcFILE_OUT, "TOB L4", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==4 && tk_id>0", LAchain);
    fitChargeN( pcFILE_OUT, "TOB L5", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==5 && tk_id>0", LAchain);
    fitChargeN( pcFILE_OUT, "TOB L6", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==6 && tk_id>0", LAchain);
    fitChargeN( pcFILE_OUT, "TOB L1S", "clusterchg/clusternoise", "monostereo == 1 && type == 5 && layer==1 && tk_id>0", LAchain);
    fitChargeN( pcFILE_OUT, "TOB L2S", "clusterchg/clusternoise", "monostereo == 1 && type == 5 && layer==2 && tk_id>0", LAchain);

    setPageTitle("cluster charge-to-noise for off track clusters");
    fitChargeN( pcFILE_OUT, "TOB L1", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==1 && tk_id==0", LAchain);
    fitChargeN( pcFILE_OUT, "TOB L2", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==2 && tk_id==0", LAchain);
    fitChargeN( pcFILE_OUT, "TOB L3", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==3 && tk_id==0", LAchain);
    fitChargeN( pcFILE_OUT, "TOB L4", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==4 && tk_id==0", LAchain);
    fitChargeN( pcFILE_OUT, "TOB L5", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==5 && tk_id==0", LAchain);
    fitChargeN( pcFILE_OUT, "TOB L6", "clusterchg/clusternoise", "monostereo == 0 && type == 5 && layer==6 && tk_id==0", LAchain);
    fitChargeN( pcFILE_OUT, "TOB L1S", "clusterchg/clusternoise", "monostereo == 1 && type == 5 && layer==1 && tk_id==0", LAchain);
    fitChargeN( pcFILE_OUT, "TOB L2S", "clusterchg/clusternoise", "monostereo == 1 && type == 5 && layer==2 && tk_id==0", LAchain);

    // Normalized clusterCharge values
    // Normalized clusterCharge values
    setPageTitle("cluster charge for track clusters (normalized)");
    fitCharge( pcFILE_OUT, "TOB L1", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 5 && layer==1 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L2", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 5 && layer==2 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L3", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 5 && layer==3 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L4", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 5 && layer==4 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L5", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 5 && layer==5 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L6", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 5 && layer==6 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L1S", "clusterchg * fabs( cos( angle ) )", "monostereo == 1 && type == 5 && layer==1 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L2S", "clusterchg * fabs( cos( angle ) )", "monostereo == 1 && type == 5 && layer==2 && tk_id>0", LAchain);

    setPageTitle("cluster charge-to-noise for track clusters (normalized)");
    fitCharge( pcFILE_OUT, "TOB L1", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 5 && layer==1 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L2", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 5 && layer==2 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L3", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 5 && layer==3 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L4", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 5 && layer==4 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L5", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 5 && layer==5 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L6", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 5 && layer==6 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L1S", "clusterchg * fabs( cos( angle ) )", "monostereo == 1 && type == 5 && layer==1 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TOB L2S", "clusterchg * fabs( cos( angle ) )", "monostereo == 1 && type == 5 && layer==2 && tk_id>0", LAchain);
  }

  // Work in progress
  // ----------------
  if (TID_ON) {
    // TID , does not have layers, but wheels. 3 wheels, the first 2 are stereo.
    // -------------------------------------------------------------------------
    // Setup to have 2 pads along x and 3 along y.
    setPadsLayout(pcFILE_OUT, 2, 3);

    setPageTitle("hitspertrack for wheels");
    createPage( LAchain, pcFILE_OUT, "hitspertrack", "type == 4 && monostereo == 0 && wheel==1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "hitspertrack", "type == 4 && monostereo == 0 && wheel==2 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "hitspertrack", "type == 4 && monostereo == 0 && wheel==3 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "hitspertrack", "type == 4 && monostereo == 1 && wheel==1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "hitspertrack", "type == 4 && monostereo == 1 && wheel==2 && tk_id>0");

    setPageTitle("TID chi2 for wheels");
    createPage( LAchain, pcFILE_OUT, "chi2>>TIDchi2w1(80,0,80)", "type == 4 && wheel==1 && monostereo == 0 && tk_id>0 && chi2<50");
    createPage( LAchain, pcFILE_OUT, "chi2>>TIDchi2w2(80,0,80)", "type == 4 && wheel==2 && monostereo == 0 && tk_id>0 && chi2<50");
    createPage( LAchain, pcFILE_OUT, "chi2>>TIDchi2w3(80,0,80)", "type == 4 && wheel==3 && monostereo == 0 && tk_id>0 && chi2<50");
    createPage( LAchain, pcFILE_OUT, "chi2>>TIDchi2w1S(80,0,80)", "type == 4 && wheel==1 && monostereo == 1 && tk_id>0 && chi2<50");
    createPage( LAchain, pcFILE_OUT, "chi2>>TIDchi2w2S(80,0,80)", "type == 4 && wheel==2 && monostereo == 1 && tk_id>0 && chi2<50");
  
    setPageTitle("track pt vs eta for wheels");
    createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 4 && wheel==1 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 4 && wheel==2 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 4 && wheel==3 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 4 && wheel==1 && monostereo == 1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "pt:eta", "type == 4 && wheel==2 && monostereo == 1 && tk_id>0");

    setPageTitle("cluster position for all clusters");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIDclusterposl1(200,0,800)", "type == 4 && wheel==1 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIDclusterposl2(200,0,800)", "type == 4 && wheel==2 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIDclusterposl3(200,0,800)", "type == 4 && wheel==3 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIDclusterposl1S(200,0,800)", "type == 4 && wheel==1 && monostereo == 1");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIDclusterposl2S(200,0,800)", "type == 4 && wheel==2 && monostereo == 1");

    setPageTitle("cluster position for track clusters");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIDclusterpostkl1(200,0,800)", "type == 4 && wheel==1 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIDclusterpostkl2(200,0,800)", "type == 4 && wheel==2 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIDclusterpostkl3(200,0,800)", "type == 4 && wheel==3 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIDclusterpostkl1S(200,0,800)", "type == 4 && wheel==1 && monostereo == 1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterpos>>TIDclusterpostkl2S(200,0,800)", "type == 4 && wheel==2 && monostereo == 1 && tk_id>0");

    setPageTitle("cluster noise vs position for all clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 4 && wheel==1 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 4 && wheel==2 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 4 && wheel==3 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 4 && wheel==1 && monostereo == 1");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 4 && wheel==2 && monostereo == 1");

    setPageTitle("cluster noise vs position for track clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 4 && wheel==1 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 4 && wheel==2 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 4 && wheel==3 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 4 && wheel==1 && monostereo == 1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:clusterpos", "type == 4 && wheel==2 && monostereo == 1 && tk_id>0");

    setPageTitle("cluster size for all clusters");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacW1(20,0,20)", "type == 4 && wheel==1 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacW2(20,0,20)", "type == 4 && wheel==2 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacW3(20,0,20)", "type == 4 && wheel==3 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacW1S(20,0,20)", "type == 4 && wheel==1 && monostereo == 1");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacW2S(20,0,20)", "type == 4 && wheel==2 && monostereo == 1");

    setPageTitle("cluster size for track clusters");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacaW1(20,0,20)", "type == 4 && wheel==1 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacaW2(20,0,20)", "type == 4 && wheel==2 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacaW3(20,0,20)", "type == 4 && wheel==3 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacaW1S(20,0,20)", "type == 4 && wheel==1 && monostereo == 1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "size>>hcsfacaW2S(20,0,20)", "type == 4 && wheel==2 && monostereo == 1 && tk_id>0");

    setPageTitle("cluster charge-to-noise for all clusters");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise", "type == 4 && wheel==1 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise", "type == 4 && wheel==2 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise", "type == 4 && wheel==3 && monostereo == 0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise", "type == 4 && wheel==1 && monostereo == 1");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise", "type == 4 && wheel==2 && monostereo == 1");

    setPageTitle("cluster charge-to-noise for track clusters");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise", "type == 4 && wheel==1 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise", "type == 4 && wheel==2 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise", "type == 4 && wheel==3 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise", "type == 4 && wheel==1 && monostereo == 1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusterchg/clusternoise", "type == 4 && wheel==2 && monostereo == 1 && tk_id>0");

    setPageTitle("track angle");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 4 && wheel==1 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 4 && wheel==2 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 4 && wheel==3 && monostereo == 0 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 4 && wheel==1 && monostereo == 1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "angle-int(angle/90)*180", "type == 4 && wheel==2 && monostereo == 1 && tk_id>0");

    setPageTitle("cluster noise vs run number for track clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 4 && wheel==1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 4 && wheel==2 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 4 && wheel==3 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 0 && type == 4 && wheel==4 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 1 && type == 4 && wheel==1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise:run", "monostereo == 1 && type == 4 && wheel==2 && tk_id>0");

    setPageTitle("cluster eta for all clusters");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 4 && wheel==1 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 4 && wheel==2 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 4 && wheel==3 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 1 && type == 4 && wheel==1 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 1 && type == 4 && wheel==2 && clustereta>0");

    setPageTitle("cluster eta for track clusters");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 4 && wheel==1 && tk_id>0 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 4 && wheel==2 && tk_id>0 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 0 && type == 4 && wheel==3 && tk_id>0 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 1 && type == 4 && wheel==1 && tk_id>0 && clustereta>0");
    createPage( LAchain, pcFILE_OUT, "clustereta", "monostereo == 1 && type == 4 && wheel==2 && tk_id>0 && clustereta>0");

    setPageTitle("cluster noise for all clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 0 && type == 4 && wheel==1");
    createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 0 && type == 4 && wheel==2");
    createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 0 && type == 4 && wheel==3");
    createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 1 && type == 4 && wheel==1");
    createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 1 && type == 4 && wheel==2");

    setPageTitle("cluster noise for track clusters");
    createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 0 && type == 4 && wheel==1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 0 && type == 4 && wheel==2 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 0 && type == 4 && wheel==3 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 1 && type == 4 && wheel==1 && tk_id>0");
    createPage( LAchain, pcFILE_OUT, "clusternoise", "monostereo == 1 && type == 4 && wheel==2 && tk_id>0");

    // Fits
    setPageTitle("cluster charge for all clusters");
    fitCharge( pcFILE_OUT, "TID W1", "clusterchg", "monostereo == 0 && type == 4 && wheel==1");
    fitCharge( pcFILE_OUT, "TID W2", "clusterchg", "monostereo == 0 && type == 4 && wheel==2");
    fitCharge( pcFILE_OUT, "TID W3", "clusterchg", "monostereo == 0 && type == 4 && wheel==3");
    fitCharge( pcFILE_OUT, "TID W1S", "clusterchg", "monostereo == 1 && type == 4 && wheel==1");
    fitCharge( pcFILE_OUT, "TID W2S", "clusterchg", "monostereo == 1 && type == 4 && wheel==2");

    setPageTitle("cluster charge for track clusters");
    fitCharge( pcFILE_OUT, "TID W1", "clusterchg", "monostereo == 0 && type == 4 && wheel==1 && tk_id>0");
    fitCharge( pcFILE_OUT, "TID W2", "clusterchg", "monostereo == 0 && type == 4 && wheel==2 && tk_id>0");
    fitCharge( pcFILE_OUT, "TID W3", "clusterchg", "monostereo == 0 && type == 4 && wheel==3 && tk_id>0");
    fitCharge( pcFILE_OUT, "TID W1S", "clusterchg", "monostereo == 1 && type == 4 && wheel==1 && tk_id>0");
    fitCharge( pcFILE_OUT, "TID W2S", "clusterchg", "monostereo == 1 && type == 4 && wheel==2 && tk_id>0");

    setPageTitle("cluster charge-to-noise for all clusters");
    fitChargeN( pcFILE_OUT, "TID W1", "clusterchg/clusternoise", "monostereo == 0 && type == 4 && wheel==1", LAchain);
    fitChargeN( pcFILE_OUT, "TID W2", "clusterchg/clusternoise", "monostereo == 0 && type == 4 && wheel==2", LAchain);
    fitChargeN( pcFILE_OUT, "TID W3", "clusterchg/clusternoise", "monostereo == 0 && type == 4 && wheel==3", LAchain);
    fitChargeN( pcFILE_OUT, "TID W1S", "clusterchg/clusternoise", "monostereo == 1 && type == 4 && wheel==1", LAchain);
    fitChargeN( pcFILE_OUT, "TID W2S", "clusterchg/clusternoise", "monostereo == 1 && type == 4 && wheel==2", LAchain);

    setPageTitle("cluster charge-to-noise for track clusters");
    fitChargeN( pcFILE_OUT, "TID W1", "clusterchg/clusternoise", "monostereo == 0 && type == 4 && wheel==1 && tk_id>0", LAchain);
    fitChargeN( pcFILE_OUT, "TID W2", "clusterchg/clusternoise", "monostereo == 0 && type == 4 && wheel==2 && tk_id>0", LAchain);
    fitChargeN( pcFILE_OUT, "TID W3", "clusterchg/clusternoise", "monostereo == 0 && type == 4 && wheel==3 && tk_id>0", LAchain);
    fitChargeN( pcFILE_OUT, "TID W1S", "clusterchg/clusternoise", "monostereo == 1 && type == 4 && wheel==1 && tk_id>0", LAchain);
    fitChargeN( pcFILE_OUT, "TID W2S", "clusterchg/clusternoise", "monostereo == 1 && type == 4 && wheel==2 && tk_id>0", LAchain);

    // Normalized clusterCharge values
    setPageTitle("cluster charge for track clusters (normalized)");
    fitCharge( pcFILE_OUT, "TID W1", "clusterchg * fabs( cos( angle ) )", "clusterchg * cos( 1.0 * angle / 180 * 3.14) < 500 && monostereo == 0 && type == 4 && wheel==1 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TID W2", "clusterchg * fabs( cos( angle ) )", "clusterchg * cos( 1.0 * angle / 180 * 3.14) < 500 && monostereo == 0 && type == 4 && wheel==2 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TID W3", "clusterchg * fabs( cos( angle ) )", "clusterchg * cos( 1.0 * angle / 180 * 3.14) < 500 && monostereo == 0 && type == 4 && wheel==3 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TID W1S", "clusterchg * fabs( cos( angle ) )", "clusterchg * cos( 1.0 * angle / 180 * 3.14) < 500 && monostereo == 1 && type == 4 && wheel==1 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TID W2S", "clusterchg * fabs( cos( angle ) )", "clusterchg * cos( 1.0 * angle / 180 * 3.14) < 500 && monostereo == 1 && type == 4 && wheel==2 && tk_id>0", LAchain);

    setPageTitle("cluster charge-to-noise for track clusters (normalized)");
    fitCharge( pcFILE_OUT, "TID W1", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 4 && wheel==1 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TID W2", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 4 && wheel==2 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TID W3", "clusterchg * fabs( cos( angle ) )", "monostereo == 0 && type == 4 && wheel==3 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TID W1S", "clusterchg * fabs( cos( angle ) )", "monostereo == 1 && type == 4 && wheel==1 && tk_id>0", LAchain);
    fitCharge( pcFILE_OUT, "TID W2S", "clusterchg * fabs( cos( angle ) )", "monostereo == 1 && type == 4 && wheel==2 && tk_id>0", LAchain);
  }

  if (TEC_ON) {
    // TEC
    // ---
  }
  // ---------------

  closeFile( pcFILE_OUT);

  //   poFileIn->Close();

  delete poCanvas;
  //   delete poFileIn;

  tkchain = 0;
  LAchain = 0;
}

void fitCharge( const char *pcFILE_OUT,
		std::string oHistName,
		std::string oLeafName,
		std::string oCuts,
		TChain * LAchain,
		int ymax = 0 ) {
  gStyle->SetStatH(0.2);
  gStyle->SetStatW(0.2);
  createPagePre();

  TH1F* chrg = new TH1F( "chrg", oHistName.c_str(),  50, 0., 500.);
  oLeafName += ">>chrg";

  if ( ymax != 0 ) {
    chrg->SetMaximum(ymax);
  }

  LAchain->Draw( oLeafName.c_str(), oCuts.c_str());

  langaus( chrg);

  createPagePost( pcFILE_OUT);
  gStyle->SetStatH(0.15);  
  gStyle->SetStatW(0.15);
}

void fitChargeN( const char *pcFILE_OUT,
		 std::string oHistName,
		 std::string oLeafName,
		 std::string oCuts,
		 TChain * LAchain,
		 int ymax = 0 ) {
  gStyle->SetStatH(0.2);
  gStyle->SetStatW(0.2);
  createPagePre();

  TH1F* chrg = new TH1F( "chrg", oHistName.c_str(),  50, 0., 100.);
  oLeafName += ">>chrg";

  if ( ymax != 0 ) {
    chrg->SetMaximum(ymax);
  }

  LAchain->Draw( oLeafName.c_str(), oCuts.c_str());

  langausN( chrg);

  createPagePost( pcFILE_OUT);
  gStyle->SetStatH(0.15);  
  gStyle->SetStatW(0.15);
}

void SetStyle() {
  TStyle *CMSStyle = new TStyle("CMS-Style","The Perfect Style for Plots ;-)");
  gStyle = CMSStyle;

  // Canvas
  CMSStyle->SetCanvasColor     (0);
  CMSStyle->SetCanvasBorderSize(0);
  CMSStyle->SetCanvasBorderMode(0);
  CMSStyle->SetCanvasDefH      (700);
  CMSStyle->SetCanvasDefW      (700);
  CMSStyle->SetCanvasDefX      (100);
  CMSStyle->SetCanvasDefY      (100);

  // Pads
  CMSStyle->SetPadColor       (0);
  CMSStyle->SetPadBorderSize  (0);
  CMSStyle->SetPadBorderMode  (0);
  CMSStyle->SetPadBottomMargin(0.18);
  CMSStyle->SetPadTopMargin   (0.08);
  CMSStyle->SetPadLeftMargin  (0.16);
  CMSStyle->SetPadRightMargin (0.05);
  CMSStyle->SetPadGridX       (0);
  CMSStyle->SetPadGridY       (0);
  CMSStyle->SetPadTickX       (0);
  CMSStyle->SetPadTickY       (0);

  // Frames
  CMSStyle->SetFrameFillStyle ( 0);
  CMSStyle->SetFrameFillColor ( 0);
  CMSStyle->SetFrameLineColor ( 1);
  CMSStyle->SetFrameLineStyle ( 0);
  CMSStyle->SetFrameLineWidth ( 1);
  CMSStyle->SetFrameBorderSize(10);
  CMSStyle->SetFrameBorderMode( 0);

  // Histograms
  CMSStyle->SetHistFillColor(2);
  CMSStyle->SetHistFillStyle(0);
  CMSStyle->SetHistLineColor(1);
  CMSStyle->SetHistLineStyle(0);
  CMSStyle->SetHistLineWidth(2);
  CMSStyle->SetNdivisions(505);

  // Functions
  CMSStyle->SetFuncColor(1);
  CMSStyle->SetFuncStyle(0);
  CMSStyle->SetFuncWidth(2);

  // Various
  CMSStyle->SetTitleSize  (0.050,"X");
  CMSStyle->SetTitleOffset(1.000,"X");
  CMSStyle->SetTitleFillColor (0);
  CMSStyle->SetLabelOffset(0.003,"X");
  CMSStyle->SetLabelSize  (0.050,"X");
  CMSStyle->SetLabelFont  (42   ,"X");

  CMSStyle->SetStripDecimals(kFALSE);

  CMSStyle->SetTitleSize  (0.050,"Y");
  CMSStyle->SetTitleOffset(1.500,"Y");
  CMSStyle->SetLabelOffset(0.008,"Y");
  CMSStyle->SetLabelSize  (0.050,"Y");
  CMSStyle->SetLabelFont  (42   ,"Y");

  CMSStyle->SetOptStat("emr");
  CMSStyle->SetOptFit();
  CMSStyle->SetStatColor(0);
  CMSStyle->SetStatBorderSize( 1);
  
  CMSStyle->SetTitleFont  (42);
  CMSStyle->SetTitleBorderSize( 1);

  CMSStyle->SetHistFillColor(kYellow);
}
