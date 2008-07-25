#include "DQM/SiStripHistoricInfoClient/interface/fitME.h"




//-----------------------------------------------------------------------------------------------
Double_t langaufun(Double_t *x, Double_t *par) 
//-----------------------------------------------------------------------------------------------
{
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

      // Landau Distribution Production
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

//-----------------------------------------------------------------------------------------------
Int_t langaupro(Double_t *params, Double_t &maxx, Double_t &FWHM) 
//-----------------------------------------------------------------------------------------------
{
      std::cout << "inside langaupro " << std::endl;
   // Seaches for the location (x value) at the maximum of the 
   // Landau and its full width at half-maximum.
   //
   // The search is probably not very efficient, but it's a first try.

   Double_t p,x,fy,fxr,fxl;
   Double_t step;
   Double_t l,lold,dl;
   Int_t i = 0;
   const Int_t MAXCALLS = 10000;
   const Double_t dlStop = 1e-3; // relative change < .001

   // Search for maximum
   p = params[1] - 0.1 * params[0];
   step = 0.05 * params[0];
   lold = -2.0;
   l    = -1.0;

   dl = (l-lold)/lold;    // FIXME catch divide by zero
   while ( (TMath::Abs(dl)>dlStop ) && (i < MAXCALLS) ) {
      i++;

      lold = l;
      x = p + step;
      l = langaufun(&x,params);
      dl = (l-lold)/lold; // FIXME catch divide by zero
        
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

   dl = (l-lold)/lold;   // FIXME catch divide by zero
   while ( ( TMath::Abs(dl)>dlStop ) && (i < MAXCALLS) ) {
      i++;

      lold = l;
      x = p + step;
      l = TMath::Abs(langaufun(&x,params) - fy);
      dl = (l-lold)/lold; // FIXME catch divide by zero
 
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

   dl = (l-lold)/lold;    // FIXME catch divide by zero
   while ( ( TMath::Abs(dl)>dlStop ) && (i < MAXCALLS) ) {
      i++;

      lold = l;
      x = p + step;
      l = TMath::Abs(langaufun(&x,params) - fy);
      dl = (l-lold)/lold; // FIXME catch divide by zero
 
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


//-----------------------------------------------------------------------------------------------
Double_t Gauss(Double_t *x, Double_t *par)
//-----------------------------------------------------------------------------------------------
// The noise function: a gaussian
{
   Double_t arg = 0;
   if (par[2]) arg = (x[0] - par[1])/par[2];

   Double_t noise = par[0]*TMath::Exp(-0.5*arg*arg);
   return noise;
}



//-----------------------------------------------------------------------------------------------
fitME::fitME(MonitorElement* ME)
//-----------------------------------------------------------------------------------------------
{
  htoFit = ME->getTH1F();
  
  pLanGausS=new Double_t[4];
  pGausS=new Double_t[3];
  epGausS=new Double_t[3];
  epLanGausS=new Double_t[4];
  pLanConv=new Double_t[2];  
  langausFit=0;
  gausFit=0;
  chi2GausS = -9999.;
  nDofGausS = -9999;
}


//-----------------------------------------------------------------------------------------------
fitME::~fitME()
//-----------------------------------------------------------------------------------------------
{
  if ( pLanGausS!=0 ) delete [] pLanGausS;
  if ( epLanGausS!=0 ) delete [] epLanGausS;
  if ( pGausS!=0 ) delete [] pGausS;
  if ( epGausS!=0 ) delete [] epGausS;
  if ( pLanConv!=0) delete[] pLanConv;
  if ( langausFit!=0 ) delete langausFit;
  if ( gausFit!=0 ) delete gausFit;
}

  
 
//-----------------------------------------------------------------------------------------------
Stat_t fitME::doFit()
//-----------------------------------------------------------------------------------------------
{

pLanGausS[0]=0; pLanGausS[1]=0; pLanGausS[2]=0; pLanGausS[3]=0;
epLanGausS[0]=0; epLanGausS[1]=0; epLanGausS[2]=0; epLanGausS[3]=0;
 
 
 if (htoFit->GetEntries()!=0) {
		  std::cout<<"Fitting "<< htoFit->GetTitle() <<std::endl;
		  // Setting fit range and start values
		  Double_t fr[2];
		  Double_t sv[4], pllo[4], plhi[4];
		  fr[0]=0.5*htoFit->GetMean();
		  fr[1]=3.0*htoFit->GetMean();
	      
		  // (EM) parameters setting good for signal only 
		  Int_t imax=htoFit->GetMaximumBin();
		  Double_t xmax=htoFit->GetBinCenter(imax);
		  Double_t ymax=htoFit->GetBinContent(imax);
		  Int_t i[2];
		  Int_t iArea[2];
	      
		  i[0]=htoFit->GetXaxis()->FindBin(fr[0]);
		  i[1]=htoFit->GetXaxis()->FindBin(fr[1]);
		  
		  iArea[0]=htoFit->GetXaxis()->FindBin(fr[0]);
		  iArea[1]=htoFit->GetXaxis()->FindBin(fr[1]);
		  Double_t AreaFWHM=htoFit->Integral(iArea[0],iArea[1],"width");
		  
		  sv[1]=xmax;
		  sv[2]=htoFit->Integral(i[0],i[1],"width");
		  sv[3]=AreaFWHM/(4*ymax);
		  sv[0]=sv[3];
		  
		  plhi[0]=25.0; plhi[1]=200.0; plhi[2]=1000000.0; plhi[3]=50.0;
		  pllo[0]=1.5 ; pllo[1]=10.0 ; pllo[2]=1.0      ; pllo[3]= 1.0;
		  
		  // create different landau+gaussians for different runs
		  Char_t FunName[100];
		  sprintf(FunName,"FitfcnLG_%s",htoFit->GetName());  
		  TF1 *ffitold = (TF1*)gROOT->GetListOfFunctions()->FindObject(FunName);
		  if (ffitold) delete ffitold;
		  
		  langausFit = new TF1(FunName,langaufun,fr[0],fr[1],4);
		  langausFit->SetParameters(sv);
		  langausFit->SetParNames("Width","MP","Area","GSigma");
		  
		  for (Int_t i=0; i<4; i++) {
		    langausFit->SetParLimits(i,pllo[i],plhi[i]);
		  }  
		  
		  htoFit->Fit(langausFit,"R0");  // "R" fit in a range,"0" quiet fit
		  
		  langausFit->SetRange(fr[0],fr[1]);
		  pLanGausS=langausFit->GetParameters();
		  epLanGausS=langausFit->GetParErrors();
		  
		  chi2GausS =langausFit->GetChisquare();  // obtain chi^2
		  nDofGausS = langausFit->GetNDF();           // obtain ndf
		  
		  Double_t sPeak, sFWHM;
		  langaupro(pLanGausS,sPeak,sFWHM);
		  pLanConv[0]=sPeak;
		  pLanConv[1]=sFWHM;
		  std::cout << "langaupro:  max  " << sPeak << std::endl;
		  std::cout << "langaupro:  FWHM " << sFWHM << std::endl;
		  
		  TCanvas *cAll = new TCanvas("Fit",htoFit->GetTitle(),1);
		  //Char_t fitFileName[60];
		  //sprintf(fitFileName,"Fits/Run_%d/%s/Fit_%s.png",RunNumber,SubDetName,htoFit->GetTitle());
		  htoFit->Draw("pe");
		  htoFit->SetStats(100);
		  langausFit->Draw("lsame");
		  gStyle->SetOptFit(1111111);
		  
		  //cAll->Print(fitFileName,"png");
		}
		else {  
		  pLanGausS[0]=-10; pLanGausS[1]=-10; pLanGausS[2]=-10; pLanGausS[3]=-10;
		  epLanGausS[0]=-10; epLanGausS[1]=-10; epLanGausS[2]=-10; epLanGausS[3]=-10;
		  pLanConv[0]=-10;   pLanConv[1]=-10;   chi2GausS=-10;  nDofGausS=-10;    
		}

  return htoFit->GetEntries();

  
}

  
//-----------------------------------------------------------------------------------------------
Stat_t fitME::doNoiseFit()
//-----------------------------------------------------------------------------------------------
{


if (htoFit->GetEntries()!=0) {
		
// Setting fit range and start values
		  Double_t fr[2];
		  Double_t sv[3], pllo[3], plhi[3];
		  fr[0]=htoFit->GetMean()-5*htoFit->GetRMS();
		  fr[1]=htoFit->GetMean()+5*htoFit->GetRMS();
		  
		  Int_t imax=htoFit->GetMaximumBin();
		  Double_t xmax=htoFit->GetBinCenter(imax);
		  Double_t ymax=htoFit->GetBinContent(imax);
		  Int_t i[2];
		  Int_t iArea[2];
		  
		  i[0]=htoFit->GetXaxis()->FindBin(fr[0]);
		  i[1]=htoFit->GetXaxis()->FindBin(fr[1]);
		  
		  iArea[0]=htoFit->GetXaxis()->FindBin(fr[0]);
		  iArea[1]=htoFit->GetXaxis()->FindBin(fr[1]);
		  Double_t AreaFWHM=htoFit->Integral(iArea[0],iArea[1],"width");
		  
		  sv[2]=AreaFWHM/(4*ymax);
		  sv[1]=xmax;
		  sv[0]=htoFit->Integral(i[0],i[1],"width");
		  
		  plhi[0]=1000000.0; plhi[1]=10.0; plhi[2]=10.;
		  pllo[0]=1.5 ; pllo[1]=0.1; pllo[2]=0.3;
		  Char_t FunName[100];
		  sprintf(FunName,"FitfcnLG_%s",htoFit->GetName());
		  TF1 *ffitold = (TF1*)gROOT->GetListOfFunctions()->FindObject(FunName);
		  if (ffitold) delete ffitold;
		  
		  gausFit = new TF1(FunName,Gauss,fr[0],fr[1],3);
		  gausFit->SetParameters(sv);
		  gausFit->SetParNames("Constant","GaussPeak","Sigma");
		  
		  for (Int_t i=0; i<3; i++) {
		    gausFit->SetParLimits(i,pllo[i],plhi[i]);
		  }
		  htoFit->Fit(gausFit,"R0");
		  
		  gausFit->SetRange(fr[0],fr[1]);
		  pGausS=gausFit->GetParameters();
		  epGausS=gausFit->GetParErrors();
		  
		  chi2GausS =langausFit->GetChisquare(); // obtain chi^2
		  nDofGausS = langausFit->GetNDF();// obtain ndf
		 
}
else {
		  pGausS[0]=-10; pGausS[1]=-10; pGausS[2]=-10;
		  epGausS[0]=-10; epGausS[1]=-10; epGausS[2]=-10;
		}
	
return htoFit->GetEntries();
}
