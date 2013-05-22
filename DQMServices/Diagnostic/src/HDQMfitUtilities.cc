#include "DQMServices/Diagnostic/interface/HDQMfitUtilities.h"

namespace HDQMUtil{
  //-----------------------------------------------------------------------------------------------
  double langaufun(double *x, double *par){
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
    double invsq2pi = 0.3989422804014;   // (2 pi)^(-1/2)
    double mpshift  = -0.22278298;       // Landau maximum location

    // Control constants
    double np = 100.0;      // number of convolution steps
    double sc =   5.0;      // convolution extends to +-sc Gaussian sigmas

    // Variables
    double xx;
    double mpc;
    double fland;
    double sum = 0.0;
    double xlow,xupp;
    double step;
    double i;


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
  int32_t langaupro(double *params, double &maxx, double &FWHM) {
    edm::LogInfo("fitUtility") << "inside langaupro " << std::endl;
    // Seaches for the location (x value) at the maximum of the 
    // Landau and its full width at half-maximum.
    //
    // The search is probably not very efficient, but it's a first try.

    double p,x,fy,fxr,fxl;
    double step;
    double l,lold,dl;
    int32_t i = 0;
    const int32_t MAXCALLS = 10000;
    const double dlStop = 1e-3; // relative change < .001

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
  double Gauss(double *x, double *par)
  {
    // The noise function: a gaussian

    double arg = 0;
    if (par[2]) arg = (x[0] - par[1])/par[2];

    double noise = par[0]*TMath::Exp(-0.5*arg*arg);
    return noise;
  }

}

//-----------------------------------------------------------------------------------------------
HDQMfitUtilities::HDQMfitUtilities():langausFit(0),gausFit(0){
  init();
}

//-----------------------------------------------------------------------------------------------
void HDQMfitUtilities::init(){
  pLanGausS[0]=0 ;   pLanGausS[1]=0;   pLanGausS[2]=0;   pLanGausS[3]=0;
  epLanGausS[0]=0;  epLanGausS[1]=0;  epLanGausS[2]=0;  epLanGausS[3]=0;
  pGausS[0]=0 ;   pGausS[1]=0;   pGausS[2]=0;
  epGausS[0]=0;  epGausS[1]=0;  epGausS[2]=0;
  pLanConv[0]=0;  pLanConv[1]=0;

  //if ( langausFit!=0 ) delete langausFit;
  //if ( gausFit!=0 ) delete gausFit;

  chi2GausS = -9999.;
  nDofGausS = -9999;
}

//-----------------------------------------------------------------------------------------------
HDQMfitUtilities::~HDQMfitUtilities(){
  if ( langausFit!=0 ) delete langausFit;
  if ( gausFit!=0 ) delete gausFit;
}

  
 
//-----------------------------------------------------------------------------------------------
double HDQMfitUtilities::doLanGaussFit(TH1F* htoFit){
  init();
 
  // if (htoFit->GetEntries()!=0) {
  // Check for the entries excluding over/underflows
  if (htoFit->Integral()!=0) {
     edm::LogInfo("fitUtility")<<"Fitting "<< htoFit->GetTitle() <<std::endl;
    // Setting fit range and start values
    double fr[2];
    double sv[4], pllo[4], plhi[4];
    fr[0]=0.5*htoFit->GetMean();
    fr[1]=3.0*htoFit->GetMean();
	      
    // (EM) parameters setting good for signal only 
    int32_t imax=htoFit->GetMaximumBin();
    double xmax=htoFit->GetBinCenter(imax);
    double ymax=htoFit->GetBinContent(imax);
    int32_t i[2];
    int32_t iArea[2];
	      
    i[0]=htoFit->GetXaxis()->FindBin(fr[0]);
    i[1]=htoFit->GetXaxis()->FindBin(fr[1]);
		  
    iArea[0]=htoFit->GetXaxis()->FindBin(fr[0]);
    iArea[1]=htoFit->GetXaxis()->FindBin(fr[1]);
    double AreaFWHM=htoFit->Integral(iArea[0],iArea[1],"width");
		  
    sv[1]=xmax;
    sv[2]=htoFit->Integral(i[0],i[1],"width");
    sv[3]=AreaFWHM/(4*ymax);
    sv[0]=sv[3];
		  
    plhi[0]=25.0; plhi[1]=200.0; plhi[2]=1000000.0; plhi[3]=50.0;
    pllo[0]=1.5 ; pllo[1]=10.0 ; pllo[2]=1.0      ; pllo[3]= 1.0;
		  
    Char_t FunName[100];
    sprintf(FunName,"FitfcnLG_%s",htoFit->GetName());  
		  
    langausFit = new TF1(FunName,HDQMUtil::langaufun,fr[0],fr[1],4);
    langausFit->SetParameters(sv);
    langausFit->SetParNames("Width","MP","Area","GSigma");
		  
    for (int32_t i=0; i<4; i++) {
      langausFit->SetParLimits(i,pllo[i],plhi[i]);
    }  
		  
    try{
      htoFit->Fit(langausFit,"R0");  // "R" fit in a range,"0" quiet fit
      
      langausFit->SetRange(fr[0],fr[1]);
      langausFit->GetParameters(pLanGausS);
      std::memcpy((void*) epLanGausS, (void*) langausFit->GetParErrors(), 4*sizeof(double));
      
      chi2GausS =langausFit->GetChisquare();  // obtain chi^2
      nDofGausS = langausFit->GetNDF();           // obtain ndf
      
      double sPeak, sFWHM;
      HDQMUtil::langaupro(pLanGausS,sPeak,sFWHM);
      pLanConv[0]=sPeak;
      pLanConv[1]=sFWHM;
      edm::LogInfo("fitUtility") << "langaupro:  max  " << sPeak << std::endl;
      edm::LogInfo("fitUtility") << "langaupro:  FWHM " << sFWHM << std::endl;
    }
    catch(cms::Exception& iException){
      edm::LogError("fitUtility") << "problem in fitting " << htoFit->GetTitle() << " \n\tDefault values of the parameters will be used";
      pLanGausS[0]=-9999; pLanGausS[1]=-9999; pLanGausS[2]=-9999; pLanGausS[3]=-9999;
      epLanGausS[0]=-9999; epLanGausS[1]=-9999; epLanGausS[2]=-9999; epLanGausS[3]=-9999;
      pLanConv[0]=-9999;   pLanConv[1]=-9999;   
      chi2GausS=-9999;  nDofGausS=-9999;    
    }
  }
  else {  
    pLanGausS[0]=-9999; pLanGausS[1]=-9999; pLanGausS[2]=-9999; pLanGausS[3]=-9999;
    epLanGausS[0]=-9999; epLanGausS[1]=-9999; epLanGausS[2]=-9999; epLanGausS[3]=-9999;
    pLanConv[0]=-9999;   pLanConv[1]=-9999;   
    chi2GausS=-9999;  nDofGausS=-9999;    
  }

  return htoFit->GetEntries();

}

  
//-----------------------------------------------------------------------------------------------
double HDQMfitUtilities::doGaussFit(TH1F* htoFit){
  init();
  // if (htoFit->GetEntries()!=0) {
  if (htoFit->Integral()!=0) {
    
    // Setting fit range and start values
    double fr[2];
    double sv[3], pllo[3], plhi[3];
    fr[0]=htoFit->GetMean()-5*htoFit->GetRMS();
    fr[1]=htoFit->GetMean()+5*htoFit->GetRMS();
		  
    int32_t imax=htoFit->GetMaximumBin();
    double xmax=htoFit->GetBinCenter(imax);
    double ymax=htoFit->GetBinContent(imax);
    int32_t i[2];
    int32_t iArea[2];
		  
    i[0]=htoFit->GetXaxis()->FindBin(fr[0]);
    i[1]=htoFit->GetXaxis()->FindBin(fr[1]);
		  
    iArea[0]=htoFit->GetXaxis()->FindBin(fr[0]);
    iArea[1]=htoFit->GetXaxis()->FindBin(fr[1]);
    double AreaFWHM=htoFit->Integral(iArea[0],iArea[1],"width");
		  
    sv[2]=AreaFWHM/(4*ymax);
    sv[1]=xmax;
    sv[0]=htoFit->Integral(i[0],i[1],"width");
		  
    plhi[0]=1000000.0; plhi[1]=10.0; plhi[2]=10.;
    pllo[0]=1.5 ; pllo[1]=0.1; pllo[2]=0.3;
    Char_t FunName[100];
    sprintf(FunName,"FitfcnLG_%s",htoFit->GetName());
    gausFit = new TF1(FunName,HDQMUtil::Gauss,fr[0],fr[1],3);
    gausFit->SetParameters(sv);
    gausFit->SetParNames("Constant","GaussPeak","Sigma");
		  
    for (int32_t i=0; i<3; i++) {
      gausFit->SetParLimits(i,pllo[i],plhi[i]);
    }
 
    try{
      htoFit->Fit(gausFit,"R0");
      
      gausFit->SetRange(fr[0],fr[1]);
      gausFit->GetParameters(pGausS);
      std::memcpy((void*) epGausS, (void*) gausFit->GetParErrors(), 3*sizeof(double));
      
      chi2GausS =langausFit->GetChisquare(); // obtain chi^2
      nDofGausS = langausFit->GetNDF();// obtain ndf
    }
    catch(cms::Exception& iException){
      edm::LogError("fitUtility") << "problem in fitting " << htoFit->GetTitle() << " \n\tDefault values of the parameters will be used";
      pGausS[0]=-9999; pGausS[1]=-9999; pGausS[2]=-9999;
      epGausS[0]=-9999; epGausS[1]=-9999; epGausS[2]=-9999;
      chi2GausS=-9999;  nDofGausS=-9999;    
    }
 		 
  }
  else {
    pGausS[0]=-9999; pGausS[1]=-9999; pGausS[2]=-9999;
    epGausS[0]=-9999; epGausS[1]=-9999; epGausS[2]=-9999;
    chi2GausS=-9999;  nDofGausS=-9999;    
  }
  
  return htoFit->GetEntries();
}

//-----------------------------------------------------------------------------------------------
double HDQMfitUtilities::getLanGaussPar(std::string s){
  if(s=="landau_width")
    return pLanGausS[0];
  else if(s=="mpv")
    return pLanGausS[1];
  else if(s=="area")
    return pLanGausS[2];
  else if(s=="gauss_sigma")
    return pLanGausS[3];
  else
    return -99999;
}

double HDQMfitUtilities::getLanGaussParErr(std::string s){
  if(s=="landau_width")
    return epLanGausS[0];
  else if(s=="mpv")
    return epLanGausS[1];
  else if(s=="area")
    return epLanGausS[2];
  else if(s=="gauss_sigma")
    return epLanGausS[3];
  else
    return -99999;
}

double HDQMfitUtilities::getLanGaussConv(std::string s) {
  if(s=="mpv")
    return pLanConv[0];
  else if(s=="fwhm")
    return pLanConv[1];
  else
    return -99999;
}

double HDQMfitUtilities::getGaussPar(std::string s) {
  if(s=="area")
    return pGausS[0];
  else if(s=="mean")
    return pGausS[1];
  else if(s=="sigma")
    return pGausS[2];
  else
    return -99999;
}

double HDQMfitUtilities::getGaussParErr(std::string s) {
  if(s=="area")
    return epGausS[0];
  else if(s=="mean")
    return epGausS[1];
  else if(s=="sigma")
    return epGausS[2];
  else
    return -99999;
}
