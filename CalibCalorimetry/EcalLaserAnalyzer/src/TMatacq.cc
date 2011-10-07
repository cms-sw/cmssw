/* 
 *  \class TMatacq
 *
 *  $Date: 2010/04/12 14:17:13 $
 *  \author: Patrice Verrecchia - CEA/Saclay
 */
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMatacq.h>

#include <iostream>
#include <math.h>
#include "TVectorD.h"
#include "TF1.h"
#include "TH1D.h"
#include "TVirtualFFT.h"

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMarkov.h>

using namespace std;
//ClassImp(TMatacq)

void TMatacq::init()
{
  for(int k=0;k<NMAXSAMP;k++)
       bong[k]=0.;

  for(int k=0;k<=100;k++)
       bing[k]=0;

  for(int k=0;k<NSPARAB;k++)
       t[k]= (double) k;

  return ;
}

// Constructor...
TMatacq::TMatacq(int Ntot, int Nsamp1, int Nsamp2, int cut, int Nbef, int Naft, int niv1, int niv2, int niv3, int nevl, int NSlide)
{
  fNsamples= Ntot;
  presample= Nsamp1;
  endsample= Nsamp2;
  nsigcut= (double) cut;
  fNum_samp_bef_max= Nbef;
  fNum_samp_aft_max= Naft;
  level1= ((double) niv1)/100.;
  level2= ((double) niv2)/100.;
  level3= ((double) niv3)/100.;
  nevlasers= nevl;
  slidingmean=0.0;
  nslide=NSlide;
  for(int k=0;k<nevlasers;k++)
       status[k]=0;
  for(int k=0;k<nevlasers;k++)
       status[k+nevlasers]=0;

  nevmtq0=0; nevmtq1=0;
  
  // Define TF1 and TH1 for doFit2
  
  double max=double(NSAMP);
  double min=0.0;
  double fmax=1000.; // freq max = 1GHz when sampling at 1ns
  flandau = new TF1("flandau","landau",min,max);
  fg1     = new TF1("fg1","[0]*exp(-(x-[1])*(x-[1])/2./[2]/[2])",0.,fmax);
  fexp    = new TF1("fexp","expo",0.,MATACQ_LENGTH_MAX);
  fpol1   = new TF1("fpol1","pol1",0.,fmax);
  fpol2   = new TF1("fpol2","[0]+[1]*(x-[2])*(x-[2])",0.,MATACQ_LENGTH_MAX);
  hdph    = new TH1D("hdph","hdph",FFT_SIZE-1,0.,fmax);
  hmod    = new TH1D("hmod","hmod",FFT_SIZE-1,0.,fmax);
  htmp = new TH1D("htmp","htmp",NSAMP, min, max);
  int size = FFT_SIZE;
  fft_f = TVirtualFFT::FFT(1, &size, "C2CF M K");
  fft_b = TVirtualFFT::FFT(1, &size, "C2CB M K");
  
}

// Destructor
TMatacq::~TMatacq()
{ 
}

int TMatacq::rawPulseAnalysis(int Nsamp, double *adc)  // GHM
{
  using namespace std;

  double adcprim[NSAMP];
  //  cout << "Entering rawPulseAnalysis" << endl;

  int k,ithr;
  double dsum=0.,dsum2=0.;
  double dtest=0., dtest2=0.;

  //  cout << "calling init" << endl;
  init();
  //  cout << ".......done" << endl;

  if(Nsamp != fNsamples) {
      printf("found different number of samples fNsamples=%d Nsamp=%d\n",fNsamples,Nsamp);
      return 100;
  }
  ped_cyc=new double[20];
  ped_cycprim=new double[20];
  
  int is=(2559-440);
  for(int i=0; i<fNsamples; i++){
    adcprim[is]=adc[i];
    if(++is>=MATACQ_LENGTH_MAX)is-=MATACQ_LENGTH_MAX;  
  }

  // Compute cyclic pedestals on first 200 samples :
  
  for(int i=0; i<20; i++)ped_cycprim[i]=0.;
  for(int i=100; i<300; i++) ped_cycprim[i%20]+=adcprim[i];
  for(int i=0; i<20; i++)ped_cycprim[i]/=10.;

  is=(2559-440);
  for(int i=0; i<20; i++){
    int iprim=i+is;
    iprim=iprim%20;
    ped_cyc[i]=ped_cycprim[iprim];
  }  

  laser_qmax=0.;
  
  // Remove cyclic pedestal: 
 
  for(int i=0; i<fNsamples; i++){ 
    fadc[i]=adc[i]-ped_cyc[i%20]; 
    
    if(i<presample){ 
      dsum+= ped_cyc[i%20];
      dsum2+= ped_cyc[i%20]*ped_cyc[i%20];
      dtest+=fadc[i];
      dtest2+=fadc[i]*fadc[i]; 
    }
    if(fabs(fadc[i])>fabs(laser_qmax))
      {
	laser_qmax=fadc[i];
	laser_imax=i;
      }
    if(fabs(laser_qmax)<1. || laser_imax<FFT_START)
      {    
	laser_qmax=1000.;
	laser_imax=1450;
      }     
  }
  
  //bl=dsum/((double) presample);
  //double ss= (dsum2/((double) presample)-bl*bl);
  //if(ss<0.) ss=0.;
  //sigbl=sqrt(ss);

  bl=dtest/((double) presample);
  double ss= (dtest2/((double) presample)-bl*bl);
  sigbl=sqrt(ss);

  for(ithr=0,k=presample;k<endsample;k++) {
    //cout<<" ithr="<<ithr<<" k="<< k<<" fadc[k]="<<fadc[k]<< "  "<<nsigcut*sigbl<<" "<<presample<<" "<< endsample<< endl;
    
    if(fadc[k] > nsigcut*sigbl && ithr == 0) {
      ithr=1; firstsample=k;
    }
  }
  
  if(ithr == 0){
    //cout<<" Bad RawPulseAnalysis 101"<< endl;
    return 101;
  }
  for(ithr=0,k=firstsample;k<Nsamp;k++) {
    if(fadc[k] < nsigcut*sigbl && ithr == 0) {
      ithr=1; lastsample=k;
    }
  }
  if(ithr == 0) lastsample= Nsamp;

  if(lastsample > firstsample+NMAXSAMP) lastsample= firstsample+NMAXSAMP;

  val_max=0.; samplemax=0;
  for (int is=firstsample;is<lastsample;is++) {
    bong[is-firstsample]= fadc[is] ;
    if(bong[is-firstsample] > val_max) {
      val_max= bong[is-firstsample]; samplemax=is;
    }
  }
  if(samplemax == 0){
    //    cout<<" Bad RawPulseAnalysis 103"<< endl;
    return 103;
  }
  if(samplemax > lastsample){
    //    cout<<" Bad RawPulseAnalysis 104 "<<lastsample<<" " <<firstsample<< endl;
    return 104;
  }
  if(samplemax < firstsample){
    //    cout<<" Bad RawPulseAnalysis 105 "<<lastsample<<" " <<firstsample<< endl;
    return 105;
  }
  
  
  int endslide=samplemax -nslide;
  int beginslide=nslide;
  int islidingmean=0;
  slidingmean=0.0;
  
  for(int i=beginslide;i<endslide;i++) {
    slidingmean+= fadc[i];
    islidingmean+=1;
  }
  if( islidingmean!=0) slidingmean/=double(islidingmean);
  
  return 0;
}
int TMatacq::findPeak()
{
   int k; int nbinf=0; int jfind=0;
   int nbsup= 0;
   double thres= val_max*level1;

   for(k=0,jfind=0;k<NMAXSAMP;k++) {
       if(jfind == 0) {
           if(bong[k] > thres) {
	       nbinf=k; jfind=1;
	   }
       }
   }
   if(jfind == 0) nbinf=0;

   for(k=NMAXSAMP,jfind=0;k>nbinf;k--) {
       if(jfind == 0) {
            if(bong[k] > thres) {
	        nbsup=k; jfind=1;
	    }
       }
   }
   if(nbsup == 0) nbsup=nbinf;

   double sumpkval=1.;
   pkval= 0.;
   sigpkval=0.5;
   if(nbsup == nbinf) {
     //     cout<< " Bad Peak 301: "<<nbsup<<" "<<nbinf<< endl;
       return 301;
   } else {
       if(nbinf > 4) nbinf-=3;
       else nbinf=1;
       if(nbsup < NMAXSAMP-4) nbsup+=3;
       else nbsup=NMAXSAMP;

       for(k=0;k<nbinf;k++)
	    bong[k]=0.;
       for(k=nbsup;k<NMAXSAMP;k++)
            bong[k]=0.;

       for(k=0,jfind=0;k<NMAXSAMP;k++) {
	    if(bong[k] > 0.) jfind++;
       }
       if(jfind == 0) {
	 
	 //	 cout<< " Bad Peak 302 "<<endl;
	    return 302;
       } else if(jfind == 1) {
	 //	 cout<< " Bad Peak 303 "<<endl;
	    return 303;
       } else {

	    for(k=0;k<NMAXSAMP;k++) {
              if(k < 100) 
	          bing[k+1]= (int) bong[k];
	    }

            TMarkov *peak = new TMarkov();

            peak -> peakFinder(&bing[0]);
            pkval= peak -> getPeakValue(0);
            sigpkval= peak -> getPeakValue(1);

            delete peak;

            sumpkval= 0.0;
        
            if(sumpkval > 1000.) 
	         sumpkval=10.;

            pkval+= (firstsample -1);
       }
   }

   return 0;
}

int TMatacq::doFit()
{
 

  int testneg=0;
  ampl=0.; timeatmax=0.; 
  int heresamplemax= samplemax-firstsample;

  int beginfit= heresamplemax - fNum_samp_bef_max;
  int endfit= heresamplemax +  fNum_samp_aft_max;

  int nval= endfit-beginfit +1;
  if(nval != NSPARAB) return 201;
  for(int kn=beginfit;kn<=endfit;kn++) {
      if(bong[kn] <= 0) testneg=1;
  }
  if(testneg == 1) return 202;

  for(int i=0;i<nval;i++) {
     val[i]= bong[beginfit+i];
     fv1[i]= 1.;
     fv2[i]= (double) (i);
     fv3[i]= ((double)(i))*((double)(i));
  }

  TVectorD y(nval,val);
  TVectorD f1(nval,fv1);
  TVectorD f2(nval,fv2);
  TVectorD f3(nval,fv3);
 
  double bj[3];
  bj[0]= f1*y; bj[1]=f2*y; bj[2]= f3*y;
  TVectorD b(3,bj);

  double aij[9];
  aij[0]= f1*f1; aij[1]=f1*f2; aij[2]=f1*f3;
  aij[3]= f2*f1; aij[4]=f2*f2; aij[5]=f2*f3;
  aij[6]= f3*f1; aij[7]=f3*f2; aij[8]=f3*f3;
  TMatrixD a(3,3,aij);
 
  double det1;
  a.InvertFast(&det1);

  TVectorD c= a*b;

  double *par= c.GetMatrixArray();
  if(par[2] != 0.) {
    timeatmax= -par[1]/(2.*par[2]);
    ampl= par[0]-par[2]*(timeatmax*timeatmax);
  }

  if(ampl <= 0.) {
      ampl=bong[heresamplemax];
      return 203;
  }

  if((int)timeatmax > NSPARAB)
      return 204;
  if((int)timeatmax < 0)
      return 205;

  timeatmax+= (beginfit + firstsample);

  int tplus[3], tminus[3];
  double xampl[3];
  xampl[0]=0.2*ampl;
  xampl[1]=0.5*ampl;
  xampl[2]=0.8*ampl;
  
  int hitplus[3];
  int hitminus[3];
  double width[3];
  
  for(int i=0;i<3;i++){
    hitplus[i]=0;
    hitminus[i]=0;
    width[i]=0.0;
    tplus[i]=firstsample+lastsample;
    tminus[i]=0; 
  }
  
  // calculate first estimate of half width
  int im = heresamplemax;
  int iplusbound = firstsample+lastsample-im;
  
  for(int j=0;j<3;j++){
    
    for(int i=1;i<im;i++){
      
      
      if (bong[im-i]<xampl[j] && bong[im-i+1]>xampl[j]) {
	tminus[j]=im-i;
	hitminus[j]++;
	i=im;
      }
    }
    

    for(int i=0;i<iplusbound;i++){

      if (bong[im+i]>xampl[j] && bong[im+i+1]<xampl[j]){
	tplus[j]=im+i;
	hitplus[j]++;
	i=iplusbound;
      }
    }
    
    // interpolate to get a better estimate
    
    double slopeplus  = double( bong[tplus[j] +1] - bong[tplus[j] ] );
    double slopeminus = double( bong[tminus[j]+1] - bong[tminus[j]] );
  
  
    double timeminus=double(tminus[j])+0.5;
    if(slopeminus!=0) timeminus=tminus[j]+(xampl[j]-double(bong[tminus[j]]))/slopeminus;
    
    
    double timeplus=double(tplus[j])+0.5;
    if(slopeplus!=0) timeplus=tplus[j]+(xampl[j]-double(bong[tplus[j]]))/slopeplus;
    
    width[j]=timeplus-timeminus;
    
  }
  
  width20=width[0];
  width50=width[1];
  width80=width[2];
  
  return 0;
}

int TMatacq::doFit2()
{
  ampl=0.; timeatmax=0.;  
  double rex[FFT_SIZE],imx[FFT_SIZE],rey[FFT_SIZE],imy[FFT_SIZE];
  double mod[FFT_SIZE],phase[FFT_SIZE];
  double pi=asin(1.)*2.;
  float fadc2[MATACQ_LENGTH_MAX];
  float fadcprim[MATACQ_LENGTH_MAX];
  
  int is=(2559-440);
  for(int i=0; i<MATACQ_LENGTH_MAX;i++){
    fadcprim[is]=fadc[i];
    if(++is>=MATACQ_LENGTH_MAX)is-=MATACQ_LENGTH_MAX;
  }

  // Remove noise in the frequency domain : apply FFT, clean and reverse FFT
  //=======================================

  htmp->Reset();

  for(int i=0; i<fNsamples;i++){
    htmp->SetBinContent(i+1,fadcprim[i]);
    // cout<<" Begin loop TMatacq : " << i<<"  " <<fadc[i]<< endl;
  }

  htmp->Fit(fexp,"NQ","W",2000.,2500.);
 

  for(int i=0; i<FFT_SIZE; i++) 
  {    
    if(i+FFT_START<2300)
      rex[i]=fadcprim[FFT_START+i];
    else 
      rex[i]=fexp->Eval((double)(i+FFT_START));
    imx[i]=0.;
    rey[i]=0.;
    imy[i]=0.;
  }    
  fft_f->SetPointsComplex(rex, imx);
  fft_f->Transform();
  fft_f->GetPointsComplex(rey, imy);
  for(int i=0; i<FFT_SIZE; i++) 
  {    
    rey[i]/=FFT_SIZE;
    imy[i]/=FFT_SIZE;
    mod[i]=sqrt(rey[i]*rey[i]+imy[i]*imy[i]);
    phase[i]=atan2(imy[i],rey[i]);
  }    

// Extrapolate the frequency spectrum from a region where we have no (not yet) noise f<19 MHz)
// The modulus seems to have a gaussian shape and the phase increase linearly.


  hmod->Reset();
  for(int i=1; i<=FFT_SIZE/2.; i++) hmod->SetBinContent(i-1+1,mod[i]); // In the FFT output array, the first bin has a special meaning (DC content)
  fg1->SetParameter(0,mod[1]*2.);                                      // So, start at 1 and shift all indices accordingly
  fg1->SetParameter(1,-10.);
  fg1->SetParameter(2,20.);
  hmod->Fit(fg1,"NQ","W",8.,19.);

  double ph_old=phase[0];
  hdph->Reset();
  for(int i=1; i<=FFT_SIZE/2; i++)
  {
    int j=i-1;
    double ph=phase[i];
    double dph=ph_old-ph;
    if(dph<0.)dph+=2*pi;
    if(dph>2.*pi)dph-=2*pi;
    ph_old=ph;
    hdph->SetBinContent(j+1,dph);
  }

  hdph->Fit(fpol1,"NQ","W",5.,19.); 
  
  // printf("Fit phase step : %f rad/MHz + %f rad/MHz2\n",fpol1->GetParameter(0),fpol1->GetParameter(1));

  double ph_step=fpol1->Eval(20.);
  //printf("Average phase step : %f rad/MHz\n",ph_step);

  double cor_ref=-1.;
  for(int i=0; i<=FFT_SIZE/2; i++)
  {
    int j=i-1;
    double freq=hmod->GetBinCenter(j+1);
    
    if(freq>=19.)
    {
      double cor_mod=fg1->Eval(freq);
      //cout <<"freq="<< freq<<" cor_mod="<< cor_mod<<" cor_ref=" << cor_ref<< endl;
      if(cor_ref<0.)cor_ref=mod[i]/cor_mod;
      mod[i]=cor_mod*cor_ref;
      phase[i]=phase[i-1]-ph_step;
      if(phase[i]<-pi)phase[i]+=2*pi;
      if(phase[i]>+pi)phase[i]-=2*pi;
      rey[i]=mod[i]*cos(phase[i]);
      imy[i]=mod[i]*sin(phase[i]);
      mod[FFT_SIZE-i]=mod[i];
      phase[FFT_SIZE-i]=-phase[i];
      rey[FFT_SIZE-i]=rey[i];
      imy[FFT_SIZE-i]=-imy[i];
    }
  }
  fft_b->SetPointsComplex(rey, imy);
  fft_b->Transform();
  fft_b->GetPointsComplex(rex, imx);

// Overwrite pulse with filtered one :
  
  for(int i=0; i<MATACQ_LENGTH_MAX;i++)fadc2[i]=fadcprim[i];
  for(int i=0; i<FFT_SIZE && i<MATACQ_LENGTH_MAX; i++) fadc2[FFT_START+i]=rex[i];
  for(int i=0; i<FFT_SIZE && i<MATACQ_LENGTH_MAX; i++) htmp->SetBinContent(FFT_START+i+1,rex[i]);

//Recompute maximum and max position :
  laser_qmax=0.;
  for(int i=0; i<MATACQ_LENGTH_MAX; i++)
  {
    if(fabs(fadc2[i])>fabs(laser_qmax))
    {
      laser_qmax=fadc2[i];
      laser_imax=i;
    }
  }
  if(fabs(laser_qmax)<1. || laser_imax<FFT_START)
  {
    laser_qmax=1000.;
    laser_imax=1450;
  }

// Try a pol2 fit on max +-5 samples :
  laser_tmax=laser_imax;
  double fit_window=5.;
  fpol2->SetParameter(0,(double)laser_qmax);
  fpol2->SetParameter(1,0.);
  fpol2->SetParameter(2,(double)laser_tmax);
  htmp->Fit(fpol2,"Q","",laser_tmax-fit_window,laser_tmax+fit_window+1.); 
  laser_tmax=fpol2->GetParameter(2);
  laser_qmax=fpol2->GetParameter(0);

  // Compute w80, w20 and w50 :
  //=========================
 
  int nbin=fNsamples;
  int imin50=0;
  int imin80=0;
  int imin20=0;
  int imax50=0;
  int imax80=0;
  int imax20=0;

  for(int i=0;i<nbin;i++)
    {
      if(fabs(fadc2[i])>fabs(laser_qmax)*0.50 && imin50==0) imin50=i;
      if(fabs(fadc2[i])>fabs(laser_qmax)*level3 && imin80==0) imin80=i;
      if(fabs(fadc2[i])>fabs(laser_qmax)*level2 && imin20==0) imin20=i;
      if(fabs(fadc2[i])<fabs(laser_qmax)*0.50 && imin50!=0 && imax50==0) imax50=i;
      if(fabs(fadc2[i])<fabs(laser_qmax)*level3 && imin80!=0 && imax80==0) imax80=i;
      if(fabs(fadc2[i])<fabs(laser_qmax)*level2 && imin20!=0 && imax20==0) imax20=i;
    }
  
  double f20=0., f80=0., f50=0.;
  if(imin20>0)f20=(double)imin20-(fadc2[imin20]-level2*laser_qmax)/
		(fadc2[imin20]-fadc2[imin20-1]);
  if(imin80>0)f80=(double)imin80-(fadc2[imin80]-level3*laser_qmax)/
		(fadc2[imin80]-fadc2[imin80-1]);
  if(imin50>0)f50=(double)imin50-(fadc2[imin50]-0.50*laser_qmax)/
		(fadc2[imin50]-fadc2[imin50-1]);
  
  double g20=(double)nbin, g80=(double)nbin, g50=(double)nbin;
  if(imax20<nbin)g20=(double)imax20-(fadc2[imax20]-level2*laser_qmax)/
		   (fadc2[imax20]-fadc2[imax20-1]);
  if(imax80<nbin)g80=(double)imax80-(fadc2[imax80]-level3*laser_qmax)/
		   (fadc2[imax80]-fadc2[imax80-1]);
  if(imax50<nbin)g50=(double)imax50-(fadc2[imax50]-0.50*laser_qmax)/
		   (fadc2[imax50]-fadc2[imax50-1]);

  is=(2559-440);
  ampl=laser_qmax;
  timeatmax=laser_tmax-double(is);

  // FIXME: check this...
  if(timeatmax<0) timeatmax+=double(MATACQ_LENGTH_MAX);
  if(timeatmax>=MATACQ_LENGTH_MAX) timeatmax-=double(MATACQ_LENGTH_MAX);
  
  if(fabs(ampl)<10.)
    {
      //printf("Laser amplitude too low, skip event\n");
      return(-1);
    }
  
  width20=g20-f20;
  width50=g50-f50;
  width80=g80-f80;

  return 0;
}

int TMatacq::compute_trise()
{
  int error;
  trise= 0.;

  double t20= interpolate(ampl*level2);
  if(t20 < 0.) {
    error= (int) -t20;
    return error;
  }
  double t80= interpolate(ampl*level3);
  if(t80 < 0.) {
    error= (int) -t80;
    return error;
  }
  trise= t80 - t20;

  return 0;
}





double TMatacq::interpolate(double amplx)
{
  double T;
  int kmax= (int) pkval - firstsample;

  int bin_low=0;
  for(int k=0;k<kmax;k++)
      if(0. < bong[k] && bong[k] < amplx) {
          bin_low=k;
      }
  if(bin_low == 0) return -301.;
  int bin_high=0;
  for(int k=kmax;k>=0;k--)
      if(bong[k] > amplx) {
          bin_high=k;
      }
  if(bin_high == 0) return -302.;
  if(bin_high < bin_low) return -303.;


  if(bin_low == bin_high) {
      T= (double) bin_high;
  } else {
      double slope= (bong[bin_high]-bong[bin_low])/((double) (bin_high-bin_low));
      T= (double) bin_high - (bong[bin_high] - amplx)/slope;
  }

  return T;
}

void TMatacq::enterdata(int anevt)
{
  if(anevt < 2*nevlasers) {
      if(anevt < nevlasers) {
          nevmtq0++;
          status[nevmtq0-1]= anevt;
          comp_trise[nevmtq0-1]= trise;
          comp_peak[nevmtq0-1]= pkval;
      } else {
          nevmtq1++;
          status[nevmtq0+nevmtq1-1]= anevt;
          comp_trise[nevmtq0+nevmtq1-1]= trise;
          comp_peak[nevmtq0+nevmtq1-1]= pkval;
      }
  } else {
      if(anevt < 3*nevlasers) {
          nevmtq0++;
          status[nevmtq0-1]= anevt;
          comp_trise[nevmtq0-1]= trise;
          comp_peak[nevmtq0-1]= pkval;
      } else {
          nevmtq1++;
          status[nevmtq0+nevmtq1-1]= anevt;
          comp_trise[nevmtq0+nevmtq1-1]= trise;
          comp_peak[nevmtq0+nevmtq1-1]= pkval;
      }
  }
}

void TMatacq::printmatacqData(int gRunNumber, int color, int timestart)
{
     FILE *fmatacq;
     char filename[80];
     int i;
     double ss;
     sprintf(filename,"runMatacq%d.pedestal",gRunNumber);
     fmatacq = fopen(filename, "w");
     if(fmatacq == NULL) printf("Error while opening file : %s\n",filename);

     double sumtrise=0.; double sumtrise2=0.;
     int timestop= timestart+3;
     double mintrise=10000.;
     double maxtrise=0.;
     for(i=0;i<nevmtq0;i++) {
       if(comp_trise[i] > maxtrise) {
	   maxtrise=comp_trise[i];
       }
       if(comp_trise[i] < mintrise) {
	   mintrise= comp_trise[i];
       }
       sumtrise+=comp_trise[i];
       sumtrise2+=comp_trise[i]*comp_trise[i];
     }
     meantrise= sumtrise/((double) nevmtq0);
     ss= (sumtrise2/((double) nevmtq0) - meantrise*meantrise);
     if(ss < 0.) ss=0.;
     sigtrise=sqrt(ss);
     fprintf(fmatacq, "%d %d %d %d %f %f %f %f\n",
           nevmtq0,color,timestart,timestop,meantrise,sigtrise,mintrise,maxtrise);

     sumtrise=0.; sumtrise2=0.;
     mintrise=10000.;
     maxtrise=0.;
     for(i=nevmtq0;i<nevmtq0+nevmtq1;i++) {
       if(comp_trise[i] > maxtrise) {
	   maxtrise=comp_trise[i];
       }
       if(comp_trise[i] < mintrise) {
	   mintrise= comp_trise[i];
       }
       sumtrise+=comp_trise[i];
       sumtrise2+=comp_trise[i]*comp_trise[i];
     }
     meantrise= sumtrise/((double) nevmtq1);
     ss= (sumtrise2/((double) nevmtq1) - meantrise*meantrise);
     if(ss < 0.) ss=0.;
     sigtrise=sqrt(ss);
     fprintf(fmatacq, "%d %d %d %d %f %f %f %f\n",
           nevmtq1,color,timestart,timestop,meantrise,sigtrise,mintrise,maxtrise);

     int iret=fclose(fmatacq);
     printf(" Closing file : %d\n",iret);
}

int TMatacq::countBadPulses(int gRunNumber)
{
  FILE *fmatacq;
  char filename[80];
  sprintf(filename,"badevtsMatacq%d.dat",gRunNumber);
  fmatacq = fopen(filename, "w");
  if(fmatacq == NULL) printf("Error while opening file : %s\n",filename);

  int nevbad=0;
  for(int i=0;i<nevmtq0+nevmtq1;i++) {
       if(comp_trise[i] < meantrise - 3.*sigtrise) {
	   nevbad++;
	   fprintf(fmatacq,"%d \n",status[i]);
           status[i]=-1;
       }
       if(comp_trise[i] > meantrise + 3.*sigtrise) {
	   nevbad++;
           fprintf(fmatacq,"%d \n",status[i]);
           status[i]=-1;
       }
  }

  int iret=fclose(fmatacq);
  printf(" Closing file : %d\n",iret);
  return nevbad;
}

void TMatacq::printitermatacqData(int gRunNumber, int color, int timestart)
{
     FILE *fmatacq;
     char filename[80];
     int i;
     double ss;
     sprintf(filename,"runiterMatacq%d.pedestal",gRunNumber);
     fmatacq = fopen(filename, "w");
     if(fmatacq == NULL) printf("Error while opening file : %s\n",filename);

     int nevmtqgood=0;
     double sumtrise=0.; double sumtrise2=0.;
     int timestop= timestart+3;
     double mintrise=10000.;
     double maxtrise=0.;
     for(i=0;i<nevmtq0;i++) {
       if(status[i] >= 0) {
	   nevmtqgood++;
           if(comp_trise[i] > maxtrise) {
	       maxtrise=comp_trise[i];
           }
           if(comp_trise[i] < mintrise) {
	       mintrise= comp_trise[i];
           }
           sumtrise+=comp_trise[i];
           sumtrise2+=comp_trise[i]*comp_trise[i];
       }
     }
     meantrise= sumtrise/((double) nevmtqgood);
     ss= (sumtrise2/((double) nevmtqgood) - meantrise*meantrise);
     if(ss < 0.) ss=0.;
     sigtrise=sqrt(ss);
     fprintf(fmatacq, "%d %d %d %d %f %f %f %f\n",
           nevmtqgood,color,timestart,timestop,meantrise,sigtrise,mintrise,maxtrise);

     nevmtqgood=0;
     sumtrise=0.; sumtrise2=0.;
     mintrise=10000.;
     maxtrise=0.;
     for(i=nevmtq0;i<nevmtq0+nevmtq1;i++) {
       if(status[i] >= 0) {
	   nevmtqgood++;
           if(comp_trise[i] > maxtrise) {
	       maxtrise=comp_trise[i];
           }
           if(comp_trise[i] < mintrise) {
	       mintrise= comp_trise[i];
           }
           sumtrise+=comp_trise[i];
           sumtrise2+=comp_trise[i]*comp_trise[i];
       }
     }
     meantrise= sumtrise/((double) nevmtqgood);
     ss= (sumtrise2/((double) nevmtqgood) - meantrise*meantrise);
     if(ss < 0.) ss=0.;
     sigtrise=sqrt(ss);
     fprintf(fmatacq, "%d %d %d %d %f %f %f %f\n",
           nevmtqgood,color,timestart,timestop,meantrise,sigtrise,mintrise,maxtrise);

     int iret=fclose(fmatacq);
     printf(" Closing file : %d\n",iret);
}

