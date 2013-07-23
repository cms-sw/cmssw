/* 
 *  \class TMatacq
 *
 *  $Date: 2012/02/09 10:08:10 $
 *  \author: Patrice Verrecchia - CEA/Saclay
 */
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMatacq.h>

#include <iostream>
#include <math.h>
#include "TVectorD.h"

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
}

// Destructor
TMatacq::~TMatacq()
{ 
}

int TMatacq::rawPulseAnalysis(Int_t Nsamp, Double_t *adc)  // GHM
{
  using namespace std;

  //  std::cout << "Entering rawPulseAnalysis" << std::endl;

  int k,ithr;
  double dsum=0.,dsum2=0.;
  
  //  std::cout << "calling init" << std::endl;
  init();
  //  std::cout << ".......done" << std::endl;

  if(Nsamp != fNsamples) {
      printf("found different number of samples fNsamples=%d Nsamp=%d\n",fNsamples,Nsamp);
      return 100;
  }

  for(k=0;k<presample;k++) {
       dsum+= adc[k];
       dsum2+= adc[k]*adc[k];
  }
  bl=dsum/((double) presample);
  double ss= (dsum2/((double) presample)-bl*bl);
  if(ss<0.) ss=0.;
  sigbl=sqrt(ss);
  for(ithr=0,k=presample;k<endsample;k++) {
	if(adc[k] > (bl+nsigcut*sigbl) && ithr == 0) {
            ithr=1; firstsample=k;
	}
  }

  if(ithr == 0) return 101;

  for(ithr=0,k=firstsample;k<Nsamp;k++) {
       if(adc[k] < (bl+nsigcut*sigbl) && ithr == 0) {
             ithr=1; lastsample=k;
       }
  }
  if(ithr == 0) lastsample= Nsamp;

  if(lastsample > firstsample+NMAXSAMP) lastsample= firstsample+NMAXSAMP;

  val_max=0.; samplemax=0;
  for (Int_t is=firstsample;is<lastsample;is++) {
       bong[is-firstsample]= adc[is] - bl;
       if(bong[is-firstsample] > val_max) {
	   val_max= bong[is-firstsample]; samplemax=is;
       }
  }
  if(samplemax == 0) return 103;
  if(samplemax > lastsample) return 104;
  if(samplemax < firstsample) return 105;

  
  int endslide=samplemax -nslide;
  int beginslide=nslide;
  int islidingmean=0;
  slidingmean=0.0;
  
  for(int i=beginslide;i<endslide;i++) {
    slidingmean+= adc[i];
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
	    return 302;
       } else if(jfind == 1) {
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
  for(Int_t k=0;k<kmax;k++)
      if(0. < bong[k] && bong[k] < amplx) {
          bin_low=k;
      }
  if(bin_low == 0) return -301.;
  int bin_high=0;
  for(Int_t k=kmax;k>=0;k--)
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

void TMatacq::enterdata(Int_t anevt)
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
  for(Int_t i=0;i<nevmtq0+nevmtq1;i++) {
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
