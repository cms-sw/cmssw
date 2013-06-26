#ifndef TMatacq_H
#define TMatacq_H

#include "TObject.h"

#define NMAXSAMP 100
#define NSPARAB  16

class TMatacq: public TObject 
{

 private:	

  int fNsamples;
  int fNum_samp_bef_max;
  int fNum_samp_aft_max;

  int firstsample,lastsample,samplemax,presample,endsample;
  int bing[101];
  double nsigcut;
  double level1,level2,level3;
  double bong[NMAXSAMP];
  double t[NSPARAB],val[NSPARAB];
  double fv1[NSPARAB],fv2[NSPARAB],fv3[NSPARAB];
  double bl,sigbl,val_max;
  double ampl,timeatmax;
  double pkval,sigpkval;
  double trise;
  double width20, width50, width80;
  double meantrise,sigtrise;

  int nevmtq0,nevmtq1,nevlasers;
  int status[1200];
  double comp_trise[1200],comp_peak[1200];
  double slidingmean;
  int nslide;

  double interpolate(double);

 public:
  // Default Constructor, mainly for Root
  TMatacq(int,int,int,int,int,int,int,int,int,int,int);

  // Destructor: Does nothing
  virtual ~TMatacq();

  // Initialize 
  void init();

  int rawPulseAnalysis(Int_t, Double_t*); // GHM
  int findPeak();
  int doFit();
  int compute_trise();

  void enterdata(Int_t);
  int countBadPulses(Int_t);
  void printmatacqData(Int_t,Int_t,Int_t);
  void printitermatacqData(Int_t,Int_t,Int_t);

  int getPt1() {return firstsample;}
  int getPt2() {return lastsample;}
  int getPtm() {return samplemax;}

  double getBaseLine() {return bl;}
  double getsigBaseLine() {return sigbl;}

  double getTimpeak() {return pkval;}
  double getsigTimpeak() {return sigpkval;}

  double getAmpl() {return ampl;}
  double getTimax() {return timeatmax;}

  double getTrise() {return trise;}
  double getFwhm() {return width50;}
  double getWidth20() {return width20;}
  double getWidth80() {return width80;}
  double getSlide() {return slidingmean;}

  //  ClassDef(TMatacq,1)
};

#endif



