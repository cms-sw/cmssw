#ifndef TMatacq_H
#define TMatacq_H

#define NMAXSAMP 100
#define NSAMP 2560
#define NSPARAB  16
#define MATACQ_LENGTH_MAX 2560
#define FFT2_SIZE   2048  // Number of bins used for FFT
#define FFT_SIZE   1048  // Number of bins used for FFT
#define FFT_START  850   // Keep signal starting at 850 ns


class TH1D;
class TF1;
class TVirtualFFT;
class TMatacq
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
  double fadc[NSAMP];
  int nevmtq0,nevmtq1,nevlasers;
  int status[1200];
  double comp_trise[1200],comp_peak[1200];
  double slidingmean;
  int nslide;
  double laser_qmax;
  double laser_tmax;
  int laser_imax;
  double* ped_cyc;
  double* ped_cycprim;

  TF1  *flandau;
  TF1  *flc;
  TH1D *htmp;
  
  TF1 *fg1;
  TF1 *fexp;
  TF1 *fpol1;
  TF1 *fpol2;
  TH1D *hdph;
  TH1D *hmod;
  TVirtualFFT *fft_f;
  TVirtualFFT *fft_b;

  double interpolate(double);

 public:
  // Default Constructor, mainly for Root
  TMatacq(int,int,int,int,int,int,int,int,int,int,int);

  // Destructor: Does nothing
  virtual ~TMatacq();

  // Initialize 
  void init();

  int rawPulseAnalysis(int, double*); // GHM
  int findPeak();
  int doFit();
  int doFit2();
  int compute_trise();

  void enterdata(int);
  int countBadPulses(int);
  void printmatacqData(int,int,int);
  void printitermatacqData(int,int,int);

  int getPt1() {return firstsample;}
  int getPt2() {return lastsample;}
  int getPtm() {return samplemax;}

  double getBaseLine() {return bl;}
  double getsigBaseLine() {return sigbl;}
  double* getPedCyc() {return ped_cyc;}


  double getTimpeak() {return pkval;}
  double getsigTimpeak() {return sigpkval;}

  double getAmpl() {return ampl;}
  double getTime() {return timeatmax;}

  double getTrise() {return trise;}
  double getFwhm() {return width50;}
  double getWidth20() {return width20;}
  double getWidth80() {return width80;}
  double getSlide() {return slidingmean;}


  //  ClassDef(TMatacq,1)
};

#endif



