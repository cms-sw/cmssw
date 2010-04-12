#ifndef TPNFit_H
#define TPNFit_H
#define NMAXSAMPPN 50

class TF1;
class TH1D;

class TPNFit
{

 private:	


  int fNsamples ;
  int fNum_samp_bef_max;
  int fNum_samp_after_max;

  int firstsample,lastsample;
  double t[NMAXSAMPPN],val[NMAXSAMPPN];
  double fv1[NMAXSAMPPN],fv2[NMAXSAMPPN],fv3[NMAXSAMPPN];
  double ampl;
  double timeatmax;
  double ampl2;
  double timeatmax2;
  //  double _tau1, _tau2;

  static double fitPN_tp(double *x, double *par); 
  
  //  void setTaus(double tau1, double tau2); 
  TF1* fPN;
  // TF1* funcPN(double *x, double *par); 
  TH1D *htmp;

 public:
  // Default Constructor, mainly for Root
  TPNFit();

  // Destructor: Does nothing
  virtual ~TPNFit();

  // Initialize 
  void init(int,int,int);
  double doFit(int,double *);
  double doFit2( double *, double, double, double , double, double ); 
  double getAmpl() {return ampl;}
  double getTime() {return timeatmax;}
  double getAmpl2() {return ampl2;}
  double getTime2() {return timeatmax2;}

  //  ClassDef(TPNFit,1)
};

#endif



