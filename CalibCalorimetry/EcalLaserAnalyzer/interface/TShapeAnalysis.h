#ifndef TShapeAnalysis_H
#define TShapeAnalysis_H

#include <vector>
#include <string>
class TTree;

using namespace std;
#define fNchsel 1700

class TShapeAnalysis
{

 private:

  int _color;
  char filename[80];
  long int timestart,timestop;
  int index[fNchsel],npass[fNchsel];
  int nsamplecristal,sampbmax,sampamax,nevt;
  int presample;
  double noise;
  double alpha0,beta0;
  double alpha_val[fNchsel], beta_val[fNchsel], width_val[fNchsel], chi2_val[fNchsel];
  int flag_val[fNchsel],channel_val[fNchsel];
  double alpha_init[fNchsel], beta_init[fNchsel], width_init[fNchsel], chi2_init[fNchsel];
  int flag_init[fNchsel], eta_init[fNchsel], phi_init[fNchsel], channel_init[fNchsel];

  int dcc_init[fNchsel], tower_init[fNchsel], ch_init[fNchsel], side_init[fNchsel];

  double rawsglu[fNchsel][200][10];
  double npassok[fNchsel];
 
  double chi2cut;
  int nchsel;

  void init(double, double, double, double);
  void init(TTree *tAB, double, double, double, double);

 public:
  // Default Constructor, mainly for Root
  TShapeAnalysis(double, double, double, double);
  // Default Constructor, mainly for Root
  TShapeAnalysis(TTree *tAB, double, double, double, double);

  // Default Constructor, mainly for Root
  TShapeAnalysis(double, double, double, double, int);
  // Default Constructor, mainly for Root
  TShapeAnalysis(TTree *tAB, double, double, double, double, int);

  // Destructor: Does nothing
  virtual ~TShapeAnalysis();

  void set_const(int,int,int,int,int,double,double);
  void set_presample(int);
  void set_nch(int);
  void assignChannel(int,int);
  void putDateStart(long int);
  void putDateStop(long int);
  void getDateStart();
  void getDateStop();
  void putAllVals(int,double*, int, int);
  void putAllVals(int,double*, int, int, int, int, int, int);
  void putchannelVal(int,int);
  void putalphaVal(int,double);
  void putbetaVal(int,double);
  void putwidthVal(int,double);
  void putchi2Val(int,double);
  void putflagVal(int,int);
  void putchannelInit(int,int);
  void putalphaInit(int,double);
  void putbetaInit(int,double);
  void putwidthInit(int,double);
  void putchi2Init(int,double);
  void putflagInit(int,int);
  void putetaInit(int,int);
  void putphiInit(int,int);
  void computeShape(string namefile, TTree*);
  void computeShape(string namefile);
  void computetmaxVal(int,double*);

  std::vector<double> getVals(int);
  std::vector<double> getInitVals(int);

  //  ClassDef(TShapeAnalysis,1)
};

#endif
