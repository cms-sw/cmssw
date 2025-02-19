#ifndef TPEDestalAnalysis_H
#define TPEDestalAnalysis_H

#include "TObject.h"

#define fNpns 2
#define fNchans 400
#define ngains 3

class TPEDestalAnalysis: public TObject 
{

 private:

  int nevt;
  long int timestart[ngains],timestop[ngains];
  long int pntimestart[ngains],pntimestop[ngains];
  double valhf[ngains][fNchans+fNpns],sighf[ngains][fNchans+fNpns];
  double valbf[ngains][fNchans+fNpns],sigbf[ngains][fNchans+fNpns];
  double evts[ngains][fNchans+fNpns],evtn[ngains][fNchans+fNpns];

  double cuthflow[ngains][fNchans+fNpns],cuthfhig[ngains][fNchans+fNpns];
  double cutbflow[ngains][fNchans+fNpns],cutbfhig[ngains][fNchans+fNpns];

  void init();

 public:
  // Default Constructor, mainly for Root
  TPEDestalAnalysis();

  // Destructor: Does nothing
  virtual ~TPEDestalAnalysis();

  void reinit();
  void reinit(int);
  void putDateStart(int,long int);
  void putDateStop(int,long int);
  void putpnDateStart(int,long int);
  void putpnDateStop(int,long int);
  void getDateStart(int);
  void getDateStop(int);
  double getCuthflow(int g,int i) {return cuthflow[g][i];}
  double getCutbfhig(int g,int i) {return cutbfhig[g][i];}
  void putValues(int,int,double,double,double);
  void putValuesWithCuts(int,int,double,double,double);
  void computepedestalcuts(int,int,int,int);
  void printpedestalData(int,int,int,int,int,int);

  //  ClassDef(TPEDestalAnalysis,1)
};

#endif
