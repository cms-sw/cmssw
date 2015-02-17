#ifndef _RETINA_H_
#define _RETINA_H_

#include <vector>

#include "Hit.h"

#include "TMath.h"
#include "TStyle.h"
#include "TH2D.h"
#include "TCanvas.h"
#include "TRandom3.h"


using namespace std;


struct Hit_t {
  double x;
  double y;
  double z;
  double rho;
  short layer;
  int id;
};

struct pqPoint_i {
  int p;
  int q;
};

struct pqPoint {
  double p;
  double q;
  double w;
};

enum FitView { XY, RZ };

class Retina{

 private:

  vector<Hit_t*> hits;
  unsigned int pbins;
  unsigned int qbins;
  double pmin;
  double pmax;
  double offset;
  double qmin;
  double qmax;
  double pbinsize;
  double qbinsize;
  vector<double> sigma;
  double minWeight;
  unsigned int para;
  FitView view;

  vector <vector <double> > Grid;
  vector <pqPoint> pqCollection;

  void    makeGrid();
  double  getResponsePQ(double p, double q);
  double  getResponseXpXm(double x_plus, double x_minus);
  pqPoint findMaximumInterpolated(pqPoint_i point_i, double w);

 public:
  
  Retina(vector<Hit_t*> hits_, unsigned int pbins_, unsigned int qbins_, 
	 double pmin_, double pmax_, double qmin_, double qmax_, 
	 vector<double> sigma_, double minWeight_, unsigned int para_, FitView view_);
  ~Retina();

  void fillGrid();
  void dumpGrid(int eventNum=0,int step=1,int imax=0);
  void findMaxima();
  void printMaxima();
  vector <pqPoint> getMaxima();
  pqPoint getBestPQ();

};

#endif
