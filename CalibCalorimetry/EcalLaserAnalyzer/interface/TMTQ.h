#ifndef TMTQ_H
#define TMTQ_H

#include <vector>
using namespace std;

class TMom;

class TMTQ
{

 public:


  enum outVar { iPeak, iSigma, iFit, iAmpl, iTrise, iFwhm, iFw20, iFw80, iPed, iPedsig, iSlide, nOutVar };
  
  double cuts[2][nOutVar];

  TMom *mom[nOutVar];


  // Default Constructor, mainly for Root
  TMTQ();

  // Destructor: Does nothing
  virtual ~TMTQ();

  void  init();
  void  setCut(int, double, double);

  void  addEntry(double, double, double, double, double,  double, double, double,  double, double, double);

  vector<double> get(int);
  
  vector<double> getPeak();
  vector<double> getSigma();
  vector<double> getFit();
  vector<double> getAmpl();
  vector<double> getTrise(); 
  vector<double> getFwhm();
  vector<double> getFw20();
  vector<double> getFw80();
  vector<double> getPed();
  vector<double> getPedsig();
  vector<double> getSliding();

  
 public:
 
  //  ClassDef(TMTQ,0)
};

#endif
