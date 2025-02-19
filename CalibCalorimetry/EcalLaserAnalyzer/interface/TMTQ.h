#ifndef TMTQ_H
#define TMTQ_H

#include <vector>

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

  std::vector<double> get(int);
  
  std::vector<double> getPeak();
  std::vector<double> getSigma();
  std::vector<double> getFit();
  std::vector<double> getAmpl();
  std::vector<double> getTrise(); 
  std::vector<double> getFwhm();
  std::vector<double> getFw20();
  std::vector<double> getFw80();
  std::vector<double> getPed();
  std::vector<double> getPedsig();
  std::vector<double> getSliding();

  
 public:
 
  //  ClassDef(TMTQ,0)
};

#endif
