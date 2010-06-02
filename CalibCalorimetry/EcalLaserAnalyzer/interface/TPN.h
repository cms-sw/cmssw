#ifndef TPN_H
#define TPN_H

#include <vector>
using namespace std;

class TMom;

class TPN
{

 public:

  enum outVar { iPN, iPNoPN, iPNoPN0, iPNoPN1, nOutVar };

  double cuts[2][nOutVar];
  TMom *mom[nOutVar];

  int _nPN;

  // Default Constructor, mainly for Root
  TPN(int iPN=0);
  
  // Destructor: Does nothing
  virtual ~TPN();

  void  init();
  void  setCut(int, double, double);

  void  setPNCut(double, double);
  void  setPNoPNCut(double, double);
  void  setPNoPN0Cut(double, double);
  void  setPNoPN1Cut(double, double);

  void  addEntry(double, double, double);

  vector<double> get(int);
  vector<double> getPN();
  vector<double> getPNoPN();
  vector<double> getPNoPN0();
  vector<double> getPNoPN1();

  
  
 public:
 
  //  ClassDef(TPN,0)

};

#endif
