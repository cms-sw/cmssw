#ifndef TAPD_H
#define TAPD_H

#include <vector>

#define VARSIZE 6

class TMom;
class TAPD
{

 public:


  enum outVar { iAPD, iAPDoPN, iAPDoPN0, iAPDoPN1, iTime, iAPDoAPD, iAPDoAPD0, iAPDoAPD1, iAPDoPNCor, iAPDoPN0Cor, iAPDoPN1Cor, nOutVar };
  
  std::vector<double> _apdcuts[2][nOutVar];
  std::vector<int> _cutvars[nOutVar];
  
  TMom *mom[nOutVar];  
  double default_max[nOutVar];

  // Default Constructor, mainly for Root
  TAPD();

  // Destructor: Does nothing
  virtual ~TAPD();

  void  init();

  void  setCut(int, double, double);
  void  setCut(int , std::vector<int> , std::vector<double> , std::vector<double>);

  void  addEntry(double, double, double, double, double, double, double, double);
  void  addEntry(double, double, double, double, double, double);
  void  addEntry(double, double, double, double);

  // Simple 1D cuts on main variable at 2 sigmas
  // ===========================================

  void  setAPDCut(double, double);
  void  setAPDoPNCut(double, double);
  void  setAPDoPN0Cut(double, double);
  void  setAPDoPN1Cut(double, double); 
  void  setAPDoPNCorCut(double, double );
  void  setAPDoPN0CorCut(double, double );
  void  setAPDoPN1CorCut(double, double );
  void  setTimeCut(double, double);
  
  // More complicated 2D cuts
  // ========================= 
  void set2DCut(int, std::vector<double> ,std::vector<double> );
  void set2DAPDCut(std::vector<double>,std::vector<double>);
  void set2DAPDoPNCut(std::vector<double>,std::vector<double> );
  void set2DAPDoPN0Cut(std::vector<double>,std::vector<double> );
  void set2DAPDoPN1Cut(std::vector<double>,std::vector<double> );
  void set2DAPDoAPD0Cut(std::vector<double>,std::vector<double> );
  void set2DAPDoAPD1Cut(std::vector<double>,std::vector<double> );
  void set2DAPDoAPDCut(std::vector<double>,std::vector<double> );
  
  void set2DTimeCut(std::vector<double>,std::vector<double> );
  
  std::vector<double> get(int);
  std::vector<double> getAPD();
  std::vector<double> getAPDoPN();
  std::vector<double> getAPDoPN0();
  std::vector<double> getAPDoPN1();
  std::vector<double> getAPDoAPD();
  std::vector<double> getAPDoAPD0();
  std::vector<double> getAPDoAPD1();
  std::vector<double> getTime();
  std::vector<double> getAPDoPNCor();
  std::vector<double> getAPDoPN0Cor();
  std::vector<double> getAPDoPN1Cor();
 
  // double* get2(int ivar);
  // double* getAPD2( );
 
 public:
 
  //  ClassDef(TAPD,0)
};

#endif
