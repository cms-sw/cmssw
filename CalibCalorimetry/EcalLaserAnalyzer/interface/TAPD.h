#ifndef TAPD_H
#define TAPD_H

#include <vector>

class TMom;

class TAPD
{

 public:


  enum outVar { iAPD, iAPDoPN, iAPDoPN0, iAPDoPN1, iTime, iAPDoAPD0, iAPDoAPD1, nOutVar };
  
  std::vector<double> _apdcuts[2][nOutVar];
  std::vector<int> _cutvars[nOutVar];

  TMom *mom[nOutVar];

  // Default Constructor, mainly for Root
  TAPD();

  // Destructor: Does nothing
  virtual ~TAPD();

  void  init();

  void  setCut(int, double, double);
  void  setCut(int , const std::vector<int>& , const std::vector<double>& , const std::vector<double>&);

  void  addEntry(double, double, double, double, double, double, double);
  void  addEntry(double, double, double, double, double);

  // Simple 1D cuts on main variable at 2 sigmas
  // ===========================================

  void  setAPDCut(double, double);
  void  setAPDoPNCut(double, double);
  void  setAPDoPN0Cut(double, double);
  void  setAPDoPN1Cut(double, double);
  void  setTimeCut(double, double);

  // More complicated 2D cuts
  // ========================= 
  void set2DCut(int, const std::vector<double>& ,const std::vector<double>& );
  void set2DAPDCut(const std::vector<double>&,const std::vector<double>&);
  void set2DAPDoPNCut(const std::vector<double>&,const std::vector<double>& );
  void set2DAPDoPN0Cut(const std::vector<double>&,const std::vector<double>& );
  void set2DAPDoPN1Cut(const std::vector<double>&,const std::vector<double>& );
  void set2DAPDoAPD0Cut(const std::vector<double>&,const std::vector<double>& );
  void set2DAPDoAPD1Cut(const std::vector<double>&,const std::vector<double>& );
  void set2DTimeCut(const std::vector<double>&,const std::vector<double>& );

  std::vector<double> get(int);
  std::vector<double> getAPD();
  std::vector<double> getAPDoPN();
  std::vector<double> getAPDoPN0();
  std::vector<double> getAPDoPN1();
  std::vector<double> getAPDoAPD0();
  std::vector<double> getAPDoAPD1();
  std::vector<double> getTime();
 

  
 public:
 
  //  ClassDef(TAPD,0)
};

#endif
