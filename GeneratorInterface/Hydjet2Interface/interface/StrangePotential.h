/*                                                                            
                                                                            
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2005                                

*/

//This class is used to calculate strange potential from 
//the known initial strange density = 0 at given temperature and baryon potential.

#ifndef NAStrangePotential_h
#define NAStrangePotential_h 1

#include "StrangeDensity.h"
#include "EquationSolver.h"
#include "DatabasePDG.h"

class NAStrangePotential {
 private:
  double fTemperature;
  double fBaryonPotential;
  double fStrangeDensity;
  double fMinStrangePotential;//initial min value of strange potential 
  double fMaxStrangePotential;//initial max value of strange potential
  int fNIteration; //to find proper [minStrangePotential, maxStrangePotential] interval
  int fNSolverIteration; //to find root in [minStrangePotential,maxStrangePotential] interval
  double fTolerance;//to find root 
  DatabasePDG* fDatabase;
  NAStrangeDensity fGc;
  //compute hadron  system strange density through strange potential
  double CalculateStrangeDensity(const double strangePotential);
  //default constructor is not accesible
  NAStrangePotential(){};

 public:
  NAStrangePotential(const double initialStrangeDensity, DatabasePDG* database) :
    fStrangeDensity(initialStrangeDensity),
    fMinStrangePotential(0.0001*GeV),
    fMaxStrangePotential(0.9*GeV),
    fNIteration(100),
    fNSolverIteration(100),
    fTolerance(1.e-8),
    fDatabase(database)
      {};

    ~NAStrangePotential() {};
   
    double operator()(const double strangePotential) { 
      return (fStrangeDensity - this->CalculateStrangeDensity(strangePotential))/fStrangeDensity; 
    }	

    void SetTemperature(double value) {fTemperature = value;}
    void SetBaryonPotential(double value) {fBaryonPotential = value;}
    void SetMinStrangePotential(double value) {fMinStrangePotential = value;}
    void SetMaxStrangePotential(double value) {fMaxStrangePotential = value;}
    double CalculateStrangePotential();
};

#endif
