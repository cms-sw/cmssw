/*                                                                            
                                                                            
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2006                                

*/

#include "GeneratorInterface/Hydjet2Interface/interface/StrangePotential.h"

double NAStrangePotential::CalculateStrangePotential() {
  double minFunction = this->operator()(fMinStrangePotential);
  double maxFunction = this->operator()(fMaxStrangePotential); 
  
  int iter = 0;  
  while(minFunction < 0.0 && iter++ < fNIteration) {
    fMinStrangePotential -= 0.5*fMinStrangePotential;
    minFunction = this->operator()(fMinStrangePotential);
  }
   
  iter = 0;  
  while(minFunction*maxFunction > 0.0 && iter++ < fNIteration) {
    fMaxStrangePotential += 1.5*Abs(fMaxStrangePotential-fMinStrangePotential);
    maxFunction = this->operator()(fMaxStrangePotential);
  }
	
  if(minFunction*maxFunction > 0.0) {
    Error("StrangePotential::CalculateStrangePotential", "minFunction*maxFunction is positive!\n");
    return 0.;
  }

  NAEquationSolver<NAStrangePotential> * theSolver = 
    new NAEquationSolver<NAStrangePotential>(fNSolverIteration, fTolerance);

  theSolver->SetIntervalLimits(fMinStrangePotential, fMaxStrangePotential);
  
  if (!theSolver->Brent(*this))
    Error("StrangePotential::CalculateStrangePotential", "the root is not found!\n");
  
  double strangePotential = theSolver->GetRoot();
  delete theSolver;
  return strangePotential;
}

//calculate hadron system strange density
double NAStrangePotential::CalculateStrangeDensity(const double strangePotential)
{
  fGc.SetStrangePotential(strangePotential);
  fGc.SetTemperature(fTemperature);
  fGc.SetBaryonPotential(fBaryonPotential);
  return fGc.StrangenessDensity(fDatabase);
}
