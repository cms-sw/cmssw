/*                                                                            
                                                                            
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2006                                

*/

//This equation solver class is taken from GEANT4 and modified!!

#include "GeneratorInterface/Hydjet2Interface/interface/EquationSolver.h"

template <class Function> 
NAEquationSolver<Function>::NAEquationSolver(const NAEquationSolver & right) {
  fMaxIter = right.GetMaxIterations();
  fTolerance = right.GetTolerance();
  fA = right.GetIntervalLowerLimit();
  fB = right.GetIntervalUpperLimit();
  fRoot = right.GetRoot();
}

// operators
template <class Function> 
NAEquationSolver<Function> & NAEquationSolver<Function>::operator=(const NAEquationSolver & right) {
  fMaxIter = right.GetMaxIterations();
  fTolerance = right.GetTolerance();
  fA = right.GetIntervalLowerLimit();
  fB = right.GetIntervalUpperLimit();
  fRoot = right.GetRoot();
  return *this;
}

template <class Function> 
bool NAEquationSolver<Function>::operator==(const NAEquationSolver & right) const {
  if (this == &right) return true;
  else return false;
}

template <class Function> 
bool NAEquationSolver<Function>::operator!=(const NAEquationSolver & right) const {
  return !operator==(right);
}

