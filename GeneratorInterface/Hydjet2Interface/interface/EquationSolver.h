/*                                                                            
                                                                            
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2005                                

*/
//This equation solver class is taken from GEANT4 and modified!!

#ifndef NAEquationSolver_h
#define NAEquationSolver_h 1
#include <Rtypes.h>
#include "MathUtil.h"

#define DefaultTolerance 5.0e-14

template <class Function> 
class NAEquationSolver {

 public:
  enum {DefaultMaxIter = 100};
	
 private:
  // Maximum number of iterations
  int fMaxIter;
  double fTolerance;
  // interval limits [a,b] which should bracket the root
  double fA;
  double fB;
  // root
  double fRoot;

 public:    
  // default constructor
  NAEquationSolver() : fMaxIter(DefaultMaxIter), fTolerance(DefaultTolerance),
    fA(0.0), fB(0.0), fRoot(0.0) {};
	
    NAEquationSolver(const int iterations, const double tol) :
      fMaxIter(iterations), fTolerance(tol),
      fA(0.0), fB(0.0), fRoot(0.0) {};

      // copy constructor	
      NAEquationSolver(const NAEquationSolver & right);

      // destructor
      ~NAEquationSolver() {};
	
      // operators
      NAEquationSolver & operator=(const NAEquationSolver & right);
      bool operator==(const NAEquationSolver & right) const;
      bool operator!=(const NAEquationSolver & right) const;
		
      int GetMaxIterations(void) const {return fMaxIter;}
      void SetMaxIterations(const int iterations) {fMaxIter=iterations;}
	
      double GetTolerance(void) const {return fTolerance;}
      void SetTolerance(const double epsilon) {fTolerance = epsilon;}
  
      double GetIntervalLowerLimit(void) const {return fA;}
      double GetIntervalUpperLimit(void) const {return fB;}
	
      void SetIntervalLimits(const double Limit1, const double Limit2);

      double GetRoot(void) const {return fRoot;}	
	
      // Calculates the root by the Brent's method
      bool Brent(Function& theFunction);
};

#include "GeneratorInterface/Hydjet2Interface/src/EquationSolver.icc"

#endif
