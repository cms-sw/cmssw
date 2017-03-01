// @(#)root/minuit2:$Id$
// Author: L. Moneta Wed Oct 18 11:48:00 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class HybridMinimizer

#ifndef ROOT_Minuit2_HybridMinimizer
#define ROOT_Minuit2_HybridMinimizer

#include "Math/Minimizer.h"
#include "Minuit2/MnUserParameterState.h"
#include "Math/IFunctionfwd.h"
#include "Minuit2/MinosError.h"
#include "Minuit2/ModularFunctionMinimizer.h"
#include "Minuit2/FumiliMinimizer.h"

namespace ROOT { 

   namespace Minuit2 { 

      class ModularFunctionMinimizer; 
      class FCNBase; 
      class FunctionMinimum;

   }
}

//_____________________________________________________________________________________________________
/** 
   HybridMinimizer class implementing the ROOT::Math::Minimizer interface for 
   Minuit2 minimization algorithm. 
   In ROOT it can be instantiated using the plug-in manager (plug-in "Minuit2")
   Using a string  (used by the plugin manager) or via an enumeration 
   an one can set all the possible minimization algorithms (Migrad, Simplex, Combined, Scan and Fumili).  
*/ 

namespace PSFitter{

class HybridMinimizer : public ROOT::Math::Minimizer {

public: 

   enum EMinimizerType { 
      kMigrad, 
      kSimplex, 
      kCombined, 
      kScan,
      kFumili
   };

   /** 
      Default constructor
   */ 
   HybridMinimizer (EMinimizerType type = kMigrad); 

   /** 
      Constructor with a char (used by PM) 
   */ 
   HybridMinimizer (const char *  type); 

   /** 
      Destructor (no operations)
   */ 
   virtual ~HybridMinimizer (); 

private:
   // usually copying is non trivial, so we make this unaccessible

   /** 
      Copy constructor
   */ 
   HybridMinimizer(const HybridMinimizer &); 

   /** 
      Assignment operator
   */ 
   HybridMinimizer & operator = (const HybridMinimizer & rhs); 

public: 

   // clear resources (parameters) for consecutives minimizations
   virtual void Clear();

   /// set the function to minimize
   virtual void SetFunction(const ROOT::Math::IMultiGenFunction & func); 

   /// set gradient the function to minimize
   virtual void SetFunction(const ROOT::Math::IMultiGradFunction & func); 

   /// set free variable 
   virtual bool SetVariable(unsigned int ivar, const std::string & name, double val, double step); 

   /// set lower limit variable  (override if minimizer supports them )
   virtual bool SetLowerLimitedVariable(unsigned int  ivar , const std::string & name , double val , double step , double lower );
   /// set upper limit variable (override if minimizer supports them )
   virtual bool SetUpperLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double upper ); 
   /// set upper/lower limited variable (override if minimizer supports them )
   virtual bool SetLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double /* lower */, double /* upper */); 
   /// set fixed variable (override if minimizer supports them )
   virtual bool SetFixedVariable(unsigned int /* ivar */, const std::string & /* name */, double /* val */);  
   /// set variable
   virtual bool SetVariableValue(unsigned int ivar, double val);
   virtual bool SetVariableValues(const double * val);

   /// get name of variables (override if minimizer support storing of variable names)
   virtual std::string VariableName(unsigned int ivar) const;

   /// get index of variable given a variable given a name
   /// return -1 if variable is not found
   virtual int VariableIndex(const std::string & name) const;

   /** 
       method to perform the minimization. 
       Return false in case the minimization did not converge. In this case a 
       status code different than zero is set 
       (retrieved by the derived method Minimizer::Status() )" 
       
       status = 1    : Covariance was made pos defined
       status = 2    : Hesse is invalid
       status = 3    : Edm is above max 
       status = 4    : Reached call limit
       status = 5    : Any other failure 
   */
   virtual  bool Minimize(); 

   /// return minimum function value
   virtual double MinValue() const { return fState.Fval(); } 

   /// return expected distance reached from the minimum
   virtual double Edm() const { return fState.Edm(); }

   /// return  pointer to X values at the minimum 
   virtual const double *  X() const; 
 
   /// return pointer to gradient values at the minimum 
   virtual const double *  MinGradient() const { return 0; } // not available in Minuit2 

   /// number of function calls to reach the minimum 
   virtual unsigned int NCalls() const { return fState.NFcn(); } 

   /// this is <= Function().NDim() which is the total 
   /// number of variables (free+ constrained ones) 
   virtual unsigned int NDim() const { return fDim; }   

   /// number of free variables (real dimension of the problem) 
   /// this is <= Function().NDim() which is the total 
   virtual unsigned int NFree() const { return fState.VariableParameters(); }  

   /// minimizer provides error and error matrix
   virtual bool ProvidesError() const { return true; } 

   /// return errors at the minimum 
   virtual const double * Errors() const; 

   /** 
       return covariance matrix elements 
       if the variable is fixed or const the value is zero
       The ordering of the variables is the same as in errors and parameter value. 
       This is different from the direct interface of Minuit2 or TMinuit where the 
       values were obtained only to variable parameters
   */ 
   virtual double CovMatrix(unsigned int i, unsigned int j) const;  


   /** 
       Fill the passed array with the  covariance matrix elements 
       if the variable is fixed or const the value is zero. 
       The array will be filled as cov[i *ndim + j]
       The ordering of the variables is the same as in errors and parameter value. 
       This is different from the direct interface of Minuit2 or TMinuit where the 
       values were obtained only to variable parameters
   */ 
   virtual bool GetCovMatrix(double * cov) const;  

   /** 
       Fill the passed array with the Hessian matrix elements 
       The Hessian matrix is the matrix of the second derivatives 
       and is the inverse of the covariance matrix
       If the variable is fixed or const the values for that variables are zero. 
       The array will be filled as h[i *ndim + j]
   */ 
   virtual bool GetHessianMatrix(double * h) const;  


   /**
      return the status of the covariance matrix 
       status = -1 :  not available (inversion failed or Hesse failed) 
       status =  0 : available but not positive defined
       status =  1 : covariance only approximate
       status =  2 : full matrix but forced pos def 
       status =  3 : full accurate matrix 

    */
   virtual int CovMatrixStatus() const;
   /**
      return correlation coefficient between variable i and j.
      If the variable is fixed or const the return value is zero
    */
   virtual double Correlation(unsigned int i, unsigned int j ) const; 

   /**
      get global correlation coefficient for the variable i. This is a number between zero and one which gives 
      the correlation between the i-th variable  and that linear combination of all other variables which 
      is most strongly correlated with i.
      If the variable is fixed or const the return value is zero
    */
   virtual double GlobalCC(unsigned int i) const;

   /**
      get the minos error for parameter i, return false if Minos failed
      A minimizaiton must be performed befre, return false if no minimization has been done
      In case of Minos failed the status error is updated as following 
      status += 10 * minosStatus where the minos status is:
       status = 1    : maximum number of function calls exceeded when running for lower error
       status = 2    : maximum number of function calls exceeded when running for upper error
       status = 3    : new minimum found when running for lower error
       status = 4    : new minimum found when running for upper error
       status = 5    : any other failure 

   */
   virtual bool GetMinosError(unsigned int i, double & errLow, double & errUp, int = 0); 

   /**
      scan a parameter i around the minimum. A minimization must have been done before, 
      return false if it is not the case
    */
   virtual bool Scan(unsigned int i, unsigned int & nstep, double * x, double * y, double xmin = 0, double xmax = 0); 

   /**
      find the contour points (xi,xj) of the function for parameter i and j around the minimum
      The contour will be find for value of the function = Min + ErrorUp();
    */
   virtual bool Contour(unsigned int i, unsigned int j, unsigned int & npoints, double *xi, double *xj); 

   
   /**
      perform a full calculation of the Hessian matrix for error calculation
      If a valid minimum exists the calculation is done on the minimum point otherwise is performed 
      in the current set values of parameters
      Status code of minimizer is updated according to the following convention (in case Hesse failed)
      status += 100*hesseStatus where hesse status is: 
      status = 1 : hesse failed
      status = 2 : matrix inversion failed
      status = 3 : matrix is not pos defined 
    */
   virtual bool Hesse();


   /// return reference to the objective function
   ///virtual const ROOT::Math::IGenFunction & Function() const; 

   /// print result of minimization
   virtual void PrintResults(); 

      /// return the minimizer state (containing values, step size , etc..)
   const ROOT::Minuit2::MnUserParameterState & State() { return fState; }

   void SetMinimizerType( EMinimizerType type);

protected: 
   
   // protected function for accessing the internal Minuit2 object. Needed for derived classes

   virtual const ROOT::Minuit2::ModularFunctionMinimizer * GetMinimizer() const { return fMinimizer; } 

   virtual void SetMinimizer( ROOT::Minuit2::ModularFunctionMinimizer * m)  { fMinimizer = m; } 

   virtual const  ROOT::Minuit2::FCNBase * GetFCN() const { return fMinuitFCN; } 

   /// examine the minimum result 
   bool ExamineMinimum(const ROOT::Minuit2::FunctionMinimum & min); 

private: 
   
   unsigned int fDim;       // dimension of the function to be minimized 
   bool fUseFumili;     

   ROOT::Minuit2::MnUserParameterState fState;
   // std::vector<ROOT::Minuit2::MinosError> fMinosErrors;
   ROOT::Minuit2::ModularFunctionMinimizer * fMinimizer;
   ROOT::Minuit2::FCNBase * fMinuitFCN;
   ROOT::Minuit2::FunctionMinimum * fMinimum;
   mutable std::vector<double> fValues;
   mutable std::vector<double> fErrors;

}; 

}
#endif 
