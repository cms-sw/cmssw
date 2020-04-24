// @(#)root/minuit2:$Id$
// Author: L. Moneta Wed Oct 18 11:48:00 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// by lhx: Note copied and modifed from the Minuit2Minimizer to suit our purpose
// Changes mainly to make the SetMinimizerType public so that user can re-new to
// different minimizer...
// Implementation file for class HybridMinimizer

#include "RecoLocalCalo/HcalRecAlgos/src/HybridMinimizer.h"

#include "Math/IFunction.h"
#include "Math/IOptions.h"

#include "Minuit2/FCNAdapter.h"
#include "Minuit2/FumiliFCNAdapter.h"
#include "Minuit2/FCNGradAdapter.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnMinos.h"
#include "Minuit2/MinosError.h"
#include "Minuit2/MnHesse.h"
#include "Minuit2/MinuitParameter.h"
#include "Minuit2/MnUserFcn.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/VariableMetricMinimizer.h"
#include "Minuit2/SimplexMinimizer.h"
#include "Minuit2/CombinedMinimizer.h"
#include "Minuit2/ScanMinimizer.h"
#include "Minuit2/FumiliMinimizer.h"
#include "Minuit2/MnParameterScan.h"
#include "Minuit2/MnContours.h"
 
#include <cassert> 
#include <iostream> 
#include <algorithm>
#include <functional>

namespace PSFitter{

   using namespace ROOT::Minuit2;

   // functions needed to control siwthc off of Minuit2 printing level 
#ifdef USE_ROOT_ERROR
   int TurnOffPrintInfoLevel() { 
   // switch off Minuit2 printing of INFO message (cut off is 1001) 
   int prevErrorIgnoreLevel = gErrorIgnoreLevel; 
   if (prevErrorIgnoreLevel < 1001) { 
      gErrorIgnoreLevel = 1001; 
      return prevErrorIgnoreLevel; 
   }
   return -2;  // no op in this case  
}

void RestoreGlobalPrintLevel(int value) { 
      gErrorIgnoreLevel = value; 
}
#else
   // dummy functions
   int TurnOffPrintInfoLevel() { return -1; } 
   int ControlPrintLevel( ) { return -1;}
   void RestoreGlobalPrintLevel(int ) {} 
#endif      


HybridMinimizer::HybridMinimizer(EMinimizerType type ) : 
   Minimizer(),
   fDim(0),
   fMinimizer(nullptr),
   fMinuitFCN(nullptr),
   fMinimum(nullptr)   
{
   // Default constructor implementation depending on minimizer type 
   SetMinimizerType(type); 
}

HybridMinimizer::HybridMinimizer(const char *  type ) : 
   Minimizer(),
   fDim(0),
   fMinimizer(nullptr),
   fMinuitFCN(nullptr),
   fMinimum(nullptr)   
{   
   // constructor from a string

   std::string algoname(type);
   // tolower() is not an  std function (Windows)
   std::transform(algoname.begin(), algoname.end(), algoname.begin(), (int(*)(int)) tolower ); 

   EMinimizerType algoType = kMigrad; 
   if (algoname == "simplex")   algoType = kSimplex; 
   if (algoname == "minimize" ) algoType = kCombined; 
   if (algoname == "scan" )     algoType = kScan; 
   if (algoname == "fumili" )   algoType = kFumili;
  
   SetMinimizerType(algoType);
}

void HybridMinimizer::SetMinimizerType(EMinimizerType type) {
   // Set  minimizer algorithm type 
   fUseFumili = false;
 
   if (fMinimizer) delete fMinimizer;

   switch (type) { 
   case kMigrad: 
      //std::cout << "HybridMinimizer: minimize using MIGRAD " << std::endl;
      SetMinimizer( new ROOT::Minuit2::VariableMetricMinimizer() );
      return;
   case kSimplex: 
      //std::cout << "HybridMinimizer: minimize using SIMPLEX " << std::endl;
      SetMinimizer( new ROOT::Minuit2::SimplexMinimizer() );
      return;
   case kCombined: 
      SetMinimizer( new ROOT::Minuit2::CombinedMinimizer() );
      return;
   case kScan: 
      SetMinimizer( new ROOT::Minuit2::ScanMinimizer() );
      return;
   case kFumili:          
      SetMinimizer( new ROOT::Minuit2::FumiliMinimizer() );
      fUseFumili = true;
      return;
   default: 
      //migrad minimizer
      SetMinimizer( new ROOT::Minuit2::VariableMetricMinimizer() );
   }
}


HybridMinimizer::~HybridMinimizer() 
{
   // Destructor implementation.
   if (fMinimizer) delete fMinimizer; 
   if (fMinuitFCN) delete fMinuitFCN; 
   if (fMinimum)   delete fMinimum; 
}

HybridMinimizer::HybridMinimizer(const HybridMinimizer &) : 
   ROOT::Math::Minimizer()
{
   // Implementation of copy constructor.
}

HybridMinimizer & HybridMinimizer::operator = (const HybridMinimizer &rhs) 
{
   // Implementation of assignment operator.
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}


void HybridMinimizer::Clear() { 
   // delete the state in case of consecutive minimizations
   fState = MnUserParameterState();
   // clear also the function minimum
   if (fMinimum) delete fMinimum; 
   fMinimum = nullptr;
}


// set variables 

bool HybridMinimizer::SetVariable(unsigned int ivar, const std::string & name, double val, double step) { 
   // set a free variable. 
   // Add the variable if not existing otherwise  set value if exists already
   // this is implemented in MnUserParameterState::Add
   // if index is wrong (i.e. variable already exists but with a different index return false) but 
   // value is set for corresponding variable name

//    std::cout << " add parameter " << name << "  " <<  val << " step " << step << std::endl;

   if (step <= 0) { 
      std::string txtmsg = "Parameter " + name + "  has zero or invalid step size - consider it as constant ";
      MN_INFO_MSG2("HybridMinimizer::SetVariable",txtmsg);
      fState.Add(name, val);
   }
   else 
      fState.Add(name, val, step); 

   unsigned int minuit2Index = fState.Index(name ); 
   if ( minuit2Index != ivar) {
      std::string txtmsg("Wrong index used for the variable " + name);
      MN_INFO_MSG2("HybridMinimizer::SetVariable",txtmsg);  
      MN_INFO_VAL2("HybridMinimizer::SetVariable",minuit2Index);  
      ivar = minuit2Index;
      return false;
   }
   fState.RemoveLimits(ivar);

   return true; 
}

bool HybridMinimizer::SetLowerLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double lower ) {
   // add a lower bounded variable
   if (!SetVariable(ivar, name, val, step) ) return false;
   fState.SetLowerLimit(ivar, lower);
   return true;
}

bool HybridMinimizer::SetUpperLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double upper ) {
   // add a upper bounded variable
   if (!SetVariable(ivar, name, val, step) ) return false;
   fState.SetUpperLimit(ivar, upper);
   return true;
}



bool HybridMinimizer::SetLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double lower , double upper) {
   // add a double bound variable
   if (!SetVariable(ivar, name, val, step) ) return false;
   fState.SetLimits(ivar, lower, upper);
   return true;
}

bool HybridMinimizer::SetFixedVariable(unsigned int ivar , const std::string & name , double val ) {
   // add a fixed variable
   // need a step size otherwise treated as a constant 
   // use 10% 
   double step = ( val != 0) ? 0.1 * std::abs(val) : 0.1;
   if (!SetVariable(ivar, name, val, step ) ) { 
      ivar = fState.Index(name );      
   }
   fState.Fix(ivar);
   return true;
}

std::string HybridMinimizer::VariableName(unsigned int ivar) const { 
   // return the variable name
   if (ivar >= fState.MinuitParameters().size() ) return std::string();
   return fState.GetName(ivar);
}


int HybridMinimizer::VariableIndex(const std::string & name) const { 
   // return the variable index
   // check if variable exist
   return fState.Trafo().FindIndex(name);
}


bool HybridMinimizer::SetVariableValue(unsigned int ivar, double val) { 
   // set value for variable ivar (only for existing parameters)
   if (ivar >= fState.MinuitParameters().size() ) return false; 
   fState.SetValue(ivar, val);
   return true; 
}

bool HybridMinimizer::SetVariableValues(const double * x)  { 
   // set value for variable ivar (only for existing parameters)
   unsigned int n =  fState.MinuitParameters().size(); 
   if (n== 0) return false; 
   for (unsigned int ivar = 0; ivar < n; ++ivar) 
      fState.SetValue(ivar, x[ivar]);
   return true; 
}


void HybridMinimizer::SetFunction(const  ROOT::Math::IMultiGenFunction & func) { 
   // set function to be minimized
   if (fMinuitFCN) delete fMinuitFCN;
   fDim = func.NDim(); 
   if (!fUseFumili) {
      fMinuitFCN = new ROOT::Minuit2::FCNAdapter<ROOT::Math::IMultiGenFunction> (func, ErrorDef() );
   }
   else { 
      // for Fumili the fit method function interface is required
      const ROOT::Math::FitMethodFunction * fcnfunc = dynamic_cast<const ROOT::Math::FitMethodFunction *>(&func);
      if (!fcnfunc) {
         MN_ERROR_MSG("HybridMinimizer: Wrong Fit method function for Fumili");
         return;
      }
      fMinuitFCN = new ROOT::Minuit2::FumiliFCNAdapter<ROOT::Math::FitMethodFunction> (*fcnfunc, fDim, ErrorDef() );
   }
}

void HybridMinimizer::SetFunction(const  ROOT::Math::IMultiGradFunction & func) { 
   // set function to be minimized
   fDim = func.NDim(); 
   if (fMinuitFCN) delete fMinuitFCN;
   if (!fUseFumili) { 
      fMinuitFCN = new ROOT::Minuit2::FCNGradAdapter<ROOT::Math::IMultiGradFunction> (func, ErrorDef() );
   }
   else { 
      // for Fumili the fit method function interface is required
      const ROOT::Math::FitMethodGradFunction * fcnfunc = dynamic_cast<const ROOT::Math::FitMethodGradFunction*>(&func);
      if (!fcnfunc) {
         MN_ERROR_MSG("HybridMinimizer: Wrong Fit method function for Fumili");
         return;
      }
      fMinuitFCN = new ROOT::Minuit2::FumiliFCNAdapter<ROOT::Math::FitMethodGradFunction> (*fcnfunc, fDim, ErrorDef() );
   }
}
                                   
bool HybridMinimizer::Minimize() { 
   // perform the minimization
   // store a copy of FunctionMinimum 
   if (!fMinuitFCN) { 
      MN_ERROR_MSG2("HybridMinimizer::Minimize","FCN function has not been set");
      return false; 
  }

   assert(GetMinimizer() != nullptr );

   // delete result of previous minimization
   if (fMinimum) delete fMinimum; 
   fMinimum = nullptr;


   int maxfcn = MaxFunctionCalls(); 
   double tol = Tolerance();
   int strategyLevel = Strategy(); 
   fMinuitFCN->SetErrorDef(ErrorDef() );

   if (PrintLevel() >=1) { 
      // print the real number of maxfcn used (defined in ModularFuncitonMinimizer)
      int maxfcn_used = maxfcn; 
      if (maxfcn_used == 0) { 
         int nvar = fState.VariableParameters();
         maxfcn_used = 200 + 100*nvar + 5*nvar*nvar;
      }      
//      std::cout << "HybridMinimizer: Minimize with max-calls " << maxfcn_used 
//                << " convergence for edm < " << tol << " strategy " 
//                << strategyLevel << std::endl; 
   }

   // internal minuit messages
   MnPrint::SetLevel(PrintLevel() );

   // switch off Minuit2 printing
   int prev_level = (PrintLevel() <= 0 ) ?   TurnOffPrintInfoLevel() : -2; 

   // set the precision if needed
   if (Precision() > 0) fState.SetPrecision(Precision());

   // set strategy and add extra options if needed
   ROOT::Minuit2::MnStrategy strategy(strategyLevel);
   ROOT::Math::IOptions * minuit2Opt = ROOT::Math::MinimizerOptions::FindDefault("Minuit2");
   if (minuit2Opt) { 
      // set extra strategy options
      int nGradCycles = strategy.GradientNCycles();
      int nHessCycles = strategy.HessianNCycles();
      int nHessGradCycles = strategy.HessianGradientNCycles();

      double gradTol =  strategy.GradientTolerance();
      double gradStepTol = strategy.GradientStepTolerance();
      double hessStepTol = strategy.HessianStepTolerance();
      double hessG2Tol = strategy.HessianG2Tolerance();

      minuit2Opt->GetValue("GradientNCycles",nGradCycles);
      minuit2Opt->GetValue("HessianNCycles",nHessCycles);
      minuit2Opt->GetValue("HessianGradientNCycles",nHessGradCycles);

      minuit2Opt->GetValue("GradientTolerance",gradTol);
      minuit2Opt->GetValue("GradientStepTolerance",gradStepTol);
      minuit2Opt->GetValue("HessianStepTolerance",hessStepTol);
      minuit2Opt->GetValue("HessianG2Tolerance",hessG2Tol);

      strategy.SetGradientNCycles(nGradCycles);      
      strategy.SetHessianNCycles(nHessCycles);
      strategy.SetHessianGradientNCycles(nHessGradCycles);

      strategy.SetGradientTolerance(gradTol);
      strategy.SetGradientStepTolerance(gradStepTol);
      strategy.SetHessianStepTolerance(hessStepTol);
      strategy.SetHessianG2Tolerance(hessStepTol);

      if (PrintLevel() > 0) { 
//         std::cout << "HybridMinimizer::Minuit  - Changing default stratgey options" << std::endl;
         minuit2Opt->Print();
      }
      
   }
      
   const ROOT::Minuit2::FCNGradientBase * gradFCN = dynamic_cast<const ROOT::Minuit2::FCNGradientBase *>( fMinuitFCN ); 
   if ( gradFCN != nullptr) {
      // use gradient
      //SetPrintLevel(3);
      ROOT::Minuit2::FunctionMinimum min =  GetMinimizer()->Minimize(*gradFCN, fState, strategy, maxfcn, tol);
      fMinimum = new ROOT::Minuit2::FunctionMinimum (min);    
   }
   else {
      ROOT::Minuit2::FunctionMinimum min = GetMinimizer()->Minimize(*GetFCN(), fState, strategy, maxfcn, tol);
      fMinimum = new ROOT::Minuit2::FunctionMinimum (min);    
   }

   // check if Hesse needs to be run 
   if (fMinimum->IsValid() && IsValidError() && fMinimum->State().Error().Dcovar() != 0 ) {
      // run Hesse (Hesse will add results in the last state of fMinimum
      ROOT::Minuit2::MnHesse hesse(strategy );
      hesse( *fMinuitFCN, *fMinimum, maxfcn); 
   }

   // -2 is the highest low invalid value for gErrorIgnoreLevel
   if (prev_level > -2) RestoreGlobalPrintLevel(prev_level);
   
   fState = fMinimum->UserState(); 
   bool ok =  ExamineMinimum(*fMinimum);
   //fMinimum = 0; 
   return ok; 
}

bool  HybridMinimizer::ExamineMinimum(const ROOT::Minuit2::FunctionMinimum & min) {  
   /// study the function minimum      
   
   // debug ( print all the states) 
   int debugLevel = PrintLevel(); 
   if (debugLevel >= 3) { 
/*      
      const std::vector<ROOT::Minuit2::MinimumState>& iterationStates = min.States();
      std::cout << "Number of iterations " << iterationStates.size() << std::endl;
      for (unsigned int i = 0; i <  iterationStates.size(); ++i) {
         //std::cout << iterationStates[i] << std::endl;                                                                       
         const ROOT::Minuit2::MinimumState & st =  iterationStates[i];
         std::cout << "----------> Iteration " << i << std::endl;
         int pr = std::cout.precision(12);
         std::cout << "            FVAL = " << st.Fval() << " Edm = " << st.Edm() << " Nfcn = " << st.NFcn() << std::endl;
         std::cout.precision(pr);
         std::cout << "            Error matrix change = " << st.Error().Dcovar() << std::endl;
         std::cout << "            Parameters : ";
         // need to transform from internal to external 
         for (int j = 0; j < st.size() ; ++j) std::cout << " p" << j << " = " << fState.Int2ext( j, st.Vec()(j) );
         std::cout << std::endl;
      }
*/
   }

   fStatus = 0;
   std::string txt;
   if (min.HasMadePosDefCovar() ) { 
      txt = "Covar was made pos def";
      fStatus = 1; 
   }
   if (min.HesseFailed() ) { 
      txt = "Hesse is not valid";
      fStatus = 2; 
   }
   if (min.IsAboveMaxEdm() ) { 
      txt = "Edm is above max"; 
      fStatus = 3; 
   }
   if (min.HasReachedCallLimit() ) { 
      txt = "Reached call limit";
      fStatus = 4;
   }

   
   bool validMinimum = min.IsValid();
   if (validMinimum) { 
      // print a warning message in case something is not ok
      if (fStatus != 0 && debugLevel > 0)  MN_INFO_MSG2("HybridMinimizer::Minimize",txt);
   }
   else { 
      // minimum is not valid when state is not valid and edm is over max or has passed call limits
      if (fStatus == 0) { 
         // this should not happen
         txt = "unknown failure";  
         fStatus = 5;
      }
      std::string msg = "Minimization did NOT converge, " + txt;
      MN_INFO_MSG2("HybridMinimizer::Minimize",msg);                   
   }

   if (debugLevel >= 1) PrintResults(); 
   return validMinimum;
}


void HybridMinimizer::PrintResults() {
   // print results of minimization
   if (!fMinimum) return;
   if (fMinimum->IsValid() ) {
      // valid minimum
/*
      std::cout << "HybridMinimizer : Valid minimum - status = " << fStatus  << std::endl; 
      int pr = std::cout.precision(18);
      std::cout << "FVAL  = " << fState.Fval() << std::endl;
      std::cout << "Edm   = " << fState.Edm() << std::endl;
      std::cout.precision(pr);
      std::cout << "Nfcn  = " << fState.NFcn() << std::endl;
      for (unsigned int i = 0; i < fState.MinuitParameters().size(); ++i) {
         const MinuitParameter & par = fState.Parameter(i); 
         std::cout << par.Name() << "\t  = " << par.Value() << "\t ";
         if (par.IsFixed() )      std::cout << "(fixed)" << std::endl;
         else if (par.IsConst() ) std::cout << "(const)" << std::endl;
         else if (par.HasLimits() ) 
            std::cout << "+/-  " << par.Error() << "\t(limited)"<< std::endl; 
         else 
            std::cout << "+/-  " << par.Error() << std::endl; 
      }
*/
   }
   else { 
/*
      std::cout << "HybridMinimizer : Invalid Minimum - status = " << fStatus << std::endl; 
      std::cout << "FVAL  = " << fState.Fval() << std::endl;
      std::cout << "Edm   = " << fState.Edm() << std::endl;
      std::cout << "Nfcn  = " << fState.NFcn() << std::endl;
*/
   }
}

const double * HybridMinimizer::X() const { 
   // return values at minimum 
   const std::vector<MinuitParameter> & paramsObj = fState.MinuitParameters();
   if (paramsObj.empty()) return nullptr;
   assert(fDim == paramsObj.size());
   // be careful for multiple calls of this function. I will redo an allocation here
   // only when size of vectors has changed (e.g. after a new minimization)
   if (fValues.size() != fDim) fValues.resize(fDim);
   for (unsigned int i = 0; i < fDim; ++i) { 
      fValues[i] = paramsObj[i].Value();
   }

   return  &fValues.front(); 
}


const double * HybridMinimizer::Errors() const { 
   // return error at minimum (set to zero for fixed and constant params)
   const std::vector<MinuitParameter> & paramsObj = fState.MinuitParameters();
   if (paramsObj.empty()) return nullptr;
   assert(fDim == paramsObj.size());
   // be careful for multiple calls of this function. I will redo an allocation here
   // only when size of vectors has changed (e.g. after a new minimization)
   if (fErrors.size() != fDim)   fErrors.resize( fDim );
   for (unsigned int i = 0; i < fDim; ++i) { 
      const MinuitParameter & par = paramsObj[i]; 
      if (par.IsFixed() || par.IsConst() ) 
         fErrors[i] = 0; 
      else 
         fErrors[i] = par.Error();
   }

   return  &fErrors.front(); 
}


double HybridMinimizer::CovMatrix(unsigned int i, unsigned int j) const { 
   // get value of covariance matrices (transform from external to internal indices)
   if ( i >= fDim || i >= fDim) return 0;  
   if (  !fState.HasCovariance()    ) return 0; // no info available when minimization has failed
   if (fState.Parameter(i).IsFixed() || fState.Parameter(i).IsConst() ) return 0; 
   if (fState.Parameter(j).IsFixed() || fState.Parameter(j).IsConst() ) return 0; 
   unsigned int k = fState.IntOfExt(i); 
   unsigned int l = fState.IntOfExt(j); 
   return fState.Covariance()(k,l); 
}

bool HybridMinimizer::GetCovMatrix(double * cov) const { 
   // get value of covariance matrices 
   if ( !fState.HasCovariance()    ) return false; // no info available when minimization has failed
   for (unsigned int i = 0; i < fDim; ++i) {
      if (fState.Parameter(i).IsFixed() || fState.Parameter(i).IsConst() ) {
         for (unsigned int j = 0; j < fDim; ++j) { cov[i*fDim + j] = 0; }          
      } 
      else 
      {
         unsigned int l = fState.IntOfExt(i); 
         for (unsigned int j = 0; j < fDim; ++j) { 
            // could probably speed up this loop (if needed)
            int k = i*fDim + j;
            if (fState.Parameter(j).IsFixed() || fState.Parameter(j).IsConst() ) 
               cov[k] = 0; 
            else {
            // need to transform from external to internal indices)
            // for taking care of the removed fixed row/columns in the Minuit2 representation
               unsigned int m = fState.IntOfExt(j); 
               cov[k] =  fState.Covariance()(l,m); 
            }
         }
      }
   }
   return true;
}

bool HybridMinimizer::GetHessianMatrix(double * hess) const { 
   // get value of Hessian matrix
   // this is the second derivative matrices
   if (  !fState.HasCovariance()    ) return false; // no info available when minimization has failed
   for (unsigned int i = 0; i < fDim; ++i) {
      if (fState.Parameter(i).IsFixed() || fState.Parameter(i).IsConst() ) {
         for (unsigned int j = 0; j < fDim; ++j) { hess[i*fDim + j] = 0; }          
      } 
      else { 
         unsigned int l = fState.IntOfExt(i); 
         for (unsigned int j = 0; j < fDim; ++j) { 
            // could probably speed up this loop (if needed)
            int k = i*fDim + j;
            if (fState.Parameter(j).IsFixed() || fState.Parameter(j).IsConst() ) 
               hess[k] = 0; 
            else { 
               // need to transform from external to internal indices)
               // for taking care of the removed fixed row/columns in the Minuit2 representation
               unsigned int m = fState.IntOfExt(j); 
               hess[k] =  fState.Hessian()(l,m); 
            }
         }
      }
   }

   return true;
}


double HybridMinimizer::Correlation(unsigned int i, unsigned int j) const { 
   // get correlation between parameter i and j 
   if ( i >= fDim || i >= fDim) return 0;  
   if (  !fState.HasCovariance()    ) return 0; // no info available when minimization has failed
   if (fState.Parameter(i).IsFixed() || fState.Parameter(i).IsConst() ) return 0; 
   if (fState.Parameter(j).IsFixed() || fState.Parameter(j).IsConst() ) return 0; 
   unsigned int k = fState.IntOfExt(i); 
   unsigned int l = fState.IntOfExt(j); 
   double cij =  fState.IntCovariance()(k,l); 
   double tmp =  std::sqrt( std::abs ( fState.IntCovariance()(k,k) * fState.IntCovariance()(l,l) ) );
   if (tmp > 0 ) return cij/tmp; 
   return 0; 
}

double HybridMinimizer::GlobalCC(unsigned int i) const { 
   // get global correlation coefficient for the parameter i. This is a number between zero and one which gives 
   // the correlation between the i-th parameter  and that linear combination of all other parameters which 
   // is most strongly correlated with i.

   if ( i >= fDim || i >= fDim) return 0;  
    // no info available when minimization has failed or has some problems
   if ( !fState.HasGlobalCC()    ) return 0; 
   if (fState.Parameter(i).IsFixed() || fState.Parameter(i).IsConst() ) return 0; 
   unsigned int k = fState.IntOfExt(i); 
   return fState.GlobalCC().GlobalCC()[k]; 
}


bool HybridMinimizer::GetMinosError(unsigned int i, double & errLow, double & errUp, int runopt) { 
   // return the minos error for parameter i
   // if a minimum does not exist an error is returned
   // runopt is a flag which specifies if only lower or upper error needs to be run
   // if runopt = 0 both, = 1 only lower, + 2 only upper errors
   errLow = 0; errUp = 0; 
   bool runLower = runopt != 2;
   bool runUpper = runopt != 1;

   assert( fMinuitFCN );

   // need to know if parameter is const or fixed 
   if ( fState.Parameter(i).IsConst() || fState.Parameter(i).IsFixed() ) { 
      return false; 
   }

   int debugLevel = PrintLevel(); 
   // internal minuit messages
   MnPrint::SetLevel( debugLevel );

   // to run minos I need function minimum class 
   // redo minimization from current state
//    ROOT::Minuit2::FunctionMinimum min =  
//       GetMinimizer()->Minimize(*GetFCN(),fState, ROOT::Minuit2::MnStrategy(strategy), MaxFunctionCalls(), Tolerance());
//    fState = min.UserState();
   if (fMinimum == nullptr) { 
      MN_ERROR_MSG("HybridMinimizer::GetMinosErrors:  failed - no function minimum existing");
      return false;
   }
   
   if (!fMinimum->IsValid() ) { 
      MN_ERROR_MSG("HybridMinimizer::MINOS failed due to invalid function minimum");
      return false;
   }

   fMinuitFCN->SetErrorDef(ErrorDef() );
   // if error def has been changed update it in FunctionMinimum
   if (ErrorDef() != fMinimum->Up() ) 
      fMinimum->SetErrorDef(ErrorDef() );

   // switch off Minuit2 printing
   int prev_level = (PrintLevel() <= 0 ) ?   TurnOffPrintInfoLevel() : -2; 

   // set the precision if needed
   if (Precision() > 0) fState.SetPrecision(Precision());


   ROOT::Minuit2::MnMinos minos( *fMinuitFCN, *fMinimum);

   // run MnCross 
   MnCross low;
   MnCross up;
   int maxfcn = MaxFunctionCalls(); 
   double tol = Tolerance();

//   const char * par_name = fState.Name(i);

   // now input tolerance for migrad calls inside Minos (MnFunctionCross)
   // before it was fixed to 0.05 
   // cut off too small tolerance (they are not needed)
   tol = std::max(tol, 0.01);
   
   if (PrintLevel() >=1) { 
      // print the real number of maxfcn used (defined in MnMinos)
      int maxfcn_used = maxfcn; 
      if (maxfcn_used == 0) { 
         int nvar = fState.VariableParameters();
         maxfcn_used = 2*(nvar+1)*(200 + 100*nvar + 5*nvar*nvar);
      }
//      std::cout << "HybridMinimizer::GetMinosError for parameter " << i << "  " << par_name
//                << " using max-calls " << maxfcn_used << ", tolerance " << tol << std::endl; 
   }


   if (runLower) low = minos.Loval(i,maxfcn,tol);
   if (runUpper) up  = minos.Upval(i,maxfcn,tol);
 
   ROOT::Minuit2::MinosError me(i, fMinimum->UserState().Value(i),low, up);

   if (prev_level > -2) RestoreGlobalPrintLevel(prev_level);

   // debug result of Minos 
   // print error message in Minos


   if (debugLevel >= 1) {
/*
      if (runLower) { 
         if (!me.LowerValid() )  
            std::cout << "Minos:  Invalid lower error for parameter " << par_name << std::endl; 
         if(me.AtLowerLimit()) 
            std::cout << "Minos:  Parameter : " << par_name << "  is at Lower limit."<<std::endl;
         if(me.AtLowerMaxFcn())
            std::cout << "Minos:  Maximum number of function calls exceeded when running for lower error" <<std::endl;   
         if(me.LowerNewMin() )
            std::cout << "Minos:  New Minimum found while running Minos for lower error" <<std::endl;     

         if (debugLevel > 1)  std::cout << "Minos: Lower error for parameter " << par_name << "  :  " << me.Lower() << std::endl; 

      }
      if (runUpper) {          
         if (!me.UpperValid() )  
            std::cout << "Minos:  Invalid upper error for parameter " << par_name << std::endl; 
         if(me.AtUpperLimit()) 
            std::cout << "Minos:  Parameter " << par_name << " is at Upper limit."<<std::endl;
         if(me.AtUpperMaxFcn())
            std::cout << "Minos:  Maximum number of function calls exceeded when running for upper error" <<std::endl;   
         if(me.UpperNewMin() )
            std::cout << "Minos:  New Minimum found while running Minos for upper error" <<std::endl;              

         if (debugLevel > 1)  std::cout << "Minos: Upper error for parameter " << par_name << "  :  " << me.Upper() << std::endl;
      }
*/      
   }

   bool lowerInvalid =  (runLower && !me.LowerValid() ); 
   bool upperInvalid =  (runUpper && !me.UpperValid() );
   int mstatus = 0; 
   if (lowerInvalid || upperInvalid ) { 
      // set status accroding to bit  
      // bit 1:  lower invalid Minos errors 
      // bit 2:  uper invalid Minos error
      // bit 3:   invalid because max FCN
      // bit 4 : invalid because a new minimum has been found
      if (lowerInvalid) { 
         mstatus |= 1;
         if (me.AtLowerMaxFcn() ) mstatus |= 4; 
         if (me.LowerNewMin() ) mstatus |= 8; 
      }
      if(upperInvalid) { 
         mstatus |= 3;
         if (me.AtUpperMaxFcn() ) mstatus |= 4; 
         if (me.UpperNewMin() ) mstatus |= 8; 
      }
      //std::cout << "Error running Minos for parameter " << i << std::endl; 
      fStatus += 10*mstatus; 
   }
         
   errLow = me.Lower();
   errUp = me.Upper();

   bool isValid = (runLower && me.LowerValid() ) || (runUpper && me.UpperValid() );    
   return isValid;  
} 

bool HybridMinimizer::Scan(unsigned int ipar, unsigned int & nstep, double * x, double * y, double xmin, double xmax) { 
   // scan a parameter (variable) around the minimum value
   // the parameters must have been set before 
   // if xmin=0 && xmax == 0  by default scan around 2 sigma of the error
   // if the errors  are also zero then scan from min and max of parameter range

   if (!fMinuitFCN) { 
      MN_ERROR_MSG2("HybridMinimizer::Scan"," Function must be set before using Scan");
      return false;
   }
   
   if ( ipar > fState.MinuitParameters().size() ) { 
      MN_ERROR_MSG2("HybridMinimizer::Scan"," Invalid number. Minimizer variables must be set before using Scan");
      return false;
   }

   // switch off Minuit2 printing
   int prev_level = (PrintLevel() <= 0 ) ?   TurnOffPrintInfoLevel() : -2; 

   MnPrint::SetLevel( PrintLevel() );


   // set the precision if needed
   if (Precision() > 0) fState.SetPrecision(Precision());

   MnParameterScan scan( *fMinuitFCN, fState.Parameters() );
   double amin = scan.Fval(); // fcn value of the function before scan 

   // first value is param value
   std::vector<std::pair<double, double> > result = scan(ipar, nstep-1, xmin, xmax);

   if (prev_level > -2) RestoreGlobalPrintLevel(prev_level);

   if (result.size() != nstep) { 
      MN_ERROR_MSG2("HybridMinimizer::Scan"," Invalid result from MnParameterScan");
      return false; 
   }
   // sort also the returned points in x
   std::sort(result.begin(), result.end() );


   for (unsigned int i = 0; i < nstep; ++i ) { 
      x[i] = result[i].first; 
      y[i] = result[i].second; 
   }

   // what to do if a new minimum has been found ? 
   // use that as new minimum
   if (scan.Fval() < amin ) { 
      if (PrintLevel() > 0) MN_INFO_MSG2("HybridMinimizer::Scan","A new minimum has been found");
      fState.SetValue(ipar, scan.Parameters().Value(ipar) );
         
   }


   return true; 
}

bool HybridMinimizer::Contour(unsigned int ipar, unsigned int jpar, unsigned int & npoints, double * x, double * y) {
   // contour plot for parameter i and j
   // need a valid FunctionMinimum otherwise exits
   if (fMinimum == nullptr) { 
      MN_ERROR_MSG2("HybridMinimizer::Contour"," no function minimum existing. Must minimize function before");
      return false;
   }

   if (!fMinimum->IsValid() ) { 
      MN_ERROR_MSG2("HybridMinimizer::Contour","Invalid function minimum");
      return false;
   }
   assert(fMinuitFCN); 

   fMinuitFCN->SetErrorDef(ErrorDef() );
   // if error def has been changed update it in FunctionMinimum
   if (ErrorDef() != fMinimum->Up() ) 
      fMinimum->SetErrorDef(ErrorDef() );

   // switch off Minuit2 printing (for level of  0,1)
   int prev_level = (PrintLevel() <= 1 ) ?   TurnOffPrintInfoLevel() : -2; 

   MnPrint::SetLevel( PrintLevel() );

   // set the precision if needed
   if (Precision() > 0) fState.SetPrecision(Precision());

   // eventually one should specify tolerance in contours 
   MnContours contour(*fMinuitFCN, *fMinimum, Strategy() ); 
   
   if (prev_level > -2) RestoreGlobalPrintLevel(prev_level);

   std::vector<std::pair<double,double> >  result = contour(ipar,jpar, npoints);
   if (result.size() != npoints) { 
      MN_ERROR_MSG2("HybridMinimizer::Contour"," Invalid result from MnContours");
      return false; 
   }
   for (unsigned int i = 0; i < npoints; ++i ) { 
      x[i] = result[i].first; 
      y[i] = result[i].second; 
   }


   return true;
   

}

bool HybridMinimizer::Hesse( ) { 
    // find Hessian (full second derivative calculations)
   // the contained state will be updated with the Hessian result
   // in case a function minimum exists and is valid the result will be 
   // appended in the function minimum

   if (!fMinuitFCN) { 
      MN_ERROR_MSG2("HybridMinimizer::Hesse","FCN function has not been set");
      return false; 
   }

   int strategy = Strategy();
   int maxfcn = MaxFunctionCalls(); 

   // switch off Minuit2 printing
   int prev_level = (PrintLevel() <= 0 ) ?   TurnOffPrintInfoLevel() : -2; 

   MnPrint::SetLevel( PrintLevel() );

   // set the precision if needed
   if (Precision() > 0) fState.SetPrecision(Precision());

   ROOT::Minuit2::MnHesse hesse( strategy );

   // case when function minimum exists
   if (fMinimum  ) { 
      // run hesse and function minimum will be updated with Hesse result
      hesse( *fMinuitFCN, *fMinimum, maxfcn ); 
      fState = fMinimum->UserState(); 
   }

   else { 
      // run Hesse on point stored in current state (independent of function minimum validity)
      // (x == 0) 
      fState = hesse( *fMinuitFCN, fState, maxfcn); 
   }

   if (prev_level > -2) RestoreGlobalPrintLevel(prev_level);

   if (PrintLevel() >= 3) { 
//      std::cout << "State returned from Hesse " << std::endl;
//      std::cout << fState << std::endl; 
   }

   if (!fState.HasCovariance() ) { 
      // if false means error is not valid and this is due to a failure in Hesse
      if (PrintLevel() > 0) MN_INFO_MSG2("HybridMinimizer::Hesse","Hesse failed ");
      // update minimizer error status 
      int hstatus = 4;
      // information on error state can be retrieved only if fMinimum is available
      if (fMinimum) { 
         if (fMinimum->Error().HesseFailed() ) hstatus = 1;
         if (fMinimum->Error().InvertFailed() ) hstatus = 2;
         else if (!(fMinimum->Error().IsPosDef()) ) hstatus = 3;
      }
      fStatus += 100*hstatus; 
      return false; 
   }

   return true;       
}

int HybridMinimizer::CovMatrixStatus() const { 
   // return status of covariance matrix 
   //-1 - not available (inversion failed or Hesse failed) 
   // 0 - available but not positive defined
   // 1 - covariance only approximate
   // 2 full matrix but forced pos def 
   // 3 full accurate matrix 

   if (fMinimum) {
      // case a function minimum  is available 
      if (fMinimum->HasAccurateCovar() ) return 3; 
      else if (fMinimum->HasMadePosDefCovar() ) return 2; 
      else if (fMinimum->HasValidCovariance() ) return 1; 
      else if (fMinimum->HasCovariance() ) return 0; 
      return -1;
   }
   else { 
      // case fMinimum is not available - use state information
      return fState.CovarianceStatus();
   }
   return 0; 
}

}
