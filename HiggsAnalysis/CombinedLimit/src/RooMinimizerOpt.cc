#include "../interface/RooMinimizerOpt.h"

#include <stdexcept>
#include <RooRealVar.h>
#include <RooAbsPdf.h>
#include <RooMsgService.h>

#include <Math/MinimizerOptions.h>

RooMinimizerOpt::RooMinimizerOpt(RooAbsReal& function) :
    RooMinimizer(function)
{
    delete _fcn;
    _fcn = new RooMinimizerFcnOpt(_func,this,_verbose); 
    setEps(ROOT::Math::MinimizerOptions::DefaultTolerance());
}

Double_t
RooMinimizerOpt::edm()
{
    if (_theFitter == 0) throw std::logic_error("Must have done a fit before calling edm()");
    return _theFitter->Result().Edm();    
}




RooMinimizerFcnOpt::RooMinimizerFcnOpt(RooAbsReal *funct, RooMinimizer *context,  bool verbose) :
    RooMinimizerFcn(funct, context, verbose)
{
    _vars.resize(_floatParamList->getSize());
    std::vector<RooRealVar *>::iterator itv = _vars.begin();
    RooLinkedListIter iter = _floatParamList->iterator();
    for (TObject *a = iter.Next(); a != 0; a = iter.Next(), ++itv) {
        RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
        if (rrv == 0) throw std::logic_error(Form("Float param not a RooRealVar but a %s", a->ClassName()));
        *itv = rrv; 
    }
}

ROOT::Math::IBaseFunctionMultiDim* 
RooMinimizerFcnOpt::Clone() const
{
      return new RooMinimizerFcnOpt(_funct,_context,_verbose);
}

double
RooMinimizerFcnOpt::DoEval(const double * x) const 
{
  // Set the parameter values for this iteration
  for (int index = 0; index < _nDim; index++) {
      if (_logfile) (*_logfile) << x[index] << " " ;
      RooRealVar* par = _vars[index];
      if (par->getVal()!=x[index]) {
          if (_verbose) cout << par->GetName() << "=" << x[index] << ", " ;
          par->setVal(x[index]);
      }
  }

  // Calculate the function for these parameters
  double fvalue = _funct->getVal();
  if (RooAbsPdf::evalError() || RooAbsReal::numEvalErrors()>0) {

    if (_printEvalErrors>=0) {

      if (_doEvalErrorWall) {
        oocoutW(_context,Minimization) << "RooMinimizerFcn: Minimized function has error status." << endl 
				       << "Returning maximum FCN so far (" << _maxFCN 
				       << ") to force MIGRAD to back out of this region. Error log follows" << endl ;
      } else {
        oocoutW(_context,Minimization) << "RooMinimizerFcn: Minimized function has error status but is ignored" << endl ;
      } 

      TIterator* iter = _floatParamList->createIterator() ;
      RooRealVar* var ;
      Bool_t first(kTRUE) ;
      ooccoutW(_context,Minimization) << "Parameter values: " ;
      while((var=(RooRealVar*)iter->Next())) {
        if (first) { first = kFALSE ; } else ooccoutW(_context,Minimization) << ", " ;
        ooccoutW(_context,Minimization) << var->GetName() << "=" << var->getVal() ;
      }
      delete iter ;
      ooccoutW(_context,Minimization) << endl ;
      
      RooAbsReal::printEvalErrors(ooccoutW(_context,Minimization),_printEvalErrors) ;
      ooccoutW(_context,Minimization) << endl ;
    } 

    if (_doEvalErrorWall) {
      fvalue = _maxFCN ;
    }

    RooAbsPdf::clearEvalError() ;
    RooAbsReal::clearEvalErrorLog() ;
    _numBadNLL++ ;
  } else if (fvalue>_maxFCN) {
    _maxFCN = fvalue ;
  }
      
  // Optional logging
  if (_logfile) 
    (*_logfile) << setprecision(15) << fvalue << setprecision(4) << endl;
  if (_verbose) {
    cout << "\nprevFCN = " << setprecision(10) 
         << fvalue << setprecision(4) << "  " ;
    cout.flush() ;
  }

  return fvalue;
}
