#ifndef __RscCompModel__
#define __RscCompModel__

/**
\class RscCompModel
$Revision: 1.3 $
$Date: 2009/04/15 11:10:45 $
\author G. Schott (gregory.Schott<at>cern.ch) - Universitaet Karlsruhe
**/

#include <iostream>
#include "RooCategory.h"
#include "RooFormulaVar.h"

#if (defined (STANDALONE) or defined (__CINT__) )
   #include "RscAbsPdfBuilder.h"
#else
   #include "PhysicsTools/RooStatsCms/interface/RscAbsPdfBuilder.h"
#endif

class RscCompModel : public RscAbsPdfBuilder {

public:
  RscCompModel(TString theName, RooArgList& theVars);
  RscCompModel(TString theName, RooArgList& theVars, RooArgList& caller_vars);

  ~RscCompModel();

  RooRealVar number_components;

  RooFormulaVar* getYield();

  void readDataCard();
  void writeDataCard(ostream& out);

private:
  TString _name;

  RooArgList* _vars;

  /// Allow the instance to add to a RooArgList the read vars
  RooArgList* _caller_vars;

  Bool_t ownPDFs;

  TList* pdfs;

  //DP test the RooFormulaVar
  RooFormulaVar* _yield;
  //RooRealVar* _yield; 

  void buildPdf();
  #ifndef SWIG
  #endif /*SWIG */
//For Cint
#if (defined (STANDALONE) or defined (__CINT__) )
ClassDef(RscCompModel,1)
#endif
};

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
