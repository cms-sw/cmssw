#ifndef __RscCompModel__
#define __RscCompModel__

/**
\class RscCompModel
$Revision: 1.1.1.1 $
$Date: 2009/04/15 08:40:01 $
\author G. Schott (gregory.Schott<at>cern.ch) - Universitaet Karlsruhe
**/

#include <iostream>
#include "RooCategory.h"
#include "RooFormulaVar.h"

#include "PhysicsTools/RooStatsCms/interface/RscAbsPdfBuilder.h"

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
};

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
