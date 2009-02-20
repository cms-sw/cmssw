#ifndef __RscMultiModel__
#define __RscMultiModel__

#include <iostream>
#include "RooArgList.h"
#include "PhysicsTools/RooStatsCms/interface/RscAbsPdfBuilder.h"

/**
\class RscMultiMOdel
$Revision: 1.1 $
$Date: 2009/01/06 12:18:37 $
\author G. Schott (gregory.Schott<at>cern.ch) - Universitaet Karlsruhe
**/

class RscMultiModel : public RscAbsPdfBuilder {

public:
  RscMultiModel(TString theName, RooArgList& theVars);
  ~RscMultiModel();

  //RooRealVar* getYield();
  RooFormulaVar* getYield();

  void readDataCard();
  void writeDataCard(ostream& out);
  
private:
  TString _name;
  RooArgList* _vars;
  Bool_t ownPDFs;
  TList* pdfs;

  /// Pdf Yield as a formulavar for decomposition
  RooFormulaVar* _yield;

  /// Hold here the non extended Pdf
  RooAbsPdf* _non_ext_pdf;

  void buildPdf();

  #ifndef SWIG
  //ClassDef(RscMultiModel,1)
  #endif /*SWIG */
};

#endif
