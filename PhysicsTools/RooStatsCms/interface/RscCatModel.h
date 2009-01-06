#ifndef __RscCatModel__
#define __RscCatModel__

/**
\class RscCatModel
$Revision: 1.2 $
$Date: 2008/10/06 12:32:41 $
\author G. Schott (gregory.Schott<at>cern.ch) - Universitaet Karlsruhe
**/

#include <iostream>
#include "RooCategory.h"

#include "RscAbsPdfBuilder.h"

class RscCatModel : public RscAbsPdfBuilder {

public:
  RscCatModel(TString theName, RooRealVar& theVar, RooArgSet* discVars=0);
  ~RscCatModel();

  RooRealVar number_categories;
  RooCategory extended_pdf;

  RooRealVar* x; // discriminating variable
  RooRealVar yield;

  TList* pdfs;

  void readDataCard();
  void writeDataCard(ostream& out);
  
private:
  TString _name;
  Bool_t ownCategories;
  Bool_t extendedPdf;

  void buildPdf();
  #ifndef SWIG
  ClassDef(RscCatModel,1)
  #endif /*SWIG */
};

#endif
