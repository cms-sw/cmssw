#ifndef __RscCatModel__
#define __RscCatModel__

/**
\class RscCatModel
$Revision: 1.1.1.1 $
$Date: 2009/04/15 08:40:01 $
\author G. Schott (gregory.Schott<at>cern.ch) - Universitaet Karlsruhe
**/

#include <iostream>
#include "RooCategory.h"

#include "PhysicsTools/RooStatsCms/interface/RscAbsPdfBuilder.h"

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
  #endif /*SWIG */
};

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
