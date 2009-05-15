#ifndef __RscCatModel__
#define __RscCatModel__

/**
\class RscCatModel
$Revision: 1.3 $
$Date: 2009/04/15 12:33:55 $
\author G. Schott (gregory.Schott<at>cern.ch) - Universitaet Karlsruhe
**/

#include <iostream>
#include "RooCategory.h"

#if (defined (STANDALONE) or defined (__CINT__) )
   #include "RscAbsPdfBuilder.h"
#else
   #include "PhysicsTools/RooStatsCms/interface/RscAbsPdfBuilder.h"
#endif

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
//For Cint
#if (defined (STANDALONE) or defined (__CINT__) )
ClassDef(RscCatModel,1)
#endif
};

#endif
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
