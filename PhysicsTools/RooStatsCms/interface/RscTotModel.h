#ifndef __RscTotModel__
#define __RscTotModel__

/// RscTotModel: An analysis resumed

/**
\class RscTotModel
$Revision: 1.1 $
$Date: 2009/01/06 12:18:38 $
\author G. Schott (gregory.Schott<at>cern.ch), Danilo Piparo - Universitaet Karlsruhe
**/

#include "PhysicsTools/RooStatsCms/interface/RscCompModel.h"

#include "RooExtendPdf.h"
#include "RooStringVar.h"

class RscTotModel : public RscAbsPdfBuilder {

public:
  RscTotModel(TString theName);
  ~RscTotModel();

  RooAbsPdf* getExtendedBkgPdf();
  RooAbsPdf* getExtendedSigPdf();

  RooArgList* getVars();
  RooRealVar* getVar(TString varName);
  RooArgList* getConstraints();

  RooFormulaVar* getSigYield();
  RooFormulaVar* getBkgYield();

  //DP might be useful
  const char* getName(){return _name.Data();};

  void readDataCard();
  void writeDataCard(ostream& out);
  
private:
  TString _name;

  RooStringVar variablesString;

  RooArgList* theConstraints;
  RooArgList* theVars;

  RscCompModel* bkgModel;
  RscCompModel* sigModel;

  RooExtendPdf* sigExtendedPdf;
  RooExtendPdf* bkgExtendedPdf;

  void buildPdf();
  void buildVars();
  void buildConstraints();

  #ifndef SWIG
  //ClassDef(RscTotModel,1)
  #endif /*SWIG */
};

#endif
