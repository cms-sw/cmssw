#ifndef TauElementsOperators_H_
#define TauElementsOperators_H_

#include "DataFormats/TauReco/interface/Tau.h"

#include "TFormula.h"

using namespace edm;
using namespace std;
using namespace reco;

class TauElementsOperators{
 public:
  TauElementsOperators(Tau& theTau) : Tau_(theTau){}
  ~TauElementsOperators(){}   
  // compute size of signal cone possibly depending on E(energy) and/or ET(transverse energy) of the tau-jet candidate
  double computeConeSize(const TFormula& ConeSizeTFormula,double ConeSizeMin,double ConeSizeMax);
  TFormula computeConeSizeTFormula(const string& ConeSizeFormula,const char* errorMessage);
  void replaceSubStr(string& s,const string& oldSubStr,const string& newSubStr); 
 protected:
  Tau& Tau_;
};
#endif
