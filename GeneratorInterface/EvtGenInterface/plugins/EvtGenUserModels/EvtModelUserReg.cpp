#include "GeneratorInterface/EvtGenInterface/interface/EvtGenUserModels/EvtModelUserReg.h"
#include "GeneratorInterface/EvtGenInterface/interface/EvtGenUserModels/EvtLb2plnuLCSR.hh"
#include <list>

std::list<EvtDecayBase*> EvtModelUserReg::getUserModels(){

  // Create user models
  EvtLb2plnuLCSR* EvtLb2plnuLCSRModel = new EvtLb2plnuLCSR();

  std::list<EvtDecayBase*> extraModels;
  extraModels.push_back(EvtLb2plnuLCSRModel);

  // Return the list of models
  return extraModels;
	
}

