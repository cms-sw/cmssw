#include "GeneratorInterface/EvtGenInterface/plugins/EvtGenUserModels/EvtModelUserReg.h"
#include <list>

std::list<EvtDecayBase*> EvtModelUserReg::getUserModels() {
  // Create user models

  std::list<EvtDecayBase*> extraModels;

  // Return the list of models
  return extraModels;
}
