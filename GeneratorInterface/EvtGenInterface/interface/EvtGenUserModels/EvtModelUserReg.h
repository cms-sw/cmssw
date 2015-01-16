#ifndef gen_EvtModelUserReg_h
#define gen_EvtModelUserReg_h

#include <memory>
#include "EvtGenModels/EvtModelReg.hh"
/** 
 * Provides a list of user defined decay models to EvtGen.
*/
//typedef std::list<EvtDecayBase*> EvtModelList;

class EvtModelUserReg
{
public:
  std::list<EvtDecayBase*> getUserModels();
};

#endif /*EvtModelUserReg*/
