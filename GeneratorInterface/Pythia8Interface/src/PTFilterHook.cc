#include "Pythia8/Pythia.h"
#include "GeneratorInterface/Pythia8Interface/interface/PTFilterHook.h"

using namespace Pythia8;


//--------------------------------------------------------------------------
bool PTFilterHook::initAfterBeams() {
  filter_  = settingsPtr->flag("PTFilter:filter");
  quark_   = settingsPtr->mode("PTFilter:quarkToFilter");
  scale_   = settingsPtr->parm("PTFilter:scaleToFilter");
  quarkY_  = settingsPtr->parm("PTFilter:quarkRapidity");
  quarkPt_ = settingsPtr->parm("PTFilter:quarkPt");
  
  return true;
  
}

//--------------------------------------------------------------------------
bool PTFilterHook::checkVetoPT( int iPos, const Pythia8::Event& event) {

  if (!filter_ || iPos >3 ) return false;
  bool foundQuark = false;
  
  //look for quark
  for (int i = 0; i < event.size(); ++i) {
    if ( (abs(event[i].id()) == quark_ ) && ( abs(event[i].y()) <= quarkY_ ) ) {
       if (event[i].pT() >= quarkPt_) { 
          foundQuark = true;
          break;
       }
    }
  }

  if (!foundQuark) return true;
  //all criteria satisfied, don't veto
  return false;
}
