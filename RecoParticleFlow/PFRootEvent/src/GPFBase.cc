#include "RecoParticleFlow/PFRootEvent/interface/DisplayManager.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFBase.h"

//__________________________________________________________________
GPFBase::GPFBase( DisplayManager *display,int viewType,int ident, int color) :
  display_(display),viewId_(viewType),origId_(ident), color_(color)
{
}                
//_____________________________________________________________________
