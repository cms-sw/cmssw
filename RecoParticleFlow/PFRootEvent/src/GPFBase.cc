#include "RecoParticleFlow/PFRootEvent/interface/DisplayManager.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFBase.h"

//__________________________________________________________________
GPFBase::GPFBase( DisplayManager *display,int viewType,int ident) :
                 display_(display),viewId_(viewType),origId_(ident)
{
}		 
//_____________________________________________________________________
