#include "RecoParticleFlow/PFRootEvent/interface/DisplayManagerUpgrade.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFBaseUpgrade.h"

//__________________________________________________________________
GPFBaseUpgrade::GPFBaseUpgrade( DisplayManagerUpgrade *display,
                  int viewType,
                  int ident,
                  TAttMarker *attm)
  :display_(display),viewId_(viewType),origId_(ident),
   markerAttr_(attm),lineAttr_(0),color_(0)
{

}                
//__________________________________________________________________
GPFBaseUpgrade::GPFBaseUpgrade( DisplayManagerUpgrade *display,
                  int viewType,
                  int ident,
                  TAttMarker *attm,
                  TAttLine *attl )
  :display_(display),viewId_(viewType),origId_(ident),
   markerAttr_(attm),lineAttr_(attl),color_(0)
{
}                
//__________________________________________________________________
GPFBaseUpgrade::GPFBaseUpgrade( DisplayManagerUpgrade *display,
                  int viewType,
                  int ident,
                  int color )
  :display_(display),viewId_(viewType),origId_(ident),
   markerAttr_(0),lineAttr_(0),color_(color)
{
}                
