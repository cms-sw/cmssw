#include "Fireworks/Core/interface/fwPaths.h"
#include "Fireworks/Core/interface/fwLog.h" 	 
#include "TString.h"
#include "TSystem.h"

#include <iostream>

namespace fireworks  {

const TString datadir("/src/Fireworks/Core/") ; 	 
const TString p1 = gSystem->Getenv("CMSSW_BASE") + datadir; 	 
const TString p2 = gSystem->Getenv("CMSSW_RELEASE_BASE") + datadir;

void setPath( TString& v)
{
   if (gSystem->AccessPathName(p1 + v) == kFALSE)
   {
      v.Prepend(p1);
      return;
   }

   v.Prepend(p2);
   if (gSystem->AccessPathName(v))
   fwLog(fwlog::kError) << "Can't access path " << v << std::endl;
}
}



