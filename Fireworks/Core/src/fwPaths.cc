#include "Fireworks/Core/interface/fwPaths.h"
#include "TString.h"
#include "TSystem.h"
#include <stdio>
#include <iostream>

namespace fireworks  {
const TString p1(gSystem->Getenv("CMSSW_BASE/Fireworks/Core"));
const TString p2(gSystem->Getenv("CMSSW_RELEASE_BASE/Fireworks/Core"));


void setPath( TString& v)
{
   if (gSystem->AccessPathName(p1 + v) == kFALSE)
   {
      v.Prepend(p1);
      return;
   }

   v.Prepend(p2);
   std::cout << "set PATH " << v << std::endl;
}
}



