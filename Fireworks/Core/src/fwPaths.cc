#include "Fireworks/Core/interface/fwPaths.h"
#include "Fireworks/Core/interface/fwLog.h" 	 
#include "TString.h"
#include "TSystem.h"
#include "TPRegexp.h"
#include <iostream>
#include <fstream>

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



void getDecomposedVersion(TString& s, int* out)

{
   static TPMERegexp re("CMSSW_(\\d+)_(\\d+)_(\\d+)", "g");
   re.Match(s);

   if (re.NMatches() > 3)
   {
      out[0] = atoi(re[1]);
      out[1] = atoi(re[2]);
      out[2] = atoi(re[3]);
   }
}



int* supportedDataFormatsVersion()
{
   static int mm[] = {0, 0, 0};

   if (!mm[0]) {
      TString v;
      if (gSystem->Getenv("CMSSW_VERSION"))
      {
         v = gSystem->Getenv("CMSSW_VERSION");

      }
      else
      {
         TString versionFileName = gSystem->Getenv("CMSSW_BASE");
         versionFileName += "/src/Fireworks/Core/data/version.txt";
         ifstream fs(versionFileName);
         TString infoText;
         infoText.ReadLine(fs); infoText.ReadLine(fs);
         fs.close();
         v = &infoText[13];
      }

      getDecomposedVersion(v, &mm[0]);
   }

   return &mm[0];
}

bool acceptDataFormatsVersion(TString& n)
{
   int v[] = {0, 0, 0};
   getDecomposedVersion(n, v);
   return v[0] == supportedDataFormatsVersion()[0];
}

}
