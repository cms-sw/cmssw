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



void getDecomposedVersion(const TString& s, int* out)
{
   TPMERegexp re("CMSSW_(\\d+)_(\\d+)_");
   re.Match(s);
   if (re.NMatches() > 2)
   {
      out[0] = re[1].Atoi();
      out[1] = re[2].Atoi();
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
         std::ifstream fs(versionFileName);
         TString infoText;
         infoText.ReadLine(fs); infoText.ReadLine(fs);
         fs.close();
         v = &infoText[13];
      }

      getDecomposedVersion(v, &mm[0]);
   }

   return &mm[0];
}

bool acceptDataFormatsVersion(TString& processConfigurationVersion)
{
   int data[] = {0, 0, 0};
   getDecomposedVersion(processConfigurationVersion, data);


   int* running = supportedDataFormatsVersion();
   if ((data[0] == 6 && running[0] == 5 && running[1] > 1) ||
       (data[0] == 5 && data[1] > 1 && running[0] == 6))
      return true;
   else
      return data[0] == running[0];
}

}
