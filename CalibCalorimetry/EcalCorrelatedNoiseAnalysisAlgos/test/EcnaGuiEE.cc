//------------------------- EcnaGuiEE.cc ------------------------
//
//         E.C.N.A.  dialog box (GUI) for Endcap
// 
//         Update:13/10/2010
//
//---------------------------------------------------------------
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaGui.h"
#include <cstdlib>

#include "Riostream.h"
#include "TROOT.h"
#include "TGApplication.h"
#include "TGClient.h"
#include "TRint.h"

#include <stdlib.h>
#include <string>
#include "TSystem.h"
#include "TObject.h"
#include "TGWindow.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParPaths.h"

extern void InitGui();
VoidFuncPtr_t initfuncs[] = { InitGui, 0 };
TROOT root("GUI","GUI test environnement", initfuncs);

using namespace std;

int main(int argc, char **argv)
{
  TEcnaParPaths* pCnaParPaths = new TEcnaParPaths();
  if( pCnaParPaths->fPathForResultsRootFiles    == kTRUE &&
      pCnaParPaths->fPathForResultsAsciiFiles   == kTRUE &&
      pCnaParPaths->fPathForHistoryRunListFiles == kTRUE )
    {
      cout << "*EcnaGuiEE> Starting ROOT session" << endl;
      TRint theApp("App", &argc, argv);
      
      cout << "*EcnaGuiEE> Starting ECNA session" << endl;
      TEcnaGui* mainWin = new TEcnaGui(gClient->GetRoot(), 395, 710, "EE");
      Bool_t retVal = kTRUE;
      theApp.Run(retVal);
      cout << "*EcnaGuiEE> End of ECNA session." << endl;
      delete mainWin;
      
      cout << "*EcnaGuiEE> End of ROOT session." << endl;
      theApp.Terminate(0);
      cout << "*EcnaGuiEE> Exiting main program." << endl;
      exit(0);
    }
}
