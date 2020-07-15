//------------------------- EcnaGuiEE.cc ------------------------
//
//         E.C.N.A.  dialog box (GUI) for Endcap
//
//         Update: 21/10/2010
//
//---------------------------------------------------------------
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaGui.h"
#include <cstdlib>

#include "Riostream.h"
#include "TROOT.h"
#include "TGApplication.h"
#include "TGClient.h"
#include "TRint.h"

#include "TGWindow.h"
#include "TObject.h"
#include "TSystem.h"
#include <cstdlib>
#include <string>

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEcnaParPaths.h"

extern void InitGui();
VoidFuncPtr_t initfuncs[] = {InitGui, nullptr};
TROOT root("GUI", "GUI test environnement", initfuncs);

using namespace std;

int main(int argc, char** argv) {
  TEcnaObject* MyEcnaObjectManager = new TEcnaObject();
  TEcnaParPaths* pCnaParPaths = new TEcnaParPaths(MyEcnaObjectManager);
  if (pCnaParPaths->GetPaths() == kTRUE) {
    std::cout << "*EcnaGuiEE> Starting ROOT session" << std::endl;
    TRint theApp("App", &argc, argv);

    std::cout << "*EcnaGuiEE> Starting ECNA session" << std::endl;
    TEcnaGui* mainWin = new TEcnaGui(MyEcnaObjectManager, "EE", gClient->GetRoot(), 395, 710);
    mainWin->DialogBox();
    Bool_t retVal = kTRUE;
    theApp.Run(retVal);
    std::cout << "*EcnaGuiEE> End of ECNA session." << std::endl;
    delete mainWin;

    std::cout << "*EcnaGuiEE> End of ROOT session." << std::endl;
    theApp.Terminate(0);
    std::cout << "*EcnaGuiEE> Exiting main program." << std::endl;
    exit(0);
  }
}
