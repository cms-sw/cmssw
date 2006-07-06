#include "JetMETCorrections/TauJet/interface/TauJetHelper.h"
#include "JetMETCorrections/Utilities/interface/GetParameters.h"

#include <string>
using namespace std;

  
std::string TauJetHelper::getCalibrationType(JetParameters)
{
   std::string theType = "Kt_R1.0_EScheme_TowerEt0.5_E0.8_Jets873_2x1033PU_qcd";
   return theType;
}

