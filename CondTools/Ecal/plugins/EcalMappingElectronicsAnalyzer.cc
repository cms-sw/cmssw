/** Declaration of the analyzer class for EcalMappingElectronics
    
    \author Stefano Argiro
    $Id: EcalMappingElectronicsAnalyzer.cc,v 1.1 2008/11/14 15:46:04 argiro Exp $
 */

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalMappingElectronicsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<EcalMappingElectronicsHandler> EcalMappingElectronicsAnalyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(EcalMappingElectronicsAnalyzer);
