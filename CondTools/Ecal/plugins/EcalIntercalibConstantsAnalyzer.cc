/** Declaration of the analyzer class for EcalintercalibConstants
    
    \author Stefano Argiro
    $Id: EcalIntercalibConstantsAnalyzer.cc,v 1.1 2008/10/22 08:41:30 argiro Exp $
 */

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalIntercalibConstantsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<EcalIntercalibConstantsHandler>  
                                         EcalIntercalibConstantsAnalyzer;


//define this as a plug-in
DEFINE_FWK_MODULE(EcalIntercalibConstantsAnalyzer);
