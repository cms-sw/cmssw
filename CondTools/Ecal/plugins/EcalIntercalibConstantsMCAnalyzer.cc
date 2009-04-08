/** Declaration of the analyzer class for EcalintercalibConstants
    
    \author Stefano Argiro
    $Id: EcalIntercalibConstantsAnalyzer.cc,v 1.1 2008/11/14 15:46:04 argiro Exp $
 */

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalIntercalibConstantsMCHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<EcalIntercalibConstantsMCHandler>  
                                         EcalIntercalibConstantsMCAnalyzer;


//define this as a plug-in
DEFINE_FWK_MODULE(EcalIntercalibConstantsMCAnalyzer);
