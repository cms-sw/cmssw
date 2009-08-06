/** Declaration of the analyzer class for EcalintercalibErrors
    
    \author Stefano Argiro
    $Id: EcalIntercalibErrorsAnalyzer.cc,v 1.1 2008/10/22 08:41:30 argiro Exp $
 */

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalIntercalibErrorsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<EcalIntercalibErrorsHandler>  
                                         EcalIntercalibErrorsAnalyzer;


//define this as a plug-in
DEFINE_FWK_MODULE(EcalIntercalibErrorsAnalyzer);
