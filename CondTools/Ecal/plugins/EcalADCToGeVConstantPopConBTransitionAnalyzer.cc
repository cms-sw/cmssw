#include "CondCore/PopCon/interface/PopConBTransitionSourceHandler.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer< popcon::PopConBTransitionSourceHandler<EcalADCToGeVConstant> > EcalADCToGeVConstantPopConBTransitionAnalyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(EcalADCToGeVConstantPopConBTransitionAnalyzer);
