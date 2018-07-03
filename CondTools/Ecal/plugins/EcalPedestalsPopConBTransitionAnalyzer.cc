#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondTools/RunInfo/interface/PopConBTransitionSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer< popcon::PopConBTransitionSourceHandler<EcalPedestals> > EcalPedestalsPopConBTransitionAnalyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(EcalPedestalsPopConBTransitionAnalyzer);
