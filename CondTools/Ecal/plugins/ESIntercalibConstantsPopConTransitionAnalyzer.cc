#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "CondTools/Ecal/interface/PopConESTransitionSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer< popcon::PopConESTransitionSourceHandler<ESIntercalibConstants> > ESIntercalibConstantsPopConTransitionAnalyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(ESIntercalibConstantsPopConTransitionAnalyzer);
