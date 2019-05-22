#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondFormats/ESObjects/interface/ESRecHitRatioCuts.h"
#include "CondTools/Ecal/interface/PopConESTransitionSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer< popcon::PopConESTransitionSourceHandler<ESRecHitRatioCuts> > ESRecHitRatioCutsPopConTransitionAnalyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(ESRecHitRatioCutsPopConTransitionAnalyzer);
