#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RunInfo/interface/L1TriggerScalerHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<L1TriggerScalerHandler> L1TriggerScalerPopConAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(L1TriggerScalerPopConAnalyzer);
