#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RunInfo/interface/HLTScalerHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<lumi::HLTScalerHandler> HLTScalerPopConAnalyzer;
DEFINE_FWK_MODULE(HLTScalerPopConAnalyzer);
