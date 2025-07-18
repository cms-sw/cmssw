#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RunInfo/interface/LHCInfoPerFillPopConSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<LHCInfoPerFillPopConSourceHandler> LHCInfoPerFillPopConAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(LHCInfoPerFillPopConAnalyzer);
