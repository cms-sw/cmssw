#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTDeadFlagHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTDeadFlagHandler> DTDeadFlagPopConAnalyzer;


DEFINE_FWK_MODULE(DTDeadFlagPopConAnalyzer);


