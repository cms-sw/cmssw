#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTReadOutMappingHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<DTReadOutMappingHandler> DTReadOutMappingPopConAnalyzer;


DEFINE_FWK_MODULE(DTReadOutMappingPopConAnalyzer);

