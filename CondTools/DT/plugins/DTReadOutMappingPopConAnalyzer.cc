#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DT/interface/DTReadOutMappingHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/Common/interface/Serialization.h"
#include "CondFormats/DTObjects/interface/Serialization.h"

typedef popcon::PopConAnalyzer<DTReadOutMappingHandler> DTReadOutMappingPopConAnalyzer;


DEFINE_FWK_MODULE(DTReadOutMappingPopConAnalyzer);

