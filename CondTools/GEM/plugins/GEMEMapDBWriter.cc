#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/GEM/interface/GEMEMapSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::GEMEMapSourceHandler> GEMEMapDBWriter;
//define this as a plug-in
DEFINE_FWK_MODULE(GEMEMapDBWriter);
