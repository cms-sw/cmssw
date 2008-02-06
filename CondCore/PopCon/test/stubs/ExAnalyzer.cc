#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "ExSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::ExPedestalSource> ExPopConAnalyzer1;
//define this as a plug-in
DEFINE_FWK_MODULE(ExPopConAnalyzer1);


