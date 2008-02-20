#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "ExSourceHandler_new.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::ExPedestalSource_new> ExPopConAnalyzer2;
//define this as a plug-in
DEFINE_FWK_MODULE(ExPopConAnalyzer2);


