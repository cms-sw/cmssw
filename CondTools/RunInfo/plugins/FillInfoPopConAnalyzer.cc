#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RunInfo/interface/FillInfoPopConSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<FillInfoPopConSourceHandler> FillInfoPopConAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(FillInfoPopConAnalyzer);
