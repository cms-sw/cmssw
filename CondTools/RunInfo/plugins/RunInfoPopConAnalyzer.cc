#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RunInfo/interface/RunInfoHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<RunInfoHandler> RunInfoPopConAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(RunInfoPopConAnalyzer);
