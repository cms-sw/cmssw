#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RunInfo/interface/RunSummaryHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<RunSummaryHandler> RunSummaryPopConAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(RunSummaryPopConAnalyzer);
