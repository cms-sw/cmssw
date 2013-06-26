#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DQM/interface/DQMSummarySourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::DQMSummarySourceHandler> DQMSummaryPopConAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(DQMSummaryPopConAnalyzer);
