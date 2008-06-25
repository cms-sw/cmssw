#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/RunInfo/interface/RunNumberHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<RunNumberHandler> RunNumberPopConAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(RunNumberPopConAnalyzer);
