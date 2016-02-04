#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/DQM/interface/DQMReferenceHistogramRootFileSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::DQMReferenceHistogramRootFileSourceHandler> DQMReferenceHistogramRootFilePopConAnalyzer;
//define this as a plug-in
DEFINE_FWK_MODULE(DQMReferenceHistogramRootFilePopConAnalyzer);
