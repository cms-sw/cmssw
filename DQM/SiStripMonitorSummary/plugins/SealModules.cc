#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "DQM/SiStripMonitorSummary/plugins/SiStripCorrelateBadStripAndNoise.h"
DEFINE_FWK_MODULE(SiStripCorrelateBadStripAndNoise);

#include "DQM/SiStripMonitorSummary/plugins/SiStripCorrelateNoise.h"
DEFINE_FWK_MODULE(SiStripCorrelateNoise);

#include "DQM/SiStripMonitorSummary/plugins/SiStripPlotGain.h"
DEFINE_FWK_MODULE(SiStripPlotGain);
