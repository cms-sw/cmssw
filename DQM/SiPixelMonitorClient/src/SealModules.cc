#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelEDAClient.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelDaqInfo.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelDcsInfo.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelCertification.h"


DEFINE_FWK_MODULE(SiPixelEDAClient);
DEFINE_FWK_MODULE(SiPixelDaqInfo);
DEFINE_FWK_MODULE(SiPixelDcsInfo);
DEFINE_FWK_MODULE(SiPixelCertification);
