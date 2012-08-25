#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/Services/src/SiteLocalConfigService.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "MuonAnalysis/MomentumScaleCalibration/plugins/MuScleFit.h"
#include "MuonAnalysis/MomentumScaleCalibration/plugins/MuScleFitFilter.h"
#include "MuonAnalysis/MomentumScaleCalibration/plugins/MuScleFitGenFilter.h"
#include "MuonAnalysis/MomentumScaleCalibration/plugins/TestCorrection.h"

DEFINE_FWK_LOOPER(MuScleFit);
DEFINE_FWK_MODULE(MuScleFitFilter);
DEFINE_FWK_MODULE(MuScleFitGenFilter);
DEFINE_FWK_MODULE(TestCorrection);


