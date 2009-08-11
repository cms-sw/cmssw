#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_SEAL_MODULE();

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

// //SiStrip private (and original) approach
#include "DQMOffline/CalibTracker/plugins/SiStripBadComponentsDQMService.h"
DEFINE_ANOTHER_FWK_SERVICE(SiStripBadComponentsDQMService);

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/SiStrip/interface/SiStripPopConDbObjHandler.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
typedef popcon::PopConAnalyzer< popcon::SiStripPopConDbObjHandler< SiStripBadStrip, SiStripBadComponentsDQMService > > SiStripPopConBadComponentsDQM;
DEFINE_ANOTHER_FWK_MODULE(SiStripPopConBadComponentsDQM);

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMOffline/CalibTracker/plugins/SiStripQualityHotStripIdentifierRoot.h"
#include "DQMOffline/CalibTracker/plugins/SiStripDQMProfileToTkMapConverter.h"

#include "DQMOffline/CalibTracker/plugins/SiStripBadComponentsDQMServiceReader.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripBadComponentsDQMServiceReader);

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripQualityHotStripIdentifierRoot);
DEFINE_ANOTHER_FWK_MODULE(SiStripDQMProfileToTkMapConverter);
