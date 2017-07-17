#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"


#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

// //SiStrip private (and original) approach
#include "DQMOffline/CalibTracker/plugins/SiStripBadComponentsDQMService.h"
DEFINE_FWK_SERVICE(SiStripBadComponentsDQMService);

#include "DQMOffline/CalibTracker/plugins/SiStripPedestalsDQMService.h"
DEFINE_FWK_SERVICE(SiStripPedestalsDQMService);

#include "DQMOffline/CalibTracker/plugins/SiStripNoisesDQMService.h"
DEFINE_FWK_SERVICE(SiStripNoisesDQMService);

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/SiStrip/interface/SiStripPopConDbObjHandler.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
typedef popcon::PopConAnalyzer< popcon::SiStripPopConDbObjHandler< SiStripBadStrip, SiStripBadComponentsDQMService > > SiStripPopConBadComponentsDQM;
DEFINE_FWK_MODULE(SiStripPopConBadComponentsDQM);

#include "DQMOffline/CalibTracker/plugins/SiStripFEDErrorsDQM.h"
DEFINE_FWK_MODULE(SiStripFEDErrorsDQM);


#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
typedef popcon::PopConAnalyzer< popcon::SiStripPopConDbObjHandler< SiStripPedestals, SiStripPedestalsDQMService > > SiStripPopConPedestalsDQM;
DEFINE_FWK_MODULE(SiStripPopConPedestalsDQM);

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
typedef popcon::PopConAnalyzer< popcon::SiStripPopConDbObjHandler< SiStripNoises, SiStripNoisesDQMService > > SiStripPopConNoisesDQM;
DEFINE_FWK_MODULE(SiStripPopConNoisesDQM);

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMOffline/CalibTracker/plugins/SiStripQualityHotStripIdentifierRoot.h"

#include "DQMOffline/CalibTracker/plugins/SiStripBadComponentsDQMServiceReader.h"
DEFINE_FWK_MODULE(SiStripBadComponentsDQMServiceReader);


DEFINE_FWK_MODULE(SiStripQualityHotStripIdentifierRoot);
