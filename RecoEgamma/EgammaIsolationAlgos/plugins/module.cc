#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"
#include "EgammaTrackExtractor.h"
#include "EgammaEcalExtractor.h"
#include "EgammaHcalExtractor.h"
#include "EgammaTowerExtractor.h"
#include "EgammaRecHitExtractor.h"
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, egammaisolation::EgammaTrackExtractor, "EgammaTrackExtractor");
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, egammaisolation::EgammaEcalExtractor, "EgammaEcalExtractor");
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, egammaisolation::EgammaHcalExtractor, "EgammaHcalExtractor");
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, egammaisolation::EgammaTowerExtractor, "EgammaTowerExtractor");
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, egammaisolation::EgammaRecHitExtractor, "EgammaRecHitExtractor");
