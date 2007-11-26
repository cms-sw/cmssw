#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

using namespace cms;

#include "JetMETCorrections/JetPlusTrack/plugins/JetPlusTrackAnalysis.h"
using cms::JetPlusTrackAnalysis;
DEFINE_ANOTHER_FWK_MODULE(JetPlusTrackAnalysis);
