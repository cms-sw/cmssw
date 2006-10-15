#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "JetMETCorrections/JetPlusTrack/interface/JetPlusTrackProducer.h"
using cms::JetPlusTrack;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(JetPlusTrack)
