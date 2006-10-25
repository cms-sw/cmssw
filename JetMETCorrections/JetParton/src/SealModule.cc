#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "JetMETCorrections/JetParton/interface/JetPartonProducer.h"
using cms::JetParton;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(JetParton);
