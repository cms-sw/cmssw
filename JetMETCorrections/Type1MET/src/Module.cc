
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "JetMETCorrections/Type1MET/interface/Type1MET.h"
#include "JetMETCorrections/Type1MET/interface/MuonMET.h"
#include "JetMETCorrections/Type1MET/interface/TauMET.h"
#include "JetMETCorrections/Type1MET/interface/MuonMETValueMapProducer.h"

using cms::Type1MET;
using cms::MuonMET;
using cms::TauMET;
using cms::MuonMETValueMapProducer;

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(Type1MET);
DEFINE_ANOTHER_FWK_MODULE(MuonMET);
DEFINE_ANOTHER_FWK_MODULE(TauMET);
DEFINE_ANOTHER_FWK_MODULE(MuonMETValueMapProducer);
