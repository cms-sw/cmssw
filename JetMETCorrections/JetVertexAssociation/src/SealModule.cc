#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "JetMETCorrections/JetVertexAssociation/interface/JetVertexAssociation.h"
using cms::JetVertexAssociation;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(JetVertexAssociation);
