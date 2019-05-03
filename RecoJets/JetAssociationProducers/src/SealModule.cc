#include "FWCore/Framework/interface/MakerMacros.h"

#include "JetExtender.h"
#include "JetSignalVertexCompatibility.h"
#include "JetTracksAssociatorAtCaloFace.h"
#include "JetTracksAssociatorAtVertex.h"
#include "JetTracksAssociatorExplicit.h"

DEFINE_FWK_MODULE(JetTracksAssociatorAtVertex);
DEFINE_FWK_MODULE(JetTracksAssociatorExplicit);
DEFINE_FWK_MODULE(JetTracksAssociatorAtCaloFace);
DEFINE_FWK_MODULE(JetExtender);
DEFINE_FWK_MODULE(JetSignalVertexCompatibility);
