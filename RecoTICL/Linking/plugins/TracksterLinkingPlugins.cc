#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "TracksterLinkingbyFastJet.h"
#include "TracksterLinkingbySkeletons.h"
#include "TracksterLinkingRecovery.h"
#include "TracksterLinkingbyCornetto.h"
#include "RecoTICL/Linking/interface/TracksterLinkingPluginFactory.h"

DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory, ticl::TracksterLinkingbySkeletons, "Skeletons");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory, ticl::TracksterLinkingbyFastJet, "FastJet");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory, ticl::TracksterLinkingRecovery, "Recovery");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory, ticl::TracksterLinkingbyCornetto, "Cornetto");
