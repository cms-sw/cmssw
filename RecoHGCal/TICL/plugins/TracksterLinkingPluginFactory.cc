// #include "TracksterLinkingbySkeletons.h"
// #include "TracksterLinkingbySuperClustering.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "TracksterLinkingbyFastJet.h"
#include "RecoHGCal/TICL/plugins/TracksterLinkingPluginFactory.h"

EDM_REGISTER_VALIDATED_PLUGINFACTORY(TracksterLinkingPluginFactory, "TracksterLinkingPluginFactory");
// DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory, ticl::TracksterLinkingbySkeletons, "Skeletons");
// DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory, ticl::TracksterLinkingbySuperClustering, "SuperClustering");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory, ticl::TracksterLinkingbyFastJet, "FastJet");
