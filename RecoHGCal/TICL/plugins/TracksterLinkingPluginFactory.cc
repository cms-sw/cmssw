#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "TracksterLinkingbyFastJet.h"
#include "TracksterLinkingbySuperClusteringDNN.h"
#include "TracksterLinkingbySuperClusteringMustache.h"
#include "TracksterLinkingbySkeletons.h"
#include "TracksterLinkingPassthrough.h"
#include "RecoHGCal/TICL/plugins/TracksterLinkingPluginFactory.h"

EDM_REGISTER_VALIDATED_PLUGINFACTORY(TracksterLinkingPluginFactory, "TracksterLinkingPluginFactory");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory, ticl::TracksterLinkingbySkeletons, "Skeletons");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory,
                            ticl::TracksterLinkingbySuperClusteringDNN,
                            "SuperClusteringDNN");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory,
                            ticl::TracksterLinkingbySuperClusteringMustache,
                            "SuperClusteringMustache");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory, ticl::TracksterLinkingbyFastJet, "FastJet");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory, ticl::TracksterLinkingPassthrough, "Passthrough");
