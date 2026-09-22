#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "RecoTICL/Linking/interface/TracksterLinkingPluginFactory.h"
#include "TracksterLinkingbySuperClusteringDNN.h"
#include "TracksterLinkingbySuperClusteringMustache.h"

DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory,
                            ticl::TracksterLinkingbySuperClusteringDNN,
                            "SuperClusteringDNN");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory,
                            ticl::TracksterLinkingbySuperClusteringMustache,
                            "SuperClusteringMustache");
