#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "LinkingAlgoFactory.h"
#include "LinkingAlgoByDirectionGeometric.h"

EDM_REGISTER_VALIDATED_PLUGINFACTORY(LinkingAlgoFactory, "LinkingAlgoFactory");

DEFINE_EDM_VALIDATED_PLUGIN(LinkingAlgoFactory,
                            ticl::LinkingAlgoByDirectionGeometric,
                            "LinkingAlgoByDirectionGeometric");
