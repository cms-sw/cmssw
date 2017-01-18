#include "FastSimulation/InteractionModel/interface/InteractionModelFactory.h"

#include "FastSimulation/InteractionModel/interface/InteractionModel.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

EDM_REGISTER_PLUGINFACTORY(
    fastsim::InteractionModelFactory,
    "FastSimInteractionModelFactory"
    );
