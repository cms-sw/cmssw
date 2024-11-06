// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 07/2024

#ifndef RecoHGCal_TICL_TracksterInferenceAlgoFactory_H__
#define RecoHGCal_TICL_TracksterInferenceAlgoFactory_H__

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoBase.h"

typedef edmplugin::PluginFactory<ticl::TracksterInferenceAlgoBase*(const edm::ParameterSet&)>
    TracksterInferenceAlgoFactory;

#endif
