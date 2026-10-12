// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 07/2024

#ifndef RecoHGCal_TICL_TracksterInferenceAlgoFactory_h
#define RecoHGCal_TICL_TracksterInferenceAlgoFactory_h

#include <string>

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoTICL/Inference/interface/TracksterInferenceAlgoBase.h"

typedef edmplugin::PluginFactory<ticl::TracksterInferenceAlgoBase*(const edm::ParameterSet&,
                                                                   ticl::TICLONNXGlobalCache const*)>
    TracksterInferenceAlgoFactory;

namespace ticl {
  // The PSet description of the inference plugin T that the factory registers as `type`.
  // Modules in other packages must use it, not edm::PluginDescription with a default type.
  // A release build writes their cfi before it registers the plugins of this package.
  template <typename T>
  edm::ParameterSetDescription inferencePluginPSetDescription(std::string const& type) {
    edm::ParameterSetDescription desc;
    T::fillPSetDescription(desc);
    desc.add<std::string>("type", type);
    return desc;
  }
}  // namespace ticl

#endif
