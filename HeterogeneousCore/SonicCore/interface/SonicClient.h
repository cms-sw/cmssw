#ifndef HeterogeneousCore_SonicCore_SonicClient
#define HeterogeneousCore_SonicCore_SonicClient

#include "HeterogeneousCore/SonicCore/interface/SonicClientBase.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientTypes.h"

//convenience definition for multiple inheritance (base and types)
template <typename InputT, typename OutputT = InputT>
class SonicClient : public SonicClientBase, public SonicClientTypes<InputT, OutputT> {
public:
  //constructor
  SonicClient(const edm::ParameterSet& params, const std::string& debugName, const std::string& clientName)
      : SonicClientBase(params, debugName, clientName), SonicClientTypes<InputT, OutputT>() {}
};

#endif
