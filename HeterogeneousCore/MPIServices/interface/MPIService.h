#ifndef HeterogeneousCore_MPIServices_interface_MPIService_h
#define HeterogeneousCore_MPIServices_interface_MPIService_h

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

class MPIService {
public:
  MPIService(edm::ParameterSet const& config);
  ~MPIService();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static void required();
};

#endif  // HeterogeneousCore_MPIServices_interface_MPIService_h
