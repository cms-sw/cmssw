#ifndef HeterogeneousCore_MPIServices_interface_MPIService_h
#define HeterogeneousCore_MPIServices_interface_MPIService_h

#include <mutex>
#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

class MPIService {
public:
  MPIService(edm::ParameterSet const& config, edm::ActivityRegistry& iRegistry);
  ~MPIService();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static void required();

  std::vector<int> getRanksByProcessName(std::string const& processName);

private:
  // variables related to process hash exchange
  std::once_flag init_flag_;
  std::vector<uint64_t> all_process_hashes_;

  void abortOnError_(std::string const& termination_type);

  void exchangeProcessHashes_();
};

#endif  // HeterogeneousCore_MPIServices_interface_MPIService_h
