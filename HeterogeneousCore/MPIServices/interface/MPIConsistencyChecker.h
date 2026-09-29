#ifndef HeterogeneousCore_MPIServices_interface_MPIConsistencyChecker_h
#define HeterogeneousCore_MPIServices_interface_MPIConsistencyChecker_h

#include <mutex>
#include <string>
#include <cstring>
#include <unordered_map>
#include <vector>

// ROOT headers
#include <TBufferFile.h>
#include <TClass.h>
#include <Rtypes.h>

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

struct MPIModuleInfo {
  bool is_sender;
  int instance;
  std::string module_label;
  std::vector<std::string> product_types;
};

class MPIConsistencyChecker {
public:
  MPIConsistencyChecker(edm::ParameterSet const& config);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static void required();

  void recordMPIModuleInfo(bool is_sender,
                           std::string const& module_label,
                           std::string const& upstream_label,
                           int instance,
                           std::vector<std::string> const& product_types);
  void registerMPIPathOrigin(std::string const& controller_or_source_name);
  void reconstructMPIPaths();
  void getSerializedMPIModuleInfo(std::vector<char>& buffer, std::string const& origin_name);
  void deserializeMPIModuleInfo(std::vector<char> const& buffer, std::vector<MPIModuleInfo>& info);
  void compareMPIModules(std::vector<MPIModuleInfo> const& other,
                         std::string const& origin_name,
                         std::string const& other_process_name,
                         std::string const& this_process_name);

private:
  std::once_flag paths_reconstructed_flag_;
  std::vector<MPIModuleInfo> modules_info_;
  std::vector<std::string> module_upstream_labels_;
  std::mutex modules_info_mutex_;
  std::unordered_map<std::string, std::vector<MPIModuleInfo>> mpi_paths_mappings_;
};

#endif  // HeterogeneousCore_MPIServices_interface_MPIConsistencyChecker_h
