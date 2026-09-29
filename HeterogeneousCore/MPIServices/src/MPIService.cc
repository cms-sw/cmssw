// -*- C++ -*-
#include <cstdlib>
#include <exception>
#include <mutex>
#include <string>

#include <mpi.h>

#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAInterface.h"
#include "HeterogeneousCore/MPIServices/interface/MPIService.h"
#include "HeterogeneousCore/ROCmServices/interface/ROCmInterface.h"

namespace {
  // list the MPI thread support levels
  const char* const mpi_thread_support_level[] = {
      "MPI_THREAD_SINGLE",      // only one thread will execute (the process is single-threaded)
      "MPI_THREAD_FUNNELED",    // only the thread that called MPI_Init_thread will make MPI calls
      "MPI_THREAD_SERIALIZED",  // only one thread will make MPI library calls at one time
      "MPI_THREAD_MULTIPLE"     // multiple threads may call MPI at once with no restrictions
  };

}  // namespace

MPIService::MPIService(edm::ParameterSet const& config, edm::ActivityRegistry& iRegistry) {
  /* As of Open MPI 4.1.0, `MPI_THREAD_MULTIPLE` is supported by the following transports:
   *   - the `ob1` PML, with the following BTLs:
   *       - `self`
   *       - `sm`
   *       - `smcuda`
   *       - `tcp`
   *       - `ugni`
   *       - `usnic`
   *   - the `cm` PML, with the following MTLs:
   *       - `ofi` (Libfabric)
   *       - `portals4`
   *   - the `ucx` PML
   *
   * MPI File operations are not thread safe even if MPI is initialized for `MPI_THREAD_MULTIPLE` support.
   *
   * See https://github.com/open-mpi/ompi/blob/v4.1.0/README .
   */
  iRegistry.watchPreSourceEarlyTermination(
      [this](edm::TerminationOrigin) { abortOnError_("PreSourceEarlyTermination"); });
  iRegistry.watchPreGlobalEarlyTermination(
      [this](edm::GlobalContext const&, edm::TerminationOrigin) { abortOnError_("PreGlobalEarlyTermination"); });
  iRegistry.watchPreStreamEarlyTermination(
      [this](edm::StreamContext const&, edm::TerminationOrigin) { abortOnError_("PreStreamEarlyTermination"); });

  // If a CUDAService or a ROCmService is configured for this job, construct it
  // now, before the MPIService is constructed. This is to make sure the
  // MPIService is destructed (and MPI_Finalize() is called) *before* the
  // CUDAService/ROCmService destructors are called (specifically, before
  // cudaDeviceReset()/hipDeviceReset() are called). Otherwise MPI_Finalize()
  // would segfault on nodes with AMD GPUs.
  //
  // Triggering the construction of the cuda and rocm services is the purpose of
  // the isAvailable() call below.
  //
  // Note that the MPIService does not require a ROCmService nor a CUDAService
  // to run.
  edm::Service<CUDAInterface> cuda;
  cuda.isAvailable();
  edm::Service<ROCmInterface> rocm;
  rocm.isAvailable();

  // set the pmix_server_uri MCA parameter if specified in the configuration and not already set in the environment
  if (config.existsAs<std::string>("pmix_server_uri", false)) {
    std::string uri = config.getUntrackedParameter<std::string>("pmix_server_uri");
    // do not overwrite the environment variable if it is already set
    setenv("OMPI_MCA_pmix_server_uri", uri.c_str(), false);
  }

  // initializes the MPI execution environment, requesting multi-threading support
  int provided;
  MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
  if (provided < MPI_THREAD_MULTIPLE) {
    throw cms::Exception("UnsupportedFeature")
        << "CMSSW requires the " << mpi_thread_support_level[MPI_THREAD_MULTIPLE]
        << " multithreading support level, but the MPI library provides only the " << mpi_thread_support_level[provided]
        << " level.";
  } else {
    edm::LogInfo log("MPIService");
    log << "The MPI library provides the " << mpi_thread_support_level[provided] << " multithreading support level\n";

    // get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    log << "MPI_COMM_WORLD size: " << world_size << '\n';

    // get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    log << "MPI_COMM_WORLD rank: " << world_rank << '\n';

    // get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    log << "MPI processor name:  " << processor_name << '\n';

    // initialisation done
    log << '\n';
    log << "MPI successfully initialised";
  }
}

MPIService::~MPIService() {
  // Finalize MPI execution environment
  MPI_Finalize();
}

void MPIService::abortOnError_(std::string const& termination_type) {
  // Clean exit involves several blocking synchronisation calls in the destructors, which hang because the error is not yet propagated to the other processes.
  // The hang might also occur when failing process is inside a blocking Wait() to send or receive a ususal message.
  // Doing a flag check would solve the problem for deadlocks in the first case, but in the second case process might be already inside a Wait() call when the error occurs, therefore flag check would not help in this scenario.
  // As we don't have any recovery mechanisms anyway, it's better to simply abort the MPI job immediately to avoid deadlocks and other issues.
  edm::LogError("MPIService") << "MPIService: " << termination_type
                              << " event occured, Aborting MPI to avoid possible synchronization issues..."
                              << std::endl;
  MPI_Abort(MPI_COMM_WORLD, edm::errors::ExternalFailure);
}

void MPIService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addOptionalUntracked<std::string>("pmix_server_uri")
      ->setComment("Set the OpenMPI MCA pmix_server_uri parameter if not already set in the environment");
  descriptions.add("MPIService", desc);
  descriptions.setComment(R"(This Service provides a common interface to MPI configuration for the CMSSW job.)");
}

void MPIService::required() {
  edm::Service<MPIService> s;
  if (not s.isAvailable()) {
    throw cms::Exception("Configuration") << R"(The MPIService is required by this module.
Please add it to the configuration, for example via

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
)";
  }
}

// Ensure that processes exchange their hashes only once
void MPIService::exchangeProcessHashes_() {
  std::call_once(init_flag_, [&]() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    all_process_hashes_.resize(world_size);
    edm::Service<edm::service::TriggerNamesService> tns;
    std::string const& processName = tns->getProcessName();
    uint64_t this_process_hash = std::hash<std::string>{}(processName);
    MPI_Allgather(&this_process_hash, 1, MPI_UINT64_T, all_process_hashes_.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);
  });
}

std::vector<int> MPIService::getRanksByProcessName(std::string const& processName) {
  this->exchangeProcessHashes_();
  std::vector<int> process_indices;
  uint64_t other_process_hash = std::hash<std::string>{}(processName);
  for (size_t i = 0; i < all_process_hashes_.size(); i++) {
    if (all_process_hashes_[i] == other_process_hash) {
      process_indices.push_back(i);
    }
  }
  return process_indices;
}
