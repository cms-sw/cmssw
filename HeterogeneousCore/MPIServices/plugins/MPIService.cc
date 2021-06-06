// -*- C++ -*-

#include <mpi.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace {
  const char* const mpi_thread_support_level[] = {
      "MPI_THREAD_SINGLE",      // only one thread will execute (the process is single-threaded)
      "MPI_THREAD_FUNNELED",    // only the thread that called MPI_Init_thread will make MPI calls
      "MPI_THREAD_SERIALIZED",  // only one thread will make MPI library calls at one time
      "MPI_THREAD_MULTIPLE"     // multiple threads may call MPI at once with no restrictions
  };
}

class MPIService {
public:
  MPIService(edm::ParameterSet const& config);
  ~MPIService();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
};

MPIService::MPIService(edm::ParameterSet const& config) {
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

    log << "The MPI library provides the " << mpi_thread_support_level[provided] << " multithreading support level\n";
    log << "MPI successfully initialised";
  }
}

MPIService::~MPIService() {
  // terminate the MPI execution environment
  MPI_Finalize();
}

void MPIService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("MPIService", desc);
  descriptions.setComment(R"(This Service provides a common interface to MPI configuration for the CMSSW job.)");
}

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE_MAKER(MPIService, edm::serviceregistry::ParameterSetMaker<MPIService>);
