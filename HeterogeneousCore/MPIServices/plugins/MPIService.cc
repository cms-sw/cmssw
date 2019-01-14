// -*- C++ -*-

#include <cassert>

#include <mpi.h>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

class MPIService {
public:
  MPIService(edm::ParameterSet const& config, edm::ActivityRegistry & registry);
  ~MPIService();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
};

MPIService::MPIService(edm::ParameterSet const & config, edm::ActivityRegistry & registry)
{
  int provided;
  MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
  assert(provided == MPI_THREAD_MULTIPLE);
}

MPIService::~MPIService() {
  MPI_Finalize();
}

void
MPIService::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("MPIService", desc);
  descriptions.setComment(R"(This Service provides a common interface to MPI configuration for the CMSSW job.)");
}

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(MPIService);
