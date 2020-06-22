#ifndef HeterogeneousCore_SonicTriton_TritonClient
#define HeterogeneousCore_SonicTriton_TritonClient

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientSync.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientPseudoAsync.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientAsync.h"

#include <vector>
#include <string>
#include <exception>

#include "request_grpc.h"

//avoid possible collisions in global namespace
namespace {
  namespace ni = nvidia::inferenceserver;
  namespace nic = ni::client;
}  // namespace

template <typename Client>
class TritonClient : public Client {
public:
  struct ServerSideStats {
    uint64_t request_count_;
    uint64_t cumul_time_ns_;
    uint64_t queue_time_ns_;
    uint64_t compute_time_ns_;
  };

  //constructor
  TritonClient(const edm::ParameterSet& params);

  //helper
  bool getResults(const nic::InferContext::Result& result);

  //accessors
  unsigned nInput() const { return nInput_; }
  unsigned nOutput() const { return nOutput_; }
  unsigned batchSize() const { return batchSize_; }
  bool verbose() const { return verbose_; }
  void setBatchSize(unsigned bsize) { batchSize_ = bsize; }

  //for fillDescriptions
  static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    edm::ParameterSetDescription descClient;
    descClient.add<unsigned>("nInput");
    descClient.add<unsigned>("nOutput");
    descClient.add<std::string>("modelName");
    descClient.add<int>("modelVersion", -1);
	//server parameters should not affect the physics results
    descClient.addUntracked<unsigned>("batchSize");
    descClient.addUntracked<std::string>("address");
    descClient.addUntracked<unsigned>("port");
    descClient.addUntracked<unsigned>("timeout");
    descClient.addUntracked<bool>("verbose", false);
    descClient.addUntracked<unsigned>("allowedTries", 0);
    iDesc.add<edm::ParameterSetDescription>("Client", descClient);
  }

protected:
  unsigned allowedTries() const override { return allowedTries_; }

  void evaluate() override;

  //helper for common ops
  bool setup();

  //helper to turn triton error into warning
  bool wrap(const nic::Error& err, const std::string& msg) const;

  void reportServerSideStats(const ServerSideStats& stats) const;
  ServerSideStats summarizeServerStats(const ni::ModelStatus& start_status, const ni::ModelStatus& end_status) const;

  ni::ModelStatus getServerSideStatus() const;

  //members
  std::string url_;
  unsigned timeout_;
  std::string modelName_;
  int modelVersion_;
  unsigned batchSize_;
  unsigned nInput_;
  unsigned nOutput_;
  bool verbose_;
  unsigned allowedTries_;
  std::unique_ptr<nic::InferContext> context_;
  std::unique_ptr<nic::ServerStatusContext> serverCtx_;
  std::shared_ptr<nic::InferContext::Input> nicInput_;
};

using TritonClientSync = TritonClient<SonicClientSync<std::vector<float>>>;
using TritonClientPseudoAsync = TritonClient<SonicClientPseudoAsync<std::vector<float>>>;
using TritonClientAsync = TritonClient<SonicClientAsync<std::vector<float>>>;

#endif
