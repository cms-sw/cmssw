#ifndef HeterogeneousCore_SonicTriton_TritonClient
#define HeterogeneousCore_SonicTriton_TritonClient

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClient.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"

#include <map>
#include <vector>
#include <string>
#include <exception>
#include <unordered_map>

#include "request_grpc.h"

class TritonClient : public SonicClient<TritonInputMap, TritonOutputMap> {
public:
  using ModelStatus = nvidia::inferenceserver::ModelStatus;
  using InferContext = nvidia::inferenceserver::client::InferContext;

  struct ServerSideStats {
    uint64_t request_count_;
    uint64_t cumul_time_ns_;
    uint64_t queue_time_ns_;
    uint64_t compute_time_ns_;
  };

  //constructor
  TritonClient(const edm::ParameterSet& params);

  //helper
  bool getResults(std::map<std::string, std::unique_ptr<InferContext::Result>>& results);

  //accessors
  unsigned batchSize() const { return batchSize_; }
  bool verbose() const { return verbose_; }
  bool setBatchSize(unsigned bsize);
  void reset() override;

  //for fillDescriptions
  static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

protected:
  void evaluate() override;

  void reportServerSideStats(const ServerSideStats& stats) const;
  ServerSideStats summarizeServerStats(const ModelStatus& start_status, const ModelStatus& end_status) const;

  ModelStatus getServerSideStatus() const;

  //members
  std::string url_;
  unsigned timeout_;
  std::string modelName_;
  int modelVersion_;
  unsigned maxBatchSize_;
  unsigned batchSize_;
  bool noBatch_;
  bool verbose_;

  std::unique_ptr<InferContext> context_;
  std::unique_ptr<nvidia::inferenceserver::client::ServerStatusContext> serverCtx_;
  std::unique_ptr<InferContext::Options> options_;
};

#endif
