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

#include "grpc_client.h"
#include "grpc_service.pb.h"

class TritonClient : public SonicClient<TritonInputMap, TritonOutputMap> {
public:
  struct ServerSideStats {
    uint64_t inference_count_;
    uint64_t execution_count_;
    uint64_t success_count_;
    uint64_t cumm_time_ns_;
    uint64_t queue_time_ns_;
    uint64_t compute_input_time_ns_;
    uint64_t compute_infer_time_ns_;
    uint64_t compute_output_time_ns_;
  };

  //constructor
  TritonClient(const edm::ParameterSet& params);

  //accessors
  unsigned batchSize() const { return batchSize_; }
  bool verbose() const { return verbose_; }
  bool setBatchSize(unsigned bsize);
  void reset() override;

  //for fillDescriptions
  static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

protected:
  //helper
  bool getResults(std::shared_ptr<nvidia::inferenceserver::client::InferResult> results);

  void evaluate() override;

  void reportServerSideStats(const ServerSideStats& stats) const;
  ServerSideStats summarizeServerStats(const inference::ModelStatistics& start_status,
                                       const inference::ModelStatistics& end_status) const;

  inference::ModelStatistics getServerSideStatus() const;

  //members
  unsigned maxBatchSize_;
  unsigned batchSize_;
  bool noBatch_;
  bool verbose_;

  //IO pointers for triton
  std::vector<nvidia::inferenceserver::client::InferInput*> inputsTriton_;
  std::vector<const nvidia::inferenceserver::client::InferRequestedOutput*> outputsTriton_;

  std::unique_ptr<nvidia::inferenceserver::client::InferenceServerGrpcClient> client_;
  //stores timeout, model name and version
  nvidia::inferenceserver::client::InferOptions options_;
};

#endif
