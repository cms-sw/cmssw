#ifndef HeterogeneousCore_SonicTriton_TritonClient
#define HeterogeneousCore_SonicTriton_TritonClient

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClient.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"

#include <map>
#include <vector>
#include <string>
#include <exception>
#include <unordered_map>

#include "grpc_client.h"
#include "grpc_service.pb.h"

enum class TritonBatchMode { Rectangular = 1, Ragged = 2 };

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
  TritonClient(const edm::ParameterSet& params, const std::string& debugName);

  //destructor
  ~TritonClient() override;

  //accessors
  unsigned batchSize() const;
  TritonBatchMode batchMode() const { return batchMode_; }
  bool verbose() const { return verbose_; }
  bool useSharedMemory() const { return useSharedMemory_; }
  void setUseSharedMemory(bool useShm) { useSharedMemory_ = useShm; }
  bool setBatchSize(unsigned bsize);
  void setBatchMode(TritonBatchMode batchMode);
  void resetBatchMode();
  void reset() override;
  TritonServerType serverType() const { return serverType_; }

  //for fillDescriptions
  static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

protected:
  //helpers
  bool noOuterDim() const { return noOuterDim_; }
  unsigned outerDim() const { return outerDim_; }
  unsigned nEntries() const;
  void getResults(const std::vector<std::shared_ptr<triton::client::InferResult>>& results);
  void evaluate() override;
  template <typename F>
  bool handle_exception(F&& call);

  void reportServerSideStats(const ServerSideStats& stats) const;
  ServerSideStats summarizeServerStats(const inference::ModelStatistics& start_status,
                                       const inference::ModelStatistics& end_status) const;

  inference::ModelStatistics getServerSideStatus() const;

  //members
  unsigned maxOuterDim_;
  unsigned outerDim_;
  bool noOuterDim_;
  unsigned nEntries_;
  TritonBatchMode batchMode_;
  bool manualBatchMode_;
  bool verbose_;
  bool useSharedMemory_;
  TritonServerType serverType_;
  grpc_compression_algorithm compressionAlgo_;
  triton::client::Headers headers_;

  std::unique_ptr<triton::client::InferenceServerGrpcClient> client_;
  //stores timeout, model name and version
  std::vector<triton::client::InferOptions> options_;

private:
  friend TritonInputData;
  friend TritonOutputData;

  //private accessors only used by data
  auto client() { return client_.get(); }
  void addEntry(unsigned entry);
  void resizeEntries(unsigned entry);
};

#endif
