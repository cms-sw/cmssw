#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonException.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"
#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"

#include "grpc_client.h"
#include "grpc_service.pb.h"
#include "model_config.pb.h"

#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <experimental/iterator>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <utility>
#include <tuple>

namespace tc = triton::client;

namespace {
  grpc_compression_algorithm getCompressionAlgo(const std::string& name) {
    if (name.empty() or name.compare("none") == 0)
      return grpc_compression_algorithm::GRPC_COMPRESS_NONE;
    else if (name.compare("deflate") == 0)
      return grpc_compression_algorithm::GRPC_COMPRESS_DEFLATE;
    else if (name.compare("gzip") == 0)
      return grpc_compression_algorithm::GRPC_COMPRESS_GZIP;
    else
      throw cms::Exception("GrpcCompression")
          << "Unknown compression algorithm requested: " << name << " (choices: none, deflate, gzip)";
  }

  std::vector<std::shared_ptr<tc::InferResult>> convertToShared(const std::vector<tc::InferResult*>& tmp) {
    std::vector<std::shared_ptr<tc::InferResult>> results;
    results.reserve(tmp.size());
    std::transform(tmp.begin(), tmp.end(), std::back_inserter(results), [](tc::InferResult* ptr) {
      return std::shared_ptr<tc::InferResult>(ptr);
    });
    return results;
  }
}  // namespace

//based on https://github.com/triton-inference-server/server/blob/v2.3.0/src/clients/c++/examples/simple_grpc_async_infer_client.cc
//and https://github.com/triton-inference-server/server/blob/v2.3.0/src/clients/c++/perf_client/perf_client.cc

TritonClient::TritonClient(const edm::ParameterSet& params, const std::string& debugName)
    : SonicClient(params, debugName, "TritonClient"),
      batchMode_(TritonBatchMode::Rectangular),
      manualBatchMode_(false),
      verbose_(params.getUntrackedParameter<bool>("verbose")),
      useSharedMemory_(params.getUntrackedParameter<bool>("useSharedMemory")),
      compressionAlgo_(getCompressionAlgo(params.getUntrackedParameter<std::string>("compression"))) {
  options_.emplace_back(params.getParameter<std::string>("modelName"));
  //get appropriate server for this model
  edm::Service<TritonService> ts;

  // We save the token to be able to notify the service in case of an exception in the evaluate method.
  // The evaluate method can be called outside the frameworks TBB threadpool in the case of a retry. In
  // this case the context is not setup to access the service registry, we need the service token to
  // create the context.
  token_ = edm::ServiceRegistry::instance().presentToken();

  const auto& server =
      ts->serverInfo(options_[0].model_name_, params.getUntrackedParameter<std::string>("preferredServer"));
  serverType_ = server.type;
  edm::LogInfo("TritonDiscovery") << debugName_ << " assigned server: " << server.url;
  //enforce sync mode for fallback CPU server to avoid contention
  //todo: could enforce async mode otherwise (unless mode was specified by user?)
  if (serverType_ == TritonServerType::LocalCPU)
    setMode(SonicMode::Sync);
  isLocal_ = serverType_ == TritonServerType::LocalCPU or serverType_ == TritonServerType::LocalGPU;

  //connect to the server
  TRITON_THROW_IF_ERROR(
      tc::InferenceServerGrpcClient::Create(&client_, server.url, false, server.useSsl, server.sslOptions),
      "TritonClient(): unable to create inference context",
      isLocal_);

  //set options
  options_[0].model_version_ = params.getParameter<std::string>("modelVersion");
  options_[0].client_timeout_ = params.getUntrackedParameter<unsigned>("timeout");
  //convert to microseconds
  const auto& timeoutUnit = params.getUntrackedParameter<std::string>("timeoutUnit");
  unsigned conversion = 1;
  if (timeoutUnit == "seconds")
    conversion = 1e6;
  else if (timeoutUnit == "milliseconds")
    conversion = 1e3;
  else if (timeoutUnit == "microseconds")
    conversion = 1;
  else
    throw cms::Exception("Configuration") << "Unknown timeout unit: " << timeoutUnit;
  options_[0].client_timeout_ *= conversion;

  //get fixed parameters from local config
  inference::ModelConfig localModelConfig;
  {
    const std::string localModelConfigPath(params.getParameter<edm::FileInPath>("modelConfigPath").fullPath());
    int fileDescriptor = open(localModelConfigPath.c_str(), O_RDONLY);
    if (fileDescriptor < 0)
      throw TritonException("LocalFailure")
          << "TritonClient(): unable to open local model config: " << localModelConfigPath;
    google::protobuf::io::FileInputStream localModelConfigInput(fileDescriptor);
    localModelConfigInput.SetCloseOnDelete(true);
    if (!google::protobuf::TextFormat::Parse(&localModelConfigInput, &localModelConfig))
      throw TritonException("LocalFailure")
          << "TritonClient(): unable to parse local model config: " << localModelConfigPath;
  }

  //check batch size limitations (after i/o setup)
  //triton uses max batch size = 0 to denote a model that does not support native batching (using the outer dimension)
  //but for models that do support batching (native or otherwise), a given event may set batch size 0 to indicate no valid input is present
  //so set the local max to 1 and keep track of "no outer dim" case
  maxOuterDim_ = localModelConfig.max_batch_size();
  noOuterDim_ = maxOuterDim_ == 0;
  maxOuterDim_ = std::max(1u, maxOuterDim_);
  //propagate batch size
  setBatchSize(1);

  //compare model checksums to remote config to enforce versioning
  inference::ModelConfigResponse modelConfigResponse;
  TRITON_THROW_IF_ERROR(client_->ModelConfig(&modelConfigResponse, options_[0].model_name_, options_[0].model_version_),
                        "TritonClient(): unable to get model config",
                        isLocal_);
  inference::ModelConfig remoteModelConfig(modelConfigResponse.config());

  std::map<std::string, std::array<std::string, 2>> checksums;
  size_t fileCounter = 0;
  for (const auto& modelConfig : {localModelConfig, remoteModelConfig}) {
    const auto& agents = modelConfig.model_repository_agents().agents();
    auto agent = std::find_if(agents.begin(), agents.end(), [](auto const& a) { return a.name() == "checksum"; });
    if (agent != agents.end()) {
      const auto& params = agent->parameters();
      for (const auto& [key, val] : params) {
        // only check the requested version
        if (key.compare(0, options_[0].model_version_.size() + 1, options_[0].model_version_ + "/") == 0)
          checksums[key][fileCounter] = val;
      }
    }
    ++fileCounter;
  }
  std::vector<std::string> incorrect;
  for (const auto& [key, val] : checksums) {
    if (checksums[key][0] != checksums[key][1])
      incorrect.push_back(key);
  }
  if (!incorrect.empty())
    throw TritonException("ModelVersioning") << "The following files have incorrect checksums on the remote server: "
                                             << triton_utils::printColl(incorrect, ", ");

  //get model info
  inference::ModelMetadataResponse modelMetadata;
  TRITON_THROW_IF_ERROR(client_->ModelMetadata(&modelMetadata, options_[0].model_name_, options_[0].model_version_),
                        "TritonClient(): unable to get model metadata",
                        isLocal_);

  //get input and output (which know their sizes)
  const auto& nicInputs = modelMetadata.inputs();
  const auto& nicOutputs = modelMetadata.outputs();

  //report all model errors at once
  std::stringstream msg;
  std::string msg_str;

  //currently no use case is foreseen for a model with zero inputs or outputs
  if (nicInputs.empty())
    msg << "Model on server appears malformed (zero inputs)\n";

  if (nicOutputs.empty())
    msg << "Model on server appears malformed (zero outputs)\n";

  //stop if errors
  msg_str = msg.str();
  if (!msg_str.empty())
    throw cms::Exception("ModelErrors") << msg_str;

  //setup input map
  std::stringstream io_msg;
  if (verbose_)
    io_msg << "Model inputs: "
           << "\n";
  for (const auto& nicInput : nicInputs) {
    const auto& iname = nicInput.name();
    auto [curr_itr, success] = input_.emplace(std::piecewise_construct,
                                              std::forward_as_tuple(iname),
                                              std::forward_as_tuple(iname, nicInput, this, ts->pid()));
    auto& curr_input = curr_itr->second;
    if (verbose_) {
      io_msg << "  " << iname << " (" << curr_input.dname() << ", " << curr_input.byteSize()
             << " b) : " << triton_utils::printColl(curr_input.shape()) << "\n";
    }
  }

  //allow selecting only some outputs from server
  const auto& v_outputs = params.getUntrackedParameter<std::vector<std::string>>("outputs");
  std::unordered_set s_outputs(v_outputs.begin(), v_outputs.end());

  //setup output map
  if (verbose_)
    io_msg << "Model outputs: "
           << "\n";
  for (const auto& nicOutput : nicOutputs) {
    const auto& oname = nicOutput.name();
    if (!s_outputs.empty() and s_outputs.find(oname) == s_outputs.end())
      continue;
    auto [curr_itr, success] = output_.emplace(std::piecewise_construct,
                                               std::forward_as_tuple(oname),
                                               std::forward_as_tuple(oname, nicOutput, this, ts->pid()));
    auto& curr_output = curr_itr->second;
    if (verbose_) {
      io_msg << "  " << oname << " (" << curr_output.dname() << ", " << curr_output.byteSize()
             << " b) : " << triton_utils::printColl(curr_output.shape()) << "\n";
    }
    if (!s_outputs.empty())
      s_outputs.erase(oname);
  }

  //check if any requested outputs were not available
  if (!s_outputs.empty())
    throw cms::Exception("MissingOutput")
        << "Some requested outputs were not available on the server: " << triton_utils::printColl(s_outputs);

  //print model info
  std::stringstream model_msg;
  if (verbose_) {
    model_msg << "Model name: " << options_[0].model_name_ << "\n"
              << "Model version: " << options_[0].model_version_ << "\n"
              << "Model max outer dim: " << (noOuterDim_ ? 0 : maxOuterDim_) << "\n";
    edm::LogInfo(fullDebugName_) << model_msg.str() << io_msg.str();
  }
}

TritonClient::~TritonClient() {
  //by default: members of this class destroyed before members of base class
  //in shared memory case, TritonMemResource (member of TritonData) unregisters from client_ in its destructor
  //but input/output objects are member of base class, so destroyed after client_ (member of this class)
  //therefore, clear the maps here
  input_.clear();
  output_.clear();
}

void TritonClient::setBatchMode(TritonBatchMode batchMode) {
  unsigned oldBatchSize = batchSize();
  batchMode_ = batchMode;
  manualBatchMode_ = true;
  //this allows calling setBatchSize() and setBatchMode() in either order consistently to change back and forth
  //includes handling of change from ragged to rectangular if multiple entries already created
  setBatchSize(oldBatchSize);
}

void TritonClient::resetBatchMode() {
  batchMode_ = TritonBatchMode::Rectangular;
  manualBatchMode_ = false;
}

unsigned TritonClient::nEntries() const { return !input_.empty() ? input_.begin()->second.entries_.size() : 0; }

unsigned TritonClient::batchSize() const { return batchMode_ == TritonBatchMode::Rectangular ? outerDim_ : nEntries(); }

bool TritonClient::setBatchSize(unsigned bsize) {
  if (batchMode_ == TritonBatchMode::Rectangular) {
    if (bsize > maxOuterDim_) {
      throw TritonException("LocalFailure")
          << "Requested batch size " << bsize << " exceeds server-specified max batch size " << maxOuterDim_ << ".";
      return false;
    } else {
      outerDim_ = bsize;
      //take min to allow resizing to 0
      resizeEntries(std::min(outerDim_, 1u));
      return true;
    }
  } else {
    resizeEntries(bsize);
    outerDim_ = 1;
    return true;
  }
}

void TritonClient::resizeEntries(unsigned entry) {
  if (entry > nEntries())
    //addEntry(entry) extends the vector to size entry+1
    addEntry(entry - 1);
  else if (entry < nEntries()) {
    for (auto& element : input_) {
      element.second.entries_.resize(entry);
    }
    for (auto& element : output_) {
      element.second.entries_.resize(entry);
    }
  }
}

void TritonClient::addEntry(unsigned entry) {
  for (auto& element : input_) {
    element.second.addEntryImpl(entry);
  }
  for (auto& element : output_) {
    element.second.addEntryImpl(entry);
  }
  if (entry > 0) {
    batchMode_ = TritonBatchMode::Ragged;
    outerDim_ = 1;
  }
}

void TritonClient::reset() {
  if (!manualBatchMode_)
    batchMode_ = TritonBatchMode::Rectangular;
  for (auto& element : input_) {
    element.second.reset();
  }
  for (auto& element : output_) {
    element.second.reset();
  }
}

template <typename F>
bool TritonClient::handle_exception(F&& call) {
  //caught exceptions will be propagated to edm::WaitingTaskWithArenaHolder
  CMS_SA_ALLOW try {
    call();
    return true;
  }
  //TritonExceptions are intended/expected to be recoverable, i.e. retries should be allowed
  catch (TritonException& e) {
    e.convertToWarning();
    finish(false);
    return false;
  }
  //other exceptions are not: execution should stop if they are encountered
  catch (...) {
    finish(false, std::current_exception());
    return false;
  }
}

void TritonClient::getResults(const std::vector<std::shared_ptr<tc::InferResult>>& results) {
  for (unsigned i = 0; i < results.size(); ++i) {
    const auto& result = results[i];
    for (auto& [oname, output] : output_) {
      //set shape here before output becomes const
      if (output.variableDims()) {
        std::vector<int64_t> tmp_shape;
        TRITON_THROW_IF_ERROR(
            result->Shape(oname, &tmp_shape), "getResults(): unable to get output shape for " + oname, false);
        if (!noOuterDim_)
          tmp_shape.erase(tmp_shape.begin());
        output.setShape(tmp_shape, i);
      }
      //extend lifetime
      output.setResult(result, i);
      //compute size after getting all result entries
      if (i == results.size() - 1)
        output.computeSizes();
    }
  }
}

//default case for sync and pseudo async
void TritonClient::evaluate() {
  //undo previous signal from TritonException
  if (tries_ > 0) {
    // If we are retrying then the evaluate method is called outside the frameworks TBB thread pool.
    // So we need to setup the service token for the current thread to access the service registry.
    edm::ServiceRegistry::Operate op(token_);
    edm::Service<TritonService> ts;
    ts->notifyCallStatus(true);
  }

  //in case there is nothing to process
  if (batchSize() == 0) {
    //call getResults on an empty vector
    std::vector<std::shared_ptr<tc::InferResult>> empty_results;
    getResults(empty_results);
    finish(true);
    return;
  }

  //set up input pointers for triton (generalized for multi-request ragged batching case)
  //one vector<InferInput*> per request
  unsigned nEntriesVal = nEntries();
  std::vector<std::vector<triton::client::InferInput*>> inputsTriton(nEntriesVal);
  for (auto& inputTriton : inputsTriton) {
    inputTriton.reserve(input_.size());
  }
  for (auto& [iname, input] : input_) {
    for (unsigned i = 0; i < nEntriesVal; ++i) {
      inputsTriton[i].push_back(input.data(i));
    }
  }

  //set up output pointers similarly
  std::vector<std::vector<const triton::client::InferRequestedOutput*>> outputsTriton(nEntriesVal);
  for (auto& outputTriton : outputsTriton) {
    outputTriton.reserve(output_.size());
  }
  for (auto& [oname, output] : output_) {
    for (unsigned i = 0; i < nEntriesVal; ++i) {
      outputsTriton[i].push_back(output.data(i));
    }
  }

  //set up shared memory for output
  auto success = handle_exception([&]() {
    for (auto& element : output_) {
      element.second.prepare();
    }
  });
  if (!success)
    return;

  // Get the status of the server prior to the request being made.
  inference::ModelStatistics start_status;
  success = handle_exception([&]() {
    if (verbose())
      start_status = getServerSideStatus();
  });
  if (!success)
    return;

  if (mode_ == SonicMode::Async) {
    //non-blocking call
    success = handle_exception([&]() {
      TRITON_THROW_IF_ERROR(client_->AsyncInferMulti(
                                [start_status, this](std::vector<tc::InferResult*> resultsTmp) {
                                  //immediately convert to shared_ptr
                                  const auto& results = convertToShared(resultsTmp);
                                  //check results
                                  for (auto ptr : results) {
                                    auto success = handle_exception([&]() {
                                      TRITON_THROW_IF_ERROR(
                                          ptr->RequestStatus(), "evaluate(): unable to get result(s)", isLocal_);
                                    });
                                    if (!success)
                                      return;
                                  }

                                  if (verbose()) {
                                    inference::ModelStatistics end_status;
                                    auto success = handle_exception([&]() { end_status = getServerSideStatus(); });
                                    if (!success)
                                      return;

                                    const auto& stats = summarizeServerStats(start_status, end_status);
                                    reportServerSideStats(stats);
                                  }

                                  //check result
                                  auto success = handle_exception([&]() { getResults(results); });
                                  if (!success)
                                    return;

                                  //finish
                                  finish(true);
                                },
                                options_,
                                inputsTriton,
                                outputsTriton,
                                headers_,
                                compressionAlgo_),
                            "evaluate(): unable to launch async run",
                            isLocal_);
    });
    if (!success)
      return;
  } else {
    //blocking call
    std::vector<tc::InferResult*> resultsTmp;
    success = handle_exception([&]() {
      TRITON_THROW_IF_ERROR(
          client_->InferMulti(&resultsTmp, options_, inputsTriton, outputsTriton, headers_, compressionAlgo_),
          "evaluate(): unable to run and/or get result",
          isLocal_);
    });
    //immediately convert to shared_ptr
    const auto& results = convertToShared(resultsTmp);
    if (!success)
      return;

    if (verbose()) {
      inference::ModelStatistics end_status;
      success = handle_exception([&]() { end_status = getServerSideStatus(); });
      if (!success)
        return;

      const auto& stats = summarizeServerStats(start_status, end_status);
      reportServerSideStats(stats);
    }

    success = handle_exception([&]() { getResults(results); });
    if (!success)
      return;

    finish(true);
  }
}

void TritonClient::reportServerSideStats(const TritonClient::ServerSideStats& stats) const {
  std::stringstream msg;

  // https://github.com/triton-inference-server/server/blob/v2.3.0/src/clients/c++/perf_client/inference_profiler.cc
  const uint64_t count = stats.success_count_;
  msg << "  Inference count: " << stats.inference_count_ << "\n";
  msg << "  Execution count: " << stats.execution_count_ << "\n";
  msg << "  Successful request count: " << count << "\n";

  if (count > 0) {
    auto get_avg_us = [count](uint64_t tval) {
      constexpr uint64_t us_to_ns = 1000;
      return tval / us_to_ns / count;
    };

    const uint64_t cumm_avg_us = get_avg_us(stats.cumm_time_ns_);
    const uint64_t queue_avg_us = get_avg_us(stats.queue_time_ns_);
    const uint64_t compute_input_avg_us = get_avg_us(stats.compute_input_time_ns_);
    const uint64_t compute_infer_avg_us = get_avg_us(stats.compute_infer_time_ns_);
    const uint64_t compute_output_avg_us = get_avg_us(stats.compute_output_time_ns_);
    const uint64_t compute_avg_us = compute_input_avg_us + compute_infer_avg_us + compute_output_avg_us;
    const uint64_t overhead =
        (cumm_avg_us > queue_avg_us + compute_avg_us) ? (cumm_avg_us - queue_avg_us - compute_avg_us) : 0;

    msg << "  Avg request latency: " << cumm_avg_us << " usec"
        << "\n"
        << "  (overhead " << overhead << " usec + "
        << "queue " << queue_avg_us << " usec + "
        << "compute input " << compute_input_avg_us << " usec + "
        << "compute infer " << compute_infer_avg_us << " usec + "
        << "compute output " << compute_output_avg_us << " usec)" << std::endl;
  }

  if (!debugName_.empty())
    edm::LogInfo(fullDebugName_) << msg.str();
}

TritonClient::ServerSideStats TritonClient::summarizeServerStats(const inference::ModelStatistics& start_status,
                                                                 const inference::ModelStatistics& end_status) const {
  TritonClient::ServerSideStats server_stats;

  server_stats.inference_count_ = end_status.inference_count() - start_status.inference_count();
  server_stats.execution_count_ = end_status.execution_count() - start_status.execution_count();
  server_stats.success_count_ =
      end_status.inference_stats().success().count() - start_status.inference_stats().success().count();
  server_stats.cumm_time_ns_ =
      end_status.inference_stats().success().ns() - start_status.inference_stats().success().ns();
  server_stats.queue_time_ns_ = end_status.inference_stats().queue().ns() - start_status.inference_stats().queue().ns();
  server_stats.compute_input_time_ns_ =
      end_status.inference_stats().compute_input().ns() - start_status.inference_stats().compute_input().ns();
  server_stats.compute_infer_time_ns_ =
      end_status.inference_stats().compute_infer().ns() - start_status.inference_stats().compute_infer().ns();
  server_stats.compute_output_time_ns_ =
      end_status.inference_stats().compute_output().ns() - start_status.inference_stats().compute_output().ns();

  return server_stats;
}

inference::ModelStatistics TritonClient::getServerSideStatus() const {
  if (verbose_) {
    inference::ModelStatisticsResponse resp;
    TRITON_THROW_IF_ERROR(client_->ModelInferenceStatistics(&resp, options_[0].model_name_, options_[0].model_version_),
                          "getServerSideStatus(): unable to get model statistics",
                          isLocal_);
    return *(resp.model_stats().begin());
  }
  return inference::ModelStatistics{};
}

//for fillDescriptions
void TritonClient::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
  edm::ParameterSetDescription descClient;
  fillBasePSetDescription(descClient);
  descClient.add<std::string>("modelName");
  descClient.add<std::string>("modelVersion", "");
  descClient.add<edm::FileInPath>("modelConfigPath");
  //server parameters should not affect the physics results
  descClient.addUntracked<std::string>("preferredServer", "");
  descClient.addUntracked<unsigned>("timeout");
  descClient.ifValue(edm::ParameterDescription<std::string>("timeoutUnit", "seconds", false),
                     edm::allowedValues<std::string>("seconds", "milliseconds", "microseconds"));
  descClient.addUntracked<bool>("useSharedMemory", true);
  descClient.addUntracked<std::string>("compression", "");
  descClient.addUntracked<std::vector<std::string>>("outputs", {});
  iDesc.add<edm::ParameterSetDescription>("Client", descClient);
}
