#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"

#include "request_grpc.h"

#include <string>
#include <cmath>
#include <chrono>
#include <exception>
#include <sstream>
#include <cstring>
#include <numeric>
#include <functional>
#include <algorithm>
#include <iterator>

//little vector helper functions
namespace {
  bool any_neg(const std::vector<int64_t>& vec) {
    return std::any_of(vec.begin(), vec.end(), [](int64_t i) { return i < 0; });
  }

  int64_t dim_product(const std::vector<int64_t>& vec) {
    return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<int64_t>());
  }

  std::string print_vec(const std::vector<int64_t>& vec) {
    if (vec.empty())
      return "";
    std::stringstream msg;
    //avoid trailing delim
    std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<int64_t>(msg, ","));
    //last element
    msg << vec.back();
    return msg.str();
  }
}  // namespace

namespace ni = nvidia::inferenceserver;
namespace nic = ni::client;

//based on https://github.com/NVIDIA/triton-inference-server/blob/v1.12.0/src/clients/c++/examples/simple_callback_client.cc

template <typename Client>
TritonClient<Client>::TritonClient(const edm::ParameterSet& params)
    : url_(params.getUntrackedParameter<std::string>("address") + ":" +
           std::to_string(params.getUntrackedParameter<unsigned>("port"))),
      timeout_(params.getUntrackedParameter<unsigned>("timeout")),
      modelName_(params.getParameter<std::string>("modelName")),
      modelVersion_(params.getParameter<int>("modelVersion")),
      batchSize_(params.getUntrackedParameter<unsigned>("batchSize")),
      verbose_(params.getUntrackedParameter<bool>("verbose")),
      allowedTries_(params.getUntrackedParameter<unsigned>("allowedTries")) {
  this->clientName_ = "TritonClient";

  //connect to the server
  wrap(nic::InferGrpcContext::Create(&context_, url_, modelName_, modelVersion_, false),
       "TritonClient(): unable to create inference context",
       true);

  //get options
  wrap(nic::InferContext::Options::Create(&options_),
       "TritonClient(): unable to create inference context options",
       true);

  //get input and output (which know their sizes)
  const auto& nicInputs = context_->Inputs();
  const auto& nicOutputs = context_->Outputs();

  //report all model errors at once
  std::stringstream msg;
  std::string msg_str;

  //currently no use case is foreseen for a model with zero inputs or outputs
  //models with multiple named inputs or outputs currently aren't supported
  if (nicInputs.empty())
    msg << "Model on server appears malformed (zero inputs)\n";
  else if (nicInputs.size() > 1)
    msg << "Model has multiple inputs (not supported)\n";

  if (nicOutputs.empty())
    msg << "Model on server appears malformed (zero outputs)\n";
  else if (nicOutputs.size() > 1)
    msg << "Model has multiple outputs (not supported)\n";

  //stop if errors
  msg_str = msg.str();
  if (!msg_str.empty())
    throw cms::Exception("ModelErrors") << msg_str;

  //get sizes
  nicInput_ = nicInputs[0];
  nicOutput_ = nicOutputs[0];
  //convert google::protobuf::RepeatedField to vector
  const auto& dimsInputTmp = nicInput_->Dims();
  dimsInput_.assign(dimsInputTmp.begin(), dimsInputTmp.end());
  const auto& dimsOutputTmp = nicOutput_->Dims();
  dimsOutput_.assign(dimsOutputTmp.begin(), dimsOutputTmp.end());

  //check sizes: variable-size dimensions reported as -1 (currently not supported)
  if (any_neg(dimsInput_))
    msg << "Model has variable-size inputs (not supported)\n";
  if (any_neg(dimsOutput_))
    msg << "Model has variable-size outputs (not supported)\n";

  //stop if errors
  msg_str = msg.str();
  if (!msg_str.empty())
    throw cms::Exception("ModelErrors") << msg_str;

  //compute sizes
  nInput_ = dim_product(dimsInput_);
  nOutput_ = dim_product(dimsOutput_);

  //only used for monitoring
  bool has_server = false;
  if (verbose_) {
    //print model info (debug name not set yet)
    const auto& input_str = print_vec(dimsInput_);
    const auto& output_str = print_vec(dimsOutput_);
    edm::LogInfo(this->clientName_) << "Model name: " << modelName_ << "\n"
                                    << "Model version: " << modelVersion_ << "\n"
                                    << "Model input: " << input_str << "\n"
                                    << "Model output: " << output_str;

    has_server = wrap(nic::ServerStatusGrpcContext::Create(&serverCtx_, url_, false),
                      "TritonClient(): unable to create server context");
  }
  if (!has_server)
    serverCtx_ = nullptr;
}

template <typename Client>
bool TritonClient<Client>::wrap(const nic::Error& err, const std::string& msg, bool stop) const {
  if (!err.IsOk()) {
    //throw exception: used in constructor, should not be used in evaluate() or any method called by evaluate()
    if (stop) {
      throw cms::Exception("TritonServerFailure") << msg << ": " << err;
    } else {
      //emit warning
      edm::LogWarning("TritonServerWarning") << msg << ": " << err;
    }
  }
  return err.IsOk();
}

template <typename Client>
bool TritonClient<Client>::setup() {
  bool status = true;

  std::unique_ptr<nic::InferContext::Options> options;
  status = wrap(nic::InferContext::Options::Create(&options), "setup(): unable to create inference context options");
  if (!status)
    return status;

  options->SetBatchSize(batchSize_);
  status = wrap(options->AddRawResult(nicOutput_), "setup(): unable to add raw result");
  if (!status)
    return status;

  status = wrap(context_->SetRunOptions(*options), "setup(): unable to set run options");
  if (!status)
    return status;

  nicInput_->Reset();

  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<int64_t> input_shape;
  for (unsigned i0 = 0; i0 < batchSize_; i0++) {
    float* arr = &(this->input_[i0 * nInput_]);
    status = wrap(nicInput_->SetRaw(reinterpret_cast<const uint8_t*>(arr), nInput_ * sizeof(float)),
                  "setup(): unable to set data for batch entry " + std::to_string(i0));
    if (!status)
      return status;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  if (!this->debugName_.empty())
    edm::LogInfo(this->fullDebugName_) << "Input array time: "
                                       << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

  //should be true
  return status;
}

template <typename Client>
bool TritonClient<Client>::getResults(const nic::InferContext::Result& result) {
  bool status = true;

  auto t1 = std::chrono::high_resolution_clock::now();
  this->output_.resize(nOutput_ * batchSize_, 0.f);
  for (unsigned i0 = 0; i0 < batchSize_; i0++) {
    const uint8_t* r0;
    size_t content_byte_size;
    status = wrap(result.GetRaw(i0, &r0, &content_byte_size),
                  "getResults(): unable to get raw for entry " + std::to_string(i0));
    if (!status)
      return status;
    std::memcpy(&this->output_[i0 * nOutput_], r0, nOutput_ * sizeof(float));
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  if (!this->debugName_.empty())
    edm::LogInfo(this->fullDebugName_) << "Output array time: "
                                       << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

  //should be true
  return status;
}

//default case for sync and pseudo async
template <typename Client>
void TritonClient<Client>::evaluate() {
  //in case there is nothing to process
  if (batchSize_ == 0) {
    this->finish(true);
    return;
  }

  //common operations first
  bool status = setup();
  if (!status) {
    this->finish(false);
    return;
  }

  // Get the status of the server prior to the request being made.
  const auto& start_status = getServerSideStatus();

  //blocking call
  auto t1 = std::chrono::high_resolution_clock::now();
  std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
  status = wrap(context_->Run(&results), "evaluate(): unable to run and/or get result");
  if (!status) {
    this->output_.resize(nOutput_ * batchSize_, 0.f);
    this->finish(false);
    return;
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  if (!this->debugName_.empty())
    edm::LogInfo(this->fullDebugName_) << "Remote time: "
                                       << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

  const auto& end_status = getServerSideStatus();

  if (verbose()) {
    const auto& stats = summarizeServerStats(start_status, end_status);
    reportServerSideStats(stats);
  }

  status = getResults(*results.begin()->second);

  this->finish(status);
}

//specialization for true async
template <>
void TritonClientAsync::evaluate() {
  //in case there is nothing to process
  if (batchSize_ == 0) {
    this->finish(true);
    return;
  }

  //common operations first
  bool status = setup();
  if (!status) {
    this->finish(false);
    return;
  }

  // Get the status of the server prior to the request being made.
  const auto& start_status = getServerSideStatus();

  //non-blocking call
  auto t1 = std::chrono::high_resolution_clock::now();
  status =
      wrap(context_->AsyncRun([t1, start_status, this](nic::InferContext* ctx,
                                                       const std::shared_ptr<nic::InferContext::Request>& request) {
        //get results
        bool status = true;
        std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
        status = wrap(ctx->GetAsyncRunResults(request, &results), "evaluate(): unable to get result");
        if (!status) {
          output_.resize(nOutput_ * batchSize_, 0.f);
          finish(false);
          return;
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        if (!debugName_.empty())
          edm::LogInfo(fullDebugName_) << "Remote time: "
                                       << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        const auto& end_status = getServerSideStatus();

        if (verbose()) {
          const auto& stats = summarizeServerStats(start_status, end_status);
          reportServerSideStats(stats);
        }

        //check result
        status = getResults(*results.begin()->second);

        //finish
        finish(status);
      }),
           "evaluate(): unable to launch async run");

  //if AsyncRun failed, finish() wasn't called
  if (!status)
    this->finish(false);
}

template <typename Client>
void TritonClient<Client>::reportServerSideStats(const typename TritonClient<Client>::ServerSideStats& stats) const {
  std::stringstream msg;

  // https://github.com/NVIDIA/tensorrt-inference-server/blob/v1.12.0/src/clients/c++/perf_client/inference_profiler.cc
  const uint64_t count = stats.request_count_;
  msg << "  Request count: " << count;

  if (count > 0) {
    auto get_avg_us = [count](uint64_t tval) {
      constexpr uint64_t us_to_ns = 1000;
      return tval / us_to_ns / count;
    };

    const uint64_t cumul_avg_us = get_avg_us(stats.cumul_time_ns_);
    const uint64_t queue_avg_us = get_avg_us(stats.queue_time_ns_);
    const uint64_t compute_avg_us = get_avg_us(stats.compute_time_ns_);
    const uint64_t overhead =
        (cumul_avg_us > queue_avg_us + compute_avg_us) ? (cumul_avg_us - queue_avg_us - compute_avg_us) : 0;

    msg << "\n"
        << "  Avg request latency: " << cumul_avg_us << " usec"
        << "\n"
        << "  (overhead " << overhead << " usec + "
        << "queue " << queue_avg_us << " usec + "
        << "compute " << compute_avg_us << " usec)" << std::endl;
  }

  if (!this->debugName_.empty())
    edm::LogInfo(this->fullDebugName_) << msg.str();
}

template <typename Client>
typename TritonClient<Client>::ServerSideStats TritonClient<Client>::summarizeServerStats(
    const ni::ModelStatus& start_status, const ni::ModelStatus& end_status) const {
  // If model_version is -1 then look in the end status to find the
  // latest (highest valued version) and use that as the version.
  int64_t status_model_version = 0;
  if (modelVersion_ < 0) {
    for (const auto& vp : end_status.version_status()) {
      status_model_version = std::max(status_model_version, vp.first);
    }
  } else
    status_model_version = modelVersion_;

  typename TritonClient<Client>::ServerSideStats server_stats;
  auto vend_itr = end_status.version_status().find(status_model_version);
  if (vend_itr != end_status.version_status().end()) {
    auto end_itr = vend_itr->second.infer_stats().find(batchSize_);
    if (end_itr != vend_itr->second.infer_stats().end()) {
      uint64_t start_count = 0;
      uint64_t start_cumul_time_ns = 0;
      uint64_t start_queue_time_ns = 0;
      uint64_t start_compute_time_ns = 0;

      auto vstart_itr = start_status.version_status().find(status_model_version);
      if (vstart_itr != start_status.version_status().end()) {
        auto start_itr = vstart_itr->second.infer_stats().find(batchSize_);
        if (start_itr != vstart_itr->second.infer_stats().end()) {
          start_count = start_itr->second.success().count();
          start_cumul_time_ns = start_itr->second.success().total_time_ns();
          start_queue_time_ns = start_itr->second.queue().total_time_ns();
          start_compute_time_ns = start_itr->second.compute().total_time_ns();
        }
      }

      server_stats.request_count_ = end_itr->second.success().count() - start_count;
      server_stats.cumul_time_ns_ = end_itr->second.success().total_time_ns() - start_cumul_time_ns;
      server_stats.queue_time_ns_ = end_itr->second.queue().total_time_ns() - start_queue_time_ns;
      server_stats.compute_time_ns_ = end_itr->second.compute().total_time_ns() - start_compute_time_ns;
    }
  }
  return server_stats;
}

template <typename Client>
ni::ModelStatus TritonClient<Client>::getServerSideStatus() const {
  if (serverCtx_) {
    ni::ServerStatus server_status;
    serverCtx_->GetServerStatus(&server_status);
    auto itr = server_status.model_status().find(modelName_);
    if (itr != server_status.model_status().end())
      return itr->second;
  }
  return ni::ModelStatus{};
}

//explicit template instantiations
template class TritonClient<SonicClientSync<std::vector<float>>>;
template class TritonClient<SonicClientAsync<std::vector<float>>>;
template class TritonClient<SonicClientPseudoAsync<std::vector<float>>>;
