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

namespace ni = nvidia::inferenceserver;
namespace nic = ni::client;

//based on https://github.com/NVIDIA/triton-inference-server/blob/v1.12.0/src/clients/c++/examples/simple_callback_client.cc

template <typename Client>
TritonClient<Client>::TritonClient(const edm::ParameterSet& params)
    : Client(),
      url_(params.getParameter<std::string>("address") + ":" + std::to_string(params.getParameter<unsigned>("port"))),
      timeout_(params.getParameter<unsigned>("timeout")),
      modelName_(params.getParameter<std::string>("modelName")),
      modelVersion_(params.getParameter<int>("modelVersion")),
      batchSize_(params.getParameter<unsigned>("batchSize")),
      nInput_(params.getParameter<unsigned>("nInput")),
      nOutput_(params.getParameter<unsigned>("nOutput")),
      verbose_(params.getParameter<bool>("verbose")),
      minSeverityExit_(static_cast<Severity>(params.getParameter<int>("minSeverityExit"))),
      allowedTries_(params.getParameter<unsigned>("allowedTries")) {
  this->clientName_ = "TritonClient";
}

template <typename Client>
std::exception_ptr TritonClient<Client>::wrap(const nic::Error& err, const std::string& msg, Severity sev) const {
  std::exception_ptr eptr;
  if (!err.IsOk()) {
    if (sev > minSeverityExit_) {
      cms::Exception ex("TritonServerFailure");
      ex << msg << ": " << err;
      eptr = make_exception_ptr(ex);
    } else {
      //just emit warning
      edm::LogWarning("TritonServerWarning") << msg << ": " << err;
    }
  }
  return eptr;
}

template <typename Client>
std::exception_ptr TritonClient<Client>::setup() {
  std::exception_ptr exptr{};

  exptr = wrap(nic::InferGrpcContext::Create(&context_, url_, modelName_, modelVersion_, false),
               "setup(): unable to create inference context",
               Severity::High);
  if (exptr)
    return exptr;

  //only used for monitoring
  bool has_server = false;
  if (verbose_) {
    auto server_err = nic::ServerStatusGrpcContext::Create(&serverCtx_, url_, false);
    wrap(server_err, "setup(): unable to create server context", Severity::None);
    has_server = server_err.IsOk();
  }
  if (!has_server)
    serverCtx_ = nullptr;

  std::unique_ptr<nic::InferContext::Options> options;
  exptr = wrap(nic::InferContext::Options::Create(&options),
               "setup(): unable to create inference context options",
               Severity::Low);
  if (exptr)
    return exptr;

  options->SetBatchSize(batchSize_);
  for (const auto& output : context_->Outputs()) {
    exptr = wrap(options->AddRawResult(output), "setup(): unable to add raw result", Severity::Low);
    if (exptr)
      return exptr;
  }
  exptr = wrap(context_->SetRunOptions(*options), "setup(): unable to set run options", Severity::Low);
  if (exptr)
    return exptr;

  const auto& nicInputs = context_->Inputs();
  nicInput_ = nicInputs[0];
  nicInput_->Reset();

  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<int64_t> input_shape;
  for (unsigned i0 = 0; i0 < batchSize_; i0++) {
    float* arr = &(this->input_.data()[i0 * nInput_]);
    exptr = wrap(nicInput_->SetRaw(reinterpret_cast<const uint8_t*>(arr), nInput_ * sizeof(float)),
                 "setup(): unable to set data for batch entry " + std::to_string(i0),
                 Severity::High);
    if (exptr)
      return exptr;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  if (!this->debugName_.empty())
    edm::LogInfo(this->fullDebugName_) << "Input array time: "
                                       << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

  //should be empty
  return exptr;
}

template <typename Client>
std::exception_ptr TritonClient<Client>::getResults(const std::unique_ptr<nic::InferContext::Result>& result) {
  std::exception_ptr exptr{};

  auto t1 = std::chrono::high_resolution_clock::now();
  this->output_.resize(nOutput_ * batchSize_, 0.f);
  for (unsigned i0 = 0; i0 < batchSize_; i0++) {
    const uint8_t* r0;
    size_t content_byte_size;
    exptr = wrap(result->GetRaw(i0, &r0, &content_byte_size),
                 "getResults(): unable to get raw for entry " + std::to_string(i0),
                 Severity::High);
    if (exptr)
      return exptr;
    const float* lVal = reinterpret_cast<const float*>(r0);
    std::memcpy(&this->output_[i0 * nOutput_], lVal, nOutput_ * sizeof(float));
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  if (!this->debugName_.empty())
    edm::LogInfo(this->fullDebugName_) << "Output array time: "
                                       << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

  //should be empty
  return exptr;
}

//default case for sync and pseudo async
template <typename Client>
void TritonClient<Client>::evaluate() {
  //in case there is nothing to process
  if (this->batchSize_ == 0) {
    this->finish(true);
    return;
  }

  //common operations first
  auto exptr = setup();
  if (exptr) {
    this->finish(false, exptr);
    return;
  }

  // Get the status of the server prior to the request being made.
  const auto& start_status = getServerSideStatus();

  //blocking call
  auto t1 = std::chrono::high_resolution_clock::now();
  std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
  exptr = wrap(context_->Run(&results), "evaluate(): unable to run and/or get result", Severity::High);
  if (exptr) {
    this->output_.resize(nOutput_ * batchSize_, 0.f);
    this->finish(false, exptr);
    return;
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  if (!this->debugName_.empty())
    edm::LogInfo(this->fullDebugName_) << "Remote time: "
                                       << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

  const auto& end_status = this->getServerSideStatus();

  if (this->verbose()) {
    const auto& stats = this->summarizeServerStats(start_status, end_status);
    this->reportServerSideStats(stats);
  }

  exptr = getResults(results.begin()->second);

  this->finish(exptr == nullptr, exptr);
}

//specialization for true async
template <>
void TritonClientAsync::evaluate() {
  //in case there is nothing to process
  if (this->batchSize_ == 0) {
    this->finish(true);
    return;
  }

  //common operations first
  auto exptr = setup();
  if (exptr) {
    this->finish(false, exptr);
    return;
  }

  // Get the status of the server prior to the request being made.
  const auto& start_status = getServerSideStatus();

  //non-blocking call
  auto t1 = std::chrono::high_resolution_clock::now();
  exptr = wrap(context_->AsyncRun([t1, start_status, this](nic::InferContext* ctx,
                                                           const std::shared_ptr<nic::InferContext::Request>& request) {
    //get results
    std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
    auto exptr =
        this->wrap(ctx->GetAsyncRunResults(request, &results), "evaluate(): unable to get result", Severity::High);
    if (exptr) {
      this->output_.resize(nOutput_ * batchSize_, 0.f);
      this->finish(false, exptr);
      return;
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    if (!this->debugName_.empty())
      edm::LogInfo(this->fullDebugName_) << "Remote time: "
                                         << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    const auto& end_status = this->getServerSideStatus();

    if (this->verbose()) {
      const auto& stats = this->summarizeServerStats(start_status, end_status);
      this->reportServerSideStats(stats);
    }

    //check result
    exptr = this->getResults(results.begin()->second);

    //finish
    this->finish(exptr == nullptr, exptr);
  }),
               "evaluate(): unable to launch async run",
               Severity::High);

  //if AsyncRun failed, finish() wasn't called
  if (exptr)
    this->finish(false, exptr);
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
