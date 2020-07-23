#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"

#include "request_grpc.h"

#include <string>
#include <cmath>
#include <chrono>
#include <exception>
#include <sstream>
#include <utility>
#include <tuple>

namespace ni = nvidia::inferenceserver;
namespace nic = ni::client;

//based on https://github.com/NVIDIA/triton-inference-server/blob/v1.12.0/src/clients/c++/examples/simple_callback_client.cc

TritonClient::TritonClient(const edm::ParameterSet& params)
    : SonicClient(params),
      url_(params.getUntrackedParameter<std::string>("address") + ":" +
           std::to_string(params.getUntrackedParameter<unsigned>("port"))),
      timeout_(params.getUntrackedParameter<unsigned>("timeout")),
      modelName_(params.getParameter<std::string>("modelName")),
      modelVersion_(params.getParameter<int>("modelVersion")),
      verbose_(params.getUntrackedParameter<bool>("verbose")) {
  clientName_ = "TritonClient";
  //will get overwritten later, just used in constructor
  fullDebugName_ = clientName_;

  //connect to the server
  triton_utils::throwIfError(nic::InferGrpcContext::Create(&context_, url_, modelName_, modelVersion_, false),
                             "TritonClient(): unable to create inference context");

  //get options
  triton_utils::throwIfError(nic::InferContext::Options::Create(&options_),
                             "TritonClient(): unable to create inference context options");

  //get input and output (which know their sizes)
  const auto& nicInputs = context_->Inputs();
  const auto& nicOutputs = context_->Outputs();

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
    const auto& iname = nicInput->Name();
    const auto& curr_itr =
        input_.emplace(std::piecewise_construct, std::forward_as_tuple(iname), std::forward_as_tuple(iname, nicInput));
    if (verbose_) {
      const auto& curr_input = curr_itr.first->second;
      io_msg << "  " << iname << " (" << curr_input.dname() << ", " << curr_input.byteSize()
             << " b) : " << triton_utils::printVec(curr_input.dims()) << "\n";
    }
  }

  //setup output map
  if (verbose_)
    io_msg << "Model outputs: "
           << "\n";
  for (const auto& nicOutput : nicOutputs) {
    const auto& oname = nicOutput->Name();
    const auto& curr_itr = output_.emplace(
        std::piecewise_construct, std::forward_as_tuple(oname), std::forward_as_tuple(oname, nicOutput));
    const auto& curr_output = curr_itr.first->second;
    triton_utils::throwIfError(options_->AddRawResult(nicOutput),
                               "TritonClient(): unable to add raw result " + curr_itr.first->first);
    if (verbose_) {
      io_msg << "  " << oname << " (" << curr_output.dname() << ", " << curr_output.byteSize()
             << " b) : " << triton_utils::printVec(curr_output.dims()) << "\n";
    }
  }

  //check batch size limitations (after i/o setup)
  //triton uses max batch size = 0 to denote a model that does not support batching
  //but for models that do support batching, a given event may set batch size 0 to indicate no valid input is present
  //so set the local max to 1 and keep track of "no batch" case
  maxBatchSize_ = context_->MaxBatchSize();
  noBatch_ = maxBatchSize_ == 0;
  maxBatchSize_ = std::max(1u, maxBatchSize_);
  //check requested batch size
  setBatchSize(params.getUntrackedParameter<unsigned>("batchSize"));

  //initial server settings
  triton_utils::throwIfError(context_->SetRunOptions(*options_), "TritonClient(): unable to set run options");

  //print model info
  std::stringstream model_msg;
  if (verbose_) {
    model_msg << "Model name: " << modelName_ << "\n"
              << "Model version: " << modelVersion_ << "\n"
              << "Model max batch size: " << (noBatch_ ? 0 : maxBatchSize_) << "\n";
  }

  //only used for monitoring
  bool has_server = false;
  if (verbose_) {
    //print model info
    edm::LogInfo(fullDebugName_) << model_msg.str() << io_msg.str();

    has_server = triton_utils::warnIfError(nic::ServerStatusGrpcContext::Create(&serverCtx_, url_, false),
                                           "TritonClient(): unable to create server context");
  }
  if (!has_server)
    serverCtx_ = nullptr;
}

bool TritonClient::setBatchSize(unsigned bsize) {
  if (bsize > maxBatchSize_) {
    edm::LogWarning(fullDebugName_) << "Requested batch size " << bsize << " exceeds server-specified max batch size "
                                    << maxBatchSize_ << ". Batch size will remain as" << batchSize_;
    return false;
  } else {
    batchSize_ = bsize;
    //set for input and output
    for (auto& element : input_) {
      element.second.setBatchSize(bsize);
    }
    for (auto& element : output_) {
      element.second.setBatchSize(bsize);
    }
    //set for server (and Input objects)
    if (!noBatch_) {
      options_->SetBatchSize(batchSize_);
      triton_utils::throwIfError(context_->SetRunOptions(*options_), "setBatchSize(): unable to set run options");
    }
    return true;
  }
}

void TritonClient::reset() {
  for (auto& element : input_) {
    element.second.reset();
  }
  for (auto& element : output_) {
    element.second.reset();
  }
}

bool TritonClient::getResults(std::map<std::string, std::unique_ptr<nic::InferContext::Result>>& results) {
  for (auto& element : results) {
    const auto& oname = element.first;
    auto& result = element.second;

    //check for corresponding entry in output map
    auto itr = output_.find(oname);
    if (itr == output_.end()) {
      edm::LogError("TritonServerError") << "getResults(): no entry in output map for result " << oname;
      return false;
    }
    auto& output = itr->second;

    //set shape here before output becomes const
    if (output.variableDims()) {
      bool status =
          triton_utils::warnIfError(result->GetRawShape(&(output.shape())), "getResults(): unable to get output shape");
      if (!status)
        return status;
    }
    //transfer ownership
    output.setResult(std::move(result));
  }

  return true;
}

//default case for sync and pseudo async
void TritonClient::evaluate() {
  //in case there is nothing to process
  if (batchSize_ == 0) {
    finish(true);
    return;
  }

  // Get the status of the server prior to the request being made.
  const auto& start_status = getServerSideStatus();

  if (mode_ == SonicMode::Async) {
    //non-blocking call
    auto t1 = std::chrono::high_resolution_clock::now();
    bool status = triton_utils::warnIfError(
        context_->AsyncRun([t1, start_status, this](nic::InferContext* ctx,
                                                    const std::shared_ptr<nic::InferContext::Request>& request) {
          //get results
          std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
          bool status =
              triton_utils::warnIfError(ctx->GetAsyncRunResults(request, &results), "evaluate(): unable to get result");
          if (!status) {
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
          status = getResults(results);

          //finish
          finish(status);
        }),
        "evaluate(): unable to launch async run");

    //if AsyncRun failed, finish() wasn't called
    if (!status)
      finish(false);
  } else {
    //blocking call
    auto t1 = std::chrono::high_resolution_clock::now();
    std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
    bool status = triton_utils::warnIfError(context_->Run(&results), "evaluate(): unable to run and/or get result");
    if (!status) {
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

    status = getResults(results);

    finish(status);
  }
}

void TritonClient::reportServerSideStats(const TritonClient::ServerSideStats& stats) const {
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

  if (!debugName_.empty())
    edm::LogInfo(fullDebugName_) << msg.str();
}

TritonClient::ServerSideStats TritonClient::summarizeServerStats(const ni::ModelStatus& start_status,
                                                                 const ni::ModelStatus& end_status) const {
  // If model_version is -1 then look in the end status to find the
  // latest (highest valued version) and use that as the version.
  int64_t status_model_version = 0;
  if (modelVersion_ < 0) {
    for (const auto& vp : end_status.version_status()) {
      status_model_version = std::max(status_model_version, vp.first);
    }
  } else
    status_model_version = modelVersion_;

  TritonClient::ServerSideStats server_stats;
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

ni::ModelStatus TritonClient::getServerSideStatus() const {
  if (serverCtx_) {
    ni::ServerStatus server_status;
    serverCtx_->GetServerStatus(&server_status);
    auto itr = server_status.model_status().find(modelName_);
    if (itr != server_status.model_status().end())
      return itr->second;
  }
  return ni::ModelStatus{};
}

//for fillDescriptions
void TritonClient::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
  edm::ParameterSetDescription descClient;
  fillBasePSetDescription(descClient);
  descClient.add<std::string>("modelName");
  descClient.add<int>("modelVersion", -1);
  //server parameters should not affect the physics results
  descClient.addUntracked<unsigned>("batchSize");
  descClient.addUntracked<std::string>("address");
  descClient.addUntracked<unsigned>("port");
  descClient.addUntracked<unsigned>("timeout");
  descClient.addUntracked<bool>("verbose", false);
  iDesc.add<edm::ParameterSetDescription>("Client", descClient);
}
