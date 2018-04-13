/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
//NOTE: The memory layout of the Node class changes depending on if NDEBUG was
// set when tensorflow was compiled. The reason is Node class holds two edgeset
// class instances and edgeset adds a member data if NDEBUG is set

/*
This file is an adaptation of the original direct_session.cc file located at
https://github.com/tensorflow/tensorflow/blob/v1.6.0/tensorflow/core/common_runtime/direct_session.cc
to meet the demands of the software environment developed and used by the CMS collaboration.

Changes with respect to the original code are documented in the TBBSession.h header file.
*/

#if !defined(NDEBUG)
#define NDEBUG 1
#endif

#include "TBBSession.h"

#include <atomic>
#include <string>
#include <vector>

#include "tbb/task_group.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "FWCore/Concurrency/interface/FunctorTask.h"

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/debugger_state_interface.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb_text.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/device_tracer.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

namespace {

CMS_THREAD_SAFE auto* tbb_session_runs = monitoring::Counter<0>::New(
    "/tensorflow/core/tbb_session_runs",
    "The number of times TBBSession::Run() has been called.");

// TODO(vrv): Figure out how to unify the many different functions
// that generate RendezvousKey, since many of them have to be
// consistent with each other.
string GetRendezvousKey(const string& tensor_name,
                        const DeviceAttributes& device_info,
                        const FrameAndIter& frame_iter) {
  return strings::StrCat(device_info.name(), ";",
                         strings::FpToString(device_info.incarnation()), ";",
                         device_info.name(), ";", tensor_name, ";",
                         frame_iter.frame_id, ":", frame_iter.iter_id);
}

}  // namespace

class TBBSessionFactory : public SessionFactory {
 public:
  TBBSessionFactory() {}

  bool AcceptsOptions(const SessionOptions& options) override {
    return options.target == "tbb";
  }

  Session* NewSession(const SessionOptions& options) override {
    // Must do this before the CPU allocator is created.
    if (options.config.graph_options().build_cost_model() > 0) {
      EnableCPUAllocatorFullStats(true);
    }
    std::vector<Device*> devices;
    const Status s = DeviceFactory::AddDevices(
        options, "/job:localhost/replica:0/task:0", &devices);
    if (!s.ok()) {
      LOG(ERROR) << s;
      return nullptr;
    }

    TBBSession* session =
        new TBBSession(options, new DeviceMgr(devices), this);
    {
      mutex_lock l(sessions_lock_);
      sessions_.push_back(session);
    }
    return session;
  }

  Status Reset(const SessionOptions& options,
               const std::vector<string>& containers) override {
    std::vector<TBBSession*> sessions_to_reset;
    {
      mutex_lock l(sessions_lock_);
      // We create a copy to ensure that we don't have a deadlock when
      // session->Close calls the TBBSessionFactory.Deregister, which
      // acquires sessions_lock_.
      std::swap(sessions_to_reset, sessions_);
    }
    Status s;
    for (auto session : sessions_to_reset) {
      s.Update(session->Reset(containers));
    }
    // TODO(suharshs): Change the Reset behavior of all SessionFactories so that
    // it doesn't close the sessions?
    for (auto session : sessions_to_reset) {
      s.Update(session->Close());
    }
    return s;
  }

  void Deregister(const TBBSession* session) {
    mutex_lock l(sessions_lock_);
    sessions_.erase(std::remove(sessions_.begin(), sessions_.end(), session),
                    sessions_.end());
  }

 private:
  mutex sessions_lock_;
  std::vector<TBBSession*> sessions_ GUARDED_BY(sessions_lock_);
};

class TBBSessionRegistrar {
 public:
  TBBSessionRegistrar() {
    SessionFactory::Register("TBB_SESSION", new TBBSessionFactory());
  }
};
static TBBSessionRegistrar registrar;

std::atomic_int_fast64_t TBBSession::step_id_counter_(1);

// NOTE: On Android with a single device, there is never
// a risk of an OpKernel blocking indefinitely:
//
// 1) No operations do I/O that depends on other simultaneous kernels,
//
// 2) Recv nodes always complete immediately: The inputs are sent into
//    the local rendezvous before we start the executor, so the
//    corresponding recvs will not block.
//
// Based on these assumptions, we can use the same thread pool for
// both "non-blocking" and "blocking" OpKernels on Android.
//
// This may change down the road when we add support for multiple
// devices that run concurrently, in which case we will need to
// revisit this decision.
// Override to allow CMSSW FWK to schedule
void TBBSession::SchedClosure(tbb::task_arena& arena, tbb::task_group& g, std::function<void()> c) {
  arena.execute( [&g, &c] () {g.run( c ); } );
}

TBBSession::TBBSession(const SessionOptions& options,
                       const DeviceMgr* device_mgr,
                       TBBSessionFactory* const factory)
    : options_(options),
      device_mgr_(device_mgr),
      factory_(factory),
      cancellation_manager_(new CancellationManager()),
      operation_timeout_in_ms_(options_.config.operation_timeout_in_ms()) {
  // The default value of sync_on_finish will be flipped soon and this
  // environment variable will be removed as well.
  const Status status =
      ReadBoolFromEnvVar("TF_SYNC_ON_FINISH", true, &sync_on_finish_);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
  }
  // NOTE(mrry): We do not need to use a unique string for the session
  // handle, because TBBSession owns its devices. This may change
  // in future versions.
  session_handle_ = "tbb";
  int devices_added = 0;
  if (options.config.log_device_placement()) {
    const string mapping_str = device_mgr_->DeviceMappingString();
    if (mapping_str.empty()) {
      printf("Device mapping: no known devices.\n");
    } else {
      printf("Device mapping:\n%s", mapping_str.c_str());
    }
    LOG(INFO) << "Device mapping:\n" << mapping_str;
  }
  for (auto d : device_mgr_->ListDevices()) {
    devices_.push_back(d);
    device_set_.AddDevice(d);
    d->op_segment()->AddHold(session_handle_);

    // The first device added is special: it is the 'client device' (a
    // CPU device) from which we feed and fetch Tensors.
    if (devices_added == 0) {
      device_set_.set_client_device(d);
    }
    ++devices_added;
  }
}

TBBSession::~TBBSession() {
  if (!closed_) Close().IgnoreError();
  for (auto& it : partial_runs_) {
    it.second.reset(nullptr);
  }
  for (auto& it : executors_) {
    it.second.reset();
  }
  for (auto d : device_mgr_->ListDevices()) {
    d->op_segment()->RemoveHold(session_handle_);
  }
  for (auto d : device_mgr_->ListDevices()) {
    d->ClearResourceMgr();
  }
  functions_.clear();
  delete cancellation_manager_;

  execution_state_.reset(nullptr);
  flib_def_.reset(nullptr);
}

Status TBBSession::MaybeInitializeExecutionState(
    const GraphDef& graph, bool* out_already_initialized) {
  // If already initialized, do nothing.
  if (flib_def_ && execution_state_) {
    *out_already_initialized = true;
    return Status::OK();
  }
  // Set up the per-session execution state.
  // NOTE(mrry): The function library created here will be used for
  // all subsequent extensions of the graph.
  flib_def_.reset(
      new FunctionLibraryDefinition(OpRegistry::Global(), graph.library()));
  GraphExecutionStateOptions options;
  options.device_set = &device_set_;
  options.session_options = &options_;
  // TODO(mrry,suharshs): We explicitly copy `graph` so that
  // `MakeForBaseGraph()` can take ownership of its
  // contents. Previously this happened implicitly in calls to the
  // `GraphExecutionState`. Other sessions call
  // `MakeForBaseGraph` in such a way that we can destructively read
  // the passed-in `GraphDef`. In principle we could do the same here,
  // with a wider refactoring; we might revise the direct session so
  // that it copies the graph fewer times.
  GraphDef temp(graph);
  TF_RETURN_IF_ERROR(
      GraphExecutionState::MakeForBaseGraph(&temp, options, &execution_state_));
  graph_created_ = true;
  *out_already_initialized = false;
  return Status::OK();
}

Status TBBSession::Create(const GraphDef& graph) {
  TF_RETURN_IF_ERROR(init_error_);
  if (graph.node_size() > 0) {
    mutex_lock l(graph_def_lock_);
    if (graph_created_) {
      return errors::AlreadyExists(
          "A Graph has already been created for this session.");
    }
    return ExtendLocked(graph);
  }
  return Status::OK();
}

Status TBBSession::Extend(const GraphDef& graph) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  mutex_lock l(graph_def_lock_);
  return ExtendLocked(graph);
}

Status TBBSession::ExtendLocked(const GraphDef& graph) {
  bool already_initialized;
  // If this is the first call, we can initialize the execution state
  // with `graph` and do not need to call `Extend()`.
  TF_RETURN_IF_ERROR(
      MaybeInitializeExecutionState(graph, &already_initialized));
  if (already_initialized) {
    TF_RETURN_IF_ERROR(flib_def_->AddLibrary(graph.library()));
    std::unique_ptr<GraphExecutionState> state;
    TF_RETURN_IF_ERROR(execution_state_->Extend(graph, &state));
    execution_state_.swap(state);
  }
  return Status::OK();
}

Status TBBSession::Run(const NamedTensorList& inputs,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          std::vector<Tensor>* outputs) {
  RunMetadata run_metadata;
  return Run(RunOptions(), inputs, output_names, target_nodes, outputs,
             &run_metadata);
}

Status TBBSession::CreateDebuggerState(
    const DebugOptions& debug_options, int64 session_run_index,
    int64 executor_step_index, const std::vector<string>& input_names,
    const std::vector<string>& output_names,
    const std::vector<string>& target_names,
    std::unique_ptr<DebuggerStateInterface>* debugger_state) {
  TF_RETURN_IF_ERROR(
      DebuggerStateRegistry::CreateState(debug_options, debugger_state));
  TF_RETURN_IF_ERROR(debugger_state->get()->PublishDebugMetadata(
      debug_options.global_step(), session_run_index, executor_step_index,
      input_names, output_names, target_names));
  return Status::OK();
}

Status TBBSession::DecorateAndPublishGraphForDebug(
    const DebugOptions& debug_options, Graph* graph, Device* device) {
  std::unique_ptr<DebugGraphDecoratorInterface> decorator;
  TF_RETURN_IF_ERROR(
      DebugGraphDecoratorRegistry::CreateDecorator(debug_options, &decorator));

  TF_RETURN_IF_ERROR(decorator->DecorateGraph(graph, device));
  TF_RETURN_IF_ERROR(decorator->PublishGraph(*graph, device->name()));
  return Status::OK();
}

Status TBBSession::Run(const RunOptions& run_options,
                          const NamedTensorList& inputs,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          std::vector<Tensor>* outputs,
                          RunMetadata* run_metadata) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  tbb_session_runs->GetCell()->IncrementBy(1);
  {
    mutex_lock l(graph_def_lock_);
    if (!graph_created_) {
      return errors::InvalidArgument(
          "Session was not created with a graph before Run()!");
    }
  }

  // Extract the inputs names for this run of the session.
  std::vector<string> input_tensor_names;
  input_tensor_names.reserve(inputs.size());
  for (const auto& it : inputs) {
    input_tensor_names.push_back(it.first);
  }

  // Check if we already have an executor for these arguments.
  ExecutorsAndKeys* executors_and_keys;
  RunStateArgs run_state_args(run_options.debug_options());

  Executor::Args args;
  args.step_id = step_id_counter_.fetch_add(1);

  TF_RETURN_IF_ERROR(
      GetOrCreateExecutors(input_tensor_names, output_names,
                           target_nodes, &executors_and_keys,
                           &run_state_args));
  const int64 executor_step_count = executors_and_keys->step_count.fetch_add(1);

  std::unique_ptr<DebuggerStateInterface> debugger_state;
  if (!run_options.debug_options().debug_tensor_watch_opts().empty()) {
    TF_RETURN_IF_ERROR(CreateDebuggerState(
        run_options.debug_options(), args.step_id, executor_step_count,
        input_tensor_names, output_names, target_nodes, &debugger_state));
  }

  // Configure a call frame for the step, which we use to feed and
  // fetch values to and from the executors.
  FunctionCallFrame call_frame(executors_and_keys->input_types,
                               executors_and_keys->output_types);
  gtl::InlinedVector<Tensor, 4> feed_args(inputs.size());
  for (const auto& it : inputs) {
    if (it.second.dtype() == DT_RESOURCE) {
      Tensor tensor_from_handle;
      TF_RETURN_IF_ERROR(
          ResourceHandleToInputTensor(it.second, &tensor_from_handle));
      feed_args[executors_and_keys->input_name_to_index[it.first]] =
          tensor_from_handle;
    } else {
      feed_args[executors_and_keys->input_name_to_index[it.first]] = it.second;
    }
  }
  const Status s = call_frame.SetArgs(feed_args);
  if (errors::IsInternal(s)) {
    return errors::InvalidArgument(s.error_message());
  } else if (!s.ok()) {
    return s;
  }

  // Create a run state and start execution.
  RunState run_state(args.step_id, &devices_);
  run_state.rendez = new IntraProcessRendezvous(device_mgr_.get());
  CancellationManager step_cancellation_manager;
  args.call_frame = &call_frame;

  // Use a task_arena to avoid having unrelated tasks start
  // running on this thread (which could start deadlocks)
  tbb::task_arena taskArena;
  tbb::task_group taskGroup;
  // we are required to always call wait before destructor
  auto doneWithTaskGroup = [&taskArena, &taskGroup](void *) { taskArena.execute([&taskGroup]() { taskGroup.wait();}); };
  std::unique_ptr<tbb::task_group, decltype(doneWithTaskGroup) > guard(&taskGroup, doneWithTaskGroup);

  // Start parallel Executors.
  const size_t num_executors = executors_and_keys->items.size();
  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_executors, run_state.rendez, [&run_state](const Status& ret) {
        {
          mutex_lock l(run_state.mu_);
          run_state.status.Update(ret);
        }
        run_state.executors_done.Notify();
      });

  args.rendezvous = run_state.rendez;
  args.cancellation_manager = &step_cancellation_manager;

  args.session_state = &session_state_;
  args.tensor_store = &run_state.tensor_store;
  args.step_container = &run_state.step_container;
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordStep(args.step_id, run_state_args.handle);
  }
  args.sync_on_finish = sync_on_finish_;

  const bool do_trace = (run_options.trace_level() > RunOptions::NO_TRACE);

  bool update_cost_model = false;
  if (options_.config.graph_options().build_cost_model() > 0) {
    const int64 build_cost_model_every =
        options_.config.graph_options().build_cost_model();
    const int64 build_cost_model_after =
        options_.config.graph_options().build_cost_model_after();
    int64 measure_step_count = executor_step_count - build_cost_model_after;
    if (measure_step_count >= 0) {
      update_cost_model =
          ((measure_step_count + 1) % build_cost_model_every == 0);
    }
  }
  if (do_trace || update_cost_model ||
      run_options.report_tensor_allocations_upon_oom()) {
    run_state.collector.reset(
        new StepStatsCollector(run_metadata->mutable_step_stats()));
    args.stats_collector = run_state.collector.get();
  }

  std::unique_ptr<DeviceTracer> tracer;
  if (run_options.trace_level() >= RunOptions::HARDWARE_TRACE) {
    tracer = CreateDeviceTracer();
    // tracer may be NULL on platforms without accelerators.
    if (tracer) {
      Status s = tracer->Start();
      if (!s.ok()) {
        run_state.executors_done.Notify();
        delete barrier;
        return s;
      }
    }
  }

  // Register this step with session's cancellation manager, so that
  // `Session::Close()` will cancel the step.
  const CancellationToken cancellation_token =
      cancellation_manager_->get_cancellation_token();
  const bool already_cancelled = !cancellation_manager_->RegisterCallback(
      cancellation_token, [&step_cancellation_manager]() {
        step_cancellation_manager.StartCancel();
      });
  if (already_cancelled) {
    // NOTE(mrry): If we don't explicitly notify
    // `run_state.executors_done`, the RunState destructor would
    // block on this notification.
    run_state.executors_done.Notify();
    delete barrier;
    return errors::Cancelled("Run call was cancelled");
  }

  // pass taskArena and taskGroup to SchedClosure
  // consequently, disable TF's own thread logic inside the loop
  Executor::Args::Runner default_runner = [this, &taskArena, &taskGroup](Executor::Args::Closure c) {
    SchedClosure(taskArena, taskGroup, std::move(c));
  };
  for (const auto& item : executors_and_keys->items) {
    // TODO(zhengxq): support partial run.
    // TODO(zhengxq): if the device picks its own threadpool, we need to assign
    //     less threads to the main compute pool by default.
    // thread::ThreadPool* device_thread_pool =
    //     item.device->tensorflow_device_thread_pool();
    // if (!device_thread_pool) {
    //   args.runner = default_runner;
    // } else {
    //   args.runner = [this, device_thread_pool](Executor::Args::Closure c) {
    //     SchedClosure(device_thread_pool, std::move(c));
    //   };
    // }
    args.runner = default_runner;
    item.executor->RunAsync(args, barrier->Get());
  }

  // WaitForNotification will handle calling wait on taskGroup
  guard.release();
  WaitForNotification(taskArena, taskGroup, &run_state, &step_cancellation_manager,
                      run_options.timeout_in_ms() > 0
                          ? run_options.timeout_in_ms()
                          : operation_timeout_in_ms_);

  if (!cancellation_manager_->DeregisterCallback(cancellation_token)) {
    // The step has been cancelled: make sure we don't attempt to receive the
    // outputs as this would make it block forever.
    mutex_lock l(run_state.mu_);
    run_state.status.Update(errors::Cancelled("Run call was cancelled"));
  }

  if (tracer) {
    TF_RETURN_IF_ERROR(tracer->Stop());
    TF_RETURN_IF_ERROR(tracer->Collect(args.stats_collector));
  }

  {
    mutex_lock l(run_state.mu_);
    TF_RETURN_IF_ERROR(run_state.status);
  }

  // Receive outputs.
  if (outputs) {
    std::vector<Tensor> sorted_outputs;
    const Status s = call_frame.ConsumeRetvals(&sorted_outputs);
    if (errors::IsInternal(s)) {
      return errors::InvalidArgument(s.error_message());
    } else if (!s.ok()) {
      return s;
    }
    const bool unique_outputs =
        output_names.size() == executors_and_keys->output_name_to_index.size();
    // first_indices[i] = j implies that j is the smallest value for which
    // output_names[i] == output_names[j].
    std::vector<int> first_indices;
    if (!unique_outputs) {
      first_indices.resize(output_names.size());
      for (int i = 0; i < static_cast<int>(output_names.size()); ++i) {
        for (int j = 0; j <= i; ++j) {
          if (output_names[i] == output_names[j]) {
            first_indices[i] = j;
            break;
          }
        }
      }
    }
    outputs->clear();
    outputs->reserve(sorted_outputs.size());
    for (int i = 0; i < static_cast<int>(output_names.size()); ++i) {
      const string& output_name = output_names[i];
      if (first_indices.empty() || first_indices[i] == i) {
        outputs->emplace_back(
            std::move(sorted_outputs[executors_and_keys
                                         ->output_name_to_index[output_name]]));
      } else {
        outputs->push_back((*outputs)[first_indices[i]]);
      }
    }
  }

  // Save the output tensors of this run we choose to keep.
  TF_RETURN_IF_ERROR(
      run_state.tensor_store.SaveTensors(output_names, &session_state_));
  if (args.stats_collector) {
    args.stats_collector->Finalize();
  }

  // Build and return the cost model as instructed.
  mutex_lock l(executor_lock_);
  if (update_cost_model) {
    // Build the cost model
    std::unordered_map<string, const Graph*> device_to_graph;
    for (const PerPartitionExecutorsAndLib& partition :
         executors_and_keys->items) {
      const Graph* graph = partition.graph;
      const string device = partition.flib->device()->name();
      device_to_graph[device] = graph;
    }
    args.stats_collector->BuildCostModel(&cost_model_manager_, device_to_graph);

    // annotate stats onto cost graph.
    CostGraphDef* cost_graph = run_metadata->mutable_cost_graph();
    for (const auto& item : executors_and_keys->items) {
      TF_RETURN_IF_ERROR(
          cost_model_manager_.AddToCostGraphDef(item.graph, cost_graph));
    }
  }

  // If requested via RunOptions, output the partition graphs.
  if (run_options.output_partition_graphs()) {
    protobuf::RepeatedPtrField<GraphDef>* partition_graph_defs =
        run_metadata->mutable_partition_graphs();
    for (const PerPartitionExecutorsAndLib& exec_and_lib :
         executors_and_keys->items) {
      GraphDef* partition_graph_def = partition_graph_defs->Add();
      exec_and_lib.graph->ToGraphDef(partition_graph_def);
    }
  }

  return Status::OK();
}

Status TBBSession::ResourceHandleToInputTensor(const Tensor& resource_tensor,
                                                  Tensor* retrieved_tensor) {
  if (resource_tensor.dtype() != DT_RESOURCE) {
    return errors::InvalidArgument(strings::StrCat(
        "ResourceHandleToInputTensor() received non-DT_RESOURCE Tensor: ",
        resource_tensor.dtype()));
  }

  const ResourceHandle& resource_handle =
      resource_tensor.scalar<ResourceHandle>()();

  if (resource_handle.container() ==
      SessionState::kTensorHandleResourceTypeName) {
    return session_state_.GetTensor(resource_handle.name(), retrieved_tensor);
  } else {
    return errors::InvalidArgument(strings::StrCat(
        "Invalid resource type hash code: ", resource_handle.hash_code(),
        "(name: ", resource_handle.name(),
        " type: ", resource_handle.maybe_type_name(),
        "). Perhaps a resource tensor was being provided as a feed? That is "
        "not currently allowed. Please file an issue at "
        "https://github.com/tensorflow/tensorflow/issues/new, ideally with a "
        "short code snippet that leads to this error message."));
  }
}

Status TBBSession::GetOrCreateExecutors(
    gtl::ArraySlice<string> inputs, gtl::ArraySlice<string> outputs,
    gtl::ArraySlice<string> target_nodes, ExecutorsAndKeys** executors_and_keys,
    RunStateArgs* run_state_args) {
  int64 handle_name_counter_value = -1;
  if (LogMemory::IsEnabled() || run_state_args->is_partial_run) {
    handle_name_counter_value = handle_name_counter_.fetch_add(1);
  }

  string debug_tensor_watches_summary;
  if (!run_state_args->debug_options.debug_tensor_watch_opts().empty()) {
    debug_tensor_watches_summary = SummarizeDebugTensorWatches(
        run_state_args->debug_options.debug_tensor_watch_opts());
  }

  // Fast lookup path, no sorting.
  const string key = strings::StrCat(
      str_util::Join(inputs, ","), "->", str_util::Join(outputs, ","), "/",
      str_util::Join(target_nodes, ","), "/", run_state_args->is_partial_run,
      "/", debug_tensor_watches_summary);
  // Set the handle, if it's needed to log memory or for partial run.
  if (handle_name_counter_value >= 0) {
    run_state_args->handle =
        strings::StrCat(key, ";", handle_name_counter_value);
  }

  // See if we already have the executors for this run.
  {
    mutex_lock l(executor_lock_);  // could use reader lock
    auto it = executors_.find(key);
    if (it != executors_.end()) {
      *executors_and_keys = it->second.get();
      return Status::OK();
    }
  }

  // Slow lookup path, the unsorted key missed the cache.
  // Sort the inputs and outputs, and look up with the sorted key in case an
  // earlier call used a different order of inputs and outputs.
  //
  // We could consider some other signature instead of sorting that
  // preserves the same property to avoid the sort in the future.
  std::vector<string> inputs_sorted(inputs.begin(), inputs.end());
  std::sort(inputs_sorted.begin(), inputs_sorted.end());
  std::vector<string> outputs_sorted(outputs.begin(), outputs.end());
  std::sort(outputs_sorted.begin(), outputs_sorted.end());
  std::vector<string> tn_sorted(target_nodes.begin(), target_nodes.end());
  std::sort(tn_sorted.begin(), tn_sorted.end());

  const string sorted_key = strings::StrCat(
      str_util::Join(inputs_sorted, ","), "->",
      str_util::Join(outputs_sorted, ","), "/", str_util::Join(tn_sorted, ","),
      "/", run_state_args->is_partial_run, "/", debug_tensor_watches_summary);
  // Set the handle, if its needed to log memory or for partial run.
  if (handle_name_counter_value >= 0) {
    run_state_args->handle =
        strings::StrCat(sorted_key, ";", handle_name_counter_value);
  }

  // See if we already have the executors for this run.
  {
    mutex_lock l(executor_lock_);
    auto it = executors_.find(sorted_key);
    if (it != executors_.end()) {
      *executors_and_keys = it->second.get();
      // Insert this under the original key.
      executors_.emplace(key, it->second);
      return Status::OK();
    }
  }

  // Nothing found, so create the executors and store in the cache.
  BuildGraphOptions options;
  options.feed_endpoints = inputs_sorted;
  options.fetch_endpoints = outputs_sorted;
  options.target_nodes = tn_sorted;
  options.use_function_convention = !run_state_args->is_partial_run;
  if (!run_state_args->debug_options.debug_tensor_watch_opts().empty()) {
    options.debug_options = run_state_args->debug_options;
  }

  std::unique_ptr<FunctionInfo> func_info(new FunctionInfo);
  std::shared_ptr<ExecutorsAndKeys> ek(new ExecutorsAndKeys);

  // The executor_lock_ is intentionally released while executor is
  // being created.
  std::unordered_map<string, std::unique_ptr<Graph>> graphs;
  TF_RETURN_IF_ERROR(CreateGraphs(options, &graphs, &func_info->flib_def,
                                  run_state_args, &ek->input_types,
                                  &ek->output_types));

  if (run_state_args->is_partial_run) {
    ek->graph = std::move(run_state_args->graph);
    std::unordered_set<StringPiece, StringPieceHasher> names;
    for (const string& input : inputs) {
      TensorId id(ParseTensorName(input));
      names.emplace(id.first);
    }
    for (const string& output : outputs) {
      TensorId id(ParseTensorName(output));
      names.emplace(id.first);
    }
    for (Node* n : ek->graph->nodes()) {
      if (names.count(n->name()) > 0) {
        ek->name_to_node.insert({n->name(), n});
      }
    }
  }
  ek->items.reserve(graphs.size());
  const auto& optimizer_opts =
      options_.config.graph_options().optimizer_options();

  int graph_def_version;
  {
    mutex_lock l(graph_def_lock_);
    graph_def_version =
        execution_state_->original_graph_def().versions().producer();
  }
  func_info->proc_flr.reset(new ProcessFunctionLibraryRuntime(
      device_mgr_.get(), options_.env, graph_def_version,
      func_info->flib_def.get(), optimizer_opts));

  GraphOptimizer optimizer(optimizer_opts);
  for (auto iter = graphs.begin(); iter != graphs.end(); ++iter) {
    const string& partition_name = iter->first;
    std::unique_ptr<Graph>& partition_graph = iter->second;

    Device* device;
    TF_RETURN_IF_ERROR(device_mgr_->LookupDevice(partition_name, &device));

    ek->items.resize(ek->items.size() + 1);
    auto* item = &(ek->items.back());
    auto lib = func_info->proc_flr->GetFLR(partition_name);
    if (lib == nullptr) {
      return errors::Internal("Could not find device: ", partition_name);
    }
    item->flib = lib;

    LocalExecutorParams params;
    params.device = device;
    params.function_library = lib;
    auto opseg = device->op_segment();
    params.create_kernel = [this, lib, opseg](const NodeDef& ndef,
                                              OpKernel** kernel) {
      // We do not share the kernel via the OpSegment if the node is
      // stateless, or a function.
      // NOTE(mrry): We must not share function kernels (implemented
      // using `CallOp`) between subgraphs, because `CallOp::handle_`
      // is tied to a particular subgraph. Even if the function itself
      // is stateful, the `CallOp` that invokes it is not.
      if (!lib->IsStateful(ndef.op()) ||
          lib->GetFunctionLibraryDefinition()->Find(ndef.op()) != nullptr) {
        return lib->CreateKernel(ndef, kernel);
      }
      auto create_fn = [lib, &ndef](OpKernel** kernel) {
        return lib->CreateKernel(ndef, kernel);
      };
      // Kernels created for subgraph nodes need to be cached.  On
      // cache miss, create_fn() is invoked to create a kernel based
      // on the function library here + global op registry.
      return opseg->FindOrCreate(session_handle_, ndef.name(), kernel,
                                 create_fn);
    };
    params.delete_kernel = [lib](OpKernel* kernel) {
      // If the node is stateful, opseg owns it. Otherwise, delete it.
      if (kernel && !lib->IsStateful(kernel->type_string())) {
        delete kernel;
      }
    };
    params.node_outputs_cb = node_outputs_callback_;

    optimizer.Optimize(lib, options_.env, device, &iter->second,
                       /*shape_map=*/nullptr);

    // EXPERIMENTAL: tfdbg inserts debug nodes in the graph.
    if (!options.debug_options.debug_tensor_watch_opts().empty()) {
      TF_RETURN_IF_ERROR(DecorateAndPublishGraphForDebug(
          options.debug_options, partition_graph.get(), params.device));
    }

    TF_RETURN_IF_ERROR(EnsureMemoryTypes(DeviceType(device->device_type()),
                                         device->name(),
                                         partition_graph.get()));
    // NewLocalExecutor takes ownership of partition_graph.
    item->graph = partition_graph.get();
    item->executor = nullptr;
    item->device = device;
    Executor* executor;
    TF_RETURN_IF_ERROR(
        NewLocalExecutor(params, partition_graph.release(), &executor));
    item->executor.reset(executor);
  }

  // Cache the mapping from input/output names to graph elements to
  // avoid recomputing it every time.
  if (!run_state_args->is_partial_run) {
    // For regular `Run()`, we use the function calling convention, and so
    // maintain a mapping from input/output names to
    // argument/return-value ordinal index.
    for (size_t i = 0; i < inputs_sorted.size(); ++i) {
      const string& input = inputs_sorted[i];
      ek->input_name_to_index[input] = i;
    }
    for (size_t i = 0; i < outputs_sorted.size(); ++i) {
      const string& output = outputs_sorted[i];
      ek->output_name_to_index[output] = i;
    }
  } else {
    // For `PRun()`, we use the rendezvous calling convention, and so
    // maintain a mapping from input/output names to rendezvous keys.
    //
    // We always use the first device as the device name portion of the
    // key, even if we're feeding another graph.
    for (size_t i = 0; i < inputs_sorted.size(); ++i) {
      const string& input = inputs_sorted[i];
      ek->input_name_to_rendezvous_key[input] = GetRendezvousKey(
          input, device_set_.client_device()->attributes(), FrameAndIter(0, 0));
    }
    for (size_t i = 0; i < outputs_sorted.size(); ++i) {
      const string& output = outputs_sorted[i];
      ek->output_name_to_rendezvous_key[output] =
          GetRendezvousKey(output, device_set_.client_device()->attributes(),
                           FrameAndIter(0, 0));
    }
  }

  // Reacquire the lock, try to insert into the map.
  mutex_lock l(executor_lock_);
  functions_.push_back(std::move(func_info));

  // Another thread may have created the entry before us, in which case we will
  // reuse the already created one.
  auto insert_result = executors_.emplace(sorted_key, ek);
  // Insert the value under the original key, so the fast path lookup will work
  // if the user uses the same order of inputs, outputs, and targets again.
  executors_.emplace(key, insert_result.first->second);
  *executors_and_keys = insert_result.first->second.get();

  return Status::OK();
}

Status TBBSession::CreateGraphs(
    const BuildGraphOptions& subgraph_options,
    std::unordered_map<string, std::unique_ptr<Graph>>* outputs,
    std::unique_ptr<FunctionLibraryDefinition>* flib_def,
    RunStateArgs* run_state_args, DataTypeVector* input_types,
    DataTypeVector* output_types) {
  mutex_lock l(graph_def_lock_);
  std::unique_ptr<ClientGraph> client_graph;

  std::unique_ptr<GraphExecutionState> temp_exec_state_holder;
  GraphExecutionState* execution_state = nullptr;
  if (options_.config.graph_options().place_pruned_graph()) {
    // Because we are placing pruned graphs, we need to create a
    // new GraphExecutionState for every new unseen graph,
    // and then place it.
    GraphExecutionStateOptions prune_options;
    prune_options.device_set = &device_set_;
    prune_options.session_options = &options_;
    prune_options.stateful_placements = stateful_placements_;
    TF_RETURN_IF_ERROR(GraphExecutionState::MakeForPrunedGraph(
        execution_state_->original_graph_def().library(), prune_options,
        execution_state_->original_graph_def(), subgraph_options,
        &temp_exec_state_holder, &client_graph));
    execution_state = temp_exec_state_holder.get();
  } else {
    execution_state = execution_state_.get();
    TF_RETURN_IF_ERROR(
        execution_state->BuildGraph(subgraph_options, &client_graph));
  }

  if (subgraph_options.feed_endpoints.size() !=
      client_graph->feed_types.size()) {
    return errors::Internal(
        "Graph pruning failed: requested number of feed endpoints = ",
        subgraph_options.feed_endpoints.size(),
        " versus number of pruned feed endpoints = ",
        client_graph->feed_types.size());
  }
  if (subgraph_options.fetch_endpoints.size() !=
      client_graph->fetch_types.size()) {
    return errors::Internal(
        "Graph pruning failed: requested number of fetch endpoints = ",
        subgraph_options.fetch_endpoints.size(),
        " versus number of pruned fetch endpoints = ",
        client_graph->fetch_types.size());
  }

  auto current_stateful_placements = execution_state->GetStatefulPlacements();
  // Update our current state based on the execution_state's
  // placements.  If there are any mismatches for a node,
  // we should fail, as this should never happen.
  for (auto placement_pair : current_stateful_placements) {
    const string& node_name = placement_pair.first;
    const string& placement = placement_pair.second;
    auto iter = stateful_placements_.find(node_name);
    if (iter == stateful_placements_.end()) {
      stateful_placements_.insert(std::make_pair(node_name, placement));
    } else if (iter->second != placement) {
      return errors::Internal(
          "Stateful placement mismatch. "
          "Current assignment of ",
          node_name, " to ", iter->second, " does not match ", placement);
    }
  }

  stateful_placements_ = execution_state->GetStatefulPlacements();

  // Remember the graph in run state if this is a partial run.
  if (run_state_args->is_partial_run) {
    run_state_args->graph.reset(new Graph(flib_def_.get()));
    CopyGraph(*execution_state->full_graph(), run_state_args->graph.get());
  }

  // Partition the graph across devices.
  PartitionOptions popts;
  popts.node_to_loc = [](const Node* node) {
    assert(node != nullptr);
    return node->assigned_device_name();
  };
  popts.new_name = [this](const string& prefix) {
    return strings::StrCat(prefix, "/_", edge_name_counter_.fetch_add(1));
  };
  popts.get_incarnation = [](const string& name) {
    // The direct session does not have changing incarnation numbers.
    // Just return '1'.
    return 1;
  };
  popts.flib_def = &client_graph->graph.flib_def();
  popts.control_flow_added = false;

  std::unordered_map<string, GraphDef> partitions;
  TF_RETURN_IF_ERROR(Partition(popts, &client_graph->graph, &partitions));

  std::vector<string> device_names;
  for (auto device : devices_) {
    // Extract the LocalName from the device.
    device_names.push_back(DeviceNameUtils::LocalName(device->name()));
  }

  // Check for valid partitions.
  for (const auto& partition : partitions) {
    const string local_partition_name =
        DeviceNameUtils::LocalName(partition.first);
    if (std::count(device_names.begin(), device_names.end(),
                   local_partition_name) == 0) {
      return errors::InvalidArgument(
          "Creating a partition for ", local_partition_name,
          " which doesn't exist in the list of available devices. Available "
          "devices: ",
          str_util::Join(device_names, ","));
    }
  }

  for (const auto& partition : partitions) {
    std::unique_ptr<Graph> device_graph(
        new Graph(client_graph->flib_def.get()));
    GraphConstructorOptions device_opts;
    // There are internal operations (e.g., send/recv) that we now allow.
    device_opts.allow_internal_ops = true;
    device_opts.expect_device_spec = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(device_opts, partition.second,
                                              device_graph.get()));
    outputs->emplace(partition.first, std::move(device_graph));
  }

  GraphOptimizationPassOptions optimization_options;
  optimization_options.session_options = &options_;
  optimization_options.flib_def = client_graph->flib_def.get();
  optimization_options.partition_graphs = outputs;
  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_PARTITIONING, optimization_options));

  Status s;
  for (auto& partition : *outputs) {
    const string& partition_name = partition.first;
    std::unique_ptr<Graph>* graph = &partition.second;

    VLOG(2) << "Created " << DebugString(graph->get()) << " for "
            << partition_name;

    // Give the device an opportunity to rewrite its subgraph.
    Device* d;
    s = device_mgr_->LookupDevice(partition_name, &d);
    if (!s.ok()) break;
    s = d->MaybeRewriteGraph(graph);
    if (!s.ok()) {
      break;
    }
  }
  *flib_def = std::move(client_graph->flib_def);
  std::swap(*input_types, client_graph->feed_types);
  std::swap(*output_types, client_graph->fetch_types);
  return s;
}

::tensorflow::Status TBBSession::ListDevices(
    std::vector<DeviceAttributes>* response) {
  response->clear();
  response->reserve(devices_.size());
  for (Device* d : devices_) {
    const DeviceAttributes& attrs = d->attributes();
    response->emplace_back(attrs);
  }
  return ::tensorflow::Status::OK();
}

::tensorflow::Status TBBSession::Reset(
    const std::vector<string>& containers) {
  device_mgr_->ClearContainers(containers);
  return ::tensorflow::Status::OK();
}

::tensorflow::Status TBBSession::Close() {
  cancellation_manager_->StartCancel();
  {
    mutex_lock l(closed_lock_);
    if (closed_) return ::tensorflow::Status::OK();
    closed_ = true;
  }
  if (factory_ != nullptr) factory_->Deregister(this);
  return ::tensorflow::Status::OK();
}

TBBSession::RunState::RunState(
    const std::vector<string>& pending_input_names,
    const std::vector<string>& pending_output_names, int64 step_id,
    const std::vector<Device*>* devices)
    : step_container(step_id, [devices](const string& name) {
        for (auto d : *devices) {
          if (!d->resource_manager()->Cleanup(name).ok()) {
            // Do nothing...
          }
        }
      }) {
  // Initially all the feeds and fetches are pending.
  for (auto& name : pending_input_names) {
    pending_inputs[name] = false;
  }
  for (auto& name : pending_output_names) {
    pending_outputs[name] = false;
  }
}

TBBSession::RunState::RunState(int64 step_id,
                                  const std::vector<Device*>* devices)
    : RunState({}, {}, step_id, devices) {}

TBBSession::RunState::~RunState() {
  if (rendez != nullptr) {
    if (!executors_done.HasBeenNotified()) {
      rendez->StartAbort(errors::Cancelled("PRun cancellation"));
      executors_done.WaitForNotification();
    }
    rendez->Unref();
  }
}

bool TBBSession::RunState::PendingDone() const {
  for (const auto& it : pending_inputs) {
    if (!it.second) return false;
  }
  for (const auto& it : pending_outputs) {
    if (!it.second) return false;
  }
  return true;
}

void TBBSession::WaitForNotification(tbb::task_arena& arena, tbb::task_group& taskGroup,
    RunState* run_state, CancellationManager* cm, int64 timeout_in_ms) {
  // Doing the wait in the arena adds this thread to the arena
  // and therefore tasks associated to the group can run on this thread
  arena.execute([&taskGroup]() { taskGroup.wait();} );

  const Status status =
      WaitForNotification(&run_state->executors_done, timeout_in_ms);
  if (!status.ok()) {
    {
      mutex_lock l(run_state->mu_);
      run_state->status.Update(status);
    }
    cm->StartCancel();
    // We must wait for the executors to complete, because they have borrowed
    // references to `cm` and other per-step state. After this notification, it
    // is safe to clean up the step.
    run_state->executors_done.WaitForNotification();
  }
}

::tensorflow::Status TBBSession::WaitForNotification(
    Notification* notification, int64 timeout_in_ms) {
  if (timeout_in_ms > 0) {
    const int64 timeout_in_us = timeout_in_ms * 1000;
    const bool notified =
        WaitForNotificationWithTimeout(notification, timeout_in_us);
    if (!notified) {
      return Status(error::DEADLINE_EXCEEDED,
                    "Timed out waiting for notification");
    }
  } else {
    notification->WaitForNotification();
  }
  return Status::OK();
}

}  // namespace tensorflow
