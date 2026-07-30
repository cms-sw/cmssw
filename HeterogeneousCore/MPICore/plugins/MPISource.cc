// C++ headers
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <condition_variable>

#include <atomic>
#include <map>
#include <mutex>
#include <thread>

// tbb headers
#include <tbb/concurrent_queue.h>

// MPI headers
#include <mpi.h>

// CMSSW headers
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/EventToProcessBlockIndexes.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/ProductProvenanceRetriever.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Sources/interface/ProducerSourceBase.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/MPICore/interface/MPIChannel.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"
#include "HeterogeneousCore/MPICore/interface/conversion.h"
#include "HeterogeneousCore/MPICore/interface/messages.h"
#include "HeterogeneousCore/MPIServices/interface/MPIService.h"

class MPISource : public edm::ProducerSourceBase {
public:
  explicit MPISource(edm::ParameterSet const& config, edm::InputSourceDescription const& desc);
  ~MPISource() override;
  using InputSource::processHistoryRegistryForUpdate;
  using InputSource::productRegistryUpdate;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool setRunAndEventInfo(edm::EventID& id, edm::TimeValue_t& time, edm::EventAuxiliary::ExperimentType&) override;
  void produce(edm::Event&) override;

  struct EventItem {
    edm::EventAuxiliary aux;
    // temporary value used to pass information from setRunAndEventInfo() to produce()
    MPIChannel* channel = nullptr;
  };

  struct LumiQueue {
    tbb::concurrent_queue<EventItem> events;
    // set to true when LumiComplete received
    std::atomic<bool> complete{false};
  };

  using RunLumi = std::pair<unsigned int, unsigned int>;

  unsigned int nextThreadID_ = 0;
  struct StreamWorker {
    unsigned int thread_id;
    // control channel used to receive stream transitions and event headers
    std::unique_ptr<MPIChannel> ctrlStream;
    std::vector<std::unique_ptr<MPIChannel>> dataChannels;
    // dedicated receiver thread for this controller stream
    std::thread thread;
    std::atomic<bool> stop{false};
  };

  void startThread();
  void receiverThreadsLoop(StreamWorker& worker);
  void controlThreadLoop();
  std::shared_ptr<LumiQueue> getOrCreateQueue(unsigned int run, unsigned int lumi);
  void markLumiComplete(unsigned int run, unsigned int lumi);

  enum Mode { kInvalid = 0, kCommWorld, kIntercommunicator };
  static constexpr const char* ModeDescription[] = {"Invalid", "CommWorld", "Intercommunicator"};
  Mode parseMode(std::string const& label) {
    if (label == ModeDescription[kCommWorld])
      return kCommWorld;
    else if (label == ModeDescription[kIntercommunicator])
      return kIntercommunicator;
    else
      return kInvalid;
  }

  char port_[MPI_MAX_PORT_NAME];
  MPI_Comm comm_ = MPI_COMM_NULL;
  MPIChannel controller_;
  std::vector<std::vector<std::unique_ptr<MPIChannel>>> channels_;
  edm::EDPutTokenT<MPIToken> token_;
  Mode mode_;

  std::mutex historyMutex_;
  edm::ProcessHistory history_;

  MPIChannel* channel_ = nullptr;

  unsigned int nControllerStreams_{0};
  // one per controller stream
  std::vector<std::unique_ptr<StreamWorker>> workers_;
  std::vector<std::unique_ptr<MPIChannel>> controllerChannels_;

  // Per-(run,lumi) queues
  std::map<RunLumi, std::shared_ptr<LumiQueue>> lumiQueues_;
  std::mutex mapMutex_;
  // wakes framework thread on new event or lumi completion
  std::condition_variable mainCV_;
  bool noMoreLumis_ = false;

  // Control thread handle
  std::thread controlThread_;

  // Temporary storage for produce()
  EventItem lastEventItem_;
};

MPISource::MPISource(edm::ParameterSet const& config, edm::InputSourceDescription const& desc)
    :  // note that almost all configuration parameters passed to IDGeneratorSourceBase via ProducerSourceBase will
       // effectively be ignored, because this ConfigurableSource will explicitly set the run, lumi, and event
       // numbers, the timestamp, and the event type
      edm::ProducerSourceBase(config, desc, false),
      token_(produces<MPIToken>()),
      mode_(parseMode(config.getUntrackedParameter<std::string>("mode")))  //
{
  // Make sure that MPI is initialised.
  MPIService::required();

  // Make sure the EDM MPI types are available.
  EDM_MPI_build_types();

  if (mode_ == kCommWorld) {
    // All processes are in MPI_COMM_WORLD.
    edm::LogInfo("MPI") << "MPISource in " << ModeDescription[mode_] << " mode.";

    // Check how many processes are there in MPI_COMM_WORLD
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check the rank of this process.
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    edm::LogInfo("MPI") << "MPIController Comm World size: " << size;

    // All processes exchange the hashes of their names.
    // One follower process has to make one communication channel with the controller process
    // If controller process is not unique, error is thrown
    auto controller_name = config.getParameter<std::string>("controllerProcessName");
    if (controller_name.empty()) {
      throw edm::Exception(edm::errors::Configuration)
          << "ERROR: Controller process name cannot be empty. Aborting MPISource...";
    }

    edm::Service<edm::service::TriggerNamesService> tns;
    std::string const& this_process_name = tns->getProcessName();
    if (controller_name == this_process_name) {
      throw edm::Exception(edm::errors::Configuration)
          << "ERROR: controller and follower processes cannot have the same name. Aborting MPISource...";
    }

    edm::Service<MPIService> mpiservice;
    std::vector<int> controller_indices = mpiservice->getRanksByProcessName(controller_name);
    int remote = -1;
    if (controller_indices.empty()) {
      throw edm::Exception(edm::errors::Configuration)
          << "ERROR: No controller process with name " << controller_name << " found. Aborting...";
    } else if (controller_indices.size() == 1) {
      remote = controller_indices[0];
    } else {
      throw edm::Exception(edm::errors::Configuration)
          << "ERROR: Multiple controller processes with name " << controller_name
          << " were found. Currently, only one controller process is supported. Aborting...";
    }

    // Create a new communicator that spans only this process and the one with the given remote rank.
    int ranks[2] = {remote, rank};
    MPI_Group world_group, comm_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, 2, ranks, &comm_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, comm_group, 0, &comm_);
    MPI_Group_free(&world_group);
    MPI_Group_free(&comm_group);
    edm::LogInfo("MPI") << "The MPIController process and MPISource have ranks " << remote << ", " << rank
                        << " in MPI_COMM_WORLD, mapped to ranks 0, 1 in their private communicator.";
    // The remote process always has rank 0 in the new communicator.
    remote = 0;
    controller_ = MPIChannel(comm_, remote);
  } else if (mode_ == kIntercommunicator) {
    // Use an intercommunicator to let two groups of processes communicate with each other.
    // The current implementation supports only two processes: one controller and one source.
    edm::LogInfo("MPI") << "MPISource in " << ModeDescription[mode_] << " mode.";

    // Check how many processes are there in MPI_COMM_WORLD
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1) {
      throw edm::Exception(edm::errors::Configuration)
          << "The current implementation supports only two processes: one controller and one source.";
    }

    // Open a server-side port.
    MPI_Open_port(MPI_INFO_NULL, port_);

    // Publish the port under the name indicated by the parameter "server".
    std::string name = config.getUntrackedParameter<std::string>("name", "server");
    MPI_Info port_info;
    MPI_Info_create(&port_info);
    MPI_Info_set(port_info, "ompi_global_scope", "true");
    MPI_Info_set(port_info, "ompi_unique", "true");
    MPI_Publish_name(name.c_str(), port_info, port_);

    // Create an intercommunicator and accept a client connection.
    edm::LogInfo("MPI") << "Waiting for a connection to the MPI server at port " << port_;

    MPI_Comm_accept(port_, MPI_INFO_NULL, 0, MPI_COMM_SELF, &comm_);
    edm::LogInfo("MPI") << "Connection accepted.";
    controller_ = MPIChannel(comm_, 0);
  } else {
    // Invalid mode.
    throw edm::Exception(edm::errors::Configuration)
        << "Invalid mode \"" << config.getUntrackedParameter<std::string>("mode") << "\"";
  }

  // Wait for a client to connect.
  MPI_Status status;
  EDM_MPI_Empty_t buffer;
  MPI_Recv(&buffer, 1, EDM_MPI_Empty, MPI_ANY_SOURCE, EDM_MPI_Connect, comm_, &status);
  edm::LogInfo("MPI") << "connected from " << status.MPI_SOURCE;

  // Receive the number of streams that each follower process will handle
  controller_.receiveStreamCount(nControllerStreams_);
  edm::LogInfo("MPI") << "Source will handle " << nControllerStreams_ << " streams";

  // Each controller stream gets an independent control channel plus one data
  // channel per controller slot (WIP: fixed 3). The duplicated communicators allow different
  // receiver threads to operate without sharing MPIChannel instances.
  controllerChannels_.resize(nControllerStreams_);
  channels_.resize(nControllerStreams_);
  for (unsigned int slot = 0; slot < nControllerStreams_; ++slot) {
    controllerChannels_[slot] = controller_.duplicate(slot);
    channels_[slot].reserve(3);
    for (int i = 0; i < 3; ++i) {
      channels_[slot].emplace_back(controllerChannels_[slot]->duplicate(i));
    }
  }

  // Start the control thread that handles BeginStream/Disconnect/LumiComplete
  // messages that are global to the source rather than associated with a
  // specific controller stream
  controlThread_ = std::thread(&MPISource::controlThreadLoop, this);
}

MPISource::~MPISource() {
  for (auto& worker : workers_) {
    if (worker && worker->thread.joinable()) {
      worker->stop = true;
      worker->thread.join();
    }
    if (worker) {
      if (worker->ctrlStream)
        worker->ctrlStream->reset();
      for (auto& c : worker->dataChannels) {
        if (c)
          c->reset();
      }
    }
  }

  // Stop control thread
  if (controlThread_.joinable()) {
    controlThread_.join();
    controller_.reset();
  }

  if (mode_ == kIntercommunicator) {
    // Close the intercommunicator.
    MPI_Comm_disconnect(&comm_);

    // Unpublish and close the port.
    MPI_Info port_info;
    MPI_Info_create(&port_info);
    MPI_Info_set(port_info, "ompi_global_scope", "true");
    MPI_Info_set(port_info, "ompi_unique", "true");
    MPI_Unpublish_name("server", port_info, port_);
    MPI_Close_port(port_);
  }
}

void MPISource::controlThreadLoop() {
  while (true) {
    MPI_Status status;
    MPI_Message message;
    controller_.probeAny(message, status);
    switch (status.MPI_TAG) {
      // BeginStream message
      case EDM_MPI_BeginStream: {
        // receive the message header
        EDM_MPI_Empty_t buf;
        MPI_Mrecv(&buf, 1, EDM_MPI_Empty, &message, &status);

        // launches a new worker thread
        startThread();

        // receive the next message
        break;
      }

      // LumiBlockComplete message
      case EDM_MPI_LuminosityBlockComplete: {
        // receive the message header
        EDM_MPI_LuminosityBlockAuxiliary_t buf;
        MPI_Mrecv(&buf, 1, EDM_MPI_LuminosityBlockAuxiliary, &message, &status);

        // mark the (run-lumi) LumiQueue as complete
        edm::LuminosityBlockAuxiliary aux;
        edmFromBuffer(buf, aux);
        markLumiComplete(aux.run(), aux.luminosityBlock());

        // receive the next message
        break;
      }

      // Connect message
      case EDM_MPI_Connect: {
        // receive the message header
        EDM_MPI_Empty_t buffer;
        MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);

        // the Connect message is unexpected here (see above)
        throw cms::Exception("InvalidValue")
            << "The MPISource has received an EDM_MPI_Connect message after the initial connection";
        return;
      }

      // Disconnect message
      case EDM_MPI_Disconnect: {
        // receive the message header
        EDM_MPI_Empty_t buf;
        MPI_Mrecv(&buf, 1, EDM_MPI_Empty, &message, &status);
        {
          std::lock_guard<std::mutex> lock(mapMutex_);
          noMoreLumis_ = true;
        }
        mainCV_.notify_one();

        // control thread done
        return;
      }

      // unexpected message
      default: {
        throw cms::Exception("InvalidValue")
            << "The MPISource has received an unknown message with tag " << status.MPI_TAG;
        return;
      }
    }
  }
}

void MPISource::startThread() {
  workers_.push_back(std::make_unique<StreamWorker>());
  StreamWorker& worker = *workers_.back();

  worker.thread_id = nextThreadID_;
  worker.ctrlStream = std::move(controllerChannels_[nextThreadID_]);
  worker.dataChannels = std::move(channels_[nextThreadID_]);
  worker.stop = false;

  // Launch the thread that will run the worker loop
  worker.thread = std::thread(&MPISource::receiverThreadsLoop, this, std::ref(worker));

  // After this, channels_ is empty (ownership transferred to workers)
  ++nextThreadID_;
}

std::shared_ptr<MPISource::LumiQueue> MPISource::getOrCreateQueue(unsigned int run, unsigned int lumi) {
  // Caller must hold mapMutex_
  RunLumi key{run, lumi};
  auto it = lumiQueues_.find(key);
  if (it != lumiQueues_.end())
    return it->second;

  auto q = std::make_shared<LumiQueue>();
  lumiQueues_[key] = q;
  return q;
}

void MPISource::markLumiComplete(unsigned int run, unsigned int lumi) {
  std::lock_guard<std::mutex> lock(mapMutex_);
  auto it = lumiQueues_.find({run, lumi});
  if (it != lumiQueues_.end()) {
    it->second->complete = true;
    mainCV_.notify_one();
  } else {
    // create it to avoid missing the completion
    auto q = std::make_shared<LumiQueue>();
    q->complete = true;
    lumiQueues_[{run, lumi}] = q;
  }
}

void MPISource::receiverThreadsLoop(StreamWorker& worker) {
  while (!worker.stop.load()) {
    MPI_Status status;
    MPI_Message message;
    worker.ctrlStream->probeAny(message, status);
    switch (status.MPI_TAG) {
      // BeginRun message
      case EDM_MPI_BeginRun: {
        EDM_MPI_RunAuxiliary_t buffer;
        MPI_Mrecv(&buffer, 1, EDM_MPI_RunAuxiliary, &message, &status);
        // TODO this is currently not used
        edm::RunAuxiliary runAuxiliary;
        edmFromBuffer(buffer, runAuxiliary);

        // receive the ProcessHistory
        std::lock_guard<std::mutex> lock(historyMutex_);  // worker threads share history_
        history_.clear();
        worker.ctrlStream->receiveProduct(0, history_);
        history_.initializeTransients();
        /*
        if (processHistoryRegistryForUpdate().registerProcessHistory(history_)) {
          edm::LogInfo("MPI") << "new ProcessHistory registered: " << history_;
        }
        */

        // receive the next message
        break;
      }

      // EndRun message
      case EDM_MPI_EndRun: {
        // receive the RunAuxiliary message
        EDM_MPI_RunAuxiliary_t buffer;
        MPI_Mrecv(&buffer, 1, EDM_MPI_RunAuxiliary, &message, &status);

        // receive the next message
        break;
      }

      // BeginLuminosityBlock message
      case EDM_MPI_BeginLuminosityBlock: {
        EDM_MPI_LuminosityBlockAuxiliary_t buffer;
        MPI_Mrecv(&buffer, 1, EDM_MPI_LuminosityBlockAuxiliary, &message, &status);
        // TODO this is currently not used
        edm::LuminosityBlockAuxiliary aux;
        edmFromBuffer(buffer, aux);
        {
          std::lock_guard<std::mutex> lk(mapMutex_);
          getOrCreateQueue(aux.run(), aux.luminosityBlock());  // ensures it exists
        }
        mainCV_.notify_one();  // main thread may be waiting for a new lumi

        // receive the next message
        break;
      }

      // EndLuminosityBlock message
      case EDM_MPI_EndLuminosityBlock: {
        // receive the LuminosityBlockAuxiliary
        EDM_MPI_LuminosityBlockAuxiliary_t buffer;
        MPI_Mrecv(&buffer, 1, EDM_MPI_LuminosityBlockAuxiliary, &message, &status);

        // Nothing else, LumiQueue closure handled by LumiComplete control message
        // receive the next message
        break;
      }

      // ProcessEvent message
      case EDM_MPI_ProcessEvent: {
        // receive the EventAuxiliary
        edm::EventAuxiliary aux;
        unsigned int ctrlSlot;
        worker.ctrlStream->receiveEvent(aux, ctrlSlot, message);
        EventItem item;
        item.aux = aux;

        // use the same communicator that the MPIController will use for this event
        item.channel = worker.dataChannels[ctrlSlot].get();

        // extract the rank of the other process (currently unused)
        int source = status.MPI_SOURCE;
        (void)source;

        // get the current LumiQueue, or create it if it wasn't before
        // push the EventItem in the tbb::concurrent_queue
        std::shared_ptr<LumiQueue> q;
        {
          std::lock_guard<std::mutex> lk(mapMutex_);
          q = getOrCreateQueue(aux.run(), aux.luminosityBlock());
        }
        q->events.push(std::move(item));

        mainCV_.notify_one();  // wake framework thread

        // receive the next message
        break;
      }

      // EndStream message
      case EDM_MPI_EndStream: {
        // receive the message header
        EDM_MPI_Empty_t buf;
        MPI_Mrecv(&buf, 1, EDM_MPI_Empty, &message, &status);

        // stop the current worker thread, linked to the ended stream
        worker.stop = true;

        // receive the next message
        break;
      }

      // unexpected message
      default: {
        throw cms::Exception("InvalidValue")
            << "The MPISource has received an unknown message with tag " << status.MPI_TAG;
        return;
      }
    }
  }
}

// Events are exposed to the framework (run-lumi) pair by (run-lumi) pair.
// The ordered map guarantees that the lowest (run-lumi) is processed before
// later luminosity blocks.
// A lumi cannot be skipped merely because its queue is temporarily empty:
// it must first be marked complete, otherwise more events may still arrive.
bool MPISource::setRunAndEventInfo(edm::EventID& event,
                                   edm::TimeValue_t& time,
                                   edm::EventAuxiliary::ExperimentType& type) {
  std::unique_lock<std::mutex> lock(mapMutex_);

  while (true) {
    // Wait until there is at least one lumi (with an event) or we know no more will come
    mainCV_.wait(lock, [this] {
      if (noMoreLumis_ && lumiQueues_.empty())
        return true;
      // Check if any lumi has an event or if any complete lumi exists that we can discard
      for (const auto& pair : lumiQueues_) {
        if (!pair.second->events.empty() || pair.second->complete)
          return true;
      }
      return false;
    });

    if (noMoreLumis_ && lumiQueues_.empty()) {
      return false;
    }

    // Find the first lumi (map gives ordered (run,lumi))
    auto it = lumiQueues_.begin();
    RunLumi currentKey = it->first;
    std::shared_ptr<LumiQueue> currentQ = it->second;

    // Try to pop an event
    EventItem item;
    if (currentQ->events.try_pop(item)) {
      lastEventItem_ = std::move(item);

      // store the channel to use it in produce()
      channel_ = lastEventItem_.channel;

      // fill the event details
      event = lastEventItem_.aux.id();
      time = lastEventItem_.aux.time().value();
      type = lastEventItem_.aux.experimentType();

      // If the queue is now empty and the lumi is complete, remove it
      if (currentQ->complete && currentQ->events.empty()) {
        lumiQueues_.erase(currentKey);
      }
      return true;
    }

    // No event popped. If the lumi is complete, discard it and retry
    if (currentQ->complete) {
      lumiQueues_.erase(currentKey);
      continue;  // loop again immediately. Will pick next lumi
    }

    // Lumi is not complete and has no events, wait for more data.
    // The condition variable will be signalled by worker threads when they push
    // a new event or by the control thread when it marks this lumi complete
    mainCV_.wait(lock, [&currentQ, this] { return !currentQ->events.empty() || currentQ->complete || noMoreLumis_; });
    // After waking up, re-evaluate the state
  }
}

void MPISource::produce(edm::Event& event) {
  // Wait for the barrier to be cleared by the MPI software in the local process.
  channel_->wait();

  // The destructor of the last copy of the token will call channel_->sync().
  // The channel is ready to receive a new event after the call is made by both local and remote processes.
  event.emplace(token_, *channel_);
  channel_ = nullptr;
}

void MPISource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  descriptions.setComment(
      "This module connects to an \"MPIController\" in a separate CMSSW job, receives all Run, LuminosityBlock and "
      "Event transitions from the remote process and reproduces them in the local one.");

  edm::ParameterSetDescription desc;
  edm::ProducerSourceBase::fillDescription(desc);
  desc.ifValue(
          edm::ParameterDescription<std::string>("mode", "CommWorld", false),
          ModeDescription[kCommWorld] >>
                  edm::ParameterDescription<std::string>(
                      "controllerProcessName",
                      "",
                      true,
                      edm::Comment("Process name of the controller process corresponding to this MPISource.\n"
                                   "Only one process with this name is expected.\n")) or
              ModeDescription[kIntercommunicator] >> edm::ParameterDescription<std::string>("name", "server", false))
      ->setComment(
          "Valid modes are CommWorld (use MPI_COMM_WORLD) and Intercommunicator (use an MPI name server to setup an "
          "intercommunicator).");

  descriptions.add("source", desc);
}

#include "FWCore/Framework/interface/InputSourceMacros.h"
DEFINE_FWK_INPUT_SOURCE(MPISource);
