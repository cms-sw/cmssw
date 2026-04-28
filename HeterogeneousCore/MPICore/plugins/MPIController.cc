// C++ headers
#include <array>
#include <optional>
#include <string>
#include <vector>

// fmt library
#include <fmt/ranges.h>

// MPI headers
#include <mpi.h>

// CMSSW headers
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Reflection/interface/ObjectWithDict.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Guid.h"
#include "HeterogeneousCore/MPICore/interface/MPIChannel.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"
#include "HeterogeneousCore/MPICore/interface/messages.h"
#include "HeterogeneousCore/MPIServices/interface/MPIService.h"

/* MPIController class
 *
 * This module runs inside a CMSSW job (the "controller") and connects to one or more "MPISource" in one or more CMSSW jobs (the "followers").
 * Each follower is informed of all transitions seen by the controller, and can replicate them in its own process.
 */

// TODO: change to an edm::global module
class MPIController : public edm::one::EDProducer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  explicit MPIController(edm::ParameterSet const& config);
  ~MPIController() override;

  void beginJob() override;
  void endJob() override;

  void beginRun(edm::Run const& run, edm::EventSetup const& setup) override;
  void endRun(edm::Run const& run, edm::EventSetup const& setup) override;

  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) override;

  void produce(edm::Event& event, edm::EventSetup const& setup) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
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

  MPI_Comm comm_ = MPI_COMM_NULL;
  std::vector<MPIChannel> followers_;
  std::vector<std::vector<std::unique_ptr<MPIChannel>>> channels_;
  edm::EDPutTokenT<MPIToken> token_;
  Mode mode_;
};

MPIController::MPIController(edm::ParameterSet const& config)
    : token_(produces<MPIToken>()),
      mode_(parseMode(config.getUntrackedParameter<std::string>("mode")))  //
{
  // Make sure that MPI is initialised.
  MPIService::required();

  // Make sure the EDM MPI types are available.
  EDM_MPI_build_types();

  if (mode_ == kCommWorld) {
    // All processes are in MPI_COMM_WORLD.
    edm::LogInfo("MPI") << "MPIController in " << ModeDescription[mode_] << " mode.";

    // Check how many processes are there in MPI_COMM_WORLD.
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check the rank of this process.
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    edm::LogInfo("MPI") << "MPIController sees world size " << size;

    // Determine the ranks of the follower processes.
    auto follower_name = config.getParameter<std::string>("followerProcessName");
    if (follower_name.empty()) {
      throw edm::Exception(edm::errors::Configuration)
          << "ERROR: Follower process name cannot be empty. Aborting MPIController...";
    }

    edm::Service<edm::service::TriggerNamesService> tns;
    std::string const& this_process_name = tns->getProcessName();
    if (follower_name == this_process_name) {
      throw edm::Exception(edm::errors::Configuration)
          << "ERROR: controller and follower processes cannot have the same name. Aborting MPIController...";
    }

    edm::Service<MPIService> mpiservice;
    auto followers = mpiservice->getRanksByProcessName(follower_name);
    if (followers.empty()) {
      throw edm::Exception(edm::errors::Configuration)
          << "ERROR: No follower process with name " << follower_name << " found. Aborting...";
    }

    if (followers.size() == static_cast<size_t>(size)) {
      throw edm::Exception(edm::errors::Configuration)
          << "The number of found followers equals to the world size. "
          << "Possible reason could be process names' hash collision. "
          << "Please check process names in follower and controller. Aborting...";
    }

    for (int follower : followers) {
      // Create a new communicator that spans only this process and one of its followers.
      std::array<int, 2> ranks = {rank, follower};
      MPI_Group world_group, comm_group;
      MPI_Comm_group(MPI_COMM_WORLD, &world_group);
      MPI_Group_incl(world_group, ranks.size(), ranks.data(), &comm_group);
      // Note: module construction is serialised, so we can leave the tag set to 0.
      MPI_Comm comm = MPI_COMM_NULL;
      MPI_Comm_create_group(MPI_COMM_WORLD, comm_group, 0, &comm);
      MPI_Group_free(&world_group);
      MPI_Group_free(&comm_group);
      edm::LogInfo("MPI") << "The controller and follower processes have ranks " << rank << ", " << follower
                          << " in MPI_COMM_WORLD, mapped to ranks 0, 1 in their private communicator.";
      // The follower process always has rank 1 in the new communicator.
      follower = 1;
      followers_.emplace_back(comm, follower);
      channels_.emplace_back();
    }
  } else if (mode_ == kIntercommunicator) {
    // Use an intercommunicator to let two groups of processes communicate with each other.
    // The current implementation supports only two processes: one controller and one source.
    edm::LogInfo("MPI") << "MPIController in " << ModeDescription[mode_] << " mode.";

    // Check how many processes are there in MPI_COMM_WORLD
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1) {
      throw edm::Exception(edm::errors::Configuration)
          << "The current implementation supports only two processes: one controller and one source.";
    }

    // Look for the port under the name indicated by the parameter "server".
    std::string name = config.getUntrackedParameter<std::string>("name", "server");
    char port[MPI_MAX_PORT_NAME];
    MPI_Lookup_name(name.c_str(), MPI_INFO_NULL, port);
    edm::LogInfo("MPI") << "Trying to connect to the MPI server on port " << port;

    // Create an intercommunicator and connect to the server.
    MPI_Comm_connect(port, MPI_INFO_NULL, 0, MPI_COMM_SELF, &comm_);
    MPI_Comm_remote_size(comm_, &size);
    if (size != 1) {
      throw edm::Exception(edm::errors::Configuration)
          << "The current implementation supports only two processes: one controller and one source.";
    }
    edm::LogInfo("MPI") << "Client connected to " << size << (size == 1 ? " server" : " servers");
    followers_.emplace_back(comm_, 0);
    channels_.emplace_back();
  } else {
    // Invalid mode.
    throw edm::Exception(edm::errors::Configuration)
        << "Invalid mode \"" << config.getUntrackedParameter<std::string>("mode") << "\"";
  }
}

MPIController::~MPIController() {
  // Disconnect the per-stream communicators.
  for (auto& channels : channels_) {
    for (auto& channel : channels) {
      channel->reset();
    }
  }

  // Disconnect the per-follower communicators.
  for (auto& follower : followers_) {
    follower.reset();
  }

  // Close the intercommunicator.
  if (mode_ == kIntercommunicator) {
    MPI_Comm_disconnect(&comm_);
  }
}

void MPIController::beginJob() {
  // signal the connection
  for (auto& follower : followers_) {
    follower.sendConnect();
  }

  /* is there a way to access all known process histories ?
  edm::ProcessHistoryRegistry const& registry = * edm::ProcessHistoryRegistry::instance();
  edm::LogInfo("MPI") << "ProcessHistoryRegistry:";
  for (auto const& keyval: registry) {
    edm::LogInfo("MPI") << keyval.first << ": " << keyval.second;
  }
  */
}

void MPIController::endJob() {
  // signal the disconnection
  for (auto& follower : followers_) {
    follower.sendDisconnect();
  }
}

void MPIController::beginRun(edm::Run const& run, edm::EventSetup const& setup) {
  // signal a new run, and transmit the RunAuxiliary
  /* FIXME
   * Ideally the ProcessHistoryID stored in the run.runAuxiliary() should be the correct one, and
   * we could simply do

  for (auto& follower: followers_) {
    follower.sendBeginRun(run.runAuxiliary());
    follower.sendProduct(0, run.processHistory());
  }

   * Instead, it looks like the ProcessHistoryID stored in the run.runAuxiliary() is that of the
   * _parent_ process.
   * So, we make a copy of the RunAuxiliary, set the ProcessHistoryID to the correct value, and
   * transmit the modified RunAuxiliary.
   */
  auto aux = run.runAuxiliary();
  aux.setProcessHistoryID(run.processHistory().id());
  for (auto& follower : followers_) {
    follower.sendBeginRun(aux);

    // transmit the ProcessHistory
    follower.sendProduct(0, run.processHistory());
  }
}

void MPIController::endRun(edm::Run const& run, edm::EventSetup const& setup) {
  // signal the end of run
  /* FIXME
   * Ideally the ProcessHistoryID stored in the run.runAuxiliary() should be the correct one, and
   * we could simply do

  for (auto& follower: followers_) {
    follower.sendEndRun(run.runAuxiliary());
  }

   * Instead, it looks like the ProcessHistoryID stored in the run.runAuxiliary() is that of the
   * _parent_ process.
   * So, we make a copy of the RunAuxiliary, set the ProcessHistoryID to the correct value, and
   * transmit the modified RunAuxiliary.
   */
  auto aux = run.runAuxiliary();
  aux.setProcessHistoryID(run.processHistory().id());
  for (auto& follower : followers_) {
    follower.sendEndRun(aux);
  }
}

void MPIController::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
  // signal a new luminosity block, and transmit the LuminosityBlockAuxiliary
  /* FIXME
   * Ideally the ProcessHistoryID stored in the lumi.luminosityBlockAuxiliary() should be the
   * correct one, and we could simply do

  for (auto& follower: followers_) {
    follower.sendBeginLuminosityBlock(lumi.luminosityBlockAuxiliary());
  }

   * Instead, it looks like the ProcessHistoryID stored in the lumi.luminosityBlockAuxiliary() is
   * that of the _parent_ process.
   * So, we make a copy of the LuminosityBlockAuxiliary, set the ProcessHistoryID to the correct
   * value, and transmit the modified LuminosityBlockAuxiliary.
   */
  auto aux = lumi.luminosityBlockAuxiliary();
  aux.setProcessHistoryID(lumi.processHistory().id());
  for (auto& follower : followers_) {
    follower.sendBeginLuminosityBlock(aux);
  }
}

void MPIController::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
  // The MPIController is a "one" module that supports only a single luminosity block at a time.
  // Before proceeding to the next luminosity block, make sure that all events from the current
  // one have been processed.
  for (auto& channels : channels_) {
    for (auto& channel : channels) {
      channel->wait();
    }
  }

  // signal the end of luminosity block
  /* FIXME
   * Ideally the ProcessHistoryID stored in the lumi.luminosityBlockAuxiliary() should be the
   * correct one, and we could simply do

  for (auto& follower: followers_) {
    follower.sendEndLuminosityBlock(lumi.luminosityBlockAuxiliary());
  }

   * Instead, it looks like the ProcessHistoryID stored in the lumi.luminosityBlockAuxiliary() is
   * that of the _parent_ process.
   * So, we make a copy of the LuminosityBlockAuxiliary, set the ProcessHistoryID to the correct
   * value, and transmit the modified LuminosityBlockAuxiliary.
   */
  auto aux = lumi.luminosityBlockAuxiliary();
  aux.setProcessHistoryID(lumi.processHistory().id());
  for (auto& follower : followers_) {
    follower.sendEndLuminosityBlock(aux);
  }
}

void MPIController::produce(edm::Event& event, edm::EventSetup const& setup) {
  LogDebug("MPI")  //
      << "processing run " << event.run() << ", lumi " << event.luminosityBlock() << ", event "
      << event.id().event()                                                                                 //
      << "\nprocess history:    " << event.processHistory()                                                 //
      << "\nprocess history id: " << event.processHistory().id()                                            //
      << "\nprocess history id: " << event.eventAuxiliary().processHistoryID() << " (from eventAuxiliary)"  //
      << "\nisRealData " << event.eventAuxiliary().isRealData()                                             //
      << "\nexperimentType " << event.eventAuxiliary().experimentType()                                     //
      << "\nbunchCrossing " << event.eventAuxiliary().bunchCrossing()                                       //
      << "\norbitNumber " << event.eventAuxiliary().orbitNumber()                                           //
      << "\nstoreNumber " << event.eventAuxiliary().storeNumber()                                           //
      << "\nprocessHistoryID " << event.eventAuxiliary().processHistoryID()                                 //
      << "\nprocessGUID " << edm::Guid(event.eventAuxiliary().processGUID(), true).toString();

  // Choose the follower associated to the framework stream, in a round-robin fashion.
  unsigned int sid = event.streamID().value();
  auto& follower = followers_[sid % followers_.size()];
  auto& channels = channels_[sid % followers_.size()];

  // Look for a channel that is ready to send a new event
  unsigned int slot = channels.size();
  bool found = false;
  for (unsigned int i = 0; i < channels.size(); ++i) {
    if (channels[i]->ready()) {
      slot = i;
      found = true;
      break;
    }
  }

  // Signal a new event, and transmit the EventAuxiliary and channel slot to use.
  follower.sendEvent(event.eventAuxiliary(), slot);

  // If no channels were ready, allocate a new one.
  if (not found) {
    // Note: this is done after sending the slot to the MPISource,
    // so that it may call controller_.duplicate(slot) at the same time.
    channels.emplace_back(follower.duplicate(slot));
  }

  // The destructor of the last copy of the token will call channels[slot]->sync().
  // The channel is ready to send a new event after the call is made by both local and remote processes.
  event.emplace(token_, *channels[slot]);
}

void MPIController::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  descriptions.setComment(
      "This module connects to an \"MPISource\" in a separate CMSSW job, and transmits all Run, LuminosityBlock and "
      "Event transitions from the current process to the remote one.");

  edm::ParameterSetDescription desc;
  desc.ifValue(
          edm::ParameterDescription<std::string>("mode", "CommWorld", false),
          ModeDescription[kCommWorld] >>
                  edm::ParameterDescription<std::string>(
                      "followerProcessName",
                      "",
                      true,
                      edm::Comment("All processes with this process name should act as followers, "
                                   "and should be configured with an MPISource that follows this controller.")) or
              ModeDescription[kIntercommunicator] >> edm::ParameterDescription<std::string>("name", "server", false))
      ->setComment(
          "Valid modes are CommWorld (use MPI_COMM_WORLD) and Intercommunicator (use an MPI name server to setup an "
          "intercommunicator).");

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(MPIController);
