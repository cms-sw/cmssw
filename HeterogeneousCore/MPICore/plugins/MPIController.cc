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
#include "HeterogeneousCore/MPICore/interface/api.h"
#include "HeterogeneousCore/MPICore/interface/messages.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"
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
  std::vector<MPIChannel> channels_;
  std::vector<std::optional<MPIChannel>> streams_;
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
    edm::LogAbsolute("MPI") << "MPIController in " << ModeDescription[mode_] << " mode.";

    // Check how many processes are there in MPI_COMM_WORLD.
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check the rank of this process.
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Determine the rank of the other process.
    auto followers = config.getUntrackedParameter<std::vector<int32_t>>("followers");
    if (followers.empty()) {
      // When there are only two proccesses, we can assume the ranks to be 0 and 1,
      // and we can infer the other process rank from our own.
      if (size == 2) {
        followers = {1 - rank};
      } else {
        throw edm::Exception(edm::errors::Configuration)
            << "An empty list of remote processes is valid only where there are exactly two processes.";
      }
    }
    if (followers.size() >= static_cast<size_t>(size)) {
      throw edm::Exception(edm::errors::Configuration)
          << "The number of remote processes is invalid. Please specify at most " << size - 1 << "remote processes.";
    }
    std::vector<int32_t> invalid;
    for (int follower : followers) {
      if (follower < 0 or follower >= size) {
        invalid.push_back(follower);
      }
    }
    if (invalid.size() == 1) {
      throw edm::Exception(edm::errors::Configuration)
          << fmt::format("The remote process {} is invalid. Valid ranks are 0 to {}.", invalid.front(), size - 1);
    } else if (invalid.size() > 1) {
      throw edm::Exception(edm::errors::Configuration) << fmt::format(
          "The remote processes {} are invalid. Valid ranks are 0 to {}.", fmt::join(invalid, ", "), size - 1);
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
      edm::LogAbsolute("MPI") << "The controller and follower processes have ranks " << rank << ", " << follower
                              << " in MPI_COMM_WORLD, mapped to ranks 0, 1 in their private communicator.";
      // The follower process always has rank 1 in the new communicator.
      follower = 1;
      channels_.emplace_back(comm, follower);
    }
  } else if (mode_ == kIntercommunicator) {
    // Use an intercommunicator to let two groups of processes communicate with each other.
    // The current implementation supports only two processes: one controller and one source.
    edm::LogAbsolute("MPI") << "MPISource in " << ModeDescription[mode_] << " mode.";

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
    edm::LogAbsolute("MPI") << "Trying to connect to the MPI server on port " << port;

    // Create an intercommunicator and connect to the server.
    MPI_Comm_connect(port, MPI_INFO_NULL, 0, MPI_COMM_SELF, &comm_);
    MPI_Comm_remote_size(comm_, &size);
    if (size != 1) {
      throw edm::Exception(edm::errors::Configuration)
          << "The current implementation supports only two processes: one controller and one source.";
    }
    edm::LogAbsolute("MPI") << "Client connected to " << size << (size == 1 ? " server" : " servers");
    channels_ = {MPIChannel(comm_, 0)};
  } else {
    // Invalid mode.
    throw edm::Exception(edm::errors::Configuration)
        << "Invalid mode \"" << config.getUntrackedParameter<std::string>("mode") << "\"";
  }
}

MPIController::~MPIController() {
  // Disconnect the per-stream communicators.
  for (auto& stream : streams_) {
    // TODO move this to end stream
    stream->reset();
  }

  // Close the intercommunicator.
  if (mode_ == kIntercommunicator) {
    MPI_Comm_disconnect(&comm_);
  }
}

void MPIController::beginJob() {
  // signal the connection
  for (auto& channel : channels_) {
    channel.sendConnect();
  }

  /* is there a way to access all known process histories ?
  edm::ProcessHistoryRegistry const& registry = * edm::ProcessHistoryRegistry::instance();
  edm::LogAbsolute("MPI") << "ProcessHistoryRegistry:";
  for (auto const& keyval: registry) {
    edm::LogAbsolute("MPI") << keyval.first << ": " << keyval.second;
  }
  */
}

void MPIController::endJob() {
  // signal the disconnection
  for (auto& channel : channels_) {
    channel.sendDisconnect();
  }
}

void MPIController::beginRun(edm::Run const& run, edm::EventSetup const& setup) {
  // signal a new run, and transmit the RunAuxiliary
  /* FIXME
   * Ideally the ProcessHistoryID stored in the run.runAuxiliary() should be the correct one, and
   * we could simply do

  for (auto& channel: channels_) {
    channel.sendBeginRun(run.runAuxiliary());
    channel.sendProduct(0, run.processHistory());
  }

   * Instead, it looks like the ProcessHistoryID stored in the run.runAuxiliary() is that of the
   * _parent_ process.
   * So, we make a copy of the RunAuxiliary, set the ProcessHistoryID to the correct value, and
   * transmit the modified RunAuxiliary.
   */
  auto aux = run.runAuxiliary();
  aux.setProcessHistoryID(run.processHistory().id());
  for (auto& channel : channels_) {
    channel.sendBeginRun(aux);

    // transmit the ProcessHistory
    channel.sendProduct(0, run.processHistory());
  }
}

void MPIController::endRun(edm::Run const& run, edm::EventSetup const& setup) {
  // signal the end of run
  /* FIXME
   * Ideally the ProcessHistoryID stored in the run.runAuxiliary() should be the correct one, and
   * we could simply do

  for (auto& channel: channels_) {
    channel.sendEndRun(run.runAuxiliary());
  }

   * Instead, it looks like the ProcessHistoryID stored in the run.runAuxiliary() is that of the
   * _parent_ process.
   * So, we make a copy of the RunAuxiliary, set the ProcessHistoryID to the correct value, and
   * transmit the modified RunAuxiliary.
   */
  auto aux = run.runAuxiliary();
  aux.setProcessHistoryID(run.processHistory().id());
  for (auto& channel : channels_) {
    channel.sendEndRun(aux);
  }
}

void MPIController::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
  // signal a new luminosity block, and transmit the LuminosityBlockAuxiliary
  /* FIXME
   * Ideally the ProcessHistoryID stored in the lumi.luminosityBlockAuxiliary() should be the
   * correct one, and we could simply do

  for (auto& channel: channels_) {
    channel.sendBeginLuminosityBlock(lumi.luminosityBlockAuxiliary());
  }

   * Instead, it looks like the ProcessHistoryID stored in the lumi.luminosityBlockAuxiliary() is
   * that of the _parent_ process.
   * So, we make a copy of the LuminosityBlockAuxiliary, set the ProcessHistoryID to the correct
   * value, and transmit the modified LuminosityBlockAuxiliary.
   */
  auto aux = lumi.luminosityBlockAuxiliary();
  aux.setProcessHistoryID(lumi.processHistory().id());
  for (auto& channel : channels_) {
    channel.sendBeginLuminosityBlock(aux);
  }
}

void MPIController::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
  // signal the end of luminosity block
  /* FIXME
   * Ideally the ProcessHistoryID stored in the lumi.luminosityBlockAuxiliary() should be the
   * correct one, and we could simply do

  for (auto& channel: channels_) {
    channel.sendEndLuminosityBlock(lumi.luminosityBlockAuxiliary());
  }

   * Instead, it looks like the ProcessHistoryID stored in the lumi.luminosityBlockAuxiliary() is
   * that of the _parent_ process.
   * So, we make a copy of the LuminosityBlockAuxiliary, set the ProcessHistoryID to the correct
   * value, and transmit the modified LuminosityBlockAuxiliary.
   */
  auto aux = lumi.luminosityBlockAuxiliary();
  aux.setProcessHistoryID(lumi.processHistory().id());
  for (auto& channel : channels_) {
    channel.sendEndLuminosityBlock(aux);
  }
}

void MPIController::produce(edm::Event& event, edm::EventSetup const& setup) {
  {
    edm::LogInfo log("MPI");
    log << "processing run " << event.run() << ", lumi " << event.luminosityBlock() << ", event " << event.id().event();
    log << "\nprocess history:    " << event.processHistory();
    log << "\nprocess history id: " << event.processHistory().id();
    log << "\nprocess history id: " << event.eventAuxiliary().processHistoryID() << " (from eventAuxiliary)";
    log << "\nisRealData " << event.eventAuxiliary().isRealData();
    log << "\nexperimentType " << event.eventAuxiliary().experimentType();
    log << "\nbunchCrossing " << event.eventAuxiliary().bunchCrossing();
    log << "\norbitNumber " << event.eventAuxiliary().orbitNumber();
    log << "\nstoreNumber " << event.eventAuxiliary().storeNumber();
    log << "\nprocessHistoryID " << event.eventAuxiliary().processHistoryID();
    log << "\nprocessGUID " << edm::Guid(event.eventAuxiliary().processGUID(), true).toString();
  }

  // use the channel associated to the framework stream
  unsigned int sid = event.streamID().value();

  // signal a new event, and transmit the EventAuxiliary
  auto& channel = channels_[sid % channels_.size()];
  channel.sendEvent(event.eventAuxiliary(), event.streamID().value());

  // keep a duplicate of the MPIChannel for each framework stream
  if (sid >= streams_.size()) {
    streams_.resize(sid + 1);
  }
  if (not streams_[sid]) {
    // TODO move this to begin stream
    streams_[sid] = channel.duplicate();
  }
  // create a new channel object reusing the same communicator that will synchronise at the end pf the event
  event.emplace(token_, streams_[sid]->syncChannel());
}

void MPIController::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  descriptions.setComment(
      "This module connects to an \"MPISource\" in a separate CMSSW job, and transmits all Run, LuminosityBlock and "
      "Event transitions from the current process to the remote one.");

  edm::ParameterSetDescription desc;
  desc.ifValue(
          edm::ParameterDescription<std::string>("mode", "CommWorld", false),
          ModeDescription[kCommWorld] >>
                  edm::ParameterDescription<std::vector<int32_t>>(
                      "followers",
                      {},
                      false,
                      edm::Comment("Ranks of the remote \"follower\" processes.\n"
                                   "When there are two or more follower processes, framework streams are associated to "
                                   "each follower in a round-robin fashion.\n"
                                   "When there is only one remote process, pass an empty list to autodetect its rank "
                                   "based on the rank of the current process.")) or
              ModeDescription[kIntercommunicator] >> edm::ParameterDescription<std::string>("name", "server", false))
      ->setComment(
          "Valid modes are CommWorld (use MPI_COMM_WORLD) and Intercommunicator (use an MPI name server to setup an "
          "intercommunicator).");

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(MPIController);
