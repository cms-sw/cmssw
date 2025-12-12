#include <iostream>
#include <memory>
#include <sstream>

#include <mpi.h>

#include <TBufferFile.h>
#include <TClass.h>

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
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"
#include "HeterogeneousCore/MPIServices/interface/MPIService.h"

#include "api.h"
#include "messages.h"

/* MPIController class
 *
 * This module runs inside a CMSSW job (the "controller") and connects to an "MPISource" in a separate CMSSW job (the "follower").
 * The follower is informed of all transitions seen by the controller, and can replicate them in its own process.
 *
 * Current limitations:
 *   - support a single "follower"
 *
 * Future work:
 *   - support multiple "followers"
 */

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
  MPIChannel channel_;
  edm::EDPutTokenT<MPIToken> token_;
  Mode mode_;
};

MPIController::MPIController(edm::ParameterSet const& config)
    : token_(produces<MPIToken>()),
      mode_(parseMode(config.getUntrackedParameter<std::string>("mode")))  //
{
  // make sure that MPI is initialised
  MPIService::required();

  // make sure the EDM MPI types are available
  EDM_MPI_build_types();

  if (mode_ == kCommWorld) {
    // All processes are in MPI_COMM_WORLD.
    // The current implementation supports only two processes: one controller and one source.
    edm::LogAbsolute("MPI") << "MPIController in " << ModeDescription[mode_] << " mode.";

    // Check how many processes are there in MPI_COMM_WORLD
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
      throw edm::Exception(edm::errors::Configuration)
          << "The current implementation supports only two processes: one controller and one source.";
    }

    // Check the rank of this process, and determine the rank of the other process.
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    edm::LogAbsolute("MPI") << "MPIController has rank " << rank << " in MPI_COMM_WORLD.";
    int other_rank = 1 - rank;
    comm_ = MPI_COMM_WORLD;
    channel_ = MPIChannel(comm_, other_rank);
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
    channel_ = MPIChannel(comm_, 0);
  } else {
    // Invalid mode.
    throw edm::Exception(edm::errors::Configuration)
        << "Invalid mode \"" << config.getUntrackedParameter<std::string>("mode") << "\"";
  }
}

MPIController::~MPIController() {
  // Close the intercommunicator.
  if (mode_ == kIntercommunicator) {
    MPI_Comm_disconnect(&comm_);
  }
}

void MPIController::beginJob() {
  // signal the connection
  channel_.sendConnect();

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
  channel_.sendDisconnect();
}

void MPIController::beginRun(edm::Run const& run, edm::EventSetup const& setup) {
  // signal a new run, and transmit the RunAuxiliary
  /* FIXME
   * Ideally the ProcessHistoryID stored in the run.runAuxiliary() should be the correct one, and
   * we could simply do

  channel_.sendBeginRun(run.runAuxiliary());

   * Instead, it looks like the ProcessHistoryID stored in the run.runAuxiliary() is that of the
   * _parent_ process.
   * So, we make a copy of the RunAuxiliary, set the ProcessHistoryID to the correct value, and
   * transmit the modified RunAuxiliary.
   */
  auto aux = run.runAuxiliary();
  aux.setProcessHistoryID(run.processHistory().id());
  channel_.sendBeginRun(aux);

  // transmit the ProcessHistory
  channel_.sendProduct(0, run.processHistory());
}

void MPIController::endRun(edm::Run const& run, edm::EventSetup const& setup) {
  // signal the end of run
  /* FIXME
   * Ideally the ProcessHistoryID stored in the run.runAuxiliary() should be the correct one, and
   * we could simply do

  channel_.sendEndRun(run.runAuxiliary());

   * Instead, it looks like the ProcessHistoryID stored in the run.runAuxiliary() is that of the
   * _parent_ process.
   * So, we make a copy of the RunAuxiliary, set the ProcessHistoryID to the correct value, and
   * transmit the modified RunAuxiliary.
   */
  auto aux = run.runAuxiliary();
  aux.setProcessHistoryID(run.processHistory().id());
  channel_.sendEndRun(aux);
}

void MPIController::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
  // signal a new luminosity block, and transmit the LuminosityBlockAuxiliary
  /* FIXME
   * Ideally the ProcessHistoryID stored in the lumi.luminosityBlockAuxiliary() should be the
   * correct one, and we could simply do

  channel_.sendBeginLuminosityBlock(lumi.luminosityBlockAuxiliary());

   * Instead, it looks like the ProcessHistoryID stored in the lumi.luminosityBlockAuxiliary() is
   * that of the _parent_ process.
   * So, we make a copy of the LuminosityBlockAuxiliary, set the ProcessHistoryID to the correct
   * value, and transmit the modified LuminosityBlockAuxiliary.
   */
  auto aux = lumi.luminosityBlockAuxiliary();
  aux.setProcessHistoryID(lumi.processHistory().id());
  channel_.sendBeginLuminosityBlock(aux);
}

void MPIController::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
  // signal the end of luminosity block
  /* FIXME
   * Ideally the ProcessHistoryID stored in the lumi.luminosityBlockAuxiliary() should be the
   * correct one, and we could simply do

  channel_.sendEndLuminosityBlock(lumi.luminosityBlockAuxiliary());

   * Instead, it looks like the ProcessHistoryID stored in the lumi.luminosityBlockAuxiliary() is
   * that of the _parent_ process.
   * So, we make a copy of the LuminosityBlockAuxiliary, set the ProcessHistoryID to the correct
   * value, and transmit the modified LuminosityBlockAuxiliary.
   */
  auto aux = lumi.luminosityBlockAuxiliary();
  aux.setProcessHistoryID(lumi.processHistory().id());
  channel_.sendEndLuminosityBlock(aux);
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

  // signal a new event, and transmit the EventAuxiliary
  channel_.sendEvent(event.eventAuxiliary());

  // duplicate the MPIChannel and put the copy into the Event
  std::shared_ptr<MPIChannel> link(new MPIChannel(channel_.duplicate()), [](MPIChannel* ptr) {
    ptr->reset();
    delete ptr;
  });
  event.emplace(token_, std::move(link));
}

void MPIController::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  descriptions.setComment(
      "This module connects to an \"MPISource\" in a separate CMSSW job, and transmits all Runs, LuminosityBlocks and "
      "Events from the current process to the remote one.");

  edm::ParameterSetDescription desc;
  desc.ifValue(
          edm::ParameterDescription<std::string>("mode", "CommWorld", false),
          ModeDescription[kCommWorld] >> edm::EmptyGroupDescription() or
              ModeDescription[kIntercommunicator] >> edm::ParameterDescription<std::string>("name", "server", false))
      ->setComment(
          "Valid modes are CommWorld (use MPI_COMM_WORLD) and Intercommunicator (use an MPI name server to setup an "
          "intercommunicator).");

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(MPIController);
