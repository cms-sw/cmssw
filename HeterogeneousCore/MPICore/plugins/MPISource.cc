// C++ headers
#include <memory>
#include <stdexcept>
#include <string>

// ROOT headers
#include <TBuffer.h>
#include <TBufferFile.h>
#include <TClass.h>

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
#include "FWCore/MessageLogger/interface/ErrorObj.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"
#include "FWCore/Sources/interface/ProducerSourceBase.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"
#include "HeterogeneousCore/MPIServices/interface/MPIService.h"

// local headers
#include "api.h"
#include "conversion.h"
#include "messages.h"

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
  MPIChannel channel_;
  edm::EDPutTokenT<MPIToken> token_;
  Mode mode_;

  edm::ProcessHistory history_;
};

MPISource::MPISource(edm::ParameterSet const& config, edm::InputSourceDescription const& desc)
    :  // note that almost all configuration parameters passed to IDGeneratorSourceBase via ProducerSourceBase will
       // effectively be ignored, because this ConfigurableSource will explicitly set the run, lumi, and event
       // numbers, the timestamp, and the event type
      edm::ProducerSourceBase(config, desc, false),
      token_(produces<MPIToken>()),
      mode_(parseMode(config.getUntrackedParameter<std::string>("mode")))  //
{
  // make sure that MPI is initialised
  MPIService::required();

  // Make sure the EDM MPI types are available.
  EDM_MPI_build_types();

  if (mode_ == kCommWorld) {
    // All processes are in MPI_COMM_WORLD.
    // The current implementation supports only two processes: one controller and one source.
    edm::LogAbsolute("MPI") << "MPISource in " << ModeDescription[mode_] << " mode.";

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
    edm::LogAbsolute("MPI") << "MPISource has rank " << rank << " in MPI_COMM_WORLD.";
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
    edm::LogAbsolute("MPI") << "Waiting for a connection to the MPI server at port " << port_;

    MPI_Comm_accept(port_, MPI_INFO_NULL, 0, MPI_COMM_SELF, &comm_);
    edm::LogAbsolute("MPI") << "Connection accepted.";
    channel_ = MPIChannel(comm_, 0);
  } else {
    // Invalid mode.
    throw edm::Exception(edm::errors::Configuration)
        << "Invalid mode \"" << config.getUntrackedParameter<std::string>("mode") << "\"";
  }

  // Wait for a client to connect.
  MPI_Status status;
  EDM_MPI_Empty_t buffer;
  MPI_Recv(&buffer, 1, EDM_MPI_Empty, MPI_ANY_SOURCE, EDM_MPI_Connect, comm_, &status);
  edm::LogAbsolute("MPI") << "connected from " << status.MPI_SOURCE;
}

MPISource::~MPISource() {
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

//MPISource::ItemTypeInfo MPISource::getNextItemType() {
bool MPISource::setRunAndEventInfo(edm::EventID& event,
                                   edm::TimeValue_t& time,
                                   edm::EventAuxiliary::ExperimentType& type) {
  while (true) {
    MPI_Status status;
    MPI_Message message;
    MPI_Mprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_, &message, &status);
    switch (status.MPI_TAG) {
      // Connect message
      case EDM_MPI_Connect: {
        // receive the message header
        EDM_MPI_Empty_t buffer;
        MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);

        // the Connect message is unexpected here (see above)
        throw cms::Exception("InvalidValue")
            << "The MPISource has received an EDM_MPI_Connect message after the initial connection";
        return false;
      }

      // Disconnect message
      case EDM_MPI_Disconnect: {
        // receive the message header
        EDM_MPI_Empty_t buffer;
        MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);

        // signal the end of the input data
        return false;
      }

      // BeginStream message
      case EDM_MPI_BeginStream: {
        // receive the message header
        EDM_MPI_Empty_t buffer;
        MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);

        // receive the next message
        break;
      }

      // EndStream message
      case EDM_MPI_EndStream: {
        // receive the message header
        EDM_MPI_Empty_t buffer;
        MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);

        // receive the next message
        break;
      }

      // BeginRun message
      case EDM_MPI_BeginRun: {
        // receive the RunAuxiliary
        EDM_MPI_RunAuxiliary_t buffer;
        MPI_Mrecv(&buffer, 1, EDM_MPI_RunAuxiliary, &message, &status);
        // TODO this is currently not used
        edm::RunAuxiliary runAuxiliary;
        edmFromBuffer(buffer, runAuxiliary);

        // receive the ProcessHistory
        history_.clear();
        channel_.receiveProduct(0, history_);
        history_.initializeTransients();
        if (processHistoryRegistryForUpdate().registerProcessHistory(history_)) {
          edm::LogAbsolute("MPI") << "new ProcessHistory registered: " << history_;
        }

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
        // receive the LuminosityBlockAuxiliary
        EDM_MPI_LuminosityBlockAuxiliary_t buffer;
        MPI_Mrecv(&buffer, 1, EDM_MPI_LuminosityBlockAuxiliary, &message, &status);
        // TODO this is currently not used
        edm::LuminosityBlockAuxiliary luminosityBlockAuxiliary;
        edmFromBuffer(buffer, luminosityBlockAuxiliary);

        // receive the next message
        break;
      }

      // EndLuminosityBlock message
      case EDM_MPI_EndLuminosityBlock: {
        // receive the LuminosityBlockAuxiliary
        EDM_MPI_LuminosityBlockAuxiliary_t buffer;
        MPI_Mrecv(&buffer, 1, EDM_MPI_LuminosityBlockAuxiliary, &message, &status);

        // receive the next message
        break;
      }

      // ProcessEvent message
      case EDM_MPI_ProcessEvent: {
        // receive the EventAuxiliary
        edm::EventAuxiliary aux;
        status = channel_.receiveEvent(aux, message);

        // extract the rank of the other process (currently unused)
        int source = status.MPI_SOURCE;
        (void)source;

        // fill the event details
        event = aux.id();
        time = aux.time().value();
        type = aux.experimentType();

        // signal a new event
        return true;
      }

      // unexpected message
      default: {
        throw cms::Exception("InvalidValue")
            << "The MPISource has received an unknown message with tag " << status.MPI_TAG;
        return false;
      }
    }
  }
}

void MPISource::produce(edm::Event& event) {
  // duplicate the MPIChannel and put the copy into the Event
  std::shared_ptr<MPIChannel> channel(new MPIChannel(channel_.duplicate()), [](MPIChannel* ptr) {
    ptr->reset();
    delete ptr;
  });
  event.emplace(token_, std::move(channel));
}

void MPISource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  descriptions.setComment(
      "This module connects to an \"MPIController\" in a separate CMSSW job, receives all Runs, LuminosityBlocks and "
      "Events from the remote process and reproduces them in the local one.");

  edm::ParameterSetDescription desc;
  edm::ProducerSourceBase::fillDescription(desc);
  desc.ifValue(
          edm::ParameterDescription<std::string>("mode", "CommWorld", false),
          ModeDescription[kCommWorld] >> edm::EmptyGroupDescription() or
              ModeDescription[kIntercommunicator] >> edm::ParameterDescription<std::string>("name", "server", false))
      ->setComment(
          "Valid modes are CommWorld (use MPI_COMM_WORLD) and Intercommunicator (use an MPI name server to setup an "
          "intercommunicator).");

  descriptions.add("source", desc);
}

#include "FWCore/Framework/interface/InputSourceMacros.h"
DEFINE_FWK_INPUT_SOURCE(MPISource);
