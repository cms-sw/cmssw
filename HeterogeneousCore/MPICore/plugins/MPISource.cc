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
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Sources/interface/ProducerSourceBase.h"
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

  char port_[MPI_MAX_PORT_NAME];
  MPI_Comm comm_ = MPI_COMM_NULL;
  MPIChannel channel_;
  edm::EDPutTokenT<MPIToken> token_;

  edm::ProcessHistory history_;
};

MPISource::MPISource(edm::ParameterSet const& config, edm::InputSourceDescription const& desc)
    :  // note that almost all configuration parameters passed to IDGeneratorSourceBase via ProducerSourceBase will
       // effectively be ignored, because this ConfigurableSource will explicitly set the run, lumi, and event
       // numbers, the timestamp, and the event type
      edm::ProducerSourceBase(config, desc, false),
      token_(produces<MPIToken>())  //
{
  // make sure that MPI is initialised
  MPIService::required();

  // FIXME move into the MPIService ?
  // make sure the EDM MPI types are available
  EDM_MPI_build_types();

  // open a server-side port
  MPI_Open_port(MPI_INFO_NULL, port_);

  // publish the port under the name "server"
  MPI_Info port_info;
  MPI_Info_create(&port_info);
  MPI_Info_set(port_info, "ompi_global_scope", "true");
  MPI_Info_set(port_info, "ompi_unique", "true");
  MPI_Publish_name("server", port_info, port_);

  // create an intercommunicator and accept a client connection
  edm::LogAbsolute("MPI") << "waiting for a connection to the MPI server at port " << port_;
  MPI_Comm_accept(port_, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &comm_);
  channel_ = MPIChannel(comm_, 0);

  // wait for a client to connect
  MPI_Status status;
  EDM_MPI_Empty_t buffer;
  MPI_Recv(&buffer, 1, EDM_MPI_Empty, MPI_ANY_SOURCE, EDM_MPI_Connect, comm_, &status);
  edm::LogAbsolute("MPI") << "connected from " << status.MPI_SOURCE;
}

MPISource::~MPISource() {
  // close the intercommunicator
  MPI_Comm_disconnect(&comm_);

  // unpublish and close the port
  MPI_Info port_info;
  MPI_Info_create(&port_info);
  MPI_Info_set(port_info, "ompi_global_scope", "true");
  MPI_Info_set(port_info, "ompi_unique", "true");
  MPI_Unpublish_name("server", port_info, port_);
  MPI_Close_port(port_);
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
        channel_.receiveSerializedProduct(0, history_);
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
  edm::ParameterSetDescription desc;
  desc.setComment("Comunicate with another cmsRun process over MPI.");
  edm::ProducerSourceBase::fillDescription(desc);

  descriptions.add("source", desc);
}

#include "FWCore/Framework/interface/InputSourceMacros.h"
DEFINE_FWK_INPUT_SOURCE(MPISource);
