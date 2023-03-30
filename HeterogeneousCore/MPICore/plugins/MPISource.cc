#include <deque>
#include <memory>
#include <string>

#include <TBuffer.h>
#include <TBufferFile.h>
#include <TClass.h>

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
#include "FWCore/Sources/interface/PuttableSourceBase.h"
#include "HeterogeneousCore/MPIServices/interface/MPIService.h"

#include "api.h"
#include "conversion.h"
#include "messages.h"
#include "mpi.h"

/* FIXME move MPIRecv ?
namespace {
  // create a branch description for the MPIOrigin object
  edm::BranchDescription makeOriginBranchDescription(edm::ModuleDescription const& md) {
    static const auto type = edm::TypeWithDict(typeid(MPIOrigin));
    edm::BranchDescription desc(edm::InEvent,         // branch type
                                md.moduleLabel(),     // producer module label
                                md.processName(),     // process name
                                "MPIOrigin",          // class name
                                "MPIOrigin",          // friendly class name
                                "",                   // product instance label
                                md.moduleName(),      // producer module name (C++ type)
                                md.parameterSetID(),  // parameter set id of the producer
                                type);                // product type
    return desc;
  }
}  // namespace
*/

class MPISource : public edm::PuttableSourceBase {
public:
  explicit MPISource(edm::ParameterSet const& config, edm::InputSourceDescription const& desc);
  ~MPISource() override;
  using InputSource::processHistoryRegistryForUpdate;
  using InputSource::productRegistryUpdate;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  ItemType getNextItemType() override;
  std::shared_ptr<edm::RunAuxiliary> readRunAuxiliary_() override;
  std::shared_ptr<edm::LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() override;
  void readEvent_(edm::EventPrincipal& eventPrincipal) override;

  char port_[MPI_MAX_PORT_NAME];
  MPI_Comm comm_ = MPI_COMM_NULL;
  MPISender link;

  edm::ProcessHistory history_;
  /* FIXME replace with a data product that keeps track of the MPIDriver origin
  edm::BranchDescription originBranchDescription_;
  edm::ProductProvenance originProvenance_;
  */

  std::shared_ptr<edm::RunAuxiliary> runAuxiliary_;
  std::shared_ptr<edm::LuminosityBlockAuxiliary> luminosityBlockAuxiliary_;
  std::deque<edm::EventAuxiliary> eventAuxiliaries_;

  /* FIXME move MPIRecv ?
  struct DataProduct {
    DataProduct() = default;
    DataProduct(std::unique_ptr<edm::WrapperBase> product,
                edm::BranchDescription const* branchDescription,
                edm::ProductProvenance provenance)
        : product(std::move(product)), branchDescription(branchDescription), provenance(std::move(provenance)) {}

    std::unique_ptr<edm::WrapperBase> product;        // owns the wrapped product until it is put into the event
    edm::BranchDescription const* branchDescription;  // non-owning pointer
    edm::ProductProvenance provenance;                // cheap enough to be stored by value
  };

  struct EventData {
    edm::EventAuxiliary eventAuxiliary;
    std::vector<DataProduct> eventProducts;
  };

  std::deque<EventData> events_;
  */
};

MPISource::MPISource(edm::ParameterSet const& config, edm::InputSourceDescription const& desc)
    : edm::PuttableSourceBase(config, desc)
/* FIXME replace with a data product that keeps track of the MPIDriver origin
      originBranchDescription_(makeOriginBranchDescription(desc.moduleDescription_)),
      originProvenance_(originBranchDescription_.branchID()) {
  // register the MPIOrigin branch
  productRegistryUpdate().addProduct(originBranchDescription_);
  */
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
  link = MPISender(comm_, 0);

  // wait for a client to connect
  MPI_Status status;
  EDM_MPI_Empty_t buffer;
  MPI_Recv(&buffer, 1, EDM_MPI_Empty, MPI_ANY_SOURCE, EDM_MPI_Connect, comm_, &status);
  edm::LogAbsolute("MPI") << "connected from " << status.MPI_SOURCE;

  /* FIXME move MPIRecv
  // receive the branch descriptions
  MPI_Message message;
  int source = status.MPI_SOURCE;
  while (true) {
    MPI_Mprobe(source, MPI_ANY_TAG, comm_, &message, &status);
    if (status.MPI_TAG == EDM_MPI_SendComplete) {
      // all branches have been received
      MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);
      edm::LogAbsolute("MPI") << "all BranchDescription received";
      break;
    } else {
      // receive the branch description for the next event product
      assert(EDM_MPI_SendSerializedProduct == status.MPI_TAG);
      int size;
      MPI_Get_count(&status, MPI_BYTE, &size);
      TBufferFile blob{TBuffer::kRead, size};
      MPI_Mrecv(blob.Buffer(), size, MPI_BYTE, &message, &status);
      edm::BranchDescription bd;
      TClass::GetClass(typeid(edm::BranchDescription))->ReadBuffer(blob, &bd);
      bd.setDropped(false);
      bd.setProduced(false);
      bd.setOnDemand(false);
      bd.setIsProvenanceSetOnRead(true);
      bd.init();
      productRegistryUpdate().copyProduct(bd);
    }
  }
  edm::LogAbsolute("MPI") << "registered branchess:\n";
  for (auto& keyval : productRegistry()->productList()) {
    edm::LogAbsolute("MPI") << "  - " << keyval.first;
  }
  edm::LogAbsolute("MPI") << '\n';
  */
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

MPISource::ItemType MPISource::getNextItemType() {
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
      return IsInvalid;
    }

    // Disconnect message
    case EDM_MPI_Disconnect: {
      // receive the message header
      EDM_MPI_Empty_t buffer;
      MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);

      // signal the end of the input data
      return IsStop;
    }

    // BeginStream message
    case EDM_MPI_BeginStream: {
      // receive the message header
      EDM_MPI_Empty_t buffer;
      MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);

      // nothing else to do
      return getNextItemType();
    }

    // EndStream message
    case EDM_MPI_EndStream: {
      // receive the message header
      EDM_MPI_Empty_t buffer;
      MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);

      // nothing else to do
      return getNextItemType();
    }

    // BeginRun message
    case EDM_MPI_BeginRun: {
      // receive the RunAuxiliary
      EDM_MPI_RunAuxiliary_t buffer;
      MPI_Mrecv(&buffer, 1, EDM_MPI_RunAuxiliary, &message, &status);
      runAuxiliary_ = std::make_shared<edm::RunAuxiliary>();
      edmFromBuffer(buffer, *runAuxiliary_);

      // receive the ProcessHistory
      MPI_Mprobe(status.MPI_SOURCE, EDM_MPI_SendSerializedProduct, comm_, &message, &status);
      int size;
      MPI_Get_count(&status, MPI_BYTE, &size);
      TBufferFile blob{TBuffer::kRead, size};
      MPI_Mrecv(blob.Buffer(), size, MPI_BYTE, &message, &status);
      history_.clear();
      TClass::GetClass(typeid(edm::ProcessHistory))->ReadBuffer(blob, &history_);
      history_.initializeTransients();
      if (processHistoryRegistryForUpdate().registerProcessHistory(history_)) {
        edm::LogAbsolute("MPI") << "new ProcessHistory registered: " << history_;
      }

      // signal a new run
      return IsRun;
    }

    // EndRun message
    case EDM_MPI_EndRun: {
      // receive the RunAuxiliary message
      EDM_MPI_RunAuxiliary_t buffer;
      MPI_Mrecv(&buffer, 1, EDM_MPI_RunAuxiliary, &message, &status);

      // nothing else to do
      return getNextItemType();
    }

    // BeginLuminosityBlock message
    case EDM_MPI_BeginLuminosityBlock: {
      // receive the LuminosityBlockAuxiliary
      EDM_MPI_LuminosityBlockAuxiliary_t buffer;
      MPI_Mrecv(&buffer, 1, EDM_MPI_LuminosityBlockAuxiliary, &message, &status);
      luminosityBlockAuxiliary_ = std::make_shared<edm::LuminosityBlockAuxiliary>();
      edmFromBuffer(buffer, *luminosityBlockAuxiliary_);

      // signal a new lumisection
      return IsLumi;
    }

    // EndLuminosityBlock message
    case EDM_MPI_EndLuminosityBlock: {
      // receive the LuminosityBlockAuxiliary
      EDM_MPI_LuminosityBlockAuxiliary_t buffer;
      MPI_Mrecv(&buffer, 1, EDM_MPI_LuminosityBlockAuxiliary, &message, &status);

      // nothing else to do
      return getNextItemType();
    }

    // ProcessEvent message
    case EDM_MPI_ProcessEvent: {
      // allocate a new event
      //auto& event = events_.emplace_back();
      //event.eventProducts.reserve(productRegistryUpdate().size());
      auto& event = eventAuxiliaries_.emplace_back();

      // receive the EventAuxiliary
      //auto [status, stream] = link.receiveEvent(event.eventAuxiliary, message);
      auto [status, stream] = link.receiveEvent(event, message);
      int source = status.MPI_SOURCE;
      // FIXME store these in a data product that keeps track of the MPIDriver origin ?
      (void)source;
      (void)stream;

      /* FIXME replace with a data product that keeps track of the MPIDriver origin
      // store the MPI origin
      auto origin = std::make_unique<edm::Wrapper<MPIOrigin>>(edm::WrapperBase::Emplace{}, source, stream);
      event.eventProducts.emplace_back(std::move(origin), &originBranchDescription_, originProvenance_);
      */

      /* FIXME move MPIRecv
      //
      MPI_Message message;
      while (true) {
        MPI_Mprobe(source, MPI_ANY_TAG, comm_, &message, &status);
        if (EDM_MPI_SendComplete == status.MPI_TAG) {
          // all products have been received
          EDM_MPI_Empty_t buffer;
          MPI_Mrecv(&buffer, 1, EDM_MPI_Empty, &message, &status);
          edm::LogAbsolute("MPI") << "all Products received";
          break;
        } else {
          edm::BranchKey key;
          edm::ProductProvenance provenance;
          edm::ProductID pid;
          edm::WrapperBase* wrapper;
          {
            // receive the BranchKey
            assert(EDM_MPI_SendSerializedProduct == status.MPI_TAG);
            int size;
            MPI_Get_count(&status, MPI_BYTE, &size);
            TBufferFile buffer{TBuffer::kRead, size};
            MPI_Mrecv(buffer.Buffer(), size, MPI_BYTE, &message, &status);
            TClass::GetClass(typeid(edm::BranchKey))->ReadBuffer(buffer, &key);
          }

          edm::BranchDescription const& branch = productRegistry()->productList().at(key);

          MPI_Mprobe(source, MPI_ANY_TAG, comm_, &message, &status);
          {
            // receive the ProductProvenance
            assert(EDM_MPI_SendSerializedProduct == status.MPI_TAG);
            int size;
            MPI_Get_count(&status, MPI_BYTE, &size);
            TBufferFile buffer{TBuffer::kRead, size};
            MPI_Mrecv(buffer.Buffer(), size, MPI_BYTE, &message, &status);
            TClass::GetClass(typeid(edm::ProductProvenance))->ReadBuffer(buffer, &provenance);
          }
          MPI_Mprobe(source, MPI_ANY_TAG, comm_, &message, &status);
          {
            // receive the ProductID
            assert(EDM_MPI_SendSerializedProduct == status.MPI_TAG);
            int size;
            MPI_Get_count(&status, MPI_BYTE, &size);
            TBufferFile buffer{TBuffer::kRead, size};
            MPI_Mrecv(buffer.Buffer(), size, MPI_BYTE, &message, &status);
            TClass::GetClass(typeid(edm::ProductID))->ReadBuffer(buffer, &pid);
          }
          MPI_Mprobe(source, MPI_ANY_TAG, comm_, &message, &status);
          {
            // receive the product
            assert(EDM_MPI_SendSerializedProduct == status.MPI_TAG);
            int size;
            MPI_Get_count(&status, MPI_BYTE, &size);
            TBufferFile buffer{TBuffer::kRead, size};
            MPI_Mrecv(buffer.Buffer(), size, MPI_BYTE, &message, &status);
            // construct an edm::Wrapper<T> and fill it with the received product
            // TODO this would be much simpler if the MPIDriver could sent the Wrapper<T> instead of T
            edm::TypeWithDict const& type = branch.wrappedType();
            edm::ObjectWithDict object = type.construct();
            *reinterpret_cast<bool*>(reinterpret_cast<char*>(object.address()) +
                                     type.dataMemberByName("present").offset()) = true;
            branch.unwrappedType().getClass()->ReadBuffer(
                buffer, reinterpret_cast<char*>(object.address()) + type.dataMemberByName("obj").offset());
            wrapper = reinterpret_cast<edm::WrapperBase*>(object.address());
          }
          edm::LogAbsolute("MPI") << "received object for branch " << key;
          //edm::LogAbsolute("MPI") << "received object of type " << branch.unwrappedType();

          // store the received product
          event.eventProducts.emplace_back(std::unique_ptr<edm::WrapperBase>(wrapper), &branch, provenance);
        }
      }
      */

      // signal a new event
      return IsEvent;
    }

    // unexpected message
    default: {
      return IsInvalid;
    }
  }
}

std::shared_ptr<edm::RunAuxiliary> MPISource::readRunAuxiliary_() { return runAuxiliary_; }

std::shared_ptr<edm::LuminosityBlockAuxiliary> MPISource::readLuminosityBlockAuxiliary_() {
  return luminosityBlockAuxiliary_;
}

void MPISource::readEvent_(edm::EventPrincipal& eventPrincipal) {
  edm::LogAbsolute("MPI") << "number of buffered events: " << eventAuxiliaries_.size();
  auto& aux = eventAuxiliaries_.front();
  edm::ProductProvenanceRetriever prov(eventPrincipal.transitionIndex(), *productRegistry());
  eventPrincipal.fillEventPrincipal(aux,
                                    &history_,
                                    edm::EventSelectionIDVector{},
                                    edm::BranchListIndexes{},
                                    edm::EventToProcessBlockIndexes{},
                                    prov,
                                    nullptr,
                                    false);

  edm::Event event(eventPrincipal, moduleDescription(), nullptr);
  event.setProducer(this, nullptr);
  // FIXME produce the MPI "token"
  //event.emplace(...);
  commit_(event);

  /* FIXME move MPIRecv ?
  for (auto& product : event.eventProducts) {
    //edm::LogAbsolute("MPI") << "putting object for branch " << *product.branchDescription;
    eventPrincipal.put(*product.branchDescription, std::move(product.product), product.provenance);
  }
  */

  //events_.pop_front();
  eventAuxiliaries_.pop_front();
}

void MPISource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Comunicate with another cmsRun process over MPI.");
  edm::InputSource::fillDescription(desc);
  descriptions.add("source", desc);
}

#include "FWCore/Framework/interface/InputSourceMacros.h"
DEFINE_FWK_INPUT_SOURCE(MPISource);
