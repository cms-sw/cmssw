// C++ include files
#include <cassert>
#include <utility>

// CMSSW include files
#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/WrapperBaseOrphanHandle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"

// local include files
#include "api.h"

class MPIReceiver : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  MPIReceiver(edm::ParameterSet const& config)
      : upstream_(consumes<MPIToken>(config.getParameter<edm::InputTag>("upstream"))),
        token_(produces<MPIToken>()),
        instance_(config.getParameter<int32_t>("instance"))  //
  {
    // instance 0 is reserved for the MPIController / MPISource pair
    // instance values greater than 255 may not fit in the MPI tag
    if (instance_ < 1 or instance_ > 255) {
      throw cms::Exception("InvalidValue")
          << "Invalid MPIReceiver instance value, please use a value between 1 and 255";
    }

    auto const& products = config.getParameter<std::vector<edm::ParameterSet>>("products");
    products_.reserve(products.size());
    for (auto const& product : products) {
      auto const& type = product.getParameter<std::string>("type");
      auto const& label = product.getParameter<std::string>("label");
      Entry entry;
      entry.type = edm::TypeWithDict::byName(type);
      entry.wrappedType = edm::TypeWithDict::byName("edm::Wrapper<" + type + ">");
      entry.token = produces(edm::TypeID{entry.type.typeInfo()}, label);

      edm::LogVerbatim("MPIReceiver") << "receive type \"" << entry.type.name() << "\" for label \"" << label
                                      << "\" over MPI channel instance " << this->instance_;

      products_.emplace_back(std::move(entry));
    }
  }

  void acquire(edm::Event const& event, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder holder) final {
    const MPIToken& token = event.get(upstream_);

    edm::Service<edm::Async> as;
    as->runAsync(
        std::move(holder),
        [this, &token]() {
          int numProducts;
          // Receive the number of products
          token.channel()->receiveProduct(instance_, numProducts);
          // edm::LogAbsolute("MPIReceiver") << "Received number of products: " << numProducts;
          assert((numProducts == static_cast<int>(products_.size())) &&
                 "Receiver number of products is different than expected");
        },
        []() { return "Calling MPIReceiver::acquire()"; });
  }

  void produce(edm::Event& event, edm::EventSetup const&) final {
    // read the MPIToken used to establish the communication channel
    MPIToken token = event.get(upstream_);

    for (auto const& entry : products_) {
      std::unique_ptr<edm::WrapperBase> wrapper(
          reinterpret_cast<edm::WrapperBase*>(entry.wrappedType.getClass()->New()));

      // receive the data sent over the MPI channel
      // note: currently this uses a blocking probe/recv
      token.channel()->receiveProduct(instance_, entry.wrappedType, *wrapper);

      // put the data into the Event
      event.put(entry.token, std::move(wrapper));
    }

    // write a shallow copy of the channel to the output, so other modules can consume it
    // to indicate that they should run after this
    event.emplace(token_, token);
  }

private:
  struct Entry {
    edm::TypeWithDict type;
    edm::TypeWithDict wrappedType;
    edm::EDPutToken token;
  };

  // TODO consider if upstream_ should be a vector instead of a single token ?
  edm::EDGetTokenT<MPIToken> const upstream_;  // MPIToken used to establish the communication channel
  edm::EDPutTokenT<MPIToken> const token_;  // copy of the MPIToken that may be used to implement an ordering relation
  std::vector<Entry> products_;             // data to be read over the channel and put into the Event
  int32_t const instance_;                  // instance used to identify the source-destination pair
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MPIReceiver);
