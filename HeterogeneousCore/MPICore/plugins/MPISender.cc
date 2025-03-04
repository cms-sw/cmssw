#include <string>
#include <string_view>
#include <vector>

// CMSSW include files
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/ProductNamePattern.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/GenericHandle.h"
#include "FWCore/Framework/interface/WrapperBaseHandle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"

// local include files
#include "api.h"

class MPISender : public edm::global::EDProducer<> {
public:
  MPISender(edm::ParameterSet const& config)
      : upstream_(consumes<MPIToken>(config.getParameter<edm::InputTag>("upstream"))),
        token_(produces<MPIToken>()),
        patterns_(edm::productPatterns(config.getParameter<std::vector<std::string>>("products"))),
        instance_(config.getParameter<int32_t>("instance"))  //
  {
    // instance 0 is reserved for the MPIController / MPISource pair
    // instance values greater than 255 may not fit in the MPI tag
    if (instance_ < 1 or instance_ > 255) {
      throw cms::Exception("InvalidValue") << "Invalid MPISender instance value, please use a value between 1 and 255";
    }

    products_.reserve(patterns_.size());

    callWhenNewProductsRegistered([this](edm::ProductDescription const& product) {
      static const std::string_view kPathStatus("edm::PathStatus");
      static const std::string_view kEndPathStatus("edm::EndPathStatus");

      switch (product.branchType()) {
        case edm::InEvent:
          if (product.className() == kPathStatus or product.className() == kEndPathStatus)
            return;
          for (auto const& pattern : patterns_) {
            if (pattern.match(product)) {
              Entry entry;
              entry.type = product.unwrappedType();
              entry.wrappedType = product.wrappedType();
              // TODO move this to EDConsumerBase::consumes() ?
              entry.token = this->consumes(
                  edm::TypeToGet{product.unwrappedTypeID(), edm::PRODUCT_TYPE},
                  edm::InputTag{product.moduleLabel(), product.productInstanceName(), product.processName()});

              edm::LogVerbatim("MPISender")
                  << "send product \"" << product.friendlyClassName() << '_' << product.moduleLabel() << '_'
                  << product.productInstanceName() << '_' << product.processName() << "\" of type \""
                  << entry.type.name() << "\" over MPI channel instance " << instance_;

              products_.emplace_back(std::move(entry));
              break;
            }
          }
          break;

        case edm::InLumi:
        case edm::InRun:
        case edm::InProcess:
          // lumi, run and process products are not supported
          break;

        default:
          throw edm::Exception(edm::errors::LogicError)
              << "Unexpected branch type " << product.branchType() << "\nPlease contact a Framework developer\n";
      }
    });

    // TODO add an error if a pattern does not match any branches? how?
  }

  void produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const override {
    // read the MPIToken used to establish the communication channel
    MPIToken token = event.get(upstream_);

    int numProducts = static_cast<int>(products_.size());
    token.channel()->sendProduct(instance_, numProducts);

    for (auto const& entry : products_) {
      // read the products to be sent over the MPI channel
      edm::Handle<edm::WrapperBase> handle(entry.type.typeInfo());
      event.getByToken(entry.token, handle);
      edm::WrapperBase const* wrapper = handle.product();
      // send the products over MPI
      // note: currently this uses a blocking send
      token.channel()->sendProduct(instance_, entry.wrappedType, *wrapper);
    }

    // write a shallow copy of the channel to the output, so other modules can consume it
    // to indicate that they should run after this
    event.emplace(token_, token);
  }

private:
  struct Entry {
    edm::TypeWithDict type;
    edm::TypeWithDict wrappedType;
    edm::EDGetToken token;
  };

  // TODO consider if upstream_ should be a vector instead of a single token ?
  edm::EDGetTokenT<MPIToken> const upstream_;  // MPIToken used to establish the communication channel
  edm::EDPutTokenT<MPIToken> const token_;  // copy of the MPIToken that may be used to implement an ordering relation
  std::vector<edm::ProductNamePattern> patterns_;  // branches to read from the Event and send over the MPI channel
  std::vector<Entry> products_;                    // types and tokens corresponding to the branches
  int32_t const instance_;                         // instance used to identify the source-destination pair
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MPISender);
