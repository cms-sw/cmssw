#include <iomanip>
#include <iostream>
#include <string>

#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/GenericHandle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"
#include "HeterogeneousCore/MPICore/interface/WrapperHandle.h"
#include "HeterogeneousCore/MPICore/interface/serialization.h"

class GenericConsumer : public edm::global::EDProducer<> {

  public:
    explicit GenericConsumer(edm::ParameterSet const& config);
    ~GenericConsumer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    void produce(edm::StreamID, edm::Event & event, edm::EventSetup const& setup) const override;

  private:
    edm::InputTag source_;
    std::vector<edm::TypeWithDict> types_;
    std::vector<edm::TypeWithDict> wrappedTypes_;
    std::vector<edm::EDGetToken> getTokens_;
    std::vector<edm::EDPutToken> putTokens_;
};

GenericConsumer::GenericConsumer(edm::ParameterSet const& config) :
  source_(config.getParameter<edm::InputTag>("source"))
{
  callWhenNewProductsRegistered([this, this_module_label = config.getParameter<std::string>("@module_label")](edm::BranchDescription const& branch){
    if (branch.moduleLabel() == source_.label() and
        branch.productInstanceName() == source_.instance() and
        (branch.processName() == source_.process() or source_.process().empty()))
    {
      // type of the product
      types_.push_back(branch.unwrappedType());
      // type of edm::Wrapprer<Product>
      wrappedTypes_.push_back(branch.wrappedType());
      // register a token for getting the product
      std::cerr << "Will consume " << branch.unwrappedType().name() << " from " << branch.moduleLabel() << ":" << branch.productInstanceName() << ":" << branch.processName() << std::endl;
      getTokens_.push_back(consumes(edm::TypeToGet{branch.unwrappedTypeID(), edm::PRODUCT_TYPE}, source_));
      // encode the original product label in the instance
      std::string instance = source_.label();
      if (not source_.instance().empty()) {
        instance += "@";
        instance += source_.instance();
      }
      // TODO add some logic to handle the process name if different from the current one ?
      // register a token for putting the product
      std::cerr << "Will produce " << branch.unwrappedType().name() << " as " << this_module_label << ":" << instance << ":" << std::endl;
      putTokens_.push_back(produces(branch.unwrappedTypeID(), instance));
    }
  });
}

void GenericConsumer::produce(edm::StreamID sid, edm::Event & event, edm::EventSetup const& setup) const {
  for (unsigned int i = 0; i < types_.size(); ++i) {
    auto const& type = types_[i];
    //auto const& wrappedType = wrappedTypes_[i];
    auto const& getToken = getTokens_[i];
    auto const& putToken = putTokens_[i];

    edm::Handle<edm::WrapperBase> handle(type.typeInfo());
    event.getByToken(getToken, handle);
    auto read_buffer = io::serialize(*handle);          // read_buffer owns the memory

    // print the content of the buffer
    std::cerr << read_buffer << std::endl;

    // copy the buffer
    io::unique_buffer write_buffer(read_buffer.size());
    memcpy(write_buffer.data(), read_buffer.data(), read_buffer.size());

    // deserialise the Wrapper<> from the buffer and store it in a unique_ptr<WrapperBase>
    auto product = io::deserialize(write_buffer);
    event.put(putToken, std::move(product));
  }
}

void GenericConsumer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("source", edm::InputTag());
  descriptions.add(defaultModuleLabel<GenericConsumer>(), desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GenericConsumer);
