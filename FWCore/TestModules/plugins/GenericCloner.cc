/*
 * This EDProducer will clone all the event products declared by its configuration, using their ROOT dictionaries.
 *
 * The products can be specified either as module labels (e.g. "<module label>") or as branch names (e.g.
 * "<product type>_<module label>_<instance name>_<process name>").
 *
 * If a module label is used, no underscore ("_") must be present; this module will clone all the products produced by
 * that module, including those produced by the Transformer functionality (such as the implicitly copied-to-host
 * products in case of Alpaka-based modules).
 * If a branch name is used, all four fields must be present, separated by underscores; this module will clone only on
 * the matching product(s).
 *
 * Glob expressions ("?" and "*") are supported in module labels and within the individual fields of branch names,
 * similar to an OutputModule's "keep" statements.
 * Use "*" to clone all products.
 *
 * For example, in the case of Alpaka-based modules running on a device, using
 *
 *   eventProducts = cms.untracked.vstring( "module" )
 *
 * will cause "module" to run, along with automatic copy of its device products to the host, and will attempt to clone
 * all device and host products.
 * To clone only the host product, the branch can be specified explicitly with
 *
 *   eventProducts = cms.untracked.vstring( "HostProductType_module_*_*" )
 *
 * .
 */

#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <TBufferFile.h>

#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/ProductNamePattern.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/WrapperBaseHandle.h"
#include "FWCore/Framework/interface/WrapperBaseOrphanHandle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Reflection/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"

namespace edmtest {

  class GenericCloner : public edm::global::EDProducer<> {
  public:
    explicit GenericCloner(edm::ParameterSet const&);
    ~GenericCloner() override = default;

    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    struct Entry {
      edm::TypeWithDict objectType_;
      edm::TypeWithDict wrappedType_;
      edm::EDGetToken getToken_;
      edm::EDPutToken putToken_;
    };

    std::vector<edm::ProductNamePattern> eventPatterns_;
    std::vector<Entry> eventProducts_;
    std::string label_;
    bool verbose_;
  };

  GenericCloner::GenericCloner(edm::ParameterSet const& config)
      : eventPatterns_(edm::productPatterns(config.getParameter<std::vector<std::string>>("eventProducts"))),
        label_(config.getParameter<std::string>("@module_label")),
        verbose_(config.getUntrackedParameter<bool>("verbose")) {
    eventProducts_.reserve(eventPatterns_.size());

    callWhenNewProductsRegistered([this](edm::ProductDescription const& product) {
      static const std::string_view kPathStatus("edm::PathStatus");
      static const std::string_view kEndPathStatus("edm::EndPathStatus");

      switch (product.branchType()) {
        case edm::InEvent:
          if (product.className() == kPathStatus or product.className() == kEndPathStatus) {
            return;
          }
          for (auto& pattern : eventPatterns_) {
            if (pattern.match(product)) {
              // check that the product is not transient
              if (product.transient()) {
                edm::LogWarning("GenericCloner") << "Event product " << product.branchName() << " of type "
                                                 << product.unwrappedType() << " is transient, will not be cloned.";
                break;
              }
              if (verbose_) {
                edm::LogInfo("GenericCloner")
                    << "will clone Event product " << product.branchName() << " of type " << product.unwrappedType();
              }
              Entry entry;
              entry.objectType_ = product.unwrappedType();
              entry.wrappedType_ = product.wrappedType();
              // TODO move this to EDConsumerBase::consumes() ?
              entry.getToken_ = this->consumes(
                  edm::TypeToGet{product.unwrappedTypeID(), edm::PRODUCT_TYPE},
                  edm::InputTag{product.moduleLabel(), product.productInstanceName(), product.processName()});
              entry.putToken_ = this->produces(product.unwrappedTypeID(), product.productInstanceName());
              eventProducts_.emplace_back(std::move(entry));
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
              << "Unexpected product type " << product.branchType() << "\nPlease contact a Framework developer.";
      }
    });
  }

  void GenericCloner::produce(edm::StreamID /*unused*/, edm::Event& event, edm::EventSetup const& /*unused*/) const {
    for (auto& product : eventProducts_) {
      edm::Handle<edm::WrapperBase> handle(product.objectType_.typeInfo());
      event.getByToken(product.getToken_, handle);
      edm::WrapperBase const* wrapper = handle.product();

      std::unique_ptr<edm::WrapperBase> clone(
          reinterpret_cast<edm::WrapperBase*>(product.wrappedType_.getClass()->New()));

      if (wrapper->hasTrivialCopyTraits()) {
        // Use the trivialCopy traits to clone the wrapped object.

        // mark the clone as present
        clone->markAsPresent();

        // initialise the clone, if the type requires it
        if (wrapper->hasTrivialCopyProperties()) {
          clone->trivialCopyInitialize(wrapper->trivialCopyParameters());
        }

        // copy the source regions to the target
        auto sources = wrapper->trivialCopyRegions();
        auto targets = clone->trivialCopyRegions();
        assert(sources.size() == targets.size());
        for (size_t i = 0; i < sources.size(); ++i) {
          assert(sources[i].data() != nullptr);
          assert(targets[i].data() != nullptr);
          assert(targets[i].size_bytes() == sources[i].size_bytes());
          std::memcpy(targets[i].data(), sources[i].data(), sources[i].size_bytes());
        }

        // finalize the clone after the trivialCopy, if the type requires it
        clone->trivialCopyFinalize();
      } else {
        // Use ROOT-based serialisation and deserialisation to clone the wrapped object.

        // write the wrapper into a TBuffer
        TBufferFile buffer(TBuffer::kWrite);
        product.wrappedType_.getClass()->Streamer(const_cast<edm::WrapperBase*>(wrapper), buffer);

        // read back a copy of the product form the TBuffer
        buffer.SetReadMode();
        buffer.SetBufferOffset(0);
        product.wrappedType_.getClass()->Streamer(clone.get(), buffer);
      }

      // move the wrapper into the Event
      event.put(product.putToken_, std::move(clone));
    }
  }

  void GenericCloner::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    descriptions.setComment(
        R"(This EDProducer will clone all the event products declared by its configuration, using their ROOT dictionaries.

The products can be specified either as module labels (e.g. "<module label>") or as branch names (e.g. "<product type>_<module label>_<instance name>_<process name>").
If a module label is used, no underscore ("_") must be present; this module will clone all the products produced by that module, including those produced by the Transformer functionality (such as the implicitly copied-to-host products in case of Alpaka-based modules).
If a branch name is used, all four fields must be present, separated by underscores; this module will clone only on the matching product(s).

Glob expressions ("?" and "*") are supported in module labels and within the individual fields of branch names, similar to an OutputModule's "keep" statements.
Use "*" to clone all products.

For example, in the case of Alpaka-based modules running on a device, using

    eventProducts = cms.untracked.vstring( "module" )

will cause "module" to run, along with automatic copy of its device products to the host, and will attempt to clone all device and host products.
To clone only the host product, the branch can be specified explicitly with

    eventProducts = cms.untracked.vstring( "HostProductType_module_*_*" )

.)");

    edm::ParameterSetDescription desc;
    desc.add<std::vector<std::string>>("eventProducts", {})
        ->setComment("List of modules or branches whose event products will be cloned.");
    desc.addUntracked<bool>("verbose", false)
        ->setComment("Print the branch names of the products that will be cloned.");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(edmtest::GenericCloner);
