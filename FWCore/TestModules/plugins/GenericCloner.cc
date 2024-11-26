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
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <TBufferFile.h>

#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/ProductNamePattern.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/GenericHandle.h"
#include "FWCore/Framework/interface/GenericProduct.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Reflection/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/EDMException.h"

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

    callWhenNewProductsRegistered([this](edm::ProductDescription const& branch) {
      static const std::string_view kPathStatus("edm::PathStatus");
      static const std::string_view kEndPathStatus("edm::EndPathStatus");

      switch (branch.branchType()) {
        case edm::InEvent:
          if (branch.className() == kPathStatus or branch.className() == kEndPathStatus) {
            return;
          }
          for (auto& pattern : eventPatterns_) {
            if (pattern.match(branch)) {
              Entry product;
              product.objectType_ = branch.unwrappedType();
              product.wrappedType_ = branch.wrappedType();
              // TODO move this to EDConsumerBase::consumes() ?
              product.getToken_ = this->consumes(
                  edm::TypeToGet{branch.unwrappedTypeID(), edm::PRODUCT_TYPE},
                  edm::InputTag{branch.moduleLabel(), branch.productInstanceName(), branch.processName()});
              product.putToken_ = this->produces(branch.unwrappedTypeID(), branch.productInstanceName());
              eventProducts_.push_back(product);

              if (verbose_) {
                edm::LogInfo("GenericCloner")
                    << label_ << " will clone Event product " << branch.friendlyClassName() << '_'
                    << branch.moduleLabel() << '_' << branch.productInstanceName() << '_' << branch.processName();
              }
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
              << "Unexpected branch type " << branch.branchType() << "\nPlease contact a Framework developer.";
      }
    });
  }

  void GenericCloner::produce(edm::StreamID /*unused*/, edm::Event& event, edm::EventSetup const& /*unused*/) const {
    for (auto& product : eventProducts_) {
      edm::GenericHandle handle(product.objectType_);
      event.getByToken(product.getToken_, handle);
      edm::ObjectWithDict const* object = handle.product();

      // An ObjectWithDict can hold
      //   - a fundamental type,
      //   - a class type (class, struct or enum),
      //   - a pointer,
      //   - a C array.
      //
      // Pointers can be treated as fundamental types and shallow-copied.
      // C arrays are not supported as event products, so can be skipped.
      if (product.objectType_.isFundamental() or product.objectType_.isPointer()) {
        // fundamental types can simply be copied, and the copy is done implicitly by the Event::put() specialisation for GenericProduct
        event.put(product.putToken_, std::make_unique<edm::GenericProduct>(*object, product.wrappedType_));
      } else if (product.objectType_.isClass()) {
        // objects are cloned using their ROOT dictionary
        if (product.wrappedType_.dataMemberByName("obj").isTransient()) {
          throw edm::Exception(edm::errors::LogicError)
              << "Type " << product.objectType_.name() << " is transient, and cannot be cloned.";
        }

        // write the product into a TBuffer
        TBufferFile buffer(TBuffer::kWrite);
        product.objectType_.getClass()->Streamer(object->address(), buffer);

        // read back a copy of the product form the TBuffer
        buffer.SetReadMode();
        buffer.SetBufferOffset(0);
        edm::ObjectWithDict clone = product.objectType_.construct();
        product.objectType_.getClass()->Streamer(clone.address(), buffer);

        // wrap the copy in a GenericProduct, and call the Event::put() method specialisation for GenericProduct
        event.put(product.putToken_, std::make_unique<edm::GenericProduct>(clone, product.wrappedType_));

        // the cloned object has been moved into the Event; now destroy the temporary object
        clone.destruct(true);
      } else {
        // other types are not supported
        throw edm::Exception(edm::errors::LogicError)
            << "Unsupported type " << product.objectType_.name() << ".\nPlease contact a framework developer.";
      }
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
