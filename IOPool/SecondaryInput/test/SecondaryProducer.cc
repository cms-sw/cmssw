// File: SecondaryProducer.cc
// Description:  see SecondaryProducer.h
// Author:  Bill Tanenbaum
//
//--------------------------------------------

#include "IOPool/SecondaryInput/test/SecondaryProducer.h"
#include "DataFormats/Common/interface/ConvertHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "FWCore/Sources/interface/VectorInputSourceDescription.h"
#include "FWCore/Sources/interface/VectorInputSourceFactory.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>
#include <string>

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {

  // Constructor
  // make secondary input source
  SecondaryProducer::SecondaryProducer(ParameterSet const& pset) :
        productRegistry_(new SignallingProductRegistry),
        secInput_(makeSecInput(pset)),
        processConfiguration_(new ProcessConfiguration(std::string("PROD"), getReleaseVersion(), getPassID())),
        eventPrincipal_(),
        sequential_(pset.getUntrackedParameter<bool>("seq", false)),
        specified_(pset.getUntrackedParameter<bool>("specified", false)),
        firstEvent_(true),
        firstLoop_(true),
        expectedEventNumber_(sequential_ ? pset.getParameterSet("input").getUntrackedParameter<unsigned int>("skipEvents", 0) + 1 : 1) {
    processConfiguration_->setParameterSetID(ParameterSet::emptyParameterSetID());
    processConfiguration_->setProcessConfigurationID();
 
    productRegistry_->setFrozen();

    produces<edmtest::ThingCollection>();
    produces<edmtest::OtherThingCollection>("testUserTag");
    consumes<edmtest::UInt64Product>(edm::InputTag{"EventNumber"});
  }

  void SecondaryProducer::beginJob() {
    // propagate_const<T> has no reset() function
    eventPrincipal_ = std::make_unique<EventPrincipal>(
                                             secInput_->productRegistry(),
                                             std::make_shared<BranchIDListHelper>(),
                                             std::make_shared<ThinnedAssociationsHelper>(),
                                             *processConfiguration_,
                                             nullptr);

  }

  // Virtual destructor needed.
  SecondaryProducer::~SecondaryProducer() {}

  // Functions that get called by framework every event
  void SecondaryProducer::produce(Event& e, EventSetup const&) {
    using std::placeholders::_1;
    size_t fileNameHash = 0U;

    if(specified_) {
      // Just for simplicity, we use the event ID from the primary to read the secondary.
      std::vector<SecondaryEventIDAndFileInfo> events(1, SecondaryEventIDAndFileInfo(e.id(), fileNameHash));
      secInput_->loopSpecified(*eventPrincipal_, fileNameHash, events.begin(), events.end(), std::bind(&SecondaryProducer::processOneEvent, this, _1, std::ref(e)));
    } else {
      CLHEP::HepRandomEngine* engine = nullptr;
      if (!sequential_) {
        edm::Service<edm::RandomNumberGenerator> rng;
        if (!rng.isAvailable()) {
          throw cms::Exception("Configuration")
            << "SecondaryProducer requires the RandomNumberGeneratorService,\n"
               "which is not present in the configuration file.  You must add the service\n"
               "in the configuration file or remove the modules that require it.";
        }
        engine = &rng->getEngine(e.streamID());
      }
      // Just for simplicity, we use the event ID from the primary to read the secondary.
      EventID id = e.id();
      secInput_->loopOverEvents(*eventPrincipal_, fileNameHash, 1, std::bind(&SecondaryProducer::processOneEvent, this, _1, std::ref(e)), engine, &id);
    }
  }

  void SecondaryProducer::processOneEvent(EventPrincipal const& eventPrincipal, Event& e) {
    typedef edmtest::ThingCollection TC;
    typedef Wrapper<TC> WTC;

    EventNumber_t en = eventPrincipal.id().event();
    // Check that secondary source products are retrieved from the same event as the EventAuxiliary
    BasicHandle bhandle = eventPrincipal.getByLabel(PRODUCT_TYPE, TypeID(typeid(edmtest::UInt64Product)),
                                                    "EventNumber",
                                                    "",
                                                    "",
                                                    nullptr,
                                                    nullptr,
                                                    nullptr);
    assert(bhandle.isValid());
    Handle<edmtest::UInt64Product> handle;
    convert_handle<edmtest::UInt64Product>(std::move(bhandle), handle);
    assert(static_cast<EventNumber_t>(handle->value) == en);

    // Check that primary source products are retrieved from the same event as the EventAuxiliary
    e.getByLabel<edmtest::UInt64Product>("EventNumber", handle);
    assert(static_cast<EventNumber_t>(handle->value) == e.id().event());

    WrapperBase const* ep = eventPrincipal.getByLabel(PRODUCT_TYPE, TypeID(typeid(TC)),
                                               "Thing",
                                               "",
                                               "",
                                               nullptr,
                                               nullptr,
                                               nullptr).wrapper();
    assert(ep != nullptr);
    WTC const* wtp = static_cast<WTC const*>(ep);
    assert(wtp);
    TC const* tp = wtp->product();
    std::unique_ptr<TC> thing(new TC(*tp));

    // Put output into event
    e.put(std::move(thing));

    if(!sequential_ && !specified_ && firstLoop_ && en == 1) {
      expectedEventNumber_ = 1;
      firstLoop_ = false;
    }
    if(firstEvent_) {
      firstEvent_ = false;
      if(!sequential_ && !specified_) {
        expectedEventNumber_ = en;
      }
    }
    assert (expectedEventNumber_ == en);
    ++expectedEventNumber_;
  }

  std::shared_ptr<VectorInputSource> SecondaryProducer::makeSecInput(ParameterSet const& ps) {
    ParameterSet const& sec_input = ps.getParameterSet("input");
    PreallocationConfiguration dummy;
    VectorInputSourceDescription desc(productRegistry(), dummy);
    std::shared_ptr<VectorInputSource> input_(static_cast<VectorInputSource *>
      (VectorInputSourceFactory::get()->makeVectorInputSource(sec_input, desc).release()));
    return input_;
  }

  void SecondaryProducer::endJob() {
    secInput_->doEndJob();
  }

} //edm
using edm::SecondaryProducer;
DEFINE_FWK_MODULE(SecondaryProducer);
