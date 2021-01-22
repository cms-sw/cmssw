// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecord
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Sat Mar 26 18:06:32 EST 2005
//

// system include files
#include <sstream>

// user include files
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/ComponentDescription.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace {
  void throwWrongRecordType(const edm::TypeIDBase& aFromToken, const edm::eventsetup::EventSetupRecordKey& aRecord) {
    throw cms::Exception("WrongRecordType") << "A ESGetTokenGeneric token using the record " << aFromToken.name()
                                            << " was passed to record " << aRecord.type().name();
  }
}  // namespace

namespace edm {
  namespace eventsetup {

    typedef std::map<DataKey, const DataProxy*> Proxies;

    EventSetupRecord::EventSetupRecord() {}

    EventSetupRecord::~EventSetupRecord() {}

    bool EventSetupRecord::doGet(const ESGetTokenGeneric& aToken, bool aGetTransiently) const {
      if UNLIKELY (aToken.transitionID() != transitionID()) {
        throwWrongTransitionID();
      }
      if UNLIKELY (aToken.recordType() != key().type()) {
        throwWrongRecordType(aToken.recordType(), key());
      }
      auto proxyIndex = getTokenIndices_[aToken.index().value()];
      if UNLIKELY (proxyIndex.value() == std::numeric_limits<int>::max()) {
        return false;
      }

      const ComponentDescription* cd = nullptr;
      DataKey const* dk = nullptr;
      return nullptr != impl_->getFromProxyAfterPrefetch(proxyIndex, aGetTransiently, cd, dk);
    }

    bool EventSetupRecord::wasGotten(const DataKey& aKey) const { return impl_->wasGotten(aKey); }

    edm::eventsetup::ComponentDescription const* EventSetupRecord::providerDescription(const DataKey& aKey) const {
      return impl_->providerDescription(aKey);
    }

    void EventSetupRecord::validate(const ComponentDescription* iDesc, const ESInputTag& iTag) const {
      if (iDesc && !iTag.module().empty()) {
        bool matched = false;
        if (iDesc->label_.empty()) {
          matched = iDesc->type_ == iTag.module();
        } else {
          matched = iDesc->label_ == iTag.module();
        }
        if (!matched) {
          throw cms::Exception("EventSetupWrongModule")
              << "EventSetup data was retrieved using an ESInputTag with the values\n"
              << "  moduleLabel = '" << iTag.module() << "'\n"
              << "  dataLabel = '" << iTag.data() << "'\n"
              << "but the data matching the C++ class type and dataLabel comes from module type=" << iDesc->type_
              << " label='" << iDesc->label_ << "'.\n Please either change the ESInputTag's 'module' label to be "
              << (iDesc->label_.empty() ? iDesc->type_ : iDesc->label_) << "\n or add the EventSetup module "
              << iTag.module() << " to the configuration.";
        }
      }
    }

    void EventSetupRecord::addTraceInfoToCmsException(cms::Exception& iException,
                                                      const char* iName,
                                                      const ComponentDescription* iDescription,
                                                      const DataKey& iKey) const {
      std::ostringstream ost;
      ost << "Using EventSetup component " << iDescription->type_ << "/'" << iDescription->label_ << "' to make data "
          << iKey.type().name() << "/'" << iName << "' in record " << this->key().type().name();
      iException.addContext(ost.str());
    }

    std::exception_ptr EventSetupRecord::makeUninitializedTokenException(EventSetupRecordKey const& iRecordKey,
                                                                         TypeTag const& iDataKey) {
      cms::Exception ex("InvalidESGetToken");
      ex << "Attempted to get data using an invalid token of type ESGetToken<" << iDataKey.name() << ","
         << iRecordKey.name()
         << ">.\n"
            "Please call consumes to properly initialize the token.";
      return std::make_exception_ptr(ex);
    }

    std::exception_ptr EventSetupRecord::makeInvalidTokenException(EventSetupRecordKey const& iRecordKey,
                                                                   TypeTag const& iDataKey,
                                                                   unsigned int iTransitionID) {
      cms::Exception ex("InvalidESGetToken");
      ex << "Attempted to get data using an invalid token of type ESGetToken<" << iDataKey.name() << ","
         << iRecordKey.name() << "> that had transition ID set (" << iTransitionID
         << ") but not the index.\n"
            "This should not happen, please contact core framework developers.";
      return std::make_exception_ptr(ex);
    }

    void EventSetupRecord::throwWrongTransitionID() const {
      cms::Exception ex("ESGetTokenWrongTransition");
      ex << "The transition ID stored in the ESGetToken does not match the\n"
         << "transition where the token is being used. The associated record\n"
         << "type is: " << key().type().name() << "\n"
         << "For producers, filters and analyzers this transition ID is\n"
         << "set as a template parameter to the call to the esConsumes\n"
         << "function that creates the token. Event is the default transition.\n"
         << "Other possibilities are BeginRun, EndRun, BeginLuminosityBlock,\n"
         << "or EndLuminosityBlock. You may need multiple tokens if you want to\n"
         << "get the same data in multiple transitions. The transition ID has a\n"
         << "different meaning in ESProducers. For ESProducers, the transition\n"
         << "ID identifies the function that produces the EventSetup data (often\n"
         << "there is one function named produce but there can be multiple\n"
         << "functions with different names). For ESProducers, the ESGetToken\n"
         << "must be used in the function associated with the ESConsumesCollector\n"
         << "returned by the setWhatProduced function.";
      throw ex;
    }

    void EventSetupRecord::throwCalledGetWithoutToken(const char* iTypeName, const char* iLabel) {
      throw cms::Exception("MustUseESGetToken")
          << "Called EventSetupRecord::get without using a ESGetToken.\n While requesting data type:" << iTypeName
          << " label:'" << iLabel << "'";
    }

  }  // namespace eventsetup
}  // namespace edm
