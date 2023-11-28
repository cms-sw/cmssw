#ifndef FWCore_Framework_EventSetupRecordDetails_h
#define FWCore_Framework_EventSetupRecordDetails_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EventSetupRecordDetails
//
/**\class EventSetupRecordDetails EventSetupRecordDetails.h "__incdir__EventSetupRecordDetails.h"

 Description: Provides access to meta data of an EventSetupRecord.

 Usage:
      This is used in the ESSourceProductResolverBase user interface.

*/
//
// Original Author:  Chris Jones
//         Created:  18 May 2020
//

// system include files

// user include files
#include "FWCore/Framework/interface/EventSetupRecordImpl.h"

// forward declarations

namespace edm {
  class EventSetupRecordDetails {
  public:
    EventSetupRecordDetails() = delete;
    explicit EventSetupRecordDetails(eventsetup::EventSetupRecordImpl const* iImpl) : m_impl(iImpl) {}

    // ---------- const member functions ---------------------
    ValidityInterval validityInterval() const { return m_impl->validityInterval(); }

    eventsetup::EventSetupRecordKey const& key() const { return m_impl->key(); }

    unsigned long long cacheIdentifier() const { return m_impl->cacheIdentifier(); }

    unsigned int iovIndex() const { return m_impl->iovIndex(); }

  private:
    // ---------- member data --------------------------------
    eventsetup::EventSetupRecordImpl const* m_impl;
  };
}  // namespace edm

#endif
