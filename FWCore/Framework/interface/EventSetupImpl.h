#ifndef FWCore_Framework_EventSetupImpl_h
#define FWCore_Framework_EventSetupImpl_h
// -*- C++ -*-
//
// Package:     Framework
// Class:      EventSetupImpl
//
/**\class EventSetupImpl EventSetupImpl.h FWCore/Framework/interface/EventSetupImpl.h

 Description: Container for all Records dealing with non-RunState info

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Mar 24 13:50:04 EST 2005
//

// system include files
#include <map>
#include <optional>
#include <vector>

// user include files
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Utilities/interface/ESIndices.h"

// forward declarations

namespace edm {
  class ActivityRegistry;
  class ESInputTag;

  namespace eventsetup {
    class EventSetupProvider;
    class EventSetupRecordImpl;
  }  // namespace eventsetup

  class EventSetupImpl {
    ///Only EventSetupProvider allowed to create a EventSetup
    friend class eventsetup::EventSetupProvider;

  public:
    ~EventSetupImpl();

    EventSetupImpl(EventSetupImpl const&) = delete;
    EventSetupImpl& operator=(EventSetupImpl const&) = delete;

    // ---------- const member functions ---------------------
    eventsetup::EventSetupRecordImpl const* findImpl(const eventsetup::EventSetupRecordKey&) const;

    std::optional<eventsetup::EventSetupRecordGeneric> find(const eventsetup::EventSetupRecordKey&,
                                                            unsigned int iTransitionID,
                                                            ESProxyIndex const* getTokenIndices) const;

    ///clears the oToFill vector and then fills it with the keys for all available records
    void fillAvailableRecordKeys(std::vector<eventsetup::EventSetupRecordKey>& oToFill) const;

    ///returns true if the Record is provided by a Source or a Producer
    /// a value of true does not mean this EventSetup object holds such a record
    bool recordIsProvidedByAModule(eventsetup::EventSetupRecordKey const&) const;
    // ---------- static member functions --------------------

    friend class eventsetup::EventSetupRecordImpl;

  protected:
    void add(const eventsetup::EventSetupRecordImpl& iRecord);

    void clear();

  private:
    EventSetupImpl(ActivityRegistry const*);

    ActivityRegistry const* activityRegistry() const { return activityRegistry_; }

    void insert(const eventsetup::EventSetupRecordKey&, const eventsetup::EventSetupRecordImpl*);

    void setKeyIters(std::vector<eventsetup::EventSetupRecordKey>::const_iterator const& keysBegin,
                     std::vector<eventsetup::EventSetupRecordKey>::const_iterator const& keysEnd);

    // ---------- member data --------------------------------

    std::vector<eventsetup::EventSetupRecordKey>::const_iterator keysBegin_;
    std::vector<eventsetup::EventSetupRecordKey>::const_iterator keysEnd_;
    std::vector<eventsetup::EventSetupRecordImpl const*> recordImpls_;
    ActivityRegistry const* activityRegistry_;
  };

}  // namespace edm

#endif  // FWCore_Framework_EventSetup_h
