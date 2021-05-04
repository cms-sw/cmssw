#ifndef FWCore_Framework_DependentRecordIntervalFinder_h
#define FWCore_Framework_DependentRecordIntervalFinder_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DependentRecordIntervalFinder
//
/**\class edm::eventsetup::DependentRecordIntervalFinder

 Description: Finds the intersection of the ValidityInterval for several Providers

 Usage:
    This class is used internally to a EventSetupRecordProvider which delivers a Record that is dependent on other Records.

    If no Providers are given, then Finder will always report an invalid ValidityInterval for all IOVSyncValues

*/
//
// Author:      Chris Jones
// Created:     Sat Apr 30 19:36:59 EDT 2005
//

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Utilities/interface/propagate_const.h"

// forward declarations
namespace edm {
  namespace eventsetup {
    class EventSetupRecordProvider;

    class DependentRecordIntervalFinder : public EventSetupRecordIntervalFinder {
    public:
      DependentRecordIntervalFinder(const EventSetupRecordKey&);
      DependentRecordIntervalFinder(const DependentRecordIntervalFinder&) = delete;
      const DependentRecordIntervalFinder& operator=(const DependentRecordIntervalFinder&) = delete;
      ~DependentRecordIntervalFinder() override;

      // ---------- const member functions ---------------------
      bool haveProviders() const { return !providers_.empty(); }

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void addProviderWeAreDependentOn(std::shared_ptr<EventSetupRecordProvider>);

      void setAlternateFinder(std::shared_ptr<EventSetupRecordIntervalFinder>);

    protected:
      void setIntervalFor(const EventSetupRecordKey&, const IOVSyncValue&, ValidityInterval&) override;

    private:
      void doResetInterval(const eventsetup::EventSetupRecordKey&) override;

      // Should never be called for this class
      bool isConcurrentFinder() const override;

      bool isNonconcurrentAndIOVNeedsUpdate(const EventSetupRecordKey&, const IOVSyncValue&) const override;
      // ---------- member data --------------------------------
      typedef std::vector<edm::propagate_const<std::shared_ptr<EventSetupRecordProvider>>> Providers;
      Providers providers_;

      edm::propagate_const<std::shared_ptr<EventSetupRecordIntervalFinder>> alternate_;
      std::vector<ValidityInterval> previousIOVs_;
    };

  }  // namespace eventsetup
}  // namespace edm
#endif
