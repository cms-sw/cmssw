#ifndef FWCore_Framework_NumberOfConcurrentIOVs_h
#define FWCore_Framework_NumberOfConcurrentIOVs_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     NumberOfConcurrentIOVs
//
/** \class edm::eventsetup::NumberOfConcurrentIOVs

 Description: Calculates and holds the number of concurrent
              intervals of validity allowed for each record
              in the EventSetup.

 Usage: Used internally by the Framework

*/
//
// Original Authors:  W. David Dagenhart
//          Created:  1 February 2019

#include "FWCore/Framework/interface/EventSetupRecordKey.h"

#include <set>
#include <utility>
#include <vector>

namespace edm {

  class ParameterSet;

  namespace eventsetup {

    class EventSetupProvider;

    class NumberOfConcurrentIOVs {
    public:
      NumberOfConcurrentIOVs();

      void readConfigurationParameters(ParameterSet const* eventSetupPset,
                                       unsigned int maxConcurrentIOVs,
                                       bool dumpOptions);

      // This depends on bool's hard coded in the EventSetupRecord C++ classes
      void fillRecordsNotAllowingConcurrentIOVs(EventSetupProvider const&);

      unsigned int numberOfConcurrentIOVs(EventSetupRecordKey const&, bool printInfoMsg = false) const;

      void clear();

    private:
      // This is the single value configured in the top level options
      // parameter set that determines the number of concurrent IOVs
      // allowed for each record. This value can be overridden by
      // either of the values of the next two data members. The default
      // for this is one.
      unsigned int numberConcurrentIOVs_;

      // This is derived from a boolean value that can be hard coded
      // into the C++ definition of a class deriving from EventSetupRecord.
      // The data member is a static constexpr member of the class and
      // this data member is named allowConcurrentIOVs_.
      // If the value is defined as false, then the key to the record
      // will get stored in the set defined below. This will be used to
      // to limit the number of concurrent IOVs for a particular record
      // to be one even if the overall "numberOfConcurrentIOVs" parameter
      // is configured to be greater than one.
      std::set<EventSetupRecordKey> recordsNotAllowingConcurrentIOVs_;

      // It is possible to individually configure the number of concurrent
      // IOVs allowed for each record. This is done by adding parameters
      // to a nested parameter set in the top level options parameter set.
      // Setting these parameters is optional. This container can hold an
      // entry for all records, it can be empty, or hold an entry for
      // any subset of records. Values in this container override both of the
      // above data members.
      std::vector<std::pair<EventSetupRecordKey, unsigned int>> forceNumberOfConcurrentIOVs_;

      unsigned int maxConcurrentIOVs_ = 1;
    };

  }  // namespace eventsetup
}  // namespace edm
#endif
