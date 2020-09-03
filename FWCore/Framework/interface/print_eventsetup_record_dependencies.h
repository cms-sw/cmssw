#ifndef FWCore_Framework_print_eventsetup_record_dependencies_h
#define FWCore_Framework_print_eventsetup_record_dependencies_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     print_eventsetup_record_dependencies
//
/**\class print_eventsetup_record_dependencies print_eventsetup_record_dependencies.h FWCore/Framework/interface/print_eventsetup_record_dependencies.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Apr 27 13:40:06 CDT 2009
//

// user include files
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/DependentRecordTag.h"
#include "FWCore/Utilities/interface/mplVector.h"

// system include files

#include <iostream>
#include <string>

// forward declarations

namespace edm {
  namespace eventsetup {
    struct DependentRecordTag;
  }
  template <typename RecordT>
  void print_eventsetup_record_dependencies(std::ostream& oStream, std::string const& iIndent = std::string());

  template <typename TFirst, typename TRemaining>
  void print_eventsetup_record_dependencies(std::ostream& oStream,
                                            std::string iIndent,
                                            TFirst const*,
                                            TRemaining const*) {
    iIndent += " ";
    print_eventsetup_record_dependencies<TFirst>(oStream, iIndent);

    using Pop = edm::mpl::Pop<TRemaining>;
    if constexpr (not Pop::empty) {
      const typename Pop::Item* next(nullptr);
      const typename Pop::Remaining* remaining(nullptr);

      print_eventsetup_record_dependencies(oStream, iIndent, next, remaining);
    }
  }

  template <typename RecordT>
  void print_eventsetup_record_dependencies(std::ostream& oStream, std::string const& iIndent) {
    oStream << iIndent << edm::eventsetup::EventSetupRecordKey::makeKey<RecordT>().name() << std::endl;

    if constexpr (std::is_base_of_v<edm::eventsetup::DependentRecordTag, RecordT>) {
      using list_type = typename RecordT::list_type;
      using Pop = edm::mpl::Pop<list_type>;

      const typename Pop::Item* begin(nullptr);
      const typename Pop::Remaining* remaining(nullptr);

      print_eventsetup_record_dependencies(oStream, iIndent, begin, remaining);
    }
  }
}  // namespace edm

#endif
