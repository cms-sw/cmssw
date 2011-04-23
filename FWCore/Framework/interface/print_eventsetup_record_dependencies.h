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

// system include files
#include "boost/mpl/begin_end.hpp"
#include "boost/mpl/deref.hpp"
#include "boost/mpl/next.hpp"

#include <iostream>
#include <string>

// forward declarations

namespace edm {
   namespace eventsetup {
     struct DependentRecordTag;
   }
   template<typename RecordT>
   void print_eventsetup_record_dependencies(std::ostream& oStream, std::string const& iIndent = std::string());
   
   template<typename T>
   void print_eventsetup_record_dependencies(std::ostream&,
                                             std::string,
                                             T const*,
                                             T const*) { }

   template<typename TFirst, typename TEnd>
   void print_eventsetup_record_dependencies(std::ostream& oStream, 
                                             std::string iIndent,
                                             TFirst const*, TEnd const* iEnd) {
      iIndent +=" ";
      print_eventsetup_record_dependencies<typename boost::mpl::deref<TFirst>::type>(oStream,iIndent);
      typename boost::mpl::next< TFirst >::type const* next(0);
      print_eventsetup_record_dependencies(oStream, iIndent, next, iEnd);
   }
   
   namespace rec_dep {
      boost::mpl::false_ inherits_from_DependentRecordTag(void const*) { return boost::mpl::false_();}
      boost::mpl::true_ inherits_from_DependentRecordTag(edm::eventsetup::DependentRecordTag const*) { return boost::mpl::true_();}
   }

   template<typename RecordT>
   void print_eventsetup_record_dependencies_recursive(std::ostream& oStream, std::string const& iIndent, boost::mpl::true_) {
      typedef typename  RecordT::list_type list_type;
      
      typename boost::mpl::begin<list_type>::type const* begin(0);
      typename boost::mpl::end<list_type>::type const* end(0);
      print_eventsetup_record_dependencies(oStream, iIndent, begin, end);
   }

   template<typename RecordT>
   void print_eventsetup_record_dependencies_recursive(std::ostream&, std::string const&, boost::mpl::false_) {
      return;
   }
   
   template<typename RecordT>
   void print_eventsetup_record_dependencies(std::ostream& oStream, std::string const& iIndent) {
      oStream<<iIndent<<edm::eventsetup::EventSetupRecordKey::makeKey<RecordT>().name()<<std::endl;
      
      print_eventsetup_record_dependencies_recursive<RecordT>(oStream, iIndent, rec_dep::inherits_from_DependentRecordTag(static_cast<RecordT const*>(0)));
   }
}

#endif
