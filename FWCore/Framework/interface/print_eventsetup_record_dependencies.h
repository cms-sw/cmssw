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
// $Id$
//

// system include files
#include <iostream>
#include <string>
#include "boost/mpl/begin_end.hpp"
#include "boost/mpl/deref.hpp"
#include "boost/mpl/next.hpp"

// user include files
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

// forward declarations

namespace edm {
   template <typename RecordT>
   void print_eventsetup_record_dependencies(std::ostream& oStream, const std::string& iIndent = std::string());

   template< class TFirst, class TEnd>
   void print_eventsetup_record_dependencies(std::ostream& oStream, 
                                             std::string iIndent,
                                             const TFirst*, const TEnd* iEnd)  {
      iIndent +=" ";
      print_eventsetup_record_dependencies<typename boost::mpl::deref<TFirst>::type>(oStream,iIndent);
      const  typename boost::mpl::next< TFirst >::type * next(0);
      print_eventsetup_record_dependencies(oStream,iIndent, next, iEnd);
   }
   
   namespace rec_dep {
      boost::mpl::false_ inherits_from_DependentRecordTag(const void*) { return boost::mpl::false_();}
      boost::mpl::true_ inherits_from_DependentRecordTag(const edm::eventsetup::DependentRecordTag*) { return boost::mpl::true_();}
   }
   
   template< typename T>
   void print_eventsetup_record_dependencies(std::ostream& oStream,
                                             std::string,
                                             const T*,
                                             const T*)  { }

   template <typename RecordT>
   void print_eventsetup_record_dependencies_recursive(std::ostream& oStream, const std::string& iIndent, boost::mpl::true_) {
      typedef typename  RecordT::list_type list_type;
      
      const  typename boost::mpl::begin<list_type>::type * begin(0);
      const  typename boost::mpl::end<list_type>::type * end(0);
      print_eventsetup_record_dependencies(oStream, iIndent,begin,end);
   }

   template <typename RecordT>
   void print_eventsetup_record_dependencies_recursive(std::ostream& oStream, const std::string&, boost::mpl::false_) {
      return;
   }
   
   template <typename RecordT>
   void print_eventsetup_record_dependencies(std::ostream& oStream, const std::string& iIndent) {
      oStream<<iIndent<<edm::eventsetup::EventSetupRecordKey::makeKey<RecordT>().name()<<std::endl;
      
      print_eventsetup_record_dependencies_recursive<RecordT>(oStream, iIndent, rec_dep::inherits_from_DependentRecordTag(static_cast<const RecordT*>(0)));
   }
}


#endif
