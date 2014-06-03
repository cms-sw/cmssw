#ifndef FWCore_Framework_DependentRecordImplementation_h
#define FWCore_Framework_DependentRecordImplementation_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DependentRecordImplementation
// 
/**\class DependentRecordImplementation DependentRecordImplementation.h FWCore/Framework/interface/DependentRecordImplementation.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Apr 29 10:03:54 EDT 2005
//

// system include files
#include "boost/mpl/begin_end.hpp"
#include "boost/mpl/find.hpp"

// user include files
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/NoRecordException.h"
#include "FWCore/Framework/interface/DependentRecordTag.h"

// forward declarations
namespace edm {
namespace eventsetup {
   
template< class RecordT, class ListT>
class DependentRecordImplementation : public EventSetupRecordImplementation<RecordT>, public DependentRecordTag
{

   public:
      DependentRecordImplementation() {}
      typedef ListT list_type;
      //virtual ~DependentRecordImplementation();
      
      // ---------- const member functions ---------------------
      template<class DepRecordT>
      const DepRecordT& getRecord() const {
        //Make sure that DepRecordT is a type in ListT
        typedef typename boost::mpl::end< ListT >::type EndItrT;
        typedef typename boost::mpl::find< ListT, DepRecordT>::type FoundItrT;
        BOOST_STATIC_ASSERT((! boost::is_same<FoundItrT, EndItrT>::value));
        try {
          EventSetup const& eventSetupT = this->eventSetup();
          return eventSetupT.get<DepRecordT>();
        } catch(NoRecordException<DepRecordT>&) {
          //rethrow but this time with dependent information.
          throw NoRecordException<DepRecordT>(this->key());
        } catch(cms::Exception& e) {
          e<<"Exception occurred while getting dependent record from record \""<<
          this->key().type().name()<<"\""<<std::endl;
          throw;
        }
      }

      template<class DepRecordT>
      const DepRecordT* tryToGetRecord() const {
        //Make sure that DepRecordT is a type in ListT
        typedef typename boost::mpl::end< ListT >::type EndItrT;
        typedef typename boost::mpl::find< ListT, DepRecordT>::type FoundItrT;
        BOOST_STATIC_ASSERT((! boost::is_same<FoundItrT, EndItrT>::value));
        EventSetup const& eventSetupT = this->eventSetup();
        return eventSetupT.tryToGet<DepRecordT>();
      }

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      DependentRecordImplementation(const DependentRecordImplementation&); // stop default

      const DependentRecordImplementation& operator=(const DependentRecordImplementation&); // stop default

      // ---------- member data --------------------------------

};

  }
}

#endif
