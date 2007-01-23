#ifndef Framework_DependentRecordImplementation_h
#define Framework_DependentRecordImplementation_h
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
// $Id: DependentRecordImplementation.h,v 1.5 2005/09/01 23:30:48 wmtan Exp $
//

// system include files
#include "boost/static_assert.hpp"
#include "boost/mpl/begin_end.hpp"
#include "boost/mpl/find.hpp"

// user include files
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetup.h"

// forward declarations
namespace edm {
namespace eventsetup {
   struct DependentRecordTag {};
   
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
            EventSetup const& eventSetupT = this->eventSetup();
            //can't do the following because of a compiler error in gcc 3.*
            // return eventSetupT.get<DepRecordT>();
            const DepRecordT* temp;
            eventSetupT.getAvoidCompilerBug(temp);
            return *temp;
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
