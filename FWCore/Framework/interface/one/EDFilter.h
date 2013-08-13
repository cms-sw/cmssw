#ifndef FWCore_Framework_one_EDFilter_h
#define FWCore_Framework_one_EDFilter_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::one::EDFilter
// 
/**\class edm::one::EDFilter EDFilter.h "FWCore/Framework/interface/one/EDFilter.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 09 May 2013 19:53:55 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/one/filterAbilityToImplementor.h"

// forward declarations
namespace edm {
  namespace one {
    template< typename... T>
    class EDFilter : public filter::AbilityToImplementor<T>::Type...,
                       public virtual EDFilterBase {
      
    public:
      EDFilter() = default;
      //virtual ~EDFilter();
      
      // ---------- const member functions ---------------------
      
      // ---------- static member functions --------------------
      
      // ---------- member functions ---------------------------
      
    private:
      EDFilter(const EDFilter&) = delete;
      const EDFilter& operator=(const EDFilter&) = delete;
      
      // ---------- member data --------------------------------
      
    };
    
  }
}


#endif
