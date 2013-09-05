#ifndef FWCore_Framework_one_EDAnalyzer_h
#define FWCore_Framework_one_EDAnalyzer_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::one::EDAnalyzer
// 
/**\class edm::one::EDAnalyzer EDAnalyzer.h "FWCore/Framework/interface/one/EDAnalyzer.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 01 Aug 2013 19:53:55 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/one/analyzerAbilityToImplementor.h"

// forward declarations
namespace edm {
  namespace one {
    template< typename... T>
    class EDAnalyzer : public analyzer::AbilityToImplementor<T>::Type...,
                       public virtual EDAnalyzerBase {
      
    public:
      EDAnalyzer() = default;
      
      // ---------- const member functions ---------------------
      
      // ---------- static member functions --------------------
      
      // ---------- member functions ---------------------------
      
    private:
      EDAnalyzer(const EDAnalyzer&) = delete;
      const EDAnalyzer& operator=(const EDAnalyzer&) = delete;
      
      // ---------- member data --------------------------------
      
    };
    
  }
}


#endif
