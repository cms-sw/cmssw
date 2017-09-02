#ifndef FWCore_Framework_limited_EDAnalyzer_h
#define FWCore_Framework_limited_EDAnalyzer_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::limited::EDAnalyzer
// 
/**\class edm::limited::EDAnalyzer EDAnalyzer.h "FWCore/Framework/interface/limited/EDAnalyzer.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 18 Jul 2013 11:51:07 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/limited/analyzerAbilityToImplementor.h"
#include "FWCore/Framework/interface/moduleAbilities.h"

// forward declarations

namespace edm {
  namespace limited {
    template< typename... T>
    class EDAnalyzer : public analyzer::AbilityToImplementor<T>::Type...,
                       public virtual EDAnalyzerBase
    {
      
    public:
      EDAnalyzer(edm::ParameterSet const& iPSet) : EDAnalyzerBase(iPSet),analyzer::AbilityToImplementor<T>::Type(iPSet)...  {}
// We do this only in the case of the intel compiler as this might
// end up creating a lot of code bloat due to inline symbols being generated 
// in each DSO which uses this header.
#ifdef __INTEL_COMPILER
      virtual ~EDAnalyzer() {}
#endif
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
