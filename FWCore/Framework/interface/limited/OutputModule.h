#ifndef FWCore_Framework_limited_OutputModule_h
#define FWCore_Framework_limited_OutputModule_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::limited::OutputModule
// 
/**\class edm::limited::OutputModule OutputModule.h "FWCore/Framework/interface/limited/OutputModule.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
//

// system include files

// user include files
#include "FWCore/Framework/interface/limited/outputmoduleAbilityToImplementor.h"

// forward declarations
namespace edm {
  namespace limited {
    template< typename... T>
    class OutputModule : public virtual OutputModuleBase,
        public outputmodule::AbilityToImplementor<T>::Type...
    {
      
    public:
      OutputModule(edm::ParameterSet const& iPSet): OutputModuleBase(iPSet),
      outputmodule::AbilityToImplementor<T>::Type(iPSet)...
       {}
      // Required to work around ICC bug, but possible source of bloat in gcc.
      // We do this only in the case of the intel compiler as this might end up
      // creating a lot of code bloat due to inline symbols being generated in
      // each DSO which uses this header.
#ifdef __INTEL_COMPILER
      virtual ~OutputModule() = default;
#endif
      
      // ---------- const member functions ---------------------
      
      // ---------- static member functions --------------------
      
      // ---------- member functions ---------------------------
      
    private:
      OutputModule(const OutputModule&) = delete; // stop default
      
      const OutputModule& operator=(const OutputModule&) =delete; // stop default
      
      // ---------- member data --------------------------------
      
    };
  }
}

#endif
