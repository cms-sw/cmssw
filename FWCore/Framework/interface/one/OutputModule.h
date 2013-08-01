#ifndef FWCore_Framework_one_OutputModule_h
#define FWCore_Framework_one_OutputModule_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::one::OutputModule
// 
/**\class edm::one::OutputModule OutputModule.h "FWCore/Framework/interface/one/OutputModule.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed, 31 Jul 2013 18:21:29 GMT
// $Id$
//

// system include files

// user include files
#include "FWCore/Framework/interface/one/outputmoduleAbilityToImplementor.h"

// forward declarations
namespace edm {
  namespace one {
    template< typename... T>
    class OutputModule : public outputmodule::AbilityToImplementor<T>::Type...,
    public virtual OutputModuleBase
    {
      
    public:
      OutputModule(edm::ParameterSet const& iPSet): OutputModuleBase(iPSet),
      outputmodule::AbilityToImplementor<T>::Type(iPSet)...
       {}
      //virtual ~OutputModule();
      
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
