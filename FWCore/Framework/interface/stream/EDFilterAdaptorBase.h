#ifndef FWCore_Framework_stream_EDFilterAdaptorBase_h
#define FWCore_Framework_stream_EDFilterAdaptorBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EDFilterAdaptorBase
// 
/**\class edm::stream::EDFilterAdaptorBase EDFilterAdaptorBase.h "FWCore/Framework/interface/stream/EDFilterAdaptorBase.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 18:09:15 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/stream/ProducingModuleAdaptorBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"


// forward declarations

namespace edm {

  class ModuleCallingContext;
  class ActivityRegistry;
  class WaitingTask;
  
  namespace maker {
    template<typename T> class ModuleHolderT;
  }

  namespace stream {
    class EDFilterBase;
    class EDFilterAdaptorBase : public ProducingModuleAdaptorBase<EDFilterBase>
    {
      
    public:
      template <typename T> friend class edm::maker::ModuleHolderT;
      template <typename T> friend class edm::WorkerT;

      EDFilterAdaptorBase();
      
      // ---------- const member functions ---------------------
      
      // ---------- static member functions --------------------
      
      // ---------- member functions ---------------------------
      
      std::string workerType() const { return "WorkerT<EDFilterAdaptorBase>";}
    protected:
      using ProducingModuleAdaptorBase<EDFilterBase>::commit;

    private:
      EDFilterAdaptorBase(const EDFilterAdaptorBase&) =delete; // stop default
      
      const EDFilterAdaptorBase& operator=(const EDFilterAdaptorBase&) =delete; // stop default
      
      bool doEvent(EventPrincipal const& ep, EventSetup const& c,
                   ActivityRegistry*,
                   ModuleCallingContext const*) ;
      //For now this is a placeholder
      /*virtual*/ void preActionBeforeRunEventAsync(WaitingTask* iTask, ModuleCallingContext const& iModuleCallingContext, Principal const& iPrincipal) const {}

    };
  }
}

#endif
