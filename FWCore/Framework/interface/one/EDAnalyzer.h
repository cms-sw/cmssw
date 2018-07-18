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
    class EDAnalyzer : public virtual EDAnalyzerBase,
                       public analyzer::AbilityToImplementor<T>::Type... { 
    public:
      EDAnalyzer() = default;
#ifdef __INTEL_COMPILER
      virtual ~EDAnalyzer() = default;
#endif
      
      // ---------- const member functions ---------------------
      bool wantsGlobalRuns() const final {
        return WantsGlobalRunTransitions<T...>::value;
      }
      bool wantsGlobalLuminosityBlocks() const final {
        return WantsGlobalLuminosityBlockTransitions<T...>::value;
      }

      // ---------- static member functions --------------------
      
      // ---------- member functions ---------------------------
      SerialTaskQueue* globalRunsQueue() final { return globalRunsQueue_.queue();}
      SerialTaskQueue* globalLuminosityBlocksQueue() final { return globalLuminosityBlocksQueue_.queue();}

    private:
      EDAnalyzer(const EDAnalyzer&) = delete;
      const EDAnalyzer& operator=(const EDAnalyzer&) = delete;
      
      // ---------- member data --------------------------------
      impl::OptionalSerialTaskQueueHolder<WantsGlobalRunTransitions<T...>::value> globalRunsQueue_;
      impl::OptionalSerialTaskQueueHolder<WantsGlobalLuminosityBlockTransitions<T...>::value> globalLuminosityBlocksQueue_;

    };
    
  }
}


#endif
