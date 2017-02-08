#ifndef FWCore_Framework_global_outputmoduleAbilityToImplementor_h
#define FWCore_Framework_global_outputmoduleAbilityToImplementor_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     outputmodule::AbilityToImplementor
// 
/**\class outputmodule::AbilityToImplementor outputmoduleAbilityToImplementor.h "FWCore/Framework/interface/global/outputmoduleAbilityToImplementor.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
//

// system include files

// user include files
#include "FWCore/Framework/interface/moduleAbilities.h"
#include "FWCore/Framework/interface/global/implementors.h"
#include "FWCore/Framework/interface/global/OutputModuleBase.h"

// forward declarations

namespace edm {
  class FileBlock;
  class ModuleCallingContext;
  
  namespace global {
    namespace outputmodule {
      class InputFileWatcher : public virtual OutputModuleBase {
      public:
        InputFileWatcher(edm::ParameterSet const&iPSet): OutputModuleBase(iPSet) {}
        InputFileWatcher(InputFileWatcher const&) = delete;
        InputFileWatcher& operator=(InputFileWatcher const&) = delete;
        
      private:
        void doRespondToOpenInputFile_(FileBlock const&) override final;
        void doRespondToCloseInputFile_(FileBlock const&) override final;
        
        virtual void respondToOpenInputFile(FileBlock const&) = 0;
        virtual void respondToCloseInputFile(FileBlock const&) = 0;
      };
      
      template<typename T> struct AbilityToImplementor;
      
      template<>
      struct AbilityToImplementor<edm::WatchInputFiles> {
        typedef edm::global::outputmodule::InputFileWatcher Type;
      };

      template<typename C>
      struct AbilityToImplementor<edm::StreamCache<C>> {
        typedef edm::global::impl::StreamCacheHolder<edm::global::OutputModuleBase,C> Type;
      };

      template<typename C>
      struct AbilityToImplementor<edm::RunCache<C>> {
        typedef edm::global::impl::RunCacheHolder<edm::global::OutputModuleBase,C> Type;
      };

      template<typename C>
      struct AbilityToImplementor<edm::RunSummaryCache<C>> {
        typedef edm::global::impl::RunSummaryCacheHolder<edm::global::OutputModuleBase,C> Type;
      };

      template<typename C>
      struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
        typedef edm::global::impl::LuminosityBlockCacheHolder<edm::global::OutputModuleBase,C> Type;
      };

      template<typename C>
      struct AbilityToImplementor<edm::LuminosityBlockSummaryCache<C>> {
        typedef edm::global::impl::LuminosityBlockSummaryCacheHolder<edm::global::OutputModuleBase,C> Type;
      };

    }
  }
}


#endif
