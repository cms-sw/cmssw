#ifndef FWCore_Framework_limited_outputmoduleAbilityToImplementor_h
#define FWCore_Framework_limited_outputmoduleAbilityToImplementor_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     outputmodule::AbilityToImplementor
// 
/**\class outputmodule::AbilityToImplementor outputmoduleAbilityToImplementor.h "FWCore/Framework/interface/limited/outputmoduleAbilityToImplementor.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
//

// system include files

// user include files
#include "FWCore/Framework/interface/moduleAbilities.h"
#include "FWCore/Framework/interface/limited/implementors.h"
#include "FWCore/Framework/interface/limited/OutputModuleBase.h"

// forward declarations

namespace edm {
  class FileBlock;
  class ModuleCallingContext;
  
  namespace limited {
    namespace outputmodule {
      class InputFileWatcher : public virtual OutputModuleBase {
      public:
        InputFileWatcher(edm::ParameterSet const&iPSet): OutputModuleBase(iPSet) {}
        InputFileWatcher(InputFileWatcher const&) = delete;
        InputFileWatcher& operator=(InputFileWatcher const&) = delete;
        ~InputFileWatcher() noexcept(false) override {};
        
      private:
        void doRespondToOpenInputFile_(FileBlock const&) final;
        void doRespondToCloseInputFile_(FileBlock const&) final;
        
        virtual void respondToOpenInputFile(FileBlock const&) = 0;
        virtual void respondToCloseInputFile(FileBlock const&) = 0;
      };
      
      template<typename T> struct AbilityToImplementor;
      
      template<>
      struct AbilityToImplementor<edm::WatchInputFiles> {
        typedef edm::limited::outputmodule::InputFileWatcher Type;
      };

      template<typename C>
      struct AbilityToImplementor<edm::StreamCache<C>> {
        typedef edm::limited::impl::StreamCacheHolder<edm::limited::OutputModuleBase,C> Type;
      };

      template<typename C>
      struct AbilityToImplementor<edm::RunCache<C>> {
        typedef edm::limited::impl::RunCacheHolder<edm::limited::OutputModuleBase,C> Type;
      };

      template<typename C>
      struct AbilityToImplementor<edm::RunSummaryCache<C>> {
        typedef edm::limited::impl::RunSummaryCacheHolder<edm::limited::OutputModuleBase,C> Type;
      };

      template<typename C>
      struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
        typedef edm::limited::impl::LuminosityBlockCacheHolder<edm::limited::OutputModuleBase,C> Type;
      };

      template<typename C>
      struct AbilityToImplementor<edm::LuminosityBlockSummaryCache<C>> {
        typedef edm::limited::impl::LuminosityBlockSummaryCacheHolder<edm::limited::OutputModuleBase,C> Type;
      };

    }
  }
}


#endif
