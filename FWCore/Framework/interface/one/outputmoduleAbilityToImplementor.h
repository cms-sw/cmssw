#ifndef FWCore_Framework_one_outputmoduleAbilityToImplementor_h
#define FWCore_Framework_one_outputmoduleAbilityToImplementor_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     outputmodule::AbilityToImplementor
// 
/**\class outputmodule::AbilityToImplementor outputmoduleAbilityToImplementor.h "FWCore/Framework/interface/one/outputmoduleAbilityToImplementor.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 09 May 2013 19:39:58 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/moduleAbilities.h"
#include "FWCore/Framework/interface/one/moduleAbilities.h"
#include "FWCore/Framework/interface/one/implementors.h"
#include "FWCore/Framework/interface/one/OutputModuleBase.h"

// forward declarations

namespace edm {
  class FileBlock;
  class ModuleCallingContext;
  
  namespace one {
    namespace outputmodule {
      class RunWatcher : public virtual OutputModuleBase {
      public:
        RunWatcher(edm::ParameterSet const&iPSet): OutputModuleBase(iPSet){}
        RunWatcher(RunWatcher const&) = delete;
        RunWatcher& operator=(RunWatcher const&) = delete;
        ~RunWatcher() noexcept(false) {};
        
      private:
        virtual void doBeginRun_(RunForOutput const& r) override final;
        virtual void doEndRun_(RunForOutput const& r) override final;
        
        virtual void beginRun(edm::RunForOutput const&) = 0;
        virtual void endRun(edm::RunForOutput const&) = 0;
      };
      
      class LuminosityBlockWatcher : public virtual OutputModuleBase {
      public:
        LuminosityBlockWatcher(edm::ParameterSet const&iPSet): OutputModuleBase(iPSet) {}
        LuminosityBlockWatcher(LuminosityBlockWatcher const&) = delete;
        LuminosityBlockWatcher& operator=(LuminosityBlockWatcher const&) = delete;
        ~LuminosityBlockWatcher() noexcept(false) {};
        
      private:
        virtual void doBeginLuminosityBlock_(LuminosityBlockForOutput const& lb) override final;
        virtual void doEndLuminosityBlock_(LuminosityBlockForOutput const& lb) override final;
        
        virtual void beginLuminosityBlock(edm::LuminosityBlockForOutput const&) = 0;
        virtual void endLuminosityBlock(edm::LuminosityBlockForOutput const&) = 0;
      };

      class InputFileWatcher : public virtual OutputModuleBase {
      public:
        InputFileWatcher(edm::ParameterSet const&iPSet): OutputModuleBase(iPSet) {}
        InputFileWatcher(InputFileWatcher const&) = delete;
        InputFileWatcher& operator=(InputFileWatcher const&) = delete;
        ~InputFileWatcher() noexcept(false) {};
        
      private:
        void doRespondToOpenInputFile_(FileBlock const&) override final;
        void doRespondToCloseInputFile_(FileBlock const&) override final;
        
        virtual void respondToOpenInputFile(FileBlock const&) = 0;
        virtual void respondToCloseInputFile(FileBlock const&) = 0;
      };
      
      template<typename T> struct AbilityToImplementor;
      
      template<>
      struct AbilityToImplementor<edm::one::SharedResources> {
        typedef edm::one::impl::SharedResourcesUser<edm::one::OutputModuleBase> Type;
      };
      
      template<>
      struct AbilityToImplementor<edm::one::WatchRuns> {
        typedef edm::one::outputmodule::RunWatcher Type;
      };

      template<>
      struct AbilityToImplementor<edm::one::WatchLuminosityBlocks> {
        typedef edm::one::outputmodule::LuminosityBlockWatcher Type;
      };

      template<>
      struct AbilityToImplementor<edm::WatchInputFiles> {
        typedef edm::one::outputmodule::InputFileWatcher Type;
      };
}
  }
}


#endif
