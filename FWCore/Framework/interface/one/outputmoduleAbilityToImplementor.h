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
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/RunForOutput.h"

// forward declarations

namespace edm {
  class FileBlock;
  class ModuleCallingContext;

  namespace one {
    namespace outputmodule {
      class RunWatcher : public virtual OutputModuleBase {
      public:
        RunWatcher(edm::ParameterSet const& iPSet) : OutputModuleBase(iPSet) {}
        RunWatcher(RunWatcher const&) = delete;
        RunWatcher& operator=(RunWatcher const&) = delete;
        ~RunWatcher() noexcept(false) override{};

      private:
        void doBeginRun_(RunForOutput const& r) final;
        void doEndRun_(RunForOutput const& r) final;

        virtual void beginRun(edm::RunForOutput const&) = 0;
        virtual void endRun(edm::RunForOutput const&) = 0;
      };

      class LuminosityBlockWatcher : public virtual OutputModuleBase {
      public:
        LuminosityBlockWatcher(edm::ParameterSet const& iPSet) : OutputModuleBase(iPSet) {}
        LuminosityBlockWatcher(LuminosityBlockWatcher const&) = delete;
        LuminosityBlockWatcher& operator=(LuminosityBlockWatcher const&) = delete;
        ~LuminosityBlockWatcher() noexcept(false) override{};

      private:
        void doBeginLuminosityBlock_(LuminosityBlockForOutput const& lb) final;
        void doEndLuminosityBlock_(LuminosityBlockForOutput const& lb) final;

        virtual void beginLuminosityBlock(edm::LuminosityBlockForOutput const&) = 0;
        virtual void endLuminosityBlock(edm::LuminosityBlockForOutput const&) = 0;
      };

      class InputFileWatcher : public virtual OutputModuleBase {
      public:
        InputFileWatcher(edm::ParameterSet const& iPSet) : OutputModuleBase(iPSet) {}
        InputFileWatcher(InputFileWatcher const&) = delete;
        InputFileWatcher& operator=(InputFileWatcher const&) = delete;
        ~InputFileWatcher() noexcept(false) override{};

      private:
        void doRespondToOpenInputFile_(FileBlock const&) final;
        void doRespondToCloseInputFile_(FileBlock const&) final;

        virtual void respondToOpenInputFile(FileBlock const&) = 0;
        virtual void respondToCloseInputFile(FileBlock const&) = 0;
      };

      template <typename C>
      class RunCacheHolder : public virtual OutputModuleBase {
      public:
        RunCacheHolder(edm::ParameterSet const& iPSet) : OutputModuleBase(iPSet) {}
        RunCacheHolder(RunCacheHolder<C> const&) = delete;
        RunCacheHolder<C>& operator=(RunCacheHolder<C> const&) = delete;
        ~RunCacheHolder() noexcept(false) override{};

      protected:
        C* runCache(edm::RunIndex iID) { return cache_.get(); }
        C const* runCache(edm::RunIndex iID) const { return cache_.get(); }

      private:
        void doBeginRun_(edm::RunForOutput const& rp) final { cache_ = globalBeginRun(rp); }
        void doEndRun_(edm::RunForOutput const& rp) final {
          globalEndRun(rp);
          cache_ = nullptr;  // propagate_const<T> has no reset() function
        }

        virtual std::shared_ptr<C> globalBeginRun(edm::RunForOutput const&) const = 0;
        virtual void globalEndRun(edm::RunForOutput const&) = 0;
        //When threaded we will have a container for N items whre N is # of simultaneous runs
        edm::propagate_const<std::shared_ptr<C>> cache_;
      };

      template <typename C>
      class LuminosityBlockCacheHolder : public virtual OutputModuleBase {
      public:
        template <typename... A>
        LuminosityBlockCacheHolder(edm::ParameterSet const& iPSet) : OutputModuleBase(iPSet) {}
        LuminosityBlockCacheHolder(LuminosityBlockCacheHolder<C> const&) = delete;
        LuminosityBlockCacheHolder<C>& operator=(LuminosityBlockCacheHolder<C> const&) = delete;
        ~LuminosityBlockCacheHolder() noexcept(false) override{};

      protected:
        void preallocLumis(unsigned int iNLumis) final { caches_.reset(new std::shared_ptr<C>[iNLumis]); }

        C const* luminosityBlockCache(edm::LuminosityBlockIndex iID) const { return caches_[iID].get(); }
        C* luminosityBlockCache(edm::LuminosityBlockIndex iID) { return caches_[iID].get(); }

      private:
        void doBeginLuminosityBlock_(edm::LuminosityBlockForOutput const& lp) final {
          caches_[lp.index()] = globalBeginLuminosityBlock(lp);
        }
        void doEndLuminosityBlock_(edm::LuminosityBlockForOutput const& lp) final {
          globalEndLuminosityBlock(lp);
          caches_[lp.index()].reset();
        }

        virtual std::shared_ptr<C> globalBeginLuminosityBlock(edm::LuminosityBlockForOutput const&) const = 0;
        virtual void globalEndLuminosityBlock(edm::LuminosityBlockForOutput const&) = 0;
        std::unique_ptr<std::shared_ptr<C>[]> caches_;
      };

      template <typename T>
      struct AbilityToImplementor;

      template <>
      struct AbilityToImplementor<edm::one::SharedResources> {
        typedef edm::one::impl::SharedResourcesUser<edm::one::OutputModuleBase> Type;
      };

      template <>
      struct AbilityToImplementor<edm::one::WatchRuns> {
        typedef edm::one::outputmodule::RunWatcher Type;
      };

      template <>
      struct AbilityToImplementor<edm::one::WatchLuminosityBlocks> {
        typedef edm::one::outputmodule::LuminosityBlockWatcher Type;
      };

      template <>
      struct AbilityToImplementor<edm::WatchInputFiles> {
        typedef edm::one::outputmodule::InputFileWatcher Type;
      };

      template <typename C>
      struct AbilityToImplementor<edm::RunCache<C>> {
        typedef edm::one::outputmodule::RunCacheHolder<C> Type;
      };

      template <typename C>
      struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
        typedef edm::one::outputmodule::LuminosityBlockCacheHolder<C> Type;
      };
    }  // namespace outputmodule
  }    // namespace one
}  // namespace edm

#endif
