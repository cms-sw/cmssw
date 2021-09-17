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
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"

// forward declarations

namespace edm {
  class FileBlock;
  class ModuleCallingContext;

  namespace global {
    namespace outputmodule {

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

      template <typename T, typename C>
      class StreamCacheHolder : public virtual T {
      public:
        explicit StreamCacheHolder(edm::ParameterSet const& iPSet) : OutputModuleBase(iPSet) {}
        StreamCacheHolder(StreamCacheHolder<T, C> const&) = delete;
        StreamCacheHolder<T, C>& operator=(StreamCacheHolder<T, C> const&) = delete;
        ~StreamCacheHolder() override {
          for (auto c : caches_) {
            delete c;
          }
        }

      protected:
        C* streamCache(edm::StreamID iID) const { return caches_[iID.value()]; }

      private:
        void preallocStreams(unsigned int iNStreams) final { caches_.resize(iNStreams, static_cast<C*>(nullptr)); }
        void doBeginStream_(StreamID id) final { caches_[id.value()] = beginStream(id).release(); }
        void doEndStream_(StreamID id) final {
          endStream(id);
          delete caches_[id.value()];
          caches_[id.value()] = nullptr;
        }

        virtual std::unique_ptr<C> beginStream(edm::StreamID) const = 0;
        virtual void endStream(edm::StreamID) const {}

        //When threaded we will have a container for N items whre N is # of streams
        std::vector<C*> caches_;
      };

      template <typename T, typename C>
      class RunCacheHolder : public virtual T {
      public:
        RunCacheHolder(edm::ParameterSet const& iPSet) : OutputModuleBase(iPSet) {}
        RunCacheHolder(RunCacheHolder<T, C> const&) = delete;
        RunCacheHolder<T, C>& operator=(RunCacheHolder<T, C> const&) = delete;
        ~RunCacheHolder() noexcept(false) override{};

      protected:
        C const* runCache(edm::RunIndex iID) const { return cache_.get(); }

      private:
        void doBeginRun_(RunForOutput const& rp) final { cache_ = globalBeginRun(rp); }
        void doEndRun_(RunForOutput const& rp) final {
          globalEndRun(rp);
          cache_ = nullptr;  // propagate_const<T> has no reset() function
        }

        virtual std::shared_ptr<C> globalBeginRun(RunForOutput const&) const = 0;
        virtual void globalEndRun(RunForOutput const&) const = 0;
        //When threaded we will have a container for N items whre N is # of simultaneous runs
        edm::propagate_const<std::shared_ptr<C>> cache_;
      };

      template <typename T, typename C>
      class LuminosityBlockCacheHolder : public virtual T {
      public:
        LuminosityBlockCacheHolder(edm::ParameterSet const& iPSet) : OutputModuleBase(iPSet) {}
        LuminosityBlockCacheHolder(LuminosityBlockCacheHolder<T, C> const&) = delete;
        LuminosityBlockCacheHolder<T, C>& operator=(LuminosityBlockCacheHolder<T, C> const&) = delete;
        ~LuminosityBlockCacheHolder() noexcept(false) override{};

      protected:
        void preallocLumis(unsigned int iNLumis) final { caches_.reset(new std::shared_ptr<C>[iNLumis]); }
        C const* luminosityBlockCache(edm::LuminosityBlockIndex iID) const { return caches_[iID].get(); }

      private:
        void doBeginLuminosityBlock_(LuminosityBlockForOutput const& lp) final {
          caches_[lp.index()] = globalBeginLuminosityBlock(lp);
        }
        void doEndLuminosityBlock_(LuminosityBlockForOutput const& lp) final {
          globalEndLuminosityBlock(lp);
          caches_[lp.index()].reset();
        }

        virtual std::shared_ptr<C> globalBeginLuminosityBlock(LuminosityBlockForOutput const&) const = 0;
        virtual void globalEndLuminosityBlock(LuminosityBlockForOutput const&) const = 0;
        //When threaded we will have a container for N items whre N is # of simultaneous runs
        std::unique_ptr<std::shared_ptr<C>[]> caches_;
      };

      template <typename T>
      class ExternalWork : public virtual T {
      public:
        ExternalWork(edm::ParameterSet const& iPSet) : OutputModuleBase(iPSet) {}
        ExternalWork(ExternalWork const&) = delete;
        ExternalWork& operator=(ExternalWork const&) = delete;
        ~ExternalWork() noexcept(false) override{};

      private:
        bool hasAcquire() const override { return true; }

        void doAcquire_(StreamID id, EventForOutput const& event, WaitingTaskWithArenaHolder& holder) final {
          acquire(id, event, holder);
        }

        virtual void acquire(StreamID, EventForOutput const&, WaitingTaskWithArenaHolder) const = 0;
      };

      template <typename T>
      struct AbilityToImplementor;

      template <>
      struct AbilityToImplementor<edm::WatchInputFiles> {
        typedef edm::global::outputmodule::InputFileWatcher Type;
      };

      template <typename C>
      struct AbilityToImplementor<edm::RunCache<C>> {
        typedef edm::global::outputmodule::RunCacheHolder<edm::global::OutputModuleBase, C> Type;
      };

      template <typename C>
      struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
        typedef edm::global::outputmodule::LuminosityBlockCacheHolder<edm::global::OutputModuleBase, C> Type;
      };

      template <typename C>
      struct AbilityToImplementor<edm::StreamCache<C>> {
        typedef edm::global::outputmodule::StreamCacheHolder<edm::global::OutputModuleBase, C> Type;
      };

      template <>
      struct AbilityToImplementor<edm::ExternalWork> {
        typedef edm::global::outputmodule::ExternalWork<edm::global::OutputModuleBase> Type;
      };

    }  // namespace outputmodule
  }    // namespace global
}  // namespace edm

#endif
