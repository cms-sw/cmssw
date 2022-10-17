#ifndef FWCore_Framework_one_implementors_h
#define FWCore_Framework_one_implementors_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     implementors
//
/**\file implementors.h "FWCore/Framework/interface/one/implementors.h"

 Description: Base classes used to implement the interfaces for the edm::one::* module  abilities

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 09 May 2013 18:40:17 GMT
//

// system include files
#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>

// user include files
#include "FWCore/Common/interface/FWCoreCommonFwd.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"
#include "FWCore/Framework/interface/CacheHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputProcessBlockCacheImpl.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/TransformerBase.h"
#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ProcessBlockIndex.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "DataFormats/Common/interface/Wrapper.h"

// forward declarations

namespace edm {
  class SharedResourcesAcquirer;
  class WaitingTaskHolder;
  class ServiceWeakToken;

  namespace one {
    namespace impl {
      template <bool V>
      struct OptionalSerialTaskQueueHolder;

      template <>
      struct OptionalSerialTaskQueueHolder<true> {
        edm::SerialTaskQueue* queue() { return &queue_; }
        edm::SerialTaskQueue queue_;
      };

      template <>
      struct OptionalSerialTaskQueueHolder<false> {
        edm::SerialTaskQueue* queue() { return nullptr; }
      };

      template <typename T>
      class SharedResourcesUser : public virtual T {
      public:
        template <typename... Args>
        SharedResourcesUser(Args... args) : T(args...) {}
        SharedResourcesUser(SharedResourcesUser const&) = delete;
        SharedResourcesUser& operator=(SharedResourcesUser const&) = delete;

        ~SharedResourcesUser() override {}

      protected:
        void usesResource(std::string const& iName);
        void usesResource();

      private:
        SharedResourcesAcquirer createAcquirer() override;
        std::set<std::string> resourceNames_;
      };

      template <typename T>
      class RunWatcher : public virtual T {
      public:
        RunWatcher() = default;
        RunWatcher(RunWatcher const&) = delete;
        RunWatcher& operator=(RunWatcher const&) = delete;
        ~RunWatcher() noexcept(false) override{};

      private:
        void doBeginRun_(Run const& rp, EventSetup const& c) final;
        void doEndRun_(Run const& rp, EventSetup const& c) final;

        virtual void beginRun(edm::Run const&, edm::EventSetup const&) = 0;
        virtual void endRun(edm::Run const&, edm::EventSetup const&) = 0;
      };

      template <typename T>
      class LuminosityBlockWatcher : public virtual T {
      public:
        LuminosityBlockWatcher() = default;
        LuminosityBlockWatcher(LuminosityBlockWatcher const&) = delete;
        LuminosityBlockWatcher& operator=(LuminosityBlockWatcher const&) = delete;
        ~LuminosityBlockWatcher() noexcept(false) override{};

      private:
        void doBeginLuminosityBlock_(LuminosityBlock const& rp, EventSetup const& c) final;
        void doEndLuminosityBlock_(LuminosityBlock const& rp, EventSetup const& c) final;

        virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) = 0;
        virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) = 0;
      };

      template <typename T>
      class WatchProcessBlock : public virtual T {
      public:
        WatchProcessBlock() = default;
        WatchProcessBlock(WatchProcessBlock const&) = delete;
        WatchProcessBlock& operator=(WatchProcessBlock const&) = delete;
        ~WatchProcessBlock() noexcept(false) override {}

      private:
        void doBeginProcessBlock_(ProcessBlock const&) final;
        void doEndProcessBlock_(ProcessBlock const&) final;

        virtual void beginProcessBlock(ProcessBlock const&) {}
        virtual void endProcessBlock(ProcessBlock const&) {}
      };

      template <typename T>
      class BeginProcessBlockProducer : public virtual T {
      public:
        BeginProcessBlockProducer() = default;
        BeginProcessBlockProducer(BeginProcessBlockProducer const&) = delete;
        BeginProcessBlockProducer& operator=(BeginProcessBlockProducer const&) = delete;
        ~BeginProcessBlockProducer() noexcept(false) override{};

      private:
        void doBeginProcessBlockProduce_(ProcessBlock&) final;

        virtual void beginProcessBlockProduce(edm::ProcessBlock&) = 0;
      };

      template <typename T>
      class EndProcessBlockProducer : public virtual T {
      public:
        EndProcessBlockProducer() = default;
        EndProcessBlockProducer(EndProcessBlockProducer const&) = delete;
        EndProcessBlockProducer& operator=(EndProcessBlockProducer const&) = delete;
        ~EndProcessBlockProducer() noexcept(false) override{};

      private:
        void doEndProcessBlockProduce_(ProcessBlock&) final;

        virtual void endProcessBlockProduce(edm::ProcessBlock&) = 0;
      };

      template <typename T>
      class BeginRunProducer : public virtual T {
      public:
        BeginRunProducer() = default;
        BeginRunProducer(BeginRunProducer const&) = delete;
        BeginRunProducer& operator=(BeginRunProducer const&) = delete;
        ~BeginRunProducer() noexcept(false) override{};

      private:
        void doBeginRunProduce_(Run& rp, EventSetup const& c) final;

        virtual void beginRunProduce(edm::Run&, edm::EventSetup const&) = 0;
      };

      template <typename T>
      class EndRunProducer : public virtual T {
      public:
        EndRunProducer() = default;
        EndRunProducer(EndRunProducer const&) = delete;
        EndRunProducer& operator=(EndRunProducer const&) = delete;
        ~EndRunProducer() noexcept(false) override{};

      private:
        void doEndRunProduce_(Run& rp, EventSetup const& c) final;

        virtual void endRunProduce(edm::Run&, edm::EventSetup const&) = 0;
      };

      template <typename T>
      class BeginLuminosityBlockProducer : public virtual T {
      public:
        BeginLuminosityBlockProducer() = default;
        BeginLuminosityBlockProducer(BeginLuminosityBlockProducer const&) = delete;
        BeginLuminosityBlockProducer& operator=(BeginLuminosityBlockProducer const&) = delete;
        ~BeginLuminosityBlockProducer() noexcept(false) override{};

      private:
        void doBeginLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) final;

        virtual void beginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) = 0;
      };

      template <typename T>
      class EndLuminosityBlockProducer : public virtual T {
      public:
        EndLuminosityBlockProducer() = default;
        EndLuminosityBlockProducer(EndLuminosityBlockProducer const&) = delete;
        EndLuminosityBlockProducer& operator=(EndLuminosityBlockProducer const&) = delete;
        ~EndLuminosityBlockProducer() noexcept(false) override{};

      private:
        void doEndLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) final;

        virtual void endLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) = 0;
      };

      template <typename T, typename... CacheTypes>
      class InputProcessBlockCacheHolder : public virtual T {
      public:
        InputProcessBlockCacheHolder() = default;
        InputProcessBlockCacheHolder(InputProcessBlockCacheHolder const&) = delete;
        InputProcessBlockCacheHolder& operator=(InputProcessBlockCacheHolder const&) = delete;
        ~InputProcessBlockCacheHolder() override {}

        std::tuple<CacheHandle<CacheTypes>...> processBlockCaches(Event const& event) const {
          return cacheImpl_.processBlockCaches(event);
        }

        template <std::size_t ICacheType, typename DataType, typename Func>
        void registerProcessBlockCacheFiller(EDGetTokenT<DataType> const& token, Func&& func) {
          cacheImpl_.template registerProcessBlockCacheFiller<ICacheType, DataType, Func>(token,
                                                                                          std::forward<Func>(func));
        }

        template <typename CacheType, typename DataType, typename Func>
        void registerProcessBlockCacheFiller(EDGetTokenT<DataType> const& token, Func&& func) {
          cacheImpl_.template registerProcessBlockCacheFiller<CacheType, DataType, Func>(token,
                                                                                         std::forward<Func>(func));
        }

        // This is intended for use by Framework unit tests only
        unsigned int cacheSize() const { return cacheImpl_.cacheSize(); }

      private:
        void doSelectInputProcessBlocks(ProductRegistry const& productRegistry,
                                        ProcessBlockHelperBase const& processBlockHelperBase) final {
          cacheImpl_.selectInputProcessBlocks(productRegistry, processBlockHelperBase, *this);
        }

        void doAccessInputProcessBlock_(ProcessBlock const& pb) final {
          cacheImpl_.accessInputProcessBlock(pb);
          accessInputProcessBlock(pb);
        }

        // Alternate method to access ProcessBlocks without using the caches
        // Mostly intended for unit testing, but might have other uses...
        virtual void accessInputProcessBlock(ProcessBlock const&) {}

        void clearInputProcessBlockCaches() final { cacheImpl_.clearCaches(); }

        edm::impl::InputProcessBlockCacheImpl<CacheTypes...> cacheImpl_;
      };

      template <typename T, typename C>
      class RunCacheHolder : public virtual T {
      public:
        RunCacheHolder() = default;
        RunCacheHolder(RunCacheHolder<T, C> const&) = delete;
        RunCacheHolder<T, C>& operator=(RunCacheHolder<T, C> const&) = delete;
        ~RunCacheHolder() noexcept(false) override{};

      protected:
        void preallocRuns(unsigned int iNRuns) final { caches_.reset(new std::shared_ptr<C>[iNRuns]); }

        C* runCache(edm::RunIndex iID) { return caches_[iID].get(); }
        C const* runCache(edm::RunIndex iID) const { return caches_[iID].get(); }

      private:
        void doBeginRun_(Run const& rp, EventSetup const& c) final { caches_[rp.index()] = globalBeginRun(rp, c); }
        void doEndRun_(Run const& rp, EventSetup const& c) final {
          globalEndRun(rp, c);
          caches_[rp.index()].reset();
        }

        virtual std::shared_ptr<C> globalBeginRun(edm::Run const&, edm::EventSetup const&) const = 0;
        virtual void globalEndRun(edm::Run const&, edm::EventSetup const&) = 0;

        std::unique_ptr<std::shared_ptr<C>[]> caches_;
      };

      template <typename T, typename C>
      class LuminosityBlockCacheHolder : public virtual T {
      public:
        LuminosityBlockCacheHolder() = default;
        LuminosityBlockCacheHolder(LuminosityBlockCacheHolder<T, C> const&) = delete;
        LuminosityBlockCacheHolder<T, C>& operator=(LuminosityBlockCacheHolder<T, C> const&) = delete;
        ~LuminosityBlockCacheHolder() noexcept(false) override{};

      protected:
        void preallocLumis(unsigned int iNLumis) final { caches_.reset(new std::shared_ptr<C>[iNLumis]); }

        C const* luminosityBlockCache(edm::LuminosityBlockIndex iID) const { return caches_[iID].get(); }
        C* luminosityBlockCache(edm::LuminosityBlockIndex iID) { return caches_[iID].get(); }

      private:
        void doBeginLuminosityBlock_(LuminosityBlock const& lp, EventSetup const& c) final {
          caches_[lp.index()] = globalBeginLuminosityBlock(lp, c);
        }
        void doEndLuminosityBlock_(LuminosityBlock const& lp, EventSetup const& c) final {
          globalEndLuminosityBlock(lp, c);
          caches_[lp.index()].reset();
        }

        virtual std::shared_ptr<C> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                              edm::EventSetup const&) const = 0;
        virtual void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) = 0;
        std::unique_ptr<std::shared_ptr<C>[]> caches_;
      };

      template <typename T>
      class Accumulator : public virtual T {
      public:
        Accumulator() = default;
        Accumulator(Accumulator const&) = delete;
        Accumulator& operator=(Accumulator const&) = delete;
        ~Accumulator() noexcept(false) override{};

      private:
        bool hasAccumulator() const override { return true; }

        void produce(Event& ev, EventSetup const& es) final { accumulate(ev, es); }

        virtual void accumulate(Event const& ev, EventSetup const& es) = 0;
      };

      template <typename T>
      class Transformer : public virtual T, private TransformerBase {
      public:
        Transformer() = default;
        Transformer(Transformer const&) = delete;
        Transformer& operator=(Transformer const&) = delete;
        ~Transformer() noexcept(false) override{};

        template <typename G, typename F>
        void registerTransform(ProductRegistryHelper::BranchAliasSetterT<G> iSetter,
                               F&& iF,
                               std::string productInstance = std::string()) {
          registerTransform(edm::EDPutTokenT<G>(iSetter), std::forward<F>(iF), std::move(productInstance));
        }

        template <typename G, typename F>
        void registerTransform(edm::EDPutTokenT<G> iToken, F iF, std::string productInstance = std::string()) {
          using ReturnTypeT = decltype(iF(std::declval<G>()));
          TypeID returnType(typeid(ReturnTypeT));
          TransformerBase::registerTransformImp(*this,
                                                EDPutToken(iToken),
                                                returnType,
                                                std::move(productInstance),
                                                [f = std::move(iF)](edm::WrapperBase const& iGotProduct) {
                                                  return std::make_unique<edm::Wrapper<ReturnTypeT>>(
                                                      WrapperBase::Emplace{},
                                                      f(*static_cast<edm::Wrapper<G> const&>(iGotProduct).product()));
                                                });
        }

        template <typename G, typename P, typename F>
        void registerTransformAsync(edm::EDPutTokenT<G> iToken,
                                    P iPre,
                                    F iF,
                                    std::string productInstance = std::string()) {
          using CacheTypeT = decltype(iPre(std::declval<G>(), WaitingTaskWithArenaHolder()));
          using ReturnTypeT = decltype(iF(std::declval<CacheTypeT>()));
          TypeID returnType(typeid(ReturnTypeT));
          TransformerBase::registerTransformAsyncImp(
              *this,
              EDPutToken(iToken),
              returnType,
              std::move(productInstance),
              [p = std::move(iPre)](edm::WrapperBase const& iGotProduct, WaitingTaskWithArenaHolder iHolder) {
                return std::any(p(*static_cast<edm::Wrapper<G> const&>(iGotProduct).product(), std::move(iHolder)));
              },
              [f = std::move(iF)](std::any const& iCache) {
                auto cache = std::any_cast<CacheTypeT>(iCache);
                return std::make_unique<edm::Wrapper<ReturnTypeT>>(WrapperBase::Emplace{}, f(cache));
              });
        }

      private:
        size_t transformIndex_(edm::BranchDescription const& iBranch) const final {
          return TransformerBase::findMatchingIndex(*this, iBranch);
        }
        ProductResolverIndex transformPrefetch_(std::size_t iIndex) const final {
          return TransformerBase::prefetchImp(iIndex);
        }
        void transformAsync_(WaitingTaskHolder iTask,
                             std::size_t iIndex,
                             edm::EventForTransformer& iEvent,
                             ServiceWeakToken const& iToken) const final {
          return TransformerBase::transformImpAsync(std::move(iTask), iIndex, *this, iEvent);
        }
        void extendUpdateLookup(BranchType iBranchType, ProductResolverIndexHelper const& iHelper) override {
          if (iBranchType == InEvent) {
            TransformerBase::extendUpdateLookup(*this, this->moduleDescription(), iHelper);
          }
        }
      };
    }  // namespace impl
  }    // namespace one
}  // namespace edm

#endif
