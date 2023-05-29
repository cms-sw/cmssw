#ifndef FWCore_Framework_stream_implementors_h
#define FWCore_Framework_stream_implementors_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     implementors
//
/**\file implementors.h "FWCore/Framework/interface/stream/implementors.h"

 Description: Base classes used to implement the interfaces for the edm::stream::* module  abilities

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 11:52:34 GMT
//

// system include files
#include <cstddef>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

// user include files
#include "FWCore/Framework/interface/CacheHandle.h"
#include "FWCore/Framework/interface/stream/EDProducerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputProcessBlockCacheImpl.h"
#include "FWCore/Framework/interface/TransformerBase.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Utilities/interface/TypeID.h"

// forward declarations
namespace edm {

  class WaitingTaskWithArenaHolder;

  namespace stream {
    namespace impl {
      class EmptyType {};

      template <typename C>
      class GlobalCacheHolder {
      public:
        GlobalCacheHolder() = default;
        GlobalCacheHolder(GlobalCacheHolder<C> const&) = delete;
        GlobalCacheHolder<C>& operator=(GlobalCacheHolder<C> const&) = delete;

        void setGlobalCache(C const* iCache) { cache_ = iCache; }

      protected:
        C const* globalCache() const { return cache_; }

      private:
        C const* cache_;
      };

      template <typename... CacheTypes>
      class InputProcessBlockCacheHolder {
      public:
        InputProcessBlockCacheHolder() = default;
        InputProcessBlockCacheHolder(InputProcessBlockCacheHolder const&) = delete;
        InputProcessBlockCacheHolder& operator=(InputProcessBlockCacheHolder const&) = delete;

        std::tuple<CacheHandle<CacheTypes>...> processBlockCaches(Event const& event) const {
          return cacheImpl_->processBlockCaches(event);
        }

        template <std::size_t N>
        using CacheTypeT = typename std::tuple_element<N, std::tuple<CacheTypes...>>::type;

        template <std::size_t ICacheType, typename DataType, typename Func>
        void registerProcessBlockCacheFiller(EDGetTokenT<DataType> const& token, Func&& cacheFiller) {
          registerProcessBlockCacheFiller<ICacheType, CacheTypeT<ICacheType>, DataType, Func>(
              token, std::forward<Func>(cacheFiller));
        }

        template <typename CacheType, typename DataType, typename Func>
        void registerProcessBlockCacheFiller(EDGetTokenT<DataType> const& token, Func&& cacheFiller) {
          static_assert(edm::impl::countTypeInParameterPack<CacheType, CacheTypes...>() == 1u,
                        "If registerProcessBlockCacheFiller is called with a type template parameter\n"
                        "then that type must appear exactly once in the template parameters of InputProcessBlockCache");

          // Find the index into the parameter pack from the CacheType
          constexpr unsigned int I = edm::impl::indexInputProcessBlockCache<CacheType, CacheTypes...>();

          registerProcessBlockCacheFiller<I, CacheType, DataType, Func>(token, std::forward<Func>(cacheFiller));
        }

      private:
        template <typename T, bool, bool>
        friend struct edm::stream::CallInputProcessBlockImpl;

        void setProcessBlockCache(edm::impl::InputProcessBlockCacheImpl<CacheTypes...> const* cacheImpl) {
          cacheImpl_ = cacheImpl;
        }

        bool cacheFillersRegistered() const { return registrationInfo_ ? true : false; }
        std::vector<edm::impl::TokenInfo>& tokenInfos() { return registrationInfo_->tokenInfos_; }
        std::tuple<edm::impl::CacheFiller<CacheTypes>...>& cacheFillers() { return registrationInfo_->cacheFillers_; }

        void clearRegistration() { registrationInfo_.reset(); }

        // The next two functions exist so that it is optional whether modules
        // with this ability implement them.

        static void accessInputProcessBlock(edm::ProcessBlock const&) {}

        template <typename GlobalCacheType>
        static void accessInputProcessBlock(edm::ProcessBlock const&, GlobalCacheType*) {}

        template <std::size_t ICacheType, typename CacheType, typename DataType, typename Func>
        void registerProcessBlockCacheFiller(EDGetTokenT<DataType> const& token, Func&& cacheFiller) {
          if (!registrationInfo_) {
            registrationInfo_ = std::make_unique<RegistrationInfo>();
            tokenInfos().resize(sizeof...(CacheTypes));
          }

          if (!tokenInfos()[ICacheType].token_.isUninitialized()) {
            throw Exception(errors::LogicError)
                << "registerProcessBlockCacheFiller should only be called once per cache type";
          }

          tokenInfos()[ICacheType] = edm::impl::TokenInfo{EDGetToken(token), TypeID(typeid(DataType))};

          std::get<ICacheType>(cacheFillers()).func_ =
              std::function<std::shared_ptr<CacheType>(ProcessBlock const&, std::shared_ptr<CacheType> const&)>(
                  std::forward<Func>(cacheFiller));
        }

        // ------------ Data members --------------------

        edm::impl::InputProcessBlockCacheImpl<CacheTypes...> const* cacheImpl_;

        // The RegistrationInfo is filled while the module constructor runs.
        // Later this information is copied to the InputProcessBlockCacheImpl
        // object owned by the adaptor and then registrationInfo_ is cleared.
        // Note that this is really only needed for one of the stream instances,
        // but we fill for all streams so registerProcessBlockCacheFiller can
        // be called in the constructor. This keeps the interface as simple as
        // possible and makes it similar to the consumes interface.
        class RegistrationInfo {
        public:
          std::vector<edm::impl::TokenInfo> tokenInfos_;
          std::tuple<edm::impl::CacheFiller<CacheTypes>...> cacheFillers_;
        };
        std::unique_ptr<RegistrationInfo> registrationInfo_;
      };

      template <typename C>
      class RunCacheHolder {
      public:
        RunCacheHolder() = default;
        RunCacheHolder(RunCacheHolder<C> const&) = delete;
        RunCacheHolder<C>& operator=(RunCacheHolder<C> const&) = delete;
        void setRunCache(C const* iCache) { cache_ = iCache; }

      protected:
        C const* runCache() const { return cache_; }

      private:
        C const* cache_;
      };

      template <typename C>
      class LuminosityBlockCacheHolder {
      public:
        LuminosityBlockCacheHolder() = default;
        LuminosityBlockCacheHolder(LuminosityBlockCacheHolder<C> const&) = delete;
        LuminosityBlockCacheHolder<C>& operator=(LuminosityBlockCacheHolder<C> const&) = delete;
        void setLuminosityBlockCache(C const* iCache) { cache_ = iCache; }

      protected:
        C const* luminosityBlockCache() const { return cache_; }

      private:
        C const* cache_;
      };

      template <typename C>
      class RunSummaryCacheHolder {
      public:
        RunSummaryCacheHolder() = default;
        RunSummaryCacheHolder(RunSummaryCacheHolder<C> const&) = delete;
        RunSummaryCacheHolder<C>& operator=(RunSummaryCacheHolder<C> const&) = delete;
        virtual ~RunSummaryCacheHolder() noexcept(false) {}

      private:
        virtual void endRunSummary(edm::Run const&, edm::EventSetup const&, C*) const = 0;
      };

      template <typename C>
      class LuminosityBlockSummaryCacheHolder {
      public:
        LuminosityBlockSummaryCacheHolder() = default;
        LuminosityBlockSummaryCacheHolder(LuminosityBlockSummaryCacheHolder<C> const&) = delete;
        LuminosityBlockSummaryCacheHolder<C>& operator=(LuminosityBlockSummaryCacheHolder<C> const&) = delete;
        virtual ~LuminosityBlockSummaryCacheHolder() noexcept(false) {}

      private:
        virtual void endLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, C*) const = 0;
      };

      class WatchProcessBlock {
      public:
        WatchProcessBlock() = default;
        WatchProcessBlock(WatchProcessBlock const&) = delete;
        WatchProcessBlock& operator=(WatchProcessBlock const&) = delete;

        ///requires the following be defined in the inheriting class
        ///static void beginProcessBlockProduce(edm::ProcessBlock const&, GlobalCache*);
      };

      class BeginProcessBlockProducer {
      public:
        BeginProcessBlockProducer() = default;
        BeginProcessBlockProducer(BeginProcessBlockProducer const&) = delete;
        BeginProcessBlockProducer& operator=(BeginProcessBlockProducer const&) = delete;

        ///requires the following be defined in the inheriting class
        ///static void beginProcessBlockProduce(edm::ProcessBlock&, GlobalCache*);
      };

      class EndProcessBlockProducer {
      public:
        EndProcessBlockProducer() = default;
        EndProcessBlockProducer(EndProcessBlockProducer const&) = delete;
        EndProcessBlockProducer& operator=(EndProcessBlockProducer const&) = delete;

        ///requires the following be defined in the inheriting class
        /// static void endProcessBlockProduce(edm::ProcessBlock&, GlobalCache*)
      };

      class BeginRunProducer {
      public:
        BeginRunProducer() = default;
        BeginRunProducer(BeginRunProducer const&) = delete;
        BeginRunProducer& operator=(BeginRunProducer const&) = delete;

        ///requires the following be defined in the inheriting class
        ///static void globalBeginRunProduce(edm::Run&, edm::EventSetup const&, RunContext const* );
      };

      class EndRunProducer {
      public:
        EndRunProducer() = default;
        EndRunProducer(EndRunProducer const&) = delete;
        EndRunProducer& operator=(EndRunProducer const&) = delete;

        ///requires the following be defined in the inheriting class
        /// static void globalEndRunProduce(edm::Run&, edm::EventSetup const&, RunContext const* )
      };

      class BeginLuminosityBlockProducer {
      public:
        BeginLuminosityBlockProducer() = default;
        BeginLuminosityBlockProducer(BeginLuminosityBlockProducer const&) = delete;
        BeginLuminosityBlockProducer& operator=(BeginLuminosityBlockProducer const&) = delete;

        ///requires the following be defined in the inheriting class
        ///static void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&, LuminosityBlockContext const*)
      };

      class EndLuminosityBlockProducer {
      public:
        EndLuminosityBlockProducer() = default;
        EndLuminosityBlockProducer(EndLuminosityBlockProducer const&) = delete;
        EndLuminosityBlockProducer& operator=(EndLuminosityBlockProducer const&) = delete;

        ///requires the following be defined in the inheriting class
        ///static void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&, LuminosityBlockContext const*)
      };

      class ExternalWork {
      public:
        ExternalWork() = default;
        ExternalWork(ExternalWork const&) = delete;
        ExternalWork& operator=(ExternalWork const&) = delete;
        virtual ~ExternalWork() noexcept(false){};

        virtual void acquire(Event const&, edm::EventSetup const&, WaitingTaskWithArenaHolder) = 0;
      };

      class Transformer : private TransformerBase, public EDProducerBase {
      public:
        Transformer() = default;
        Transformer(Transformer const&) = delete;
        Transformer& operator=(Transformer const&) = delete;
        ~Transformer() noexcept(false) override{};

        template <typename G, typename F>
        void registerTransform(ProducerBase::BranchAliasSetterT<G> iSetter,
                               F&& iF,
                               std::string productInstance = std::string()) {
          registerTransform(edm::EDPutTokenT<G>(iSetter), std::forward<F>(iF), std::move(productInstance));
        }

        template <typename G, typename F>
        void registerTransform(edm::EDPutTokenT<G> iToken, F iF, std::string productInstance = std::string()) {
          using ReturnTypeT = decltype(iF(std::declval<G>()));
          TypeID returnType(typeid(ReturnTypeT));
          TransformerBase::registerTransformImp(
              *this,
              EDPutToken(iToken),
              returnType,
              std::move(productInstance),
              [f = std::move(iF)](std::any const& iGotProduct) {
                auto pGotProduct = std::any_cast<edm::WrapperBase const*>(iGotProduct);
                return std::make_unique<edm::Wrapper<ReturnTypeT>>(
                    WrapperBase::Emplace{}, f(*static_cast<edm::Wrapper<G> const*>(pGotProduct)->product()));
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

      class Accumulator : public EDProducerBase {
      public:
        Accumulator() = default;
        Accumulator(Accumulator const&) = delete;
        Accumulator& operator=(Accumulator const&) = delete;
        ~Accumulator() noexcept(false) override{};

        virtual void accumulate(Event const& ev, EventSetup const& es) = 0;

        void produce(Event& ev, EventSetup const& es) final { accumulate(ev, es); }
      };
    }  // namespace impl
  }    // namespace stream
}  // namespace edm

#endif
