#ifndef FWCore_Framework_limited_implementors_h
#define FWCore_Framework_limited_implementors_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     implementors
//
/**\file implementors.h "FWCore/Framework/interface/limited/implementors.h"

 Description: Base classes used to implement the interfaces for the edm::limited::* module  abilities

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 18 Jul 2013 11:52:34 GMT
//

// system include files
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// user include files
#include "FWCore/Common/interface/FWCoreCommonFwd.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/CacheHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputProcessBlockCacheImpl.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/TransformerBase.h"
#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/ProcessBlockIndex.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "DataFormats/Common/interface/Wrapper.h"

// forward declarations
namespace edm {
  class ServiceWeakToken;

  namespace limited {
    namespace impl {
      class EmptyType {
      public:
        EmptyType(edm::ParameterSet const&) {}
      };

      template <typename T, typename C>
      class StreamCacheHolder : public virtual T {
      public:
        StreamCacheHolder(edm::ParameterSet const& iPSet) : T(iPSet) {}
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
        void doStreamBeginRun_(StreamID id, Run const& rp, EventSetup const& c) final { streamBeginRun(id, rp, c); }
        void doStreamEndRun_(StreamID id, Run const& rp, EventSetup const& c) final { streamEndRun(id, rp, c); }
        void doStreamBeginLuminosityBlock_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c) final {
          streamBeginLuminosityBlock(id, lbp, c);
        }
        void doStreamEndLuminosityBlock_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c) final {
          streamEndLuminosityBlock(id, lbp, c);
        }

        virtual std::unique_ptr<C> beginStream(edm::StreamID) const = 0;
        virtual void streamBeginRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const {}
        virtual void streamBeginLuminosityBlock(edm::StreamID,
                                                edm::LuminosityBlock const&,
                                                edm::EventSetup const&) const {}
        virtual void streamEndLuminosityBlock(edm::StreamID,
                                              edm::LuminosityBlock const&,
                                              edm::EventSetup const&) const {}
        virtual void streamEndRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const {}
        virtual void endStream(edm::StreamID) const {}

        //When threaded we will have a container for N items whre N is # of streams
        std::vector<C*> caches_;
      };

      template <typename T, typename... CacheTypes>
      class InputProcessBlockCacheHolder : public virtual T {
      public:
        InputProcessBlockCacheHolder(edm::ParameterSet const& iPSet) : T(iPSet) {}
        InputProcessBlockCacheHolder(InputProcessBlockCacheHolder const&) = delete;
        InputProcessBlockCacheHolder& operator=(InputProcessBlockCacheHolder const&) = delete;
        ~InputProcessBlockCacheHolder() noexcept(false) override {}

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
        RunCacheHolder(edm::ParameterSet const& iPSet) : T(iPSet) {}
        RunCacheHolder(RunCacheHolder<T, C> const&) = delete;
        RunCacheHolder<T, C>& operator=(RunCacheHolder<T, C> const&) = delete;
        ~RunCacheHolder() noexcept(false) override{};

      protected:
        void preallocRuns(unsigned int iNRuns) final { caches_.reset(new std::shared_ptr<C>[iNRuns]); }

        C const* runCache(edm::RunIndex iID) const { return caches_[iID].get(); }

      private:
        void doBeginRun_(Run const& rp, EventSetup const& c) final { caches_[rp.index()] = globalBeginRun(rp, c); }
        void doEndRun_(Run const& rp, EventSetup const& c) final {
          globalEndRun(rp, c);
          caches_[rp.index()].reset();
        }

        virtual std::shared_ptr<C> globalBeginRun(edm::Run const&, edm::EventSetup const&) const = 0;
        virtual void globalEndRun(edm::Run const&, edm::EventSetup const&) const = 0;

        std::unique_ptr<std::shared_ptr<C>[]> caches_;
      };

      template <typename T, typename C>
      class LuminosityBlockCacheHolder : public virtual T {
      public:
        LuminosityBlockCacheHolder(edm::ParameterSet const& iPSet) : T(iPSet) {}
        LuminosityBlockCacheHolder(LuminosityBlockCacheHolder<T, C> const&) = delete;
        LuminosityBlockCacheHolder<T, C>& operator=(LuminosityBlockCacheHolder<T, C> const&) = delete;
        ~LuminosityBlockCacheHolder() noexcept(false) override{};

      protected:
        C const* luminosityBlockCache(edm::LuminosityBlockIndex iID) const { return caches_[iID].get(); }

      private:
        void preallocLumis(unsigned int iNLumis) final { caches_.reset(new std::shared_ptr<C>[iNLumis]); }

        void doBeginLuminosityBlock_(LuminosityBlock const& lp, EventSetup const& c) final {
          caches_[lp.index()] = globalBeginLuminosityBlock(lp, c);
        }
        void doEndLuminosityBlock_(LuminosityBlock const& lp, EventSetup const& c) final {
          globalEndLuminosityBlock(lp, c);
          caches_[lp.index()].reset();
        }

        virtual std::shared_ptr<C> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                              edm::EventSetup const&) const = 0;
        virtual void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const = 0;

        std::unique_ptr<std::shared_ptr<C>[]> caches_;
      };

      template <typename T, typename C>
      class EndRunSummaryProducer;

      template <typename T, typename C>
      class RunSummaryCacheHolder : public virtual T {
      public:
        RunSummaryCacheHolder(edm::ParameterSet const& iPSet) : T(iPSet) {}
        RunSummaryCacheHolder(RunSummaryCacheHolder<T, C> const&) = delete;
        RunSummaryCacheHolder<T, C>& operator=(RunSummaryCacheHolder<T, C> const&) = delete;
        ~RunSummaryCacheHolder() noexcept(false) override{};

      private:
        void preallocRunsSummary(unsigned int iNRuns) final { caches_.reset(new std::shared_ptr<C>[iNRuns]); }

        friend class EndRunSummaryProducer<T, C>;
        void doBeginRunSummary_(edm::Run const& rp, EventSetup const& c) final {
          caches_[rp.index()] = globalBeginRunSummary(rp, c);
        }
        void doStreamEndRunSummary_(StreamID id, Run const& rp, EventSetup const& c) final {
          //NOTE: in future this will need to be serialized
          std::lock_guard<std::mutex> guard(mutex_);
          streamEndRunSummary(id, rp, c, caches_[rp.index()].get());
        }
        void doEndRunSummary_(Run const& rp, EventSetup const& c) final {
          globalEndRunSummary(rp, c, caches_[rp.index()].get());
          maybeClearCache(rp);
        }

        virtual std::shared_ptr<C> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const = 0;
        virtual void streamEndRunSummary(StreamID, edm::Run const&, edm::EventSetup const&, C*) const = 0;

        virtual void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, C*) const = 0;

        virtual void maybeClearCache(Run const& rp) { caches_[rp.index()].reset(); }

        std::unique_ptr<std::shared_ptr<C>[]> caches_;
        std::mutex mutex_;
      };

      template <typename T, typename C>
      class EndLuminosityBlockSummaryProducer;

      template <typename T, typename C>
      class LuminosityBlockSummaryCacheHolder : public virtual T {
      public:
        LuminosityBlockSummaryCacheHolder(edm::ParameterSet const& iPSet) : T(iPSet) {}
        LuminosityBlockSummaryCacheHolder(LuminosityBlockSummaryCacheHolder<T, C> const&) = delete;
        LuminosityBlockSummaryCacheHolder<T, C>& operator=(LuminosityBlockSummaryCacheHolder<T, C> const&) = delete;
        ~LuminosityBlockSummaryCacheHolder() noexcept(false) override{};

      private:
        void preallocLumisSummary(unsigned int iNLumis) final { caches_.reset(new std::shared_ptr<C>[iNLumis]); }

        friend class EndLuminosityBlockSummaryProducer<T, C>;

        void doBeginLuminosityBlockSummary_(edm::LuminosityBlock const& lb, EventSetup const& c) final {
          caches_[lb.index()] = globalBeginLuminosityBlockSummary(lb, c);
        }

        void doStreamEndLuminosityBlockSummary_(StreamID id, LuminosityBlock const& lb, EventSetup const& c) final {
          std::lock_guard<std::mutex> guard(mutex_);
          streamEndLuminosityBlockSummary(id, lb, c, caches_[lb.index()].get());
        }
        void doEndLuminosityBlockSummary_(LuminosityBlock const& lb, EventSetup const& c) final {
          globalEndLuminosityBlockSummary(lb, c, caches_[lb.index()].get());
          maybeClearCache(lb);
        }

        virtual std::shared_ptr<C> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                                     edm::EventSetup const&) const = 0;
        virtual void streamEndLuminosityBlockSummary(StreamID,
                                                     edm::LuminosityBlock const&,
                                                     edm::EventSetup const&,
                                                     C*) const = 0;

        virtual void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, C*) const = 0;

        virtual void maybeClearCache(LuminosityBlock const& lb) { caches_[lb.index()].reset(); }

        std::unique_ptr<std::shared_ptr<C>[]> caches_;
        std::mutex mutex_;
      };

      template <typename T>
      class WatchProcessBlock : public virtual T {
      public:
        WatchProcessBlock(edm::ParameterSet const& iPSet) : T(iPSet) {}
        WatchProcessBlock(WatchProcessBlock const&) = delete;
        WatchProcessBlock& operator=(WatchProcessBlock const&) = delete;
        ~WatchProcessBlock() noexcept(false) override{};

      private:
        void doBeginProcessBlock_(ProcessBlock const&) final;
        void doEndProcessBlock_(ProcessBlock const&) final;

        virtual void beginProcessBlock(ProcessBlock const&) {}
        virtual void endProcessBlock(ProcessBlock const&) {}
      };

      template <typename T>
      class BeginProcessBlockProducer : public virtual T {
      public:
        BeginProcessBlockProducer(edm::ParameterSet const& iPSet) : T(iPSet) {}
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
        EndProcessBlockProducer(edm::ParameterSet const& iPSet) : T(iPSet) {}
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
        BeginRunProducer(edm::ParameterSet const& iPSet) : T(iPSet) {}
        BeginRunProducer(BeginRunProducer const&) = delete;
        BeginRunProducer& operator=(BeginRunProducer const&) = delete;
        ~BeginRunProducer() noexcept(false) override{};

      private:
        void doBeginRunProduce_(Run& rp, EventSetup const& c) final;

        virtual void globalBeginRunProduce(edm::Run&, edm::EventSetup const&) const = 0;
      };

      template <typename T>
      class EndRunProducer : public virtual T {
      public:
        EndRunProducer(edm::ParameterSet const& iPSet) : T(iPSet) {}
        EndRunProducer(EndRunProducer const&) = delete;
        EndRunProducer& operator=(EndRunProducer const&) = delete;
        ~EndRunProducer() noexcept(false) override{};

      private:
        void doEndRunProduce_(Run& rp, EventSetup const& c) final;

        virtual void globalEndRunProduce(edm::Run&, edm::EventSetup const&) const = 0;
      };

      template <typename T, typename C>
      class EndRunSummaryProducer : public RunSummaryCacheHolder<T, C> {
      public:
        EndRunSummaryProducer(edm::ParameterSet const& iPSet) : T(iPSet), RunSummaryCacheHolder<T, C>(iPSet) {}
        EndRunSummaryProducer(EndRunSummaryProducer const&) = delete;
        EndRunSummaryProducer& operator=(EndRunSummaryProducer const&) = delete;
        ~EndRunSummaryProducer() noexcept(false) override{};

      private:
        void doEndRunProduce_(Run& rp, EventSetup const& c) final {
          globalEndRunProduce(rp, c, RunSummaryCacheHolder<T, C>::caches_[rp.index()].get());
          RunSummaryCacheHolder<T, C>::caches_[rp.index()].reset();
        }

        virtual void globalEndRunProduce(edm::Run&, edm::EventSetup const&, C const*) const = 0;

        // Do nothing because the cache is cleared in doEndRunProduce_
        void maybeClearCache(Run const&) final {}
      };

      template <typename T>
      class BeginLuminosityBlockProducer : public virtual T {
      public:
        BeginLuminosityBlockProducer(edm::ParameterSet const& iPSet) : T(iPSet) {}
        BeginLuminosityBlockProducer(BeginLuminosityBlockProducer const&) = delete;
        BeginLuminosityBlockProducer& operator=(BeginLuminosityBlockProducer const&) = delete;
        ~BeginLuminosityBlockProducer() noexcept(false) override{};

      private:
        void doBeginLuminosityBlockProduce_(LuminosityBlock& lb, EventSetup const& c) final;
        virtual void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const = 0;
      };

      template <typename T>
      class EndLuminosityBlockProducer : public virtual T {
      public:
        EndLuminosityBlockProducer(edm::ParameterSet const& iPSet) : T(iPSet) {}
        EndLuminosityBlockProducer(EndLuminosityBlockProducer const&) = delete;
        EndLuminosityBlockProducer& operator=(EndLuminosityBlockProducer const&) = delete;
        ~EndLuminosityBlockProducer() noexcept(false) override{};

      private:
        void doEndLuminosityBlockProduce_(LuminosityBlock& lb, EventSetup const& c) final;
        virtual void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const = 0;
      };

      template <typename T, typename S>
      class EndLuminosityBlockSummaryProducer : public LuminosityBlockSummaryCacheHolder<T, S> {
      public:
        EndLuminosityBlockSummaryProducer(edm::ParameterSet const& iPSet)
            : T(iPSet), LuminosityBlockSummaryCacheHolder<T, S>(iPSet) {}
        EndLuminosityBlockSummaryProducer(EndLuminosityBlockSummaryProducer const&) = delete;
        EndLuminosityBlockSummaryProducer& operator=(EndLuminosityBlockSummaryProducer const&) = delete;
        ~EndLuminosityBlockSummaryProducer() noexcept(false) override{};

      private:
        void doEndLuminosityBlockProduce_(LuminosityBlock& lb, EventSetup const& c) final {
          globalEndLuminosityBlockProduce(lb, c, LuminosityBlockSummaryCacheHolder<T, S>::caches_[lb.index()].get());
          LuminosityBlockSummaryCacheHolder<T, S>::caches_[lb.index()].reset();
        }

        virtual void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&, S const*) const = 0;

        // Do nothing because the cache is cleared in doEndLuminosityBlockProduce_
        void maybeClearCache(LuminosityBlock const&) final {}
      };

      template <typename T>
      class Accumulator : public virtual T {
      public:
        Accumulator(edm::ParameterSet const& iPSet) : T(iPSet) {}
        Accumulator() = default;
        Accumulator(Accumulator const&) = delete;
        Accumulator& operator=(Accumulator const&) = delete;
        ~Accumulator() noexcept(false) override{};

      private:
        bool hasAccumulator() const override { return true; }

        void produce(StreamID streamID, Event& ev, EventSetup const& es) const final { accumulate(streamID, ev, es); }

        virtual void accumulate(StreamID streamID, Event const& ev, EventSetup const& es) const = 0;
      };

      template <typename T>
      class Transformer : public virtual T, private TransformerBase {
      public:
        Transformer(edm::ParameterSet const& iPSet) : T(iPSet) {}
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
    }  // namespace impl
  }    // namespace limited
}  // namespace edm

#endif
