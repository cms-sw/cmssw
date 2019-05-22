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
#include <memory>

// user include files
#include "FWCore/Framework/interface/stream/EDProducerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"

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

      private:
        ///requires the following be defined in the inheriting class
        /// static void globalEndRunProduce(edm::Run&, edm::EventSetup const&, RunContext const* )
      };

      class BeginLuminosityBlockProducer {
      public:
        BeginLuminosityBlockProducer() = default;
        BeginLuminosityBlockProducer(BeginLuminosityBlockProducer const&) = delete;
        BeginLuminosityBlockProducer& operator=(BeginLuminosityBlockProducer const&) = delete;

      private:
        ///requires the following be defined in the inheriting class
        ///static void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&, LuminosityBlockContext const*)
      };

      class EndLuminosityBlockProducer {
      public:
        EndLuminosityBlockProducer() = default;
        EndLuminosityBlockProducer(EndLuminosityBlockProducer const&) = delete;
        EndLuminosityBlockProducer& operator=(EndLuminosityBlockProducer const&) = delete;

      private:
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
