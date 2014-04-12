#ifndef FWCore_Framework_global_implementors_h
#define FWCore_Framework_global_implementors_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     implementors
// 
/**\file implementors.h "FWCore/Framework/interface/global/implementors.h"

 Description: Base classes used to implement the interfaces for the edm::global::* module  abilities

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 18 Jul 2013 11:52:34 GMT
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"

// forward declarations
namespace edm {
  
  namespace global {
    namespace impl {
      class EmptyType {};
      
      
      template <typename T, typename C>
      class StreamCacheHolder : public virtual T {
      public:
        StreamCacheHolder() = default;
        StreamCacheHolder( StreamCacheHolder<T,C> const&) = delete;
        StreamCacheHolder<T,C>& operator=(StreamCacheHolder<T,C> const&) = delete;
        ~StreamCacheHolder() {
          for(auto c: caches_){
            delete c;
          }
        }
      protected:
        C * streamCache(edm::StreamID iID) const { return caches_[iID.value()]; }
      private:
        virtual void preallocStreams(unsigned int iNStreams) override final {
          caches_.resize(iNStreams,static_cast<C*>(nullptr));
        }
        virtual void doBeginStream_(StreamID id) override final {
          caches_[id.value()] = beginStream(id).release();
        }
        virtual void doEndStream_(StreamID id) override final {
          endStream(id);
          delete caches_[id.value()];
          caches_[id.value()]=nullptr;
        }
        virtual void doStreamBeginRun_(StreamID id, Run const& rp, EventSetup const& c) override final {
          streamBeginRun(id,rp,c);
        }
        virtual void doStreamEndRun_(StreamID id, Run const& rp, EventSetup const& c) override final {
          streamEndRun(id,rp,c);
        }
        virtual void doStreamBeginLuminosityBlock_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c) override final {
          streamBeginLuminosityBlock(id,lbp,c);
        }
        virtual void doStreamEndLuminosityBlock_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c) override final {
          streamEndLuminosityBlock(id,lbp,c);
        }

        virtual std::unique_ptr<C> beginStream(edm::StreamID) const = 0;
        virtual void streamBeginRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const  {}
        virtual void streamBeginLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const {}
        virtual void streamEndLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const {}
        virtual void streamEndRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const {}
        virtual void endStream(edm::StreamID) const {}

        //When threaded we will have a container for N items whre N is # of streams
        std::vector<C*> caches_;
      };
      
      template <typename T, typename C>
      class RunCacheHolder : public virtual T {
      public:
        RunCacheHolder() = default;
        RunCacheHolder( RunCacheHolder<T,C> const&) = delete;
        RunCacheHolder<T,C>& operator=(RunCacheHolder<T,C> const&) = delete;
      protected:
        C const* runCache(edm::RunIndex iID) const { return cache_.get(); }
      private:
        void doBeginRun_(Run const& rp, EventSetup const& c) override final {
          cache_ = globalBeginRun(rp,c);
        }
        void doEndRun_(Run const& rp, EventSetup const& c) override final {
          globalEndRun(rp,c);
          cache_.reset();
        }
        
        virtual std::shared_ptr<C> globalBeginRun(edm::Run const&, edm::EventSetup const&) const = 0;
        virtual void globalEndRun(edm::Run const&, edm::EventSetup const&) const = 0;
        //When threaded we will have a container for N items whre N is # of simultaneous runs
        std::shared_ptr<C> cache_;
      };
      
      template <typename T, typename C>
      class LuminosityBlockCacheHolder : public virtual T {
      public:
        LuminosityBlockCacheHolder() = default;
        LuminosityBlockCacheHolder( LuminosityBlockCacheHolder<T,C> const&) = delete;
        LuminosityBlockCacheHolder<T,C>& operator=(LuminosityBlockCacheHolder<T,C> const&) = delete;
      protected:
        C const* luminosityBlockCache(edm::LuminosityBlockIndex iID) const { return cache_.get(); }
      private:
        void doBeginLuminosityBlock_(LuminosityBlock const& rp, EventSetup const& c) override final {
          cache_ = globalBeginLuminosityBlock(rp,c);
        }
        void doEndLuminosityBlock_(LuminosityBlock const& rp, EventSetup const& c) override final {
          globalEndLuminosityBlock(rp,c);
          cache_.reset();
        }
        
        virtual std::shared_ptr<C> globalBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const = 0;
        virtual void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const = 0;
        //When threaded we will have a container for N items whre N is # of simultaneous runs
        std::shared_ptr<C> cache_;
      };
      
      template<typename T, typename C> class EndRunSummaryProducer;
      
      template <typename T, typename C>
      class RunSummaryCacheHolder : public virtual T {
      public:
        RunSummaryCacheHolder() = default;
        RunSummaryCacheHolder( RunSummaryCacheHolder<T,C> const&) = delete;
        RunSummaryCacheHolder<T,C>& operator=(RunSummaryCacheHolder<T,C> const&) = delete;
      private:
        friend class EndRunSummaryProducer<T,C>;
        void doBeginRunSummary_(edm::Run const& rp, EventSetup const& c) override final {
          cache_ = globalBeginRunSummary(rp,c);
        }
        void doStreamEndRunSummary_(StreamID id, Run const& rp, EventSetup const& c) override final {
          //NOTE: in future this will need to be serialized
          streamEndRunSummary(id,rp,c,cache_.get());
        }
        void doEndRunSummary_(Run const& rp, EventSetup const& c) override final {
          globalEndRunSummary(rp,c,cache_.get());
        }

        virtual std::shared_ptr<C> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const = 0;
        virtual void streamEndRunSummary(StreamID, edm::Run const&, edm::EventSetup const&, C*) const = 0;

        virtual void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, C*) const = 0;

        //When threaded we will have a container for N items where N is # of simultaneous runs
        std::shared_ptr<C> cache_;
      };

      template<typename T, typename C> class EndLuminosityBlockSummaryProducer;

      
      template <typename T, typename C>
      class LuminosityBlockSummaryCacheHolder : public virtual T {
      public:
        LuminosityBlockSummaryCacheHolder() = default;
        LuminosityBlockSummaryCacheHolder( LuminosityBlockSummaryCacheHolder<T,C> const&) = delete;
        LuminosityBlockSummaryCacheHolder<T,C>& operator=(LuminosityBlockSummaryCacheHolder<T,C> const&) = delete;
      private:
        friend class EndLuminosityBlockSummaryProducer<T,C>;
        
        void doBeginLuminosityBlockSummary_(edm::LuminosityBlock const& lb, EventSetup const& c) override final {
          cache_ = globalBeginLuminosityBlockSummary(lb,c);
        }

        virtual void doStreamEndLuminosityBlockSummary_(StreamID id, LuminosityBlock const& lb, EventSetup const& c) override final
        {
          streamEndLuminosityBlockSummary(id,lb,c,cache_.get());
        }
        void doEndLuminosityBlockSummary_(LuminosityBlock const& lb, EventSetup const& c) override final {
          globalEndLuminosityBlockSummary(lb,c,cache_.get());
        }

        virtual std::shared_ptr<C> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&) const = 0;
        virtual void streamEndLuminosityBlockSummary(StreamID, edm::LuminosityBlock const&, edm::EventSetup const&, C*) const = 0;
        
        virtual void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, C*) const = 0;
        
        //When threaded we will have a container for N items where N is # of simultaneous Lumis
        std::shared_ptr<C> cache_;
      };

      
      template <typename T>
      class BeginRunProducer : public virtual T {
      public:
        BeginRunProducer() = default;
        BeginRunProducer( BeginRunProducer const&) = delete;
        BeginRunProducer& operator=(BeginRunProducer const&) = delete;
        
      private:
        void doBeginRunProduce_(Run& rp, EventSetup const& c) override final;
        
        virtual void globalBeginRunProduce(edm::Run&, edm::EventSetup const&) const = 0;
      };
      
      template <typename T>
      class EndRunProducer : public virtual T {
      public:
        EndRunProducer() = default;
        EndRunProducer( EndRunProducer const&) = delete;
        EndRunProducer& operator=(EndRunProducer const&) = delete;
        
      private:
        
        void doEndRunProduce_(Run& rp, EventSetup const& c) override final;
        
        virtual void globalEndRunProduce(edm::Run&, edm::EventSetup const&) const = 0;
      };

      template <typename T, typename C>
      class EndRunSummaryProducer : public RunSummaryCacheHolder<T,C> {
      public:
        EndRunSummaryProducer() = default;
        EndRunSummaryProducer( EndRunSummaryProducer const&) = delete;
        EndRunSummaryProducer& operator=(EndRunSummaryProducer const&) = delete;
        
      private:
        
        void doEndRunProduce_(Run& rp, EventSetup const& c) override final {
          globalEndRunProduce(rp,c,RunSummaryCacheHolder<T,C>::cache_.get());
        }
        
        virtual void globalEndRunProduce(edm::Run&, edm::EventSetup const&, C const*) const = 0;
      };

      template <typename T>
      class BeginLuminosityBlockProducer : public virtual T {
      public:
        BeginLuminosityBlockProducer() = default;
        BeginLuminosityBlockProducer( BeginLuminosityBlockProducer const&) = delete;
        BeginLuminosityBlockProducer& operator=(BeginLuminosityBlockProducer const&) = delete;
        
      private:
        void doBeginLuminosityBlockProduce_(LuminosityBlock& lb, EventSetup const& c) override final;
        virtual void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const = 0;
      };
      
      template <typename T>
      class EndLuminosityBlockProducer : public virtual T {
      public:
        EndLuminosityBlockProducer() = default;
        EndLuminosityBlockProducer( EndLuminosityBlockProducer const&) = delete;
        EndLuminosityBlockProducer& operator=(EndLuminosityBlockProducer const&) = delete;
        
      private:
        void doEndLuminosityBlockProduce_(LuminosityBlock& lb, EventSetup const& c) override final;
        virtual void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const = 0;
      };

      template <typename T, typename S>
      class EndLuminosityBlockSummaryProducer : public LuminosityBlockSummaryCacheHolder<T,S> {
      public:
        EndLuminosityBlockSummaryProducer() = default;
        EndLuminosityBlockSummaryProducer( EndLuminosityBlockSummaryProducer const&) = delete;
        EndLuminosityBlockSummaryProducer& operator=(EndLuminosityBlockSummaryProducer const&) = delete;
        
      private:
        void doEndLuminosityBlockProduce_(LuminosityBlock& lb, EventSetup const& c) override final {
          globalEndLuminosityBlockProduce(lb,c,LuminosityBlockSummaryCacheHolder<T,S>::cache_.get());
        }
        
        virtual void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&, S const*) const = 0;
      };
    }
  }
}


#endif
