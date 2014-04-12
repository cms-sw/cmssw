
/*----------------------------------------------------------------------

Toy edm::global::EDAnalyzer modules of 
edm::*Cache templates 
for testing purposes only.

----------------------------------------------------------------------*/
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edmtest {
namespace global {

namespace {
struct Cache { 
   Cache():value(0) {}
   //Using mutable since we want to update the value.
   mutable std::atomic<unsigned int> value;
};

struct UnsafeCache {
   UnsafeCache():value(0) {}
   unsigned int value;
};

} //end anonymous namespace

  class StreamIntAnalyzer: public edm::global::EDAnalyzer<edm::StreamCache<UnsafeCache>> {
  public:
    explicit StreamIntAnalyzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions"))
    {}
    const unsigned int trans_;
    mutable std::atomic<unsigned int> m_count{0};
    
    std::unique_ptr<UnsafeCache> beginStream(edm::StreamID iID) const override {
      ++m_count;
      std::unique_ptr<UnsafeCache> pCache(new UnsafeCache);
      pCache->value = iID.value();
      return pCache;
    }
    
    void streamBeginRun(edm::StreamID iID, edm::Run const&, edm::EventSetup const&) const  override{
      ++m_count;
      if ( (streamCache(iID))->value != iID.value() ) {
          throw cms::Exception("cache value")
          << "StreamIntAnalyzer cache value "
          << (streamCache(iID))->value << " but it was supposed to be " << iID;
      }
    }

    void streamBeginLuminosityBlock(edm::StreamID iID, edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
      if ( (streamCache(iID))->value != iID.value() ) {
          throw cms::Exception("cache value")
          << "StreamIntAnalyzer cache value "
          << (streamCache(iID))->value << " but it was supposed to be " << iID;
      }
     }

    void analyze(edm::StreamID iID, const edm::Event&, const edm::EventSetup&) const override {
      ++m_count;
      if ( (streamCache(iID))->value != iID.value() ) {
          throw cms::Exception("cache value")
          << "StreamIntAnalyzer cache value "
          << (streamCache(iID))->value << " but it was supposed to be " << iID;
      }       
    }

    void streamEndLuminosityBlock(edm::StreamID iID, edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
      if ( (streamCache(iID))->value != iID.value() ) {
          throw cms::Exception("cache value")
          << "StreamIntAnalyzer cache value "
          << (streamCache(iID))->value << " but it was supposed to be " << iID;
      }
    }

    void streamEndRun(edm::StreamID iID, edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      if ( (streamCache(iID))->value != iID.value() ) {
          throw cms::Exception("cache value")
          << "StreamIntAnalyzer cache value "
          << (streamCache(iID))->value << " but it was supposed to be " << iID;
      }
    }

    void endStream(edm::StreamID iID ) const override {
      ++m_count;
      if ( (streamCache(iID))->value != iID.value() ) {
          throw cms::Exception("cache value")
          << "StreamIntAnalyzer cache value "
          << (streamCache(iID))->value << " but it was supposed to be " << iID;
      }
    } 

    ~StreamIntAnalyzer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "StreamIntAnalyzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };
  
  class RunIntAnalyzer: public edm::global::EDAnalyzer<edm::RunCache<Cache>> {
  public:
    explicit RunIntAnalyzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions"))
        ,cvalue_(p.getParameter<int>("cachevalue")) 
    {}
    const unsigned int trans_; 
    const unsigned int cvalue_; 
    mutable std::atomic<unsigned int> m_count{0};
    
    std::shared_ptr<Cache> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<Cache>(new Cache);
    }

    void analyze(edm::StreamID iID, const edm::Event& iEvent, const edm::EventSetup&) const override {
      ++m_count;
      ++((runCache(iEvent.getRun().index()))->value);
       
    }

    void globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const override {
      ++m_count;
      if ( (runCache(iRun.index()))->value != cvalue_ ) {
          throw cms::Exception("cache value")
          << "RunIntAnalyzer cache value "
          << (runCache(iRun.index()))->value << " but it was supposed to be " << cvalue_;
      }
    }

    ~RunIntAnalyzer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "RunIntAnalyzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };


  class LumiIntAnalyzer: public edm::global::EDAnalyzer<edm::LuminosityBlockCache<Cache>> {
  public:
    explicit LumiIntAnalyzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) 
        ,cvalue_(p.getParameter<int>("cachevalue")) 
    {}
    const unsigned int trans_; 
    const unsigned int cvalue_; 
    mutable std::atomic<unsigned int> m_count{0};
   
    std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<Cache>(new Cache);
    }

    void analyze(edm::StreamID, const edm::Event& iEvent, const edm::EventSetup&) const override {
      ++m_count;
      ++(luminosityBlockCache(iEvent.getLuminosityBlock().index())->value);
       
    }
     
    void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&) const override {
      ++m_count;
      if( (luminosityBlockCache(iLB.index()))->value != cvalue_) {
        throw cms::Exception("cache value")
          << "LumiIntAnalyzer cache value "
          << (luminosityBlockCache(iLB.index()))->value << " but it was supposed to be " << cvalue_;
      }
    }

    ~LumiIntAnalyzer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "LumiIntAnalyzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };
  
  class RunSummaryIntAnalyzer: public edm::global::EDAnalyzer<edm::StreamCache<UnsafeCache>,edm::RunSummaryCache<UnsafeCache>> {
  public:
    explicit RunSummaryIntAnalyzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) 
        ,cvalue_(p.getParameter<int>("cachevalue")) 
    {}
    const unsigned int trans_; 
    const unsigned int cvalue_; 
    mutable std::atomic<unsigned int> m_count{0};

    std::unique_ptr<UnsafeCache> beginStream(edm::StreamID) const override {
      ++m_count;
      return std::unique_ptr<UnsafeCache>(new UnsafeCache);
    }

    std::shared_ptr<UnsafeCache> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<UnsafeCache>(new UnsafeCache);
    }
  
    void analyze(edm::StreamID iID, const edm::Event&, const edm::EventSetup&) const override {
      ++m_count;
      ++((streamCache(iID))->value);
       
    }
  
    void streamEndRunSummary(edm::StreamID iID, edm::Run const&, edm::EventSetup const&, UnsafeCache* gCache) const override {
      ++m_count;
      gCache->value += (streamCache(iID))->value;
      (streamCache(iID))->value = 0;
    }
    
    void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, UnsafeCache* gCache) const override {
      ++m_count;
      if( gCache->value  != cvalue_) {
        throw cms::Exception("cache value")
          << "RunSummaryIntAnalyzer cache value "
          << gCache->value << " but it was supposed to be " << cvalue_;
      } 
    }

    ~RunSummaryIntAnalyzer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "RunSummaryIntAnalyzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

  class LumiSummaryIntAnalyzer: public edm::global::EDAnalyzer<edm::StreamCache<UnsafeCache>,edm::LuminosityBlockSummaryCache<UnsafeCache>> {
  public:
    explicit LumiSummaryIntAnalyzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) 
        ,cvalue_(p.getParameter<int>("cachevalue")) 
    {}
    const unsigned int trans_; 
    const unsigned int cvalue_; 
    mutable std::atomic<unsigned int> m_count{0};

    std::unique_ptr<UnsafeCache> beginStream(edm::StreamID) const override {
      ++m_count;
      return std::unique_ptr<UnsafeCache>(new UnsafeCache);
    }

    std::shared_ptr<UnsafeCache> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<UnsafeCache>(new UnsafeCache);
    }
    
    void analyze(edm::StreamID iID, const edm::Event& iEvent, const edm::EventSetup&) const override {
      ++m_count;
      ++((streamCache(iID))->value);
       
    }

    void streamEndLuminosityBlockSummary(edm::StreamID iID, edm::LuminosityBlock const& iLumiBlock, edm::EventSetup const&, UnsafeCache* gCache) const override {
      ++m_count;
      gCache->value += (streamCache(iID))->value;
      (streamCache(iID))->value = 0;
    }
 
    void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, UnsafeCache* gCache) const override {
      ++m_count;
      if( gCache->value != cvalue_) {
        throw cms::Exception("cache value")
          << "LumiSummaryIntAnalyzer cache value "
          << gCache->value << " but it was supposed to be " << cvalue_;
      }
    }

    ~LumiSummaryIntAnalyzer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "LumiSummaryIntAnalyzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

}
}

DEFINE_FWK_MODULE(edmtest::global::StreamIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::global::RunIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::global::LumiIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::global::RunSummaryIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::global::LumiSummaryIntAnalyzer);

