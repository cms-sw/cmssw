
/*----------------------------------------------------------------------

Toy edm::global::EDFilter modules of 
edm::*Cache templates and edm::*Producer classes
for testing purposes only.

----------------------------------------------------------------------*/
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "FWCore/Framework/interface/global/EDFilter.h"
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
   Cache():value(0),run(0),lumi(0),strm(0),work(0) {}
   //Using mutable since we want to update the value.
   mutable std::atomic<unsigned int> value;
   mutable std::atomic<unsigned int> run;
   mutable std::atomic<unsigned int> lumi;
   mutable std::atomic<unsigned int> strm;
   mutable std::atomic<unsigned int> work;
};

struct UnsafeCache {
   UnsafeCache():value(0),run(0),lumi(0),strm(0),work(0) {}
   unsigned int value;
   unsigned int run;
   unsigned int lumi;
   unsigned int strm;
   unsigned int work;
};
} //end anonymous namespace


  class StreamIntFilter : public edm::global::EDFilter<edm::StreamCache<UnsafeCache>> {
  public:
    explicit StreamIntFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) 
    {
    produces<unsigned int>();
    }

    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
   
    std::unique_ptr<UnsafeCache> beginStream(edm::StreamID iID) const override {
      ++m_count;
      std::unique_ptr<UnsafeCache> sCache(new UnsafeCache);
      ++(sCache->strm);
      sCache->value = iID.value();
      return sCache;
    }
    
    void streamBeginRun(edm::StreamID iID, edm::Run const&, edm::EventSetup const&) const  override{
      ++m_count;
      auto sCache = streamCache(iID);
      if ( sCache->value != iID.value() ) {
          throw cms::Exception("cache value")
          << (streamCache(iID))->value << " but it was supposed to be " << iID;
      }
      if ( sCache->run != 0 || sCache->lumi !=0 || sCache->work !=0 || sCache->strm !=1 ) {
        throw cms::Exception("out of sequence")
          << "streamBeginRun out of sequence in Stream " << iID.value();
      }      
      ++(sCache->run);
     }

    void streamBeginLuminosityBlock(edm::StreamID iID, edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
      auto sCache = streamCache(iID);
      if ( sCache->value != iID.value() ) {
          throw cms::Exception("cache value")
          << (streamCache(iID))->value << " but it was supposed to be " << iID;
      }
      if ( sCache->lumi != 0 || sCache->work != 0 ) {
        throw cms::Exception("out of sequence")
          << "streamBeginLuminosityBlock out of sequence in Stream " << iID.value();
      }      
       ++(sCache->lumi);
     }

    bool filter(edm::StreamID iID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      auto sCache = streamCache(iID);
      if ( sCache->value != iID.value() ) {
          throw cms::Exception("cache value")
          << (streamCache(iID))->value << " but it was supposed to be " << iID;
      }
      ++(sCache->work);
      if ( sCache->lumi == 0 && sCache->run == 0) {
        throw cms::Exception("out of sequence")
          << "produce out of sequence in Stream " << iID.value();
      }       
       
      return true;
    }
 
    void streamEndLuminosityBlock(edm::StreamID iID, edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
      auto sCache = streamCache(iID);
      if ( sCache->value != iID.value() ) {
          throw cms::Exception("cache value")
          << (streamCache(iID))->value << " but it was supposed to be " << iID;
      }
      --(sCache->lumi);
      sCache->work = 0;
      if ( sCache->lumi != 0 || sCache->run == 0 ) {
        throw cms::Exception("out of sequence")
          << "streamEndLuminosityBlock out of sequence in Stream " << iID.value();
      }      
     }

    void streamEndRun(edm::StreamID iID, edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      auto sCache = streamCache(iID);
      if ( sCache->value != iID.value() ) {
          throw cms::Exception("cache value")
          << (streamCache(iID))->value << " but it was supposed to be " << iID;
      }
      --(sCache->run);
      sCache->work = 0;
      if ( sCache->run != 0 || sCache->lumi != 0 ) {
        throw cms::Exception("out of sequence")
          << "streamEndRun out of sequence in Stream " << iID.value();
      }      
    }

    void endStream(edm::StreamID iID) const override {
      ++m_count;
      auto sCache = streamCache(iID);
      --(sCache->strm);
      if ( sCache->value != iID.value() ) {
          throw cms::Exception("cache value")
          << (streamCache(iID))->value << " but it was supposed to be " << iID;
      }
      if ( sCache->strm != 0 || sCache->run != 0 || sCache->lumi != 0 ) {
        throw cms::Exception("out of sequence")
          << "endStream out of sequence in Stream " << iID.value();
      }      
   }

    ~StreamIntFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "StreamIntFilter transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };
  
  class RunIntFilter : public edm::global::EDFilter<edm::StreamCache<UnsafeCache>,edm::RunCache<Cache>> {
  public:
    explicit RunIntFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) 
        ,cvalue_(p.getParameter<int>("cachevalue")) 
    {    
    produces<unsigned int>();
    }
    const unsigned int trans_; 
    const unsigned int cvalue_; 
    mutable std::atomic<unsigned int> m_count{0};
    
    std::shared_ptr<Cache> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      auto rCache = std::make_shared<Cache>();
      ++(rCache->run);
      return rCache;
    }
 
    std::unique_ptr<UnsafeCache> beginStream(edm::StreamID) const override {
      return std::unique_ptr<UnsafeCache>{new UnsafeCache};
    }

    void streamBeginRun(edm::StreamID iID, edm::Run const& iRun, edm::EventSetup const&) const  override {
      auto rCache = runCache(iRun.index());
      if ( rCache->run == 0 ) {
        throw cms::Exception("out of sequence")
          << "streamBeginRun before globalBeginRun in Stream " << iID.value();
      }      
    }
 
    bool filter(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const override {
      ++m_count;
      auto rCache = runCache(iEvent.getRun().index());
      ++(rCache->value);
       
      return true;
    }

    void streamEndRun(edm::StreamID iID, edm::Run const& iRun, edm::EventSetup const&) const override {
      auto rCache = runCache(iRun.index());
      if ( rCache->run == 0 ) {
        throw cms::Exception("out of sequence")
          << "streamEndRun after globalEndRun in Stream " << iID.value();
      }      
    }


    void globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const override {
      ++m_count;
      auto rCache = runCache(iRun.index());
      if ( rCache->value != cvalue_ ) {
          throw cms::Exception("cache value")
          << "RunIntFilter cache value "
          << rCache->value << " but it was supposed to be " << cvalue_;
      }
     --(rCache->run);
    }

    ~RunIntFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "RunIntFilter transitions "     
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };


  class LumiIntFilter : public edm::global::EDFilter<edm::StreamCache<UnsafeCache>,edm::LuminosityBlockCache<Cache>> {
  public:
    explicit LumiIntFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) 
        ,cvalue_(p.getParameter<int>("cachevalue")) 
    {    
    produces<unsigned int>();
    }
    const unsigned int trans_; 
    const unsigned int cvalue_; 
    mutable std::atomic<unsigned int> m_count{0};
    
    std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
      auto lCache = std::make_shared<Cache>();
      ++(lCache->lumi);  
      return lCache;
    }

   std::unique_ptr<UnsafeCache> beginStream(edm::StreamID iID) const override {
      return std::unique_ptr<UnsafeCache>{new UnsafeCache};
    }

   void streamBeginLuminosityBlock(edm::StreamID iID, edm::LuminosityBlock const& iLB, edm::EventSetup const&) const override {
      auto lCache = luminosityBlockCache(iLB.index()); 
      if ( lCache->lumi == 0) {
        throw cms::Exception("out of sequence")
          << "streamBeginLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock" << iLB.luminosityBlock();
       }
   }


    bool filter(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const override {
      ++m_count;
      ++(luminosityBlockCache(iEvent.getLuminosityBlock().index())->value);
      return true;
    }
 
   void streamEndLuminosityBlock(edm::StreamID iID, edm::LuminosityBlock const& iLB, edm::EventSetup const&) const override {
      auto lCache = luminosityBlockCache(iLB.index()); 
      if ( lCache->lumi == 0) {
        throw cms::Exception("out of sequence")
          << "streamEndLuminosityBlock seen before globalEndLuminosityBlock in LuminosityBlock" << iLB.luminosityBlock();
       }
    }
 
    
    void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&) const override {
      ++m_count;
      auto lCache = luminosityBlockCache(iLB.index());
      --(lCache->lumi); 
      if ( lCache->lumi != 0 ) {
        throw cms::Exception("end out of sequence")
          << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock" << iLB.luminosityBlock();
      }
      if( lCache->value != cvalue_) {
        throw cms::Exception("cache value")
          << "LumiIntFilter cache value "
          << lCache->value << " but it was supposed to be " << cvalue_;
      }
    }

    ~LumiIntFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "LumiIntFilter transitions " 
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };
  
  class RunSummaryIntFilter : public edm::global::EDFilter<edm::StreamCache<Cache>,edm::RunSummaryCache<UnsafeCache>> {
  public:
    explicit RunSummaryIntFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) 
        ,cvalue_(p.getParameter<int>("cachevalue")) 
    {
    produces<unsigned int>();
      
    }
    const unsigned int trans_; 
    const unsigned int cvalue_; 
    mutable std::atomic<unsigned int> m_count{0};

    std::unique_ptr<Cache> beginStream(edm::StreamID) const override {
      ++m_count;
      return std::unique_ptr<Cache>(new Cache);
    }
    
    std::shared_ptr<UnsafeCache> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      auto gCache = std::make_shared<UnsafeCache>();
      ++(gCache->run);
      return gCache;
    }
    
    bool filter(edm::StreamID iID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      ++((streamCache(iID))->value);
       
      return true;
    }
    
    void streamEndRunSummary(edm::StreamID iID, edm::Run const&, edm::EventSetup const&, UnsafeCache* gCache) const override {
      ++m_count;
      if ( gCache->run == 0 ) {
        throw cms::Exception("out of sequence")
          << "streamEndRunSummary after globalEndRunSummary in Stream " << iID.value();
      }      
      auto sCache = streamCache(iID);
      gCache->value += sCache->value;
      sCache->value = 0;
    }
    
    void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, UnsafeCache* gCache) const override {
      ++m_count;
      if( gCache->value  != cvalue_) {
        throw cms::Exception("cache value")
          << "RunSummaryIntFilter cache value "
          << gCache->value << " but it was supposed to be " << cvalue_;
      }
      --(gCache->run);
    }

    ~RunSummaryIntFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "RunSummaryIntFilter transitions " 
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

  class LumiSummaryIntFilter : public edm::global::EDFilter<edm::StreamCache<Cache>,edm::LuminosityBlockSummaryCache<UnsafeCache>> {
  public:
    explicit LumiSummaryIntFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) 
        ,cvalue_(p.getParameter<int>("cachevalue")) 
    {	
    produces<unsigned int>();
      
    }
    const unsigned int trans_; 
    const unsigned int cvalue_; 
    mutable std::atomic<unsigned int> m_count{0};

    std::unique_ptr<Cache> beginStream(edm::StreamID) const override {
      ++m_count;
      return std::unique_ptr<Cache>(new Cache);
    }
  
    std::shared_ptr<UnsafeCache> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
      auto gCache = std::make_shared<UnsafeCache>();
      ++(gCache->lumi);
      return gCache;
    }

    bool filter(edm::StreamID iID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      ++((streamCache(iID))->value);
      return true;
    }
    
    void streamEndLuminosityBlockSummary(edm::StreamID iID, edm::LuminosityBlock const&, edm::EventSetup const&, UnsafeCache* gCache) const override {
      ++m_count;
      if ( gCache->lumi == 0 ) {
        throw cms::Exception("out of sequence")
          << "streamEndLuminosityBlockSummary after globalEndLuminosityBlockSummary in Stream " << iID.value();
      }
      auto sCache = streamCache(iID);
      gCache->value += sCache->value;
      sCache->value = 0;
    }
    
    void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, UnsafeCache* gCache) const override {
      ++m_count;
      if( gCache->value != cvalue_) {
        throw cms::Exception("cache value")
          << "LumiSummaryIntFilter cache value "
          << gCache->value << " but it was supposed to be " << cvalue_;
      }
      --(gCache->lumi);
    }

    ~LumiSummaryIntFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "LumiSummaryIntFilter transitions "     
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

  class TestBeginRunFilter : public edm::global::EDFilter<edm::RunCache<void>,edm::BeginRunProducer> {
  public:
    explicit TestBeginRunFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    produces<unsigned int>();
    }

    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
    mutable std::atomic<bool> brp{false}; 

    std::shared_ptr<void> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&) const override {
      brp = false;
      return std::shared_ptr<void>();
    }
 
    void globalBeginRunProduce(edm::Run&, edm::EventSetup const&) const override {
      ++m_count;
      brp = true;
     }

    bool filter(edm::StreamID iID, edm::Event&, edm::EventSetup const&) const override {
      if ( !brp ) {
        throw cms::Exception("out of sequence")
          << "filter before globalBeginRunProduce in Stream " << iID.value();
      }
      return true;
    }

    void globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const override {
    }

    ~TestBeginRunFilter() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "TestBeginRunFilter transitions " 
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

  class TestEndRunFilter : public edm::global::EDFilter<edm::RunCache<void>,edm::EndRunProducer> {
  public:
    explicit TestEndRunFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    produces<unsigned int>();
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
    mutable std::atomic<bool> p{false}; 

    std::shared_ptr<void> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&) const override {
      p = false;
      return std::shared_ptr<void>();
    }


    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      p = true;
      return true;
    }

    void globalEndRunProduce(edm::Run&, edm::EventSetup const&) const override {
      if ( !p ) {
        throw cms::Exception("out of sequence")
          << "endRunProduce before produce";
      }
       ++m_count;
    }

    void globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const override {
    }

    ~TestEndRunFilter() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "TestEndRunFilter transitions " 
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

  class TestBeginLumiBlockFilter : public edm::global::EDFilter<edm::LuminosityBlockCache<void>,edm::BeginLuminosityBlockProducer> {
  public:
    explicit TestBeginLumiBlockFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    produces<unsigned int>();
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
    mutable std::atomic<bool> gblp{false}; 

    std::shared_ptr<void> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&) const override {
      gblp = false;
      return std::shared_ptr<void>();
    }


    void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const {
      ++m_count;
      gblp = true;
    }

    bool filter(edm::StreamID iID, edm::Event&, const edm::EventSetup&) const override{
      if ( !gblp ) {
        throw cms::Exception("out of sequence")
          << "filter before globalBeginLuminosityBlockProduce in Stream " << iID.value();
      }
      return true;
    }

    void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&) const override {
    }

    ~TestBeginLumiBlockFilter() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "TestBeginLumiBlockFilter transitions " 
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

  class TestEndLumiBlockFilter : public edm::global::EDFilter<edm::LuminosityBlockCache<void>,edm::EndLuminosityBlockProducer> {
  public:
    explicit TestEndLumiBlockFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    produces<unsigned int>();
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
    mutable std::atomic<bool> p{false}; 

    std::shared_ptr<void> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&) const override {
      p = false;
      return std::shared_ptr<void>();
    }


    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      p = true;
      return true;
    }

    void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const override {
      if ( !p ) {
        throw cms::Exception("out of sequence")
          << "endLumiBlockProduce before produce";
      }
      ++m_count;
    }

    void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&) const override {
    }

    ~TestEndLumiBlockFilter() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "TestEndLumiBlockFilter transitions " 
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };


}
}

DEFINE_FWK_MODULE(edmtest::global::StreamIntFilter);
DEFINE_FWK_MODULE(edmtest::global::RunIntFilter);
DEFINE_FWK_MODULE(edmtest::global::LumiIntFilter);
DEFINE_FWK_MODULE(edmtest::global::RunSummaryIntFilter);
DEFINE_FWK_MODULE(edmtest::global::LumiSummaryIntFilter);
DEFINE_FWK_MODULE(edmtest::global::TestBeginRunFilter);
DEFINE_FWK_MODULE(edmtest::global::TestBeginLumiBlockFilter);
DEFINE_FWK_MODULE(edmtest::global::TestEndRunFilter);
DEFINE_FWK_MODULE(edmtest::global::TestEndLumiBlockFilter);

