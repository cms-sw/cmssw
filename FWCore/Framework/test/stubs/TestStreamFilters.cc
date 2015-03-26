
/*----------------------------------------------------------------------

Toy edm::stream::EDFilter modules of 
edm::*Cache templates and edm::*Producer classes
for testing purposes only.

----------------------------------------------------------------------*/
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "FWCore/Framework/interface/stream/EDFilter.h"
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
namespace stream {

// anonymous namespace here causes build warnings
namespace cache {
struct Cache {
   Cache():value(0),run(0),lumi(0) {}
   //Using mutable since we want to update the value.
   mutable std::atomic<unsigned int> value;
   mutable std::atomic<unsigned int> run;
   mutable std::atomic<unsigned int> lumi;
};

} //end cache namespace

  using Cache = cache::Cache;

  class GlobalIntFilter : public edm::stream::EDFilter<edm::GlobalCache<Cache>> {
  public:
    static std::atomic<unsigned int> m_count; 
    unsigned int trans_;
    static std::atomic<unsigned int> cvalue_;
    
    static std::unique_ptr<Cache> initializeGlobalCache(edm::ParameterSet const&) {
      ++m_count;
      return std::unique_ptr<Cache>{ new Cache };
    }

    GlobalIntFilter(edm::ParameterSet const& p, const Cache* iGlobal) {
      trans_ = p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      produces<unsigned int>();
    }
    
    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      ++((globalCache())->value);
       
      return true;
    }
    
    static void globalEndJob(Cache* iGlobal) {
      ++m_count;
      if(iGlobal->value != cvalue_) {
        throw cms::Exception("cache value")
          << iGlobal->value << " but it was supposed to be " << cvalue_;
      }
    }

    ~GlobalIntFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << m_count << " but it was supposed to be " << trans_;
      }
    }

    
  };

  class RunIntFilter : public edm::stream::EDFilter<edm::RunCache<Cache>> {
  public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static std::atomic<unsigned int> cvalue_;
    static std::atomic<bool> gbr;
    static std::atomic<bool> ger;
    bool br;
    bool er;

    RunIntFilter(edm::ParameterSet const&p){
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      m_count = 0;
      produces<unsigned int>();
    }

    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      ++(runCache()->value);
       
      return true;
    }
    
    static std::shared_ptr<Cache> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
      ++m_count;
      gbr = true;
      ger = false;
      auto pCache = std::make_shared<Cache>();
      ++(pCache->run);
      return pCache;
   }

    static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
       ++m_count;
      auto pCache = iContext->run();
      if ( pCache->run != 1 ) {
        throw cms::Exception("end out of sequence")
          << "globalEndRun seen before globalBeginRun in Run" << iRun.run();
      } 
      ger = true;
      gbr = false;
      if( iContext->run()->value != cvalue_) {
        throw cms::Exception("cache value")
          << iContext->run()->value << " but it was supposed to be " << cvalue_;
      }
    }

    ~RunIntFilter() {
       if(m_count != trans_) {
        throw cms::Exception("transitions")
          << m_count << " but it was supposed to be " << trans_;
      }

    }
  };


  class LumiIntFilter : public edm::stream::EDFilter<edm::LuminosityBlockCache<Cache>> {
  public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static std::atomic<unsigned int> cvalue_;
    static std::atomic<bool> gbl;
    static std::atomic<bool> gel;
    static std::atomic<bool> bl;
    static std::atomic<bool> el;

    LumiIntFilter(edm::ParameterSet const&p){
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      m_count = 0;
      produces<unsigned int>();
    }

    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      ++(luminosityBlockCache()->value);
       
      return true;
    }
    
    static std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&, RunContext const*) {
      ++m_count;
      gbl = true;
      gel = false;
      auto pCache = std::make_shared<Cache>();
      ++(pCache->lumi);
      return pCache;
   }

    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {
      bl = true;
      el = false;
      if ( !gbl ) {
        throw cms::Exception("begin out of sequence")
          << "beginLuminosityBlock seen before globalBeginLuminosityBlock";
      }
    }

 
    static void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&, LuminosityBlockContext const* iLBContext) {
      ++m_count;
      if(iLBContext->luminosityBlock()->value != cvalue_) {
        throw cms::Exception("cache value")
          << "LumiIntFilter cache value " 
          << iLBContext->luminosityBlock()->value << " but it was supposed to be " << cvalue_;
      }
    }

   static void endLuminosityBlock(edm::Run const&, edm::EventSetup const&, LuminosityBlockContext const*) {
      el = true;
      bl = false;
      if ( gel ) {
        throw cms::Exception("end out of sequence")
          << "globalEndLuminosityBlock seen before endLuminosityBlock";
      }      
   }
   

    ~LumiIntFilter() {
       if(m_count != trans_) {
        throw cms::Exception("transitions")
          << m_count<< " but it was supposed to be " << trans_;
       }
    }
  };
  
  class RunSummaryIntFilter : public edm::stream::EDFilter<edm::RunCache<Cache>,edm::RunSummaryCache<Cache>> {
  public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static std::atomic<unsigned int> cvalue_;
    static std::atomic<bool> gbr;
    static std::atomic<bool> ger;
    static std::atomic<bool> gbrs;
    static std::atomic<bool> gers;
    static std::atomic<bool> brs;
    static std::atomic<bool> ers;
    static std::atomic<bool> br;
    static std::atomic<bool> er;

    RunSummaryIntFilter(edm::ParameterSet const&p){
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      m_count = 0;
      produces<unsigned int>();
    }

    void beginRun(edm::Run const&, edm::EventSetup const&) override {
      br=true;
      er=false;
    }

    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      ++(runCache()->value);
       
      return true;
    }
    
    static std::shared_ptr<Cache> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
      ++m_count;
      gbr=true;
      ger=false;
      auto pCache = std::make_shared<Cache>();
      ++(pCache->run);
      return pCache;
   }

    static std::shared_ptr<Cache> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&, GlobalCache const*) {
      ++m_count;
      gbrs = true;
      gers = false;
      brs = true;
      ers = false;
      if ( !gbr ) {
        throw cms::Exception("begin out of sequence")
          << "globalBeginRunSummary seen before globalBeginRun";
      }
      return std::make_shared<Cache>();
   }
    
    void endRunSummary(edm::Run const&, edm::EventSetup const&, Cache* gCache) const override {
      brs=false;
      ers=true;
      gCache->value += runCache()->value;
      runCache()->value = 0;
      if ( !er ) {
        throw cms::Exception("end out of sequence")
          << "endRunSummary seen before endRun";
      }
    }
    
    static void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, RunContext const*, Cache* gCache){
      ++m_count;
      gbrs=false;
      gers=true;
      if ( !ers ) {
        throw cms::Exception("end out of sequence")
          << "globalEndRunSummary seen before endRunSummary";
      }
      if(gCache->value != cvalue_) {
        throw cms::Exception("cache value")
          << gCache->value << " but it was supposed to be " << cvalue_;
      }
    }

   static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
      ++m_count;
      gbr=false;
      ger=true;
      auto pCache = iContext->run();
      if ( pCache->run != 1 ) {
        throw cms::Exception("end out of sequence")
          << "globalEndRun seen before globalBeginRun in Run" << iRun.run();
      } 
  }

    void endRun(edm::Run const&, edm::EventSetup const&) override {
      er = true;
      br = false;
    }   



    ~RunSummaryIntFilter() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

  class LumiSummaryIntFilter : public edm::stream::EDFilter<edm::LuminosityBlockCache<Cache>,edm::LuminosityBlockSummaryCache<Cache>> {
  public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static std::atomic<unsigned int> cvalue_;
    static std::atomic<bool> gbl;
    static std::atomic<bool> gel;
    static std::atomic<bool> gbls;
    static std::atomic<bool> gels;
    static std::atomic<bool> bls;
    static std::atomic<bool> els;
    static std::atomic<bool> bl;
    static std::atomic<bool> el;

    LumiSummaryIntFilter(edm::ParameterSet const&p) {
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      m_count = 0;
      produces<unsigned int>();
    }

    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      ++(luminosityBlockCache()->value);
       
      return true;
    }

    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {
      bl = true;
      el = false;
    }

    static std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&, RunContext const*) {
      ++m_count;
      gbl = true;
      gel = false;
      auto pCache = std::make_shared<Cache>();
      ++(pCache->lumi);
      return pCache;
    }


    static std::shared_ptr<Cache> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, LuminosityBlockContext const*){
      ++m_count;
      gbls = true;
      gels = false;
      bls = true;
      els = false;
      if ( !gbl ) {
       throw cms::Exception("begin out of sequence")
         << "globalBeginLuminosityBlockSummary seen before globalBeginLuminosityBlock";
      }
      return std::make_shared<Cache>();
    }
    
    void endLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, Cache* gCache) const override {
      bls=false;
      els=true;
      gCache->value += luminosityBlockCache()->value;
      luminosityBlockCache()->value = 0;
      if ( el ) {
        throw cms::Exception("end out of sequence")
          << "endLuminosityBlock seen before endLuminosityBlockSummary";
      }
    }
    
    static void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, LuminosityBlockContext const*, Cache* gCache){
     ++m_count;
     gbls=false;
     gels=true;
      if ( !els ) {
        throw cms::Exception("end out of sequence")
          << "LumiSummaryIntFilter " 
          << "globalEndLuminosityBlockSummary seen before endLuminosityBlockSummary";
      }
      if( gCache->value != cvalue_) {
        throw cms::Exception("cache value")
          << gCache->value << " but it was supposed to be " << cvalue_;
      }
    }

   static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&, LuminosityBlockContext const* iLBContext) {
      ++m_count;
      auto pCache = iLBContext->luminosityBlock();
      if ( pCache->lumi != 1 ) {
        throw cms::Exception("end out of sequence")
          << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock" << iLB.luminosityBlock();
      }       
      gel = true;
      gbl = false;
      if ( !gels ) {
        throw cms::Exception("end out of sequence")
          << "globalEndLuminosityBlockSummary seen before globalEndLuminosityBlock";  
      }
   }

    static void endLuminosityBlock(edm::Run const&, edm::EventSetup const&, LuminosityBlockContext const*) {
      el = true;
      bl = false;
    }


    ~LumiSummaryIntFilter() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

  class TestBeginRunFilter : public edm::stream::EDFilter<edm::RunCache<Cache>,edm::BeginRunProducer> {
    public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static std::atomic<unsigned int> cvalue_;
    static std::atomic<bool> gbr;
    static std::atomic<bool> ger;

    TestBeginRunFilter(edm::ParameterSet const&p){
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      m_count = 0;
      produces<unsigned int>();
    }

  static std::shared_ptr<Cache> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
     ++m_count;
      gbr=true;
      ger=false;
      auto pCache = std::make_shared<Cache>();
      ++(pCache->run);
      return pCache;
   }

    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      return true;
    }

    static void globalBeginRunProduce(edm::Run& iRun, edm::EventSetup const&, RunContext const*) {
      ++m_count;
      if ( !gbr ) {
        throw cms::Exception("begin out of sequence")
          << "globalBeginRunProduce seen before globalBeginRun";
      }
    }

    static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
     ++m_count;
      auto pCache = iContext->run();
      if ( pCache->run != 1 ) {
        throw cms::Exception("end out of sequence")
          << "globalEndRun seen before globalBeginRun in Run" << iRun.run();
      } 
      gbr=false;
      ger=true;
    }

    ~TestBeginRunFilter() {
    if(m_count != trans_) {
       throw cms::Exception("transitions")
         << m_count<< " but it was supposed to be " << trans_;
     }
    }
  };

  class TestEndRunFilter : public edm::stream::EDFilter<edm::RunCache<Cache>,edm::EndRunProducer> {
    public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static std::atomic<unsigned int> cvalue_;
    static std::atomic<bool> gbr;
    static std::atomic<bool> ger;

  static std::shared_ptr<Cache> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
     ++m_count;
      gbr=true;
      ger=false;
      auto pCache = std::make_shared<Cache>();
      ++(pCache->run);
      return pCache;
   }

 
    TestEndRunFilter(edm::ParameterSet const&p){
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      m_count = 0;
      produces<unsigned int>();
    }

    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
       
      return true;
    }

    static void globalEndRunProduce(edm::Run& iRun, edm::EventSetup const&, RunContext const*) {
      ++m_count;
      if ( ger ) {
        throw cms::Exception("end out of sequence")
          << "globalEndRun seen before globalEndRunProduce";
      }
    }

    static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
      ++m_count;
      auto pCache = iContext->run();
      if ( pCache->run != 1 ) {
        throw cms::Exception("end out of sequence")
          << "globalEndRun seen before globalBeginRun in Run" << iRun.run();
      } 
      gbr=false;
      ger=true;
    }


    ~TestEndRunFilter() {
    if(m_count != trans_) {
       throw cms::Exception("transitions")
         << m_count<< " but it was supposed to be " << trans_;
     }
    }
  };

  class TestBeginLumiBlockFilter : public edm::stream::EDFilter<edm::LuminosityBlockCache<Cache>,edm::BeginLuminosityBlockProducer> {
    public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static std::atomic<unsigned int> cvalue_;
    static std::atomic<bool> gbl;
    static std::atomic<bool> gel;
 
    TestBeginLumiBlockFilter(edm::ParameterSet const&p){
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      m_count = 0;
      produces<unsigned int>();
    }

    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
       
      return true;
    }

    static void globalBeginLuminosityBlockProduce(edm::LuminosityBlock& , edm::EventSetup const&, LuminosityBlockContext const*) {
      ++m_count;
      if ( !gbl ) {
        throw cms::Exception("begin out of sequence")
          << "globalBeginLumiBlockProduce seen before globalBeginLumiBlock";
      }
    }

    static std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&, RunContext const*) {
      ++m_count;
      gbl = true;
      gel = false;
      auto pCache = std::make_shared<Cache>();
      ++(pCache->lumi);
      return pCache;
   }

    static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&, LuminosityBlockContext const* iLBContext) {
      ++m_count;
      auto pCache = iLBContext->luminosityBlock();
      if ( pCache->lumi != 1 ) {
        throw cms::Exception("end out of sequence")
          << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock" << iLB.luminosityBlock();
      }
      gel = true;
      gbl = false;
    }


    ~TestBeginLumiBlockFilter() {
    if(m_count != trans_) {
       throw cms::Exception("transitions")
         << m_count<< " but it was supposed to be " << trans_;
     }
    }
  };

  class TestEndLumiBlockFilter : public edm::stream::EDFilter<edm::LuminosityBlockCache<Cache>,edm::EndLuminosityBlockProducer> {
    public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static std::atomic<unsigned int> cvalue_;
    static std::atomic<bool> gbl;
    static std::atomic<bool> gel;
 
    TestEndLumiBlockFilter(edm::ParameterSet const&p){
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      m_count = 0;
      produces<unsigned int>();
    }

    bool filter(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
       
      return true;
    }

    static std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&, RunContext const*) {
      ++m_count;
      gbl = true;
      gel = false;
      auto pCache = std::make_shared<Cache>();
      ++(pCache->lumi);
      return pCache;
   }

  static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&, LuminosityBlockContext const* iLBContext) {
      ++m_count;
      auto pCache = iLBContext->luminosityBlock();
      if ( pCache->lumi != 1 ) {
        throw cms::Exception("end out of sequence")
          << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock" << iLB.luminosityBlock();
      }
      gel = true;
      gbl = false;
   }


    static void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&, LuminosityBlockContext const*) {
      ++m_count;
    }

    ~TestEndLumiBlockFilter() {
    if(m_count != trans_) {
       throw cms::Exception("transitions")
         << m_count<< " but it was supposed to be " << trans_;
     }
    }
  };





}
}
std::atomic<unsigned int> edmtest::stream::GlobalIntFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::RunIntFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::LumiIntFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::RunSummaryIntFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::LumiSummaryIntFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestBeginRunFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestEndRunFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestBeginLumiBlockFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestEndLumiBlockFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::GlobalIntFilter::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::RunIntFilter::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::LumiIntFilter::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::RunSummaryIntFilter::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::LumiSummaryIntFilter::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::TestBeginRunFilter::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::TestEndRunFilter::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::TestBeginLumiBlockFilter::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::TestEndLumiBlockFilter::cvalue_{0};
std::atomic<bool> edmtest::stream::RunIntFilter::gbr{false};
std::atomic<bool> edmtest::stream::RunIntFilter::ger{false};
std::atomic<bool> edmtest::stream::LumiIntFilter::gbl{false};
std::atomic<bool> edmtest::stream::LumiIntFilter::gel{false};
std::atomic<bool> edmtest::stream::LumiIntFilter::bl{false};
std::atomic<bool> edmtest::stream::LumiIntFilter::el{false};
std::atomic<bool> edmtest::stream::RunSummaryIntFilter::gbr{false};
std::atomic<bool> edmtest::stream::RunSummaryIntFilter::ger{false};
std::atomic<bool> edmtest::stream::RunSummaryIntFilter::gbrs{false};
std::atomic<bool> edmtest::stream::RunSummaryIntFilter::gers{false};
std::atomic<bool> edmtest::stream::RunSummaryIntFilter::brs{false};
std::atomic<bool> edmtest::stream::RunSummaryIntFilter::ers{false};
std::atomic<bool> edmtest::stream::RunSummaryIntFilter::br{false};
std::atomic<bool> edmtest::stream::RunSummaryIntFilter::er{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntFilter::gbl{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntFilter::gel{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntFilter::gbls{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntFilter::gels{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntFilter::bls{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntFilter::els{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntFilter::bl{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntFilter::el{false};
std::atomic<bool> edmtest::stream::TestBeginRunFilter::gbr{false};
std::atomic<bool> edmtest::stream::TestBeginRunFilter::ger{false};
std::atomic<bool> edmtest::stream::TestEndRunFilter::gbr{false};
std::atomic<bool> edmtest::stream::TestEndRunFilter::ger{false};
std::atomic<bool> edmtest::stream::TestBeginLumiBlockFilter::gbl{false};
std::atomic<bool> edmtest::stream::TestBeginLumiBlockFilter::gel{false};
std::atomic<bool> edmtest::stream::TestEndLumiBlockFilter::gbl{false};
std::atomic<bool> edmtest::stream::TestEndLumiBlockFilter::gel{false};
DEFINE_FWK_MODULE(edmtest::stream::GlobalIntFilter);
DEFINE_FWK_MODULE(edmtest::stream::RunIntFilter);
DEFINE_FWK_MODULE(edmtest::stream::LumiIntFilter);
DEFINE_FWK_MODULE(edmtest::stream::RunSummaryIntFilter);
DEFINE_FWK_MODULE(edmtest::stream::LumiSummaryIntFilter);
DEFINE_FWK_MODULE(edmtest::stream::TestBeginRunFilter);
DEFINE_FWK_MODULE(edmtest::stream::TestEndRunFilter);
DEFINE_FWK_MODULE(edmtest::stream::TestBeginLumiBlockFilter);
DEFINE_FWK_MODULE(edmtest::stream::TestEndLumiBlockFilter);
