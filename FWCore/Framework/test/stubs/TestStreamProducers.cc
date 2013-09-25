
/*----------------------------------------------------------------------

Toy edm::stream::EDProducer modules of 
edm::*Cache templates and edm::*Producer classes 
for testing purposes only.

----------------------------------------------------------------------*/
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "FWCore/Framework/interface/stream/EDProducer.h"
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

struct Cache { 
   Cache():value(0),run(0),lumi(0) {}
   //Using mutable since we want to update the value.
   mutable std::atomic<unsigned int> value;
   mutable std::atomic<unsigned int> run;
   mutable std::atomic<unsigned int> lumi;
};


  class GlobalIntProducer : public edm::stream::EDProducer<edm::GlobalCache<Cache>> {
  public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static unsigned int cvalue_;

    static std::unique_ptr<Cache> initializeGlobalCache(edm::ParameterSet const&) {
      ++m_count;
      return std::unique_ptr<Cache>{new Cache()};
    }

    GlobalIntProducer(edm::ParameterSet const& p, const Cache* iGlobal)  {
      trans_ = p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      produces<unsigned int>();
    }

    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      ++((globalCache())->value);
       
    }
    
    static void globalEndJob(Cache* iGlobal) {
      ++m_count;
      if(iGlobal->value != cvalue_) {
        throw cms::Exception("cache value")
          << iGlobal->value << " but it was supposed to be " << cvalue_;
      }
    }

    ~GlobalIntProducer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << m_count << " but it was supposed to be " << trans_;
      }
    }

    
  };

  class RunIntProducer : public edm::stream::EDProducer<edm::RunCache<Cache>> {
  public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static unsigned int cvalue_;
    static bool gbr;
    static bool ger;
    bool br;
    bool er;

    RunIntProducer(edm::ParameterSet const&p) {
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      produces<unsigned int>();
    }

    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      ++(runCache()->value);
       
    }

    static std::shared_ptr<Cache> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
      ++m_count;
      gbr = true;
      ger = false;
      std::shared_ptr<Cache> pCache = std::shared_ptr<Cache>{new Cache()};
      ++(pCache->run);
      return pCache;
    }

    void beginRun(edm::Run const&, edm::EventSetup const&) override {
      br = true;
      er = true;
      if ( !gbr ) {
        throw cms::Exception("begin out of sequence")
          << "beginRun seen before globalBeginRun";
      }
    }
   
    static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
      ++m_count;
      Cache const * pCache = iContext->run();
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

    void endRun(edm::Run const&, edm::EventSetup const&) override {
      er = true;
      br = false;
      if ( ger ) {
        throw cms::Exception("end out of sequence")
          << "globalEndRun seen before endRun";
      }
    }

    ~RunIntProducer() {
       if(m_count != trans_) {
        throw cms::Exception("transitions")
          << m_count << " but it was supposed to be " << trans_;
      }

    }
  };


  class LumiIntProducer : public edm::stream::EDProducer<edm::LuminosityBlockCache<Cache>> {
  public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static unsigned int cvalue_;
    static bool gbl;
    static bool gel;
    static bool bl;
    static bool el;


    LumiIntProducer(edm::ParameterSet const&p) {
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      produces<unsigned int>();
    }

    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      ++(luminosityBlockCache()->value);
       
    }
    
     static std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&, RunContext const*) {
      ++m_count;
      gbl = true;
      gel = false;
      std::shared_ptr<Cache> pCache = std::shared_ptr<Cache>{new Cache()};
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

   
    static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&, LuminosityBlockContext const* iLBContext) {
      ++m_count;
      Cache const * pCache = iLBContext->luminosityBlock();
      if ( pCache->lumi != 1 ) {
        throw cms::Exception("end out of sequence")
          << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock" << iLB.luminosityBlock();
      }       
      gel = true;
      gbl = false;
      if(iLBContext->luminosityBlock()->value != cvalue_) {
        throw cms::Exception("cache value")
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


    ~LumiIntProducer() {
       if(m_count != trans_) {
        throw cms::Exception("transitions")
          << m_count<< " but it was supposed to be " << trans_;
       }
    }
  };
  
  class RunSummaryIntProducer : public edm::stream::EDProducer<edm::RunCache<Cache>,edm::RunSummaryCache<Cache>> {
  public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static unsigned int cvalue_;
    static bool gbr;
    static bool ger;
    static bool gbrs;
    static bool gers;
    static bool brs;
    static bool ers;
    static bool br;
    static bool er;

    RunSummaryIntProducer(edm::ParameterSet const&p){
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      produces<unsigned int>();
    }

    void beginRun(edm::Run const&, edm::EventSetup const&) override {
      br=true;
      er=false;
    }

    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      ++(runCache()->value);
       
    }
    
    static std::shared_ptr<Cache> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
      ++m_count;
      gbr=true;
      ger=false;
      std::shared_ptr<Cache> pCache = std::shared_ptr<Cache>{new Cache()};
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
      return std::shared_ptr<Cache>{new Cache()};
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
    
    static void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, RunContext const*, Cache * gCache) {
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
      Cache const * pCache = iContext->run();
      if ( pCache->run != 1 ) {
        throw cms::Exception("end out of sequence")
          << "globalEndRun seen before globalBeginRun in Run" << iRun.run();
      } 
    }

    void endRun(edm::Run const&, edm::EventSetup const&) override {
      er = true;
      br = false;
    }   

    ~RunSummaryIntProducer() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << m_count<< " but it was supposed to be " << trans_;
      }
 
    }
  };

  class LumiSummaryIntProducer : public edm::stream::EDProducer<edm::LuminosityBlockCache<Cache>,edm::LuminosityBlockSummaryCache<Cache>> {
  public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static unsigned int cvalue_;
    static bool gbl;
    static bool gel;
    static bool gbls;
    static bool gels;
    static bool bls;
    static bool els;
    static bool bl;
    static bool el;

    LumiSummaryIntProducer(edm::ParameterSet const&p){
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      produces<unsigned int>();
    }

    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      ++(luminosityBlockCache()->value);
       
    }

    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {
      bl = true;
      el = false;
    }
 
    static std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&, RunContext const*) {
      ++m_count;
      gbl = true;
      gel = false;
      std::shared_ptr<Cache> pCache = std::shared_ptr<Cache>{new Cache()};
      ++(pCache->lumi);
      return pCache;
    }


    static std::shared_ptr<Cache> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, LuminosityBlockContext const*) {
      ++m_count;
      gbls = true;
      gels = false;
      bls = true;
      els = false;
      if ( !gbl ) {
       throw cms::Exception("begin out of sequence")
         << "globalBeginLuminosityBlockSummary seen before globalBeginLuminosityBlock";
      }
      return std::shared_ptr<Cache>{new Cache()};
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
    
   static void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, LuminosityBlockContext const*, Cache* gCache) {
     ++m_count;
     gbls=false;
     gels=true;
      if ( !els ) {
        throw cms::Exception("end out of sequence")
          << "globalEndLuminosityBlockSummary seen before endLuminosityBlockSummary";
      }
     if ( gCache->value != cvalue_) {
        throw cms::Exception("cache value")
          << gCache->value << " but it was supposed to be " << cvalue_;
     }
   }

   static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&, LuminosityBlockContext const* iLBContext) {
      ++m_count;
      Cache const * pCache = iLBContext->luminosityBlock();
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

    ~LumiSummaryIntProducer() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

  class TestBeginRunProducer : public edm::stream::EDProducer<edm::RunCache<Cache>,edm::BeginRunProducer> {
    public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static unsigned int cvalue_;
    static bool gbr;
    static bool ger;
    static bool gbrp;
 
    TestBeginRunProducer(edm::ParameterSet const&p) {
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      produces<unsigned int>();
    }

    static std::shared_ptr<Cache> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
      ++m_count;
      gbr=true;
      ger=false;
      gbrp=false;
      std::shared_ptr<Cache> pCache = std::shared_ptr<Cache>{new Cache()};
      ++(pCache->run);
      return pCache;
   }

    void produce(edm::Event&, edm::EventSetup const&) override {
      if ( !gbrp ) {
        throw cms::Exception("out of sequence")
          << "produce before globalBeginRunProduce";
      }
      ++m_count;               
    }

    static void globalBeginRunProduce(edm::Run& iRun, edm::EventSetup const&, RunContext const*) {
      gbrp = true;
      ++m_count;
      if ( !gbr ) {
        throw cms::Exception("begin out of sequence")
          << "globalBeginRunProduce seen before globalBeginRun";
      }
    }

    static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
      Cache const * pCache = iContext->run();
      if ( pCache->run != 1 ) {
        throw cms::Exception("end out of sequence")
          << "globalEndRun seen before globalBeginRun in Run" << iRun.run();
      } 
      ++m_count;
      gbr=false;
      ger=true;
    }


    ~TestBeginRunProducer() {
    if(m_count != trans_) {
       throw cms::Exception("transitions")
         << m_count<< " but it was supposed to be " << trans_;
     }
    }
  };

  class TestEndRunProducer : public edm::stream::EDProducer<edm::RunCache<Cache>,edm::EndRunProducer> {
    public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static unsigned int cvalue_;
    static bool gbr;
    static bool ger;
    static bool p;

    static std::shared_ptr<Cache> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
     ++m_count;
      gbr=true;
      ger=false;
      p =false;
      std::shared_ptr<Cache> pCache = std::shared_ptr<Cache>{new Cache()};
      ++(pCache->run);
      return pCache;
   }


    TestEndRunProducer(edm::ParameterSet const&p){
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      produces<unsigned int>();
    }

    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      p = true; 
    }

    static void globalEndRunProduce(edm::Run& iRun, edm::EventSetup const&, RunContext const*) {
      ++m_count;
      if ( !p ) {
        throw cms::Exception("out of sequence")
          << "globalEndRunProduce seen before produce";
      }
      if ( ger ) {
        throw cms::Exception("out of sequence")
          << "globalEndRun seen before globalEndRunProduce";
      }
    }

    static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
      ++m_count;
      Cache const * pCache = iContext->run();
      if ( pCache->run != 1 ) {
        throw cms::Exception("out of sequence")
          << "globalEndRun seen before globalBeginRun in Run" << iRun.run();
      } 
      gbr=false;
      ger=true;
    }


    ~TestEndRunProducer() {
    if(m_count != trans_) {
       throw cms::Exception("transitions")
         << m_count<< " but it was supposed to be " << trans_;
     }
    }
  };

  class TestBeginLumiBlockProducer : public edm::stream::EDProducer<edm::LuminosityBlockCache<Cache>,edm::BeginLuminosityBlockProducer> {
    public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static unsigned int cvalue_;
    static bool gbl;
    static bool gel;
    static bool gblp;
 
    TestBeginLumiBlockProducer(edm::ParameterSet const&p){
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      produces<unsigned int>();
    }

    void produce(edm::Event&, edm::EventSetup const&) override {
      if ( !gblp ) {
        throw cms::Exception("begin out of sequence")
          << "produce seen before globalBeginLumiBlockProduce";
      }
      ++m_count;
    }

    static void globalBeginLuminosityBlockProduce(edm::LuminosityBlock& , edm::EventSetup const&, LuminosityBlockContext const*) {
     ++m_count;
      if ( !gbl ) {
        throw cms::Exception("begin out of sequence")
          << "globalBeginLumiBlockProduce seen before globalBeginLumiBlock";
      }
      gblp = true;
    }

    static std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&, RunContext const*) {
      ++m_count;
      gbl = true;
      gel = false;
      gblp = false;
      std::shared_ptr<Cache> pCache = std::shared_ptr<Cache>{new Cache()};
      ++(pCache->lumi);
      return pCache;
   }

    static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&, LuminosityBlockContext const* iLBContext) {
      ++m_count;
      Cache const * pCache = iLBContext->luminosityBlock();
      if ( pCache->lumi != 1 ) {
        throw cms::Exception("end out of sequence")
          << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock" << iLB.luminosityBlock();
      }
      gel = true;
      gbl = false;
    }


    ~TestBeginLumiBlockProducer() {
    if(m_count != trans_) {
       throw cms::Exception("transitions")
         << m_count<< " but it was supposed to be " << trans_;
     }
    }
  };

  class TestEndLumiBlockProducer : public edm::stream::EDProducer<edm::LuminosityBlockCache<Cache>,edm::EndLuminosityBlockProducer> {
    public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static unsigned int cvalue_;
    static bool gbl;
    static bool gel;
    static bool p;
 
    TestEndLumiBlockProducer(edm::ParameterSet const&p){
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      produces<unsigned int>();
    }

    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      p = true;
    }

    static void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&, LuminosityBlockContext const*) {
      ++m_count;
      if ( !p ) {
        throw cms::Exception("out of sequence")
          << "globalEndLumiBlockProduce seen before produce";
      }
    }

    static std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&, RunContext const*) {
      ++m_count;
      gbl = true;
      gel = false;
      p = true;
      std::shared_ptr<Cache> pCache = std::shared_ptr<Cache>{new Cache()};
      ++(pCache->lumi);
      return pCache;
   }

    static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&, LuminosityBlockContext const* iLBContext) {
      ++m_count;
      Cache const * pCache = iLBContext->luminosityBlock();
      if ( pCache->lumi != 1 ) {
        throw cms::Exception("end out of sequence")
          << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock" << iLB.luminosityBlock();
      }
      gel = true;
      gbl = false;
   }


    ~TestEndLumiBlockProducer() {
    if(m_count != trans_) {
       throw cms::Exception("transitions")
         << m_count<< " but it was supposed to be " << trans_;
     }
    }
  };


   

}
}
std::atomic<unsigned int> edmtest::stream::GlobalIntProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::RunIntProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::LumiIntProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::RunSummaryIntProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::LumiSummaryIntProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestBeginRunProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestEndRunProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestBeginLumiBlockProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestEndLumiBlockProducer::m_count{0};
unsigned int edmtest::stream::GlobalIntProducer::cvalue_ = 0;
unsigned int edmtest::stream::RunIntProducer::cvalue_ = 0;
unsigned int edmtest::stream::LumiIntProducer::cvalue_ = 0;
unsigned int edmtest::stream::RunSummaryIntProducer::cvalue_ = 0;
unsigned int edmtest::stream::LumiSummaryIntProducer::cvalue_ = 0;
unsigned int edmtest::stream::TestBeginRunProducer::cvalue_ = 0;
unsigned int edmtest::stream::TestEndRunProducer::cvalue_ = 0;
unsigned int edmtest::stream::TestBeginLumiBlockProducer::cvalue_ = 0;
unsigned int edmtest::stream::TestEndLumiBlockProducer::cvalue_ = 0;
bool edmtest::stream::RunIntProducer::gbr=false;
bool edmtest::stream::RunIntProducer::ger=false;
bool edmtest::stream::LumiIntProducer::gbl=false;
bool edmtest::stream::LumiIntProducer::gel=false;
bool edmtest::stream::LumiIntProducer::bl=false;
bool edmtest::stream::LumiIntProducer::el=false;
bool edmtest::stream::RunSummaryIntProducer::gbr=false;
bool edmtest::stream::RunSummaryIntProducer::ger=false;
bool edmtest::stream::RunSummaryIntProducer::gbrs=false;
bool edmtest::stream::RunSummaryIntProducer::gers=false;
bool edmtest::stream::RunSummaryIntProducer::brs=false;
bool edmtest::stream::RunSummaryIntProducer::ers=false;
bool edmtest::stream::RunSummaryIntProducer::br=false;
bool edmtest::stream::RunSummaryIntProducer::er=false;
bool edmtest::stream::LumiSummaryIntProducer::gbl=false;
bool edmtest::stream::LumiSummaryIntProducer::gel=false;
bool edmtest::stream::LumiSummaryIntProducer::gbls=false;
bool edmtest::stream::LumiSummaryIntProducer::gels=false;
bool edmtest::stream::LumiSummaryIntProducer::bls=false;
bool edmtest::stream::LumiSummaryIntProducer::els=false;
bool edmtest::stream::LumiSummaryIntProducer::bl=false;
bool edmtest::stream::LumiSummaryIntProducer::el=false;
bool edmtest::stream::TestBeginRunProducer::gbr=false;
bool edmtest::stream::TestBeginRunProducer::gbrp=false;
bool edmtest::stream::TestBeginRunProducer::ger=false;
bool edmtest::stream::TestEndRunProducer::gbr=false;
bool edmtest::stream::TestEndRunProducer::ger=false;
bool edmtest::stream::TestEndRunProducer::p=false;
bool edmtest::stream::TestBeginLumiBlockProducer::gbl=false;
bool edmtest::stream::TestBeginLumiBlockProducer::gblp=false;
bool edmtest::stream::TestBeginLumiBlockProducer::gel=false;
bool edmtest::stream::TestEndLumiBlockProducer::gbl=false;
bool edmtest::stream::TestEndLumiBlockProducer::gel=false;
bool edmtest::stream::TestEndLumiBlockProducer::p=false;
DEFINE_FWK_MODULE(edmtest::stream::GlobalIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::RunIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::LumiIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::RunSummaryIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::LumiSummaryIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::TestBeginRunProducer);
DEFINE_FWK_MODULE(edmtest::stream::TestEndRunProducer);
DEFINE_FWK_MODULE(edmtest::stream::TestBeginLumiBlockProducer);
DEFINE_FWK_MODULE(edmtest::stream::TestEndLumiBlockProducer);

