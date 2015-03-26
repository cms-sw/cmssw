
/*----------------------------------------------------------------------

Toy edm::stream::EDAnalyzer modules of 
edm::*Cache templates 
for testing purposes only.

----------------------------------------------------------------------*/
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
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

  class GlobalIntAnalyzer : public edm::stream::EDAnalyzer<edm::GlobalCache<Cache>> {
  public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static std::atomic<unsigned int> cvalue_;
   
    static std::unique_ptr<Cache> initializeGlobalCache(edm::ParameterSet const& p) {
      ++m_count;
      return std::unique_ptr<Cache>{ new Cache };
    }

    GlobalIntAnalyzer(edm::ParameterSet const& p, Cache const * iGlobal)  {
      trans_ = p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
    }

    void analyze(edm::Event const&, edm::EventSetup const&) {
      ++m_count;
      ++((globalCache())->value);
       
    }
    
    static void globalEndJob(Cache * iGlobal) {
      ++m_count;
      if(iGlobal->value != cvalue_) {
        throw cms::Exception("cache value")
          << iGlobal->value << " but it was supposed to be " << cvalue_;
      }
    }

    ~GlobalIntAnalyzer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << m_count << " but it was supposed to be " << trans_;
      }
    }

    
  };

  class RunIntAnalyzer : public edm::stream::EDAnalyzer<edm::RunCache<Cache>> {
  public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static std::atomic<unsigned int> cvalue_;
    static std::atomic<bool> gbr;
    static std::atomic<bool> ger;
    bool br;
    bool er;

    RunIntAnalyzer(edm::ParameterSet const&p) {
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      m_count = 0;
    }
    
    void analyze(edm::Event const&, edm::EventSetup const&) override {
      ++m_count;
      ++(runCache()->value);
       
    }

    static std::shared_ptr<Cache> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
      ++m_count;
      gbr = true;
      ger = false;
      auto pCache = std::make_shared<Cache>();
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

    void endRun(edm::Run const&, edm::EventSetup const&) override {
      er = true;
      br = false;
      if ( ger ) {
        throw cms::Exception("end out of sequence")
          << "globalEndRun seen before endRun";
      }
    }

    ~RunIntAnalyzer() {
       if(m_count != trans_) {
        throw cms::Exception("transitions")
          << m_count << " but it was supposed to be " << trans_;
      }

    }
  };


  class LumiIntAnalyzer : public edm::stream::EDAnalyzer<edm::LuminosityBlockCache<Cache>> {
  public:
    static std::atomic<unsigned int> m_count;
    unsigned int trans_;
    static std::atomic<unsigned int> cvalue_;
    static std::atomic<bool> gbl;
    static std::atomic<bool> gel;
    static std::atomic<bool> bl;
    static std::atomic<bool> el;

    LumiIntAnalyzer(edm::ParameterSet const&p){
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      m_count = 0;
    }
     
    void analyze(edm::Event const&, edm::EventSetup const&) override {
      ++m_count;
      ++(luminosityBlockCache()->value);
       
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


    static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&, LuminosityBlockContext const* iLBContext) {
      ++m_count;
      auto pCache = iLBContext->luminosityBlock();
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

    ~LumiIntAnalyzer() {
       if(m_count != trans_) {
        throw cms::Exception("transitions")
          << m_count<< " but it was supposed to be " << trans_;
       }
    }
  };
  
  class RunSummaryIntAnalyzer : public edm::stream::EDAnalyzer<edm::RunCache<Cache>,edm::RunSummaryCache<Cache>> {
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

    RunSummaryIntAnalyzer(edm::ParameterSet const&p){
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      m_count = 0;
    }

    void analyze(edm::Event const&, edm::EventSetup const&)  override {
      ++m_count;
      ++(runCache()->value);
       
    }
    
    void beginRun(edm::Run const&, edm::EventSetup const&) override {
      br=true;
      er=false;
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
    
    static void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, RunContext const*, Cache* gCache) {
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


    ~RunSummaryIntAnalyzer() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << m_count<< " but it was supposed to be " << trans_;
      }
 
    }
  };

  class LumiSummaryIntAnalyzer : public edm::stream::EDAnalyzer<edm::LuminosityBlockCache<Cache>,edm::LuminosityBlockSummaryCache<Cache>> {
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

    LumiSummaryIntAnalyzer(edm::ParameterSet const&p){
      trans_= p.getParameter<int>("transitions");
      cvalue_ = p.getParameter<int>("cachevalue");
      m_count = 0;
    }

    void analyze(edm::Event const&, edm::EventSetup const&) override  {
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
          << "globalEndLuminosityBlockSummary seen before endLuminosityBlockSummary";
      }
     if ( gCache->value != cvalue_) {
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


    ~LumiSummaryIntAnalyzer() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  }; 


}
}
std::atomic<unsigned int> edmtest::stream::GlobalIntAnalyzer::m_count{0};
std::atomic<unsigned int> edmtest::stream::RunIntAnalyzer::m_count{0};
std::atomic<unsigned int> edmtest::stream::LumiIntAnalyzer::m_count{0};
std::atomic<unsigned int> edmtest::stream::RunSummaryIntAnalyzer::m_count{0};
std::atomic<unsigned int> edmtest::stream::LumiSummaryIntAnalyzer::m_count{0};
std::atomic<unsigned int> edmtest::stream::GlobalIntAnalyzer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::RunIntAnalyzer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::LumiIntAnalyzer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::RunSummaryIntAnalyzer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::LumiSummaryIntAnalyzer::cvalue_{0};
std::atomic<bool> edmtest::stream::RunIntAnalyzer::gbr{false};
std::atomic<bool> edmtest::stream::RunIntAnalyzer::ger{false};
std::atomic<bool> edmtest::stream::LumiIntAnalyzer::gbl{false};
std::atomic<bool> edmtest::stream::LumiIntAnalyzer::gel{false};
std::atomic<bool> edmtest::stream::LumiIntAnalyzer::bl{false};
std::atomic<bool> edmtest::stream::LumiIntAnalyzer::el{false};
std::atomic<bool> edmtest::stream::RunSummaryIntAnalyzer::gbr{false};
std::atomic<bool> edmtest::stream::RunSummaryIntAnalyzer::ger{false};
std::atomic<bool> edmtest::stream::RunSummaryIntAnalyzer::gbrs{false};
std::atomic<bool> edmtest::stream::RunSummaryIntAnalyzer::gers{false};
std::atomic<bool> edmtest::stream::RunSummaryIntAnalyzer::brs{false};
std::atomic<bool> edmtest::stream::RunSummaryIntAnalyzer::ers{false};
std::atomic<bool> edmtest::stream::RunSummaryIntAnalyzer::br{false};
std::atomic<bool> edmtest::stream::RunSummaryIntAnalyzer::er{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntAnalyzer::gbl{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntAnalyzer::gel{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntAnalyzer::gbls{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntAnalyzer::gels{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntAnalyzer::bls{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntAnalyzer::els{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntAnalyzer::bl{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntAnalyzer::el{false};
DEFINE_FWK_MODULE(edmtest::stream::GlobalIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::stream::RunIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::stream::LumiIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::stream::RunSummaryIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::stream::LumiSummaryIntAnalyzer);

