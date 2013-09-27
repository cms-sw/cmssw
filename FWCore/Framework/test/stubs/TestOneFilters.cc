
/*----------------------------------------------------------------------

Toy edm::one::EDFilter modules of 
edm::one cache classes and edm::*Producer classes
for testing purposes only.

----------------------------------------------------------------------*/
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "FWCore/Framework/interface/one/EDFilter.h"
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
namespace one {


  class SharedResourcesFilter : public edm::one::EDFilter<edm::one::SharedResources> {
  public:
    explicit SharedResourcesFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
      produces<int>();
      usesResource();
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
    bool filter(edm::Event &, edm::EventSetup const&) override {
      ++m_count;
      return true;
    }
    
   ~SharedResourcesFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "SharedResourcesFilter transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };
  
  class WatchRunsFilter : public edm::one::EDFilter<edm::one::WatchRuns> {
  public:
    explicit WatchRunsFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    bool br = false;
    bool er = false;
    const unsigned int trans_; 
    unsigned int m_count = 0;

   bool filter(edm::Event &, edm::EventSetup const&) override {
      ++m_count;
      if ( !br  || er ) {
          throw cms::Exception("out of sequence")
          << " produce before beginRun or after endRun";
      } 
      return true;
    }

   void beginRun(edm::Run const&, edm::EventSetup const&) override {
      ++m_count;
      if ( br ) {
        throw cms::Exception("out of sequence")
          << " beginRun seen multiple times";    
      }
      br = true;
      er = false;
    }

   void endRun(edm::Run const&, edm::EventSetup const&) override {  
      ++m_count;
      if ( !br ) {
        throw cms::Exception("out of sequence")
          << " endRun before beginRun";    
      }
      br = false;
      er = true;
    }

   ~WatchRunsFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "WatchRunsFilter transitions "     
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };


  class WatchLumiBlocksFilter : public edm::one::EDFilter<edm::one::WatchLuminosityBlocks> {
  public:
    explicit WatchLumiBlocksFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    bool bl = false;
    bool el = false;
    unsigned int m_count = 0;

    bool filter(edm::Event &, edm::EventSetup const&)  override {
      ++m_count;
      if ( !bl  || el ) {
          throw cms::Exception("out of sequence")
          << " produce before beginLumiBlock or after endLumiBlock";
      }              
      return true;
    }

    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override { 
      ++m_count;
      if ( bl ) {
        throw cms::Exception("out of sequence")
          << " beginLumiBlock seen mutiple times";    
      }
      bl = true;
      el = false;
     }

    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {
      ++m_count;
      if ( !bl ) {
        throw cms::Exception("out of sequence")
          << " endLumiBlock before beginLumiBlock";    
      }
      bl = false;
      el = true;
    }

   ~WatchLumiBlocksFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "WatchLumiBlocksFilter transitions " 
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };
  
  class BeginRunFilter : public edm::one::EDFilter<edm::one::WatchRuns,edm::BeginRunProducer> {
  public:
    explicit BeginRunFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
      produces<int>();
    }
    const unsigned int trans_; 
    unsigned int m_count = 0;
    bool p = false;
 
   void beginRun(edm::Run const&, edm::EventSetup const&) override { 
      p = false;
    }
 
   bool filter(edm::Event &, edm::EventSetup const&) override {
      ++m_count;
      p=true; 
      return true;
    }

   void beginRunProduce(edm::Run&, edm::EventSetup const&) override {
      if ( p ) {
        throw cms::Exception("out of sequence")
          << "produce before beginRunProduce";
      }
      ++m_count;
    }

    void endRun(edm::Run const&, edm::EventSetup const&) override { 
    }

   ~BeginRunFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "BeginRunFilter transitions " 
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

  class BeginLumiBlockFilter : public edm::one::EDFilter<edm::one::WatchLuminosityBlocks,edm::BeginLuminosityBlockProducer> {
  public:
    explicit BeginLumiBlockFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {	
    }
    const unsigned int trans_; 
    unsigned int m_count = 0;
    bool p = false;

    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override  {
      p = false;
    }
 

    bool filter(edm::Event &, edm::EventSetup const&) override {
      ++m_count;
      p = true; 
      return true;
    }

    void beginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) override {
      ++m_count;
      if ( p ) {
        throw cms::Exception("out of sequence")
          << "produce before beginLuminosityBlockProduce";
      }
    }

    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {
    }


   ~BeginLumiBlockFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "BeginLumiBlockFilter transitions "     
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

  class EndRunFilter : public edm::one::EDFilter<edm::one::WatchRuns,edm::EndRunProducer> {
  public:
    explicit EndRunFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_;
    bool erp = false; 
    unsigned int m_count = 0;

    void beginRun(edm::Run const&, edm::EventSetup const&) override { 
      erp = false;
    }

    bool filter(edm::Event &, edm::EventSetup const&) override {
      ++m_count;
      if ( erp ) {
        throw cms::Exception("out of sequence")
          << "endRunProduce before produce";
      }       
      return true;
    }

    void endRunProduce(edm::Run&, edm::EventSetup const&) override {
      ++m_count;
    }    

    void endRun(edm::Run const&, edm::EventSetup const&) override { 
    }


   ~EndRunFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "EndRunFilter transitions " 
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

  class EndLumiBlockFilter : public edm::one::EDFilter<edm::one::WatchLuminosityBlocks,edm::EndLuminosityBlockProducer> {
  public:
    explicit EndLumiBlockFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {	
    }
    const unsigned int trans_; 
    bool elbp = false;
    unsigned int m_count = 0;

    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override  {
      elbp = false;
    }
 
   bool filter(edm::Event &, edm::EventSetup const&) override {
      ++m_count;
      if ( elbp ) {
        throw cms::Exception("out of sequence")
          << "endLumiBlockProduce before produce";
      }       
      return true;
    }

    void endLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) override {
      ++m_count;
      elbp = true;
    }

    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {
    }

   ~EndLumiBlockFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "EndLumiBlockFilter transitions "     
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };


}
}

DEFINE_FWK_MODULE(edmtest::one::SharedResourcesFilter);
DEFINE_FWK_MODULE(edmtest::one::WatchRunsFilter);
DEFINE_FWK_MODULE(edmtest::one::WatchLumiBlocksFilter);
DEFINE_FWK_MODULE(edmtest::one::BeginRunFilter);
DEFINE_FWK_MODULE(edmtest::one::BeginLumiBlockFilter);
DEFINE_FWK_MODULE(edmtest::one::EndRunFilter);
DEFINE_FWK_MODULE(edmtest::one::EndLumiBlockFilter);


