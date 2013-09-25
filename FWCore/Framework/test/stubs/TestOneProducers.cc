
/*----------------------------------------------------------------------

Toy edm::one::EDProducer modules of 
edm::one cache classes and edm::*Producer classes
for testing purposes only.

----------------------------------------------------------------------*/
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "FWCore/Framework/interface/one/EDProducer.h"
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

  class SharedResourcesProducer : public edm::one::EDProducer<edm::one::SharedResources> {
  public:
    explicit SharedResourcesProducer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
      produces<int>();
      usesResource();
    }
    const unsigned int trans_; 
    unsigned int m_count = 0;

    void produce(edm::Event&, edm::EventSetup const&) override { 
      ++m_count;
    }
       
    ~SharedResourcesProducer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "SharedResourcesProducer transitions " 
          << m_count << " but it was supposed to be " << trans_;
      }
    }

  };

  
  class WatchRunsProducer : public edm::one::EDProducer<edm::one::WatchRuns> {
  public:
    explicit WatchRunsProducer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
      produces<int>();
    }
    bool br = false;
    bool er = false;
    const unsigned int trans_; 
    unsigned int m_count = 0;

    void produce(edm::Event &, edm::EventSetup const&) override  { 
      ++m_count;
      if ( !br  || er ) {
          throw cms::Exception("out of sequence")
          << " produce before beginRun or after endRun";
      } 
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
     
   ~WatchRunsProducer() {
       if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "WatchRunsProducer transitions " 
          << m_count << " but it was supposed to be " << trans_;
      }
    }
  };


  class WatchLumiBlocksProducer : public edm::one::EDProducer<edm::one::WatchLuminosityBlocks> {
  public:
    explicit WatchLumiBlocksProducer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
      produces<int>();
    }
    const unsigned int trans_; 
    bool bl = false;
    bool el = false;
    unsigned int m_count = 0;

    void produce(edm::Event&, edm::EventSetup const&) override { 
      ++m_count;
      if ( !bl  || el ) {
          throw cms::Exception("out of sequence")
          << " produce before beginLumiBlock or after endLumiBlock";
      }       
    }

    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override  {
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

    ~WatchLumiBlocksProducer() {
       if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "WatchLumiBlockProducer transitions " 
          << m_count<< " but it was supposed to be " << trans_;
       }
    }
  };
  
  class TestBeginRunProducer : public edm::one::EDProducer<edm::one::WatchRuns,edm::BeginRunProducer> {
  public:
    explicit TestBeginRunProducer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
      produces<int>();
    }
    const unsigned int trans_; 
    unsigned int m_count = 0;
    bool p = false;

    void beginRun(edm::Run const&, edm::EventSetup const&) override { 
      p = false;
    }
 
    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      p=true;
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

   ~TestBeginRunProducer() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "TestBeginRunProducer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

  class TestBeginLumiBlockProducer : public edm::one::EDProducer<edm::one::WatchLuminosityBlocks,edm::BeginLuminosityBlockProducer> {
  public:
    explicit TestBeginLumiBlockProducer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {	
      produces<int>();
    }
    const unsigned int trans_; 
    unsigned int m_count = 0;
    bool p = false;

    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override  {
      p = false;
    }
 
    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      p = true;
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

   ~TestBeginLumiBlockProducer() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "TestBeginLumiBlockProducer transitions " 
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

  class TestEndRunProducer : public edm::one::EDProducer<edm::one::WatchRuns,edm::EndRunProducer> {
  public:
    explicit TestEndRunProducer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
      produces<int>();
    }
    const unsigned int trans_;
    bool erp = false; 

    void beginRun(edm::Run const&, edm::EventSetup const&) override { 
      erp = false;
    }
 
     unsigned int m_count = 0;

    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      if ( erp ) {
        throw cms::Exception("out of sequence")
          << "endRunProduce before produce";
      }
    }

    void endRunProduce(edm::Run&, edm::EventSetup const&) override {
      ++m_count;
      erp = true;
    }

    void endRun(edm::Run const&, edm::EventSetup const&) override { 
    }

   ~TestEndRunProducer() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "TestEndRunProducer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

  class TestEndLumiBlockProducer : public edm::one::EDProducer<edm::one::WatchLuminosityBlocks,edm::EndLuminosityBlockProducer> {
  public:
    explicit TestEndLumiBlockProducer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {	
      produces<int>();
    }
    const unsigned int trans_; 
    bool elbp = false;
    unsigned int m_count = 0;

    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override  {
      elbp = false;
    }
 

    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
      if ( elbp ) {
        throw cms::Exception("out of sequence")
          << "endLumiBlockProduce before produce";
      }
    }

    void endLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) override {
      ++m_count;
      elbp = true;
    }

    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {
    }
     
   ~TestEndLumiBlockProducer() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "TestEndLumiBlockProducer transitions " 
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

}
}

DEFINE_FWK_MODULE(edmtest::one::SharedResourcesProducer);
DEFINE_FWK_MODULE(edmtest::one::WatchRunsProducer);
DEFINE_FWK_MODULE(edmtest::one::WatchLumiBlocksProducer);
DEFINE_FWK_MODULE(edmtest::one::TestBeginRunProducer);
DEFINE_FWK_MODULE(edmtest::one::TestBeginLumiBlockProducer);
DEFINE_FWK_MODULE(edmtest::one::TestEndRunProducer);
DEFINE_FWK_MODULE(edmtest::one::TestEndLumiBlockProducer);
