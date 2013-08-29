
/*----------------------------------------------------------------------

Toy edm::global modules of ints for testing purposes only.

----------------------------------------------------------------------*/
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
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

  class StreamIntGProducer : public edm::global::EDProducer<edm::StreamCache<int>> {
  public:
    explicit StreamIntGProducer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable unsigned int m_count;

    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
    
    std::unique_ptr<int> beginStream(edm::StreamID) const override {
      ++m_count;
      return std::unique_ptr<int>{};
    }
    
    virtual void streamBeginRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const  override{
      ++m_count;
    }
    virtual void streamBeginLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    virtual void streamEndLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    virtual void streamEndRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    void endStream(edm::StreamID) const override {
      ++m_count;
    }
    ~StreamIntGProducer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "StreamIntGProducer transitions " 
          << m_count << " but it was supposed to be " << trans_;
      }
      //std::cout << "StreamIntGProducer transitions " << m_count << " but it was supposed to be " << trans_ <<"\n";
    }
  };
  
  class RunIntGProducer : public edm::global::EDProducer<edm::RunCache<int>> {
  public:
    explicit RunIntGProducer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
    
    std::shared_ptr<int> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    void globalEndRun(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    ~RunIntGProducer() {
       if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "RunIntGProducer transitions " 
          << m_count << " but it was supposed to be " << trans_;
      }
       //std::cout << "RunIntGProducer transitions " << m_count << " but it was supposed to be " << trans_ <<"\n";
    }
  };


  class LumiIntGProducer : public edm::global::EDProducer<edm::LuminosityBlockCache<int>> {
  public:
    explicit LumiIntGProducer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
    
    std::shared_ptr<int> globalBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    ~LumiIntGProducer() {
       if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "LumiIntGProducer transitions " 
          << m_count<< " but it was supposed to be " << trans_;
       }
       //std::cout << "LumiIntGProducer transitions " << m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };
  
  class RunSummaryIntGProducer : public edm::global::EDProducer<edm::RunSummaryCache<int>> {
  public:
    explicit RunSummaryIntGProducer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
    
    std::shared_ptr<int> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void streamEndRunSummary(edm::StreamID, edm::Run const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    
    void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    ~RunSummaryIntGProducer() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "RunSummaryIntGProducer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

  class LumiSummaryIntGProducer : public edm::global::EDProducer<edm::LuminosityBlockSummaryCache<int>> {
  public:
    explicit LumiSummaryIntGProducer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {	
    }
    const unsigned int trans_; 
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
    
    std::shared_ptr<int> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void streamEndLuminosityBlockSummary(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    
    void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    ~LumiSummaryIntGProducer() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "LumiSummaryIntGProducer transitions " 
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "LumiSummaryIntGProducer transitions " << m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };

  class StreamIntGAnalzer : public edm::global::EDAnalyzer<edm::StreamCache<int>> {
  public:
    explicit StreamIntGAnalzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable unsigned int m_count = 0;
    void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override {
      ++m_count;
    }
    
    std::unique_ptr<int> beginStream(edm::StreamID) const override {
      ++m_count;
      return std::unique_ptr<int>{};
    }
    
    virtual void streamBeginRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const  override{
      ++m_count;
    }
    virtual void streamBeginLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    virtual void streamEndLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    virtual void streamEndRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    void endStream(edm::StreamID) const override {
      ++m_count;
    }
    ~StreamIntGAnalzer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "StreamIntGAnalzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "StreamIntGAnalzer transitions "<< m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };
  
  class RunIntGAnalzer : public edm::global::EDAnalyzer<edm::RunCache<int>> {
  public:
    explicit RunIntGAnalzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable unsigned int m_count = 0;
    void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override {
      ++m_count;
    }
    
    std::shared_ptr<int> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    void globalEndRun(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    ~RunIntGAnalzer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "RunIntGAnalzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "RunIntGAnalzer transitions "<< m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };


  class LumiIntGAnalzer : public edm::global::EDAnalyzer<edm::LuminosityBlockCache<int>> {
  public:
    explicit LumiIntGAnalzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable unsigned int m_count = 0;
    void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override {
      ++m_count;
    }
    
    std::shared_ptr<int> globalBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    ~LumiIntGAnalzer () {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "LumiIntGAnalzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "LumiIntGAnalzer transitions "<< m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };
  
  class RunSummaryIntGAnalzer : public edm::global::EDAnalyzer<edm::RunSummaryCache<int>> {
  public:
    explicit RunSummaryIntGAnalzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable unsigned int m_count = 0;
    void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override {
      ++m_count;
    }
    
    std::shared_ptr<int> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void streamEndRunSummary(edm::StreamID, edm::Run const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    
    void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    ~RunSummaryIntGAnalzer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "RunSummaryIntGAnalzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "RunSummaryIntGAnalzer transitions " << m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };

  class LumiSummaryIntGAnalzer : public edm::global::EDAnalyzer<edm::LuminosityBlockSummaryCache<int>> {
  public:
    explicit LumiSummaryIntGAnalzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {	
    }
    const unsigned int trans_; 
    mutable unsigned int m_count = 0;
    void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override {
      ++m_count;
    }
    
    std::shared_ptr<int> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void streamEndLuminosityBlockSummary(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    
    void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    ~LumiSummaryIntGAnalzer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "LumiSummaryIntGAnalzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "LumiSummaryIntGAnalzer transitions " << m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };

  class StreamIntGFilter : public edm::global::EDFilter<edm::StreamCache<int>> {
  public:
    explicit StreamIntGFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable unsigned int m_count = 0;
    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      return true;
    }
    
    std::unique_ptr<int> beginStream(edm::StreamID) const override {
      ++m_count;
      return std::unique_ptr<int>{};
    }
    
    virtual void streamBeginRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const  override{
      ++m_count;
    }
    virtual void streamBeginLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    virtual void streamEndLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    virtual void streamEndRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    void endStream(edm::StreamID) const override {
      ++m_count;
    }
    ~StreamIntGFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "StreamIntGFilter transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "StreamIntGFilter transitions " << m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };
  
  class RunIntGFilter : public edm::global::EDFilter<edm::RunCache<int>> {
  public:
    explicit RunIntGFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable unsigned int m_count = 0;
    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      return true;
    }
    
    std::shared_ptr<int> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    void globalEndRun(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    ~RunIntGFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "RunIntGFilter transitions "     
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "RunIntGFilter transitions " << m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };


  class LumiIntGFilter : public edm::global::EDFilter<edm::LuminosityBlockCache<int>> {
  public:
    explicit LumiIntGFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable unsigned int m_count = 0;
    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      return true;
    }
    
    std::shared_ptr<int> globalBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    ~LumiIntGFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "LumiIntGFilter transitions " 
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "LumiIntGFilter transitions " << m_count<< " but it was supposed to be "<< trans_ <<"\n";
    }
  };
  
  class RunSummaryIntGFilter : public edm::global::EDFilter<edm::RunSummaryCache<int>> {
  public:
    explicit RunSummaryIntGFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable unsigned int m_count = 0;
    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      return true;
    }
    
    std::shared_ptr<int> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void streamEndRunSummary(edm::StreamID, edm::Run const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    
    void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    ~RunSummaryIntGFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "RunSummaryIntGFilter transitions " 
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "RunSummaryIntGFilter transitions " << m_count << " but it was supposed to be " << trans_ <<"\n";
    }
  };

  class LumiSummaryIntGFilter : public edm::global::EDFilter<edm::LuminosityBlockSummaryCache<int>> {
  public:
    explicit LumiSummaryIntGFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {	
    }
    const unsigned int trans_; 
    mutable unsigned int m_count = 0;
    bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
      return true;
    }
    
    std::shared_ptr<int> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void streamEndLuminosityBlockSummary(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    
    void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    ~LumiSummaryIntGFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "LumiSummaryIntGFilter transitions "     
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "LumiSummaryIntGFilter transitions " << m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };

}
}

//using edmtest::global::IntGProducer;
using edmtest::global::StreamIntGProducer;
using edmtest::global::RunIntGProducer;
using edmtest::global::LumiIntGProducer;
using edmtest::global::RunSummaryIntGProducer;
using edmtest::global::LumiSummaryIntGProducer;
//DEFINE_FWK_MODULE(IntGProducer);
DEFINE_FWK_MODULE(StreamIntGProducer);
DEFINE_FWK_MODULE(RunIntGProducer);
DEFINE_FWK_MODULE(LumiIntGProducer);
DEFINE_FWK_MODULE(RunSummaryIntGProducer);
DEFINE_FWK_MODULE(LumiSummaryIntGProducer);
//using edmtest::global::IntGAnalzer;
using edmtest::global::StreamIntGAnalzer;
using edmtest::global::RunIntGAnalzer;
using edmtest::global::LumiIntGAnalzer;
using edmtest::global::RunSummaryIntGAnalzer;
using edmtest::global::LumiSummaryIntGAnalzer;
//DEFINE_FWK_MODULE(IntGAnalzer);
DEFINE_FWK_MODULE(StreamIntGAnalzer);
DEFINE_FWK_MODULE(RunIntGAnalzer);
DEFINE_FWK_MODULE(LumiIntGAnalzer);
DEFINE_FWK_MODULE(RunSummaryIntGAnalzer);
DEFINE_FWK_MODULE(LumiSummaryIntGAnalzer);
//using edmtest::global::IntGFilter;
using edmtest::global::StreamIntGFilter;
using edmtest::global::RunIntGFilter;
using edmtest::global::LumiIntGFilter;
using edmtest::global::RunSummaryIntGFilter;
using edmtest::global::LumiSummaryIntGFilter;
//DEFINE_FWK_MODULE(IntGFilter);
DEFINE_FWK_MODULE(StreamIntGFilter);
DEFINE_FWK_MODULE(RunIntGFilter);
DEFINE_FWK_MODULE(LumiIntGFilter);
DEFINE_FWK_MODULE(RunSummaryIntGFilter);
DEFINE_FWK_MODULE(LumiSummaryIntGFilter);


