
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

  class StreamIntProducer : public edm::global::EDProducer<edm::StreamCache<int>> {
  public:
    explicit StreamIntProducer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};

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
    ~StreamIntProducer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "StreamIntProducer transitions " 
          << m_count << " but it was supposed to be " << trans_;
      }
      //std::cout << "StreamIntProducer transitions " << m_count << " but it was supposed to be " << trans_ <<"\n";
    }
  };
  
  class RunIntProducer : public edm::global::EDProducer<edm::RunCache<int>> {
  public:
    explicit RunIntProducer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
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
    ~RunIntProducer() {
       if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "RunIntProducer transitions " 
          << m_count << " but it was supposed to be " << trans_;
      }
       //std::cout << "RunIntProducer transitions " << m_count << " but it was supposed to be " << trans_ <<"\n";
    }
  };


  class LumiIntProducer : public edm::global::EDProducer<edm::LuminosityBlockCache<int>> {
  public:
    explicit LumiIntProducer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
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
    ~LumiIntProducer() {
       if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "LumiIntProducer transitions " 
          << m_count<< " but it was supposed to be " << trans_;
       }
       //std::cout << "LumiIntProducer transitions " << m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };
  
  class RunSummaryIntProducer : public edm::global::EDProducer<edm::RunSummaryCache<int>> {
  public:
    explicit RunSummaryIntProducer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
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
    ~RunSummaryIntProducer() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "RunSummaryIntProducer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };

  class LumiSummaryIntProducer : public edm::global::EDProducer<edm::LuminosityBlockSummaryCache<int>> {
  public:
    explicit LumiSummaryIntProducer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {	
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
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
    ~LumiSummaryIntProducer() {
     if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "LumiSummaryIntProducer transitions " 
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "LumiSummaryIntProducer transitions " << m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };

  class StreamIntAnalzer : public edm::global::EDAnalyzer<edm::StreamCache<int>> {
  public:
    explicit StreamIntAnalzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
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
    ~StreamIntAnalzer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "StreamIntAnalzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "StreamIntAnalzer transitions "<< m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };
  
  class RunIntAnalzer : public edm::global::EDAnalyzer<edm::RunCache<int>> {
  public:
    explicit RunIntAnalzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
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
    ~RunIntAnalzer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "RunIntAnalzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "RunIntAnalzer transitions "<< m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };


  class LumiIntAnalzer : public edm::global::EDAnalyzer<edm::LuminosityBlockCache<int>> {
  public:
    explicit LumiIntAnalzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
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
    ~LumiIntAnalzer () {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "LumiIntAnalzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "LumiIntAnalzer transitions "<< m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };
  
  class RunSummaryIntAnalzer : public edm::global::EDAnalyzer<edm::RunSummaryCache<int>> {
  public:
    explicit RunSummaryIntAnalzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
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
    ~RunSummaryIntAnalzer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "RunSummaryIntAnalzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "RunSummaryIntAnalzer transitions " << m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };

  class LumiSummaryIntAnalzer : public edm::global::EDAnalyzer<edm::LuminosityBlockSummaryCache<int>> {
  public:
    explicit LumiSummaryIntAnalzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {	
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
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
    ~LumiSummaryIntAnalzer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "LumiSummaryIntAnalzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "LumiSummaryIntAnalzer transitions " << m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };

  class StreamIntFilter : public edm::global::EDFilter<edm::StreamCache<int>> {
  public:
    explicit StreamIntFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
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
    ~StreamIntFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "StreamIntFilter transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "StreamIntFilter transitions " << m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };
  
  class RunIntFilter : public edm::global::EDFilter<edm::RunCache<int>> {
  public:
    explicit RunIntFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
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
    ~RunIntFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "RunIntFilter transitions "     
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "RunIntFilter transitions " << m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };


  class LumiIntFilter : public edm::global::EDFilter<edm::LuminosityBlockCache<int>> {
  public:
    explicit LumiIntFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
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
    ~LumiIntFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "LumiIntFilter transitions " 
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "LumiIntFilter transitions " << m_count<< " but it was supposed to be "<< trans_ <<"\n";
    }
  };
  
  class RunSummaryIntFilter : public edm::global::EDFilter<edm::RunSummaryCache<int>> {
  public:
    explicit RunSummaryIntFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
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
    ~RunSummaryIntFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "RunSummaryIntFilter transitions " 
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "RunSummaryIntFilter transitions " << m_count << " but it was supposed to be " << trans_ <<"\n";
    }
  };

  class LumiSummaryIntFilter : public edm::global::EDFilter<edm::LuminosityBlockSummaryCache<int>> {
  public:
    explicit LumiSummaryIntFilter(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {	
    }
    const unsigned int trans_; 
    mutable std::atomic<unsigned int> m_count{0};
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
    ~LumiSummaryIntFilter() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "LumiSummaryIntFilter transitions "     
          << m_count<< " but it was supposed to be " << trans_;
      }
      //std::cout << "LumiSummaryIntFilter transitions " << m_count<< " but it was supposed to be " << trans_ <<"\n";
    }
  };

}
}

DEFINE_FWK_MODULE(edmtest::global::StreamIntProducer);
DEFINE_FWK_MODULE(edmtest::global::RunIntProducer);
DEFINE_FWK_MODULE(edmtest::global::LumiIntProducer);
DEFINE_FWK_MODULE(edmtest::global::RunSummaryIntProducer);
DEFINE_FWK_MODULE(edmtest::global::LumiSummaryIntProducer);
DEFINE_FWK_MODULE(edmtest::global::StreamIntAnalzer);
DEFINE_FWK_MODULE(edmtest::global::RunIntAnalzer);
DEFINE_FWK_MODULE(edmtest::global::LumiIntAnalzer);
DEFINE_FWK_MODULE(edmtest::global::RunSummaryIntAnalzer);
DEFINE_FWK_MODULE(edmtest::global::LumiSummaryIntAnalzer);
DEFINE_FWK_MODULE(edmtest::global::StreamIntFilter);
DEFINE_FWK_MODULE(edmtest::global::RunIntFilter);
DEFINE_FWK_MODULE(edmtest::global::LumiIntFilter);
DEFINE_FWK_MODULE(edmtest::global::RunSummaryIntFilter);
DEFINE_FWK_MODULE(edmtest::global::LumiSummaryIntFilter);


