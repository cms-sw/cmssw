
/*----------------------------------------------------------------------

Toy edm::global modules of Ints for testing purposes only.

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
	m_count(p.getParameter<int>("ivalue")) {
    }
    mutable unsigned int m_count = 0;
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
      std::cout << "StreamIntProducer transitions " << m_count <<" \n";
    }
  };
  
  class RunIntProducer : public edm::global::EDProducer<edm::RunCache<int>> {
  public:
    explicit RunIntProducer(edm::ParameterSet const& p) :
	m_count(p.getParameter<int>("ivalue")) {
    }
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
    ~RunIntProducer() {
      std::cout << "RunIntProducer transitions " << m_count <<" \n";
    }
  };


  class LumiIntProducer : public edm::global::EDProducer<edm::LuminosityBlockCache<int>> {
  public:
    explicit LumiIntProducer(edm::ParameterSet const& p) :
	m_count(p.getParameter<int>("ivalue")) {
    }
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
    ~LumiIntProducer() {
      std::cout << "LumiIntProducer transitions " << m_count <<" \n";
    }
  };
  
  class RunSummaryIntProducer : public edm::global::EDProducer<edm::RunSummaryCache<int>> {
  public:
    explicit RunSummaryIntProducer(edm::ParameterSet const& p) :
	m_count(p.getParameter<int>("ivalue")) {
    }
 
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
    ~RunSummaryIntProducer() {
      std::cout << "RunSummaryIntProducer transitions " << m_count <<" \n";
    }
  };

  class LumiSummaryIntProducer : public edm::global::EDProducer<edm::LuminosityBlockSummaryCache<int>> {
  public:
    explicit LumiSummaryIntProducer(edm::ParameterSet const& p) :
	m_count(p.getParameter<int>("ivalue")) {	
    }
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
    ~LumiSummaryIntProducer() {
      std::cout << "LumiSummaryIntProducer transitions " << m_count <<" \n";
    }
  };

  class StreamIntAnalyzer : public edm::global::EDAnalyzer<edm::StreamCache<int>> {
  public:
    explicit StreamIntAnalyzer(edm::ParameterSet const& p) :
	m_count(p.getParameter<int>("ivalue")) {
    }
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
    ~StreamIntAnalyzer() {
      std::cout << "StreamIntAnalyzer transitions " << m_count <<" \n";
    }
  };
  
  class RunIntAnalyzer : public edm::global::EDAnalyzer<edm::RunCache<int>> {
  public:
    explicit RunIntAnalyzer(edm::ParameterSet const& p) :
	m_count(p.getParameter<int>("ivalue")) {
    }
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
    ~RunIntAnalyzer() {
      std::cout << "RunIntAnalyzer transitions " << m_count <<" \n";
    }
  };


  class LumiIntAnalyzer : public edm::global::EDAnalyzer<edm::LuminosityBlockCache<int>> {
  public:
    explicit LumiIntAnalyzer(edm::ParameterSet const& p) :
	m_count(p.getParameter<int>("ivalue")) {
    }
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
    ~LumiIntAnalyzer () {
      std::cout << "LumiIntAnalyzer transitions " << m_count <<" \n";
    }
  };
  
  class RunSummaryIntAnalyzer : public edm::global::EDAnalyzer<edm::RunSummaryCache<int>> {
  public:
    explicit RunSummaryIntAnalyzer(edm::ParameterSet const& p) :
	m_count(p.getParameter<int>("ivalue")) {
    }
 
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
    ~RunSummaryIntAnalyzer() {
      std::cout << "RunSummaryIntAnalyzer transitions " << m_count <<" \n";
    }
  };

  class LumiSummaryIntAnalyzer : public edm::global::EDAnalyzer<edm::LuminosityBlockSummaryCache<int>> {
  public:
    explicit LumiSummaryIntAnalyzer(edm::ParameterSet const& p) :
	m_count(p.getParameter<int>("ivalue")) {	
    }
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
    ~LumiSummaryIntAnalyzer() {
      std::cout << "LumiSummaryIntAnalyzer transitions " << m_count <<" \n";
    }
  };

  class StreamIntFilter : public edm::global::EDFilter<edm::StreamCache<int>> {
  public:
    explicit StreamIntFilter(edm::ParameterSet const& p) :
	m_count(p.getParameter<int>("ivalue")) {
    }
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
    ~StreamIntFilter() {
      std::cout << "StreamIntFilter transitions " << m_count <<" \n";
    }
  };
  
  class RunIntFilter : public edm::global::EDFilter<edm::RunCache<int>> {
  public:
    explicit RunIntFilter(edm::ParameterSet const& p) :
	m_count(p.getParameter<int>("ivalue")) {
    }
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
    ~RunIntFilter() {
      std::cout << "RunIntFilter transitions " << m_count <<" \n";
    }
  };


  class LumiIntFilter : public edm::global::EDFilter<edm::LuminosityBlockCache<int>> {
  public:
    explicit LumiIntFilter(edm::ParameterSet const& p) :
	m_count(p.getParameter<int>("ivalue")) {
    }
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
    ~LumiIntFilter() {
      std::cout << "LumiIntFilter transitions " << m_count <<" \n";
    }
  };
  
  class RunSummaryIntFilter : public edm::global::EDFilter<edm::RunSummaryCache<int>> {
  public:
    explicit RunSummaryIntFilter(edm::ParameterSet const& p) :
	m_count(p.getParameter<int>("ivalue")) {
    }
 
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
    ~RunSummaryIntFilter() {
      std::cout << "RunSummaryIntFilter transitions " << m_count <<" \n";
    }
  };

  class LumiSummaryIntFilter : public edm::global::EDFilter<edm::LuminosityBlockSummaryCache<int>> {
  public:
    explicit LumiSummaryIntFilter(edm::ParameterSet const& p) :
	m_count(p.getParameter<int>("ivalue")) {	
    }
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
    ~LumiSummaryIntFilter() {
      std::cout << "LumiSummaryIntFilter transitions " << m_count <<" \n";
    }
  };

}
}

//using edmtest::global::IntProducer;
using edmtest::global::StreamIntProducer;
using edmtest::global::RunIntProducer;
using edmtest::global::LumiIntProducer;
using edmtest::global::RunSummaryIntProducer;
using edmtest::global::LumiSummaryIntProducer;
//DEFINE_FWK_MODULE(IntProducer);
DEFINE_FWK_MODULE(StreamIntProducer);
DEFINE_FWK_MODULE(RunIntProducer);
DEFINE_FWK_MODULE(LumiIntProducer);
DEFINE_FWK_MODULE(RunSummaryIntProducer);
DEFINE_FWK_MODULE(LumiSummaryIntProducer);
//using edmtest::global::IntAnalyzer;
using edmtest::global::StreamIntAnalyzer;
using edmtest::global::RunIntAnalyzer;
using edmtest::global::LumiIntAnalyzer;
using edmtest::global::RunSummaryIntAnalyzer;
using edmtest::global::LumiSummaryIntAnalyzer;
//DEFINE_FWK_MODULE(IntAnalyzer);
DEFINE_FWK_MODULE(StreamIntAnalyzer);
DEFINE_FWK_MODULE(RunIntAnalyzer);
DEFINE_FWK_MODULE(LumiIntAnalyzer);
DEFINE_FWK_MODULE(RunSummaryIntAnalyzer);
DEFINE_FWK_MODULE(LumiSummaryIntAnalyzer);
//using edmtest::global::IntFilter;
using edmtest::global::StreamIntFilter;
using edmtest::global::RunIntFilter;
using edmtest::global::LumiIntFilter;
using edmtest::global::RunSummaryIntFilter;
using edmtest::global::LumiSummaryIntFilter;
//DEFINE_FWK_MODULE(IntFilter);
DEFINE_FWK_MODULE(StreamIntFilter);
DEFINE_FWK_MODULE(RunIntFilter);
DEFINE_FWK_MODULE(LumiIntFilter);
DEFINE_FWK_MODULE(RunSummaryIntFilter);
DEFINE_FWK_MODULE(LumiSummaryIntFilter);


