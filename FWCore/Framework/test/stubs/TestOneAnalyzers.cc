
/*----------------------------------------------------------------------

Toy edm::one::EDAnalyzer modules of 
edm::one cache templates 
for testing purposes only.

----------------------------------------------------------------------*/
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
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


  class SharedResourcesAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
  public:
    explicit SharedResourcesAnalyzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
      usesResource();
    }
    const unsigned int trans_; 
    unsigned int m_count = 0;

    void analyze(edm::Event const&, edm::EventSetup const&) override {
      ++m_count;
       
    }
    
   ~SharedResourcesAnalyzer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "SharedResourcesAnalyzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };
  
  class WatchRunsAnalyzer: public edm::one::EDAnalyzer<edm::one::WatchRuns> {
  public:
    explicit WatchRunsAnalyzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    bool br = false;
    bool er = false;
    const unsigned int trans_; 
    unsigned int m_count = 0;

    void analyze(edm::Event const&, edm::EventSetup const&) override {
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
 
   ~WatchRunsAnalyzer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "WatchRunsAnalyzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };


  class WatchLumiBlocksAnalyzer: public edm::one::EDAnalyzer<edm::one::WatchLuminosityBlocks> {
  public:
    explicit WatchLumiBlocksAnalyzer(edm::ParameterSet const& p) :
	trans_(p.getParameter<int>("transitions")) {
    }
    const unsigned int trans_; 
    bool bl = false;
    bool el = false;
    unsigned int m_count = 0;

   void analyze(edm::Event const&, edm::EventSetup const&) override {
      ++m_count;
      if ( !bl  || el ) {
          throw cms::Exception("out of sequence")
          << " produce before beginLumiBlock or after endLumiBlock";
      }              
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

   ~WatchLumiBlocksAnalyzer() {
      if(m_count != trans_) {
        throw cms::Exception("transitions")
          << "WatchLumiBlocksAnalyzer transitions "
          << m_count<< " but it was supposed to be " << trans_;
      }
    }
  };
  


}
}

DEFINE_FWK_MODULE(edmtest::one::SharedResourcesAnalyzer);
DEFINE_FWK_MODULE(edmtest::one::WatchRunsAnalyzer);
DEFINE_FWK_MODULE(edmtest::one::WatchLumiBlocksAnalyzer);

