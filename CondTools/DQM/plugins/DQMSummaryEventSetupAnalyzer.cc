#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/DQMObjects/interface/DQMSummary.h"
#include "CondFormats/DataRecord/interface/DQMSummaryRcd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <iostream>

namespace edmtest {
  class DQMSummaryEventSetupAnalyzer: public edm::EDAnalyzer {
   public:
    explicit DQMSummaryEventSetupAnalyzer(const edm::ParameterSet & pset);
    explicit DQMSummaryEventSetupAnalyzer(int i);
    virtual ~DQMSummaryEventSetupAnalyzer();
    virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  };
  
  DQMSummaryEventSetupAnalyzer::DQMSummaryEventSetupAnalyzer(const edm::ParameterSet & pset) {
    std::cout << "DQMSummaryEventSetupAnalyzer" << std::endl;
  }

  DQMSummaryEventSetupAnalyzer::DQMSummaryEventSetupAnalyzer(int i) {
    std::cout << "DQMSummaryEventSetupAnalyzer" << i << std::endl;
  }
  
  DQMSummaryEventSetupAnalyzer::~DQMSummaryEventSetupAnalyzer() {
    std::cout << "~DQMSummaryEventSetupAnalyzer" << std::endl;
  }
  
  void DQMSummaryEventSetupAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
    std::cout << "### DQMSummaryEventSetupAnalyzer::analyze" << std::endl;
      std::cout << "--- RUN NUMBER: " << event.id().run() << std::endl;
      std::cout << "--- EVENT NUMBER: " << event.id().event() << std::endl;
      edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("DQMSummaryRcd"));
      if(recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
	throw cms::Exception ("Record not found") << "Record \"DQMSummaryRcd" 
						  << "\" does not exist!" << std::endl;
      }
      edm::ESHandle<DQMSummary> sum;
      std::cout << "got EShandle" << std::endl;
      setup.get<DQMSummaryRcd>().get(sum);
      std::cout <<"got the Event Setup" << std::endl;
      const DQMSummary* summary = sum.product();
      std::cout <<"got DQMSummary* "<< std::endl;
      std::cout<< "print result" << std::endl;
      summary->printAllValues();
  }
  
  DEFINE_FWK_MODULE(DQMSummaryEventSetupAnalyzer);
}
