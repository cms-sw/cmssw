// C++ common header
#include <iostream>
#include <memory>
#include <vector>
#include <fstream>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/Common/interface/FileBlob.h"
#include "CondFormats/DataRecord/interface/DQMXMLFileRcd.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

namespace edmtest {
  class DQMXMLFileEventSetupAnalyzer: public edm::EDAnalyzer {
   public:
    explicit DQMXMLFileEventSetupAnalyzer(const edm::ParameterSet & pset);
    explicit DQMXMLFileEventSetupAnalyzer(int i);
    virtual ~DQMXMLFileEventSetupAnalyzer();
    virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
    virtual void beginRun(edm::Run const&, edm::EventSetup const&) override ;
  private:
    bool init_ ;
    std::string labelToGet_ ;
  };
  
  DQMXMLFileEventSetupAnalyzer::DQMXMLFileEventSetupAnalyzer(const edm::ParameterSet &ps):labelToGet_(ps.getParameter<std::string>("labelToGet")) {
    init_ = false ;
    //std::cout << "DQMXMLFileEventSetupAnalyzer(const edm::ParameterSet &ps)" << std::endl;
  }
  
  DQMXMLFileEventSetupAnalyzer::DQMXMLFileEventSetupAnalyzer(int i) {
    init_ = false ;
    //std::cout << "DQMXMLFileEventSetupAnalyzer(int i) " << i << std::endl;
  }
  
  DQMXMLFileEventSetupAnalyzer::~DQMXMLFileEventSetupAnalyzer()
  {
    init_ = false ;
    //std::cout << "~DQMXMLFileEventSetupAnalyzer" << std::endl;
  }
  
  void DQMXMLFileEventSetupAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup)
  {
    return ;
  }

  void DQMXMLFileEventSetupAnalyzer::beginRun(edm::Run const& run , edm::EventSetup const& iSetup)
  {
    //std::cout << "DQMXMLFileEventSetupAnalyzer::beginRun()" << std::endl;    
    if(!init_)
      {
	init_ = true ;
	edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("DQMXMLFileRcd"));
	if(recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
	  throw cms::Exception ("Record not found") << "Record \"DQMXMLFileRcd" 
						    << "\" does not exist!" << std::endl;
	}
	edm::ESHandle<FileBlob> rootgeo;
	iSetup.get<DQMXMLFileRcd>().get(labelToGet_,rootgeo);
	//std::cout<<"XML FILE IN MEMORY 1 with label " << labelToGet_ <<std::endl;
	std::unique_ptr<std::vector<unsigned char> > tb1( (*rootgeo).getUncompressedBlob() );
	//here you can implement the stream for putting the TFile on disk...
	std::string outfile1("XML1_retrieved.xml") ;
	std::ofstream output1(outfile1.c_str()) ;
	output1.write((const char *)&(*tb1)[0], tb1->size());
	output1.close() ;
	
// 	iSetup.get<DQMXMLFileRcd>().get("XML2_mio",rootgeo);
// 	std::cout<<"ROOT FILE IN MEMORY 2"<<std::endl;
// 	std::unique_ptr<std::vector<unsigned char> > tb2( (*rootgeo).getUncompressedBlob() );
// 	//here you can implement the stream for putting the TFile on disk...
// 	std::string outfile2("XML2_retrieved.xml") ;
// 	std::ofstream output2(outfile2.c_str()) ;
// 	output2.write((const char *)&(*tb2)[0], tb2->size());
// 	output2.close() ;
	//	std::unique_ptr<std::vector<unsigned char> > tb( (*rootgeo).getUncompressedBlob() );
      }
  }
  
  DEFINE_FWK_MODULE(DQMXMLFileEventSetupAnalyzer);
}
