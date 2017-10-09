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
#include "CondFormats/DataRecord/interface/DQMReferenceHistogramRootFileRcd.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

namespace edmtest {
  class DQMReferenceHistogramRootFileEventSetupAnalyzer: public edm::EDAnalyzer {
   public:
    explicit DQMReferenceHistogramRootFileEventSetupAnalyzer(const edm::ParameterSet & pset);
    explicit DQMReferenceHistogramRootFileEventSetupAnalyzer(int i);
    virtual ~DQMReferenceHistogramRootFileEventSetupAnalyzer();
    virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
    virtual void beginRun(edm::Run const&, edm::EventSetup const&) override ;
  private:
    bool init_ ;
  };
  
  DQMReferenceHistogramRootFileEventSetupAnalyzer::DQMReferenceHistogramRootFileEventSetupAnalyzer(const edm::ParameterSet &ps) {
    init_ = false ;
    //std::cout << "DQMReferenceHistogramRootFileEventSetupAnalyzer(const edm::ParameterSet &ps)" << std::endl;
  }
  
  DQMReferenceHistogramRootFileEventSetupAnalyzer::DQMReferenceHistogramRootFileEventSetupAnalyzer(int i) {
    init_ = false ;
    //std::cout << "DQMReferenceHistogramRootFileEventSetupAnalyzer(int i) " << i << std::endl;
  }
  
  DQMReferenceHistogramRootFileEventSetupAnalyzer::~DQMReferenceHistogramRootFileEventSetupAnalyzer()
  {
    init_ = false ;
    //std::cout << "~DQMReferenceHistogramRootFileEventSetupAnalyzer" << std::endl;
  }
  
  void DQMReferenceHistogramRootFileEventSetupAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup)
  {
    return ;
  }

  void DQMReferenceHistogramRootFileEventSetupAnalyzer::beginRun(edm::Run const& run , edm::EventSetup const& iSetup)
  {
    //std::cout << "DQMReferenceHistogramRootFileEventSetupAnalyzer::beginRun()" << std::endl;    
    if(!init_)
      {
	init_ = true ;
	edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("DQMReferenceHistogramRootFileRcd"));
	if(recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
	  throw cms::Exception ("Record not found") << "Record \"DQMReferenceHistogramRootFileRcd" 
						    << "\" does not exist!" << std::endl;
	}
	edm::ESHandle<FileBlob> rootgeo;
	iSetup.get<DQMReferenceHistogramRootFileRcd>().get(rootgeo);
	//std::cout<<"ROOT FILE IN MEMORY"<<std::endl;
        std::unique_ptr<std::vector<unsigned char> > tb( (*rootgeo).getUncompressedBlob() );
	// char filename[128];
	// sprintf(filename, "mem:%p,%ul", &(*tb)[0], (unsigned long) tb->size());
	// edm::Service<DQMStore>()->open(filename, false, "", "Reference");
	
	//here you can implement the stream for putting the TFile on disk...
	std::string outfile("dqmreference.root") ;
	std::ofstream output(outfile.c_str()) ;
	output.write((const char *)&(*tb)[0], tb->size());
	output.close() ;
	
	DQMStore *dqm = &*edm::Service<DQMStore>();
	dqm->open(outfile, false, "", "Reference");
 	remove(outfile.c_str());
	
	std::vector<MonitorElement *> mes = dqm->getAllContents("");
	// for (std::vector<MonitorElement *>::iterator i = mes.begin(), e = mes.end(); i != e; ++i)
	//  std::cout << "ME '" << (*i)->getFullname() << "'\n";
	
	//std::cout<<"SIZE FILE = "<<tb->size()<<std::endl;
      }
  }
  
  DEFINE_FWK_MODULE(DQMReferenceHistogramRootFileEventSetupAnalyzer);
}
