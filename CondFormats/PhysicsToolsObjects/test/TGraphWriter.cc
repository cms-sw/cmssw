#include "CondFormats/PhysicsToolsObjects/test/TGraphWriter.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/PhysicsToolsObjects/interface/PhysicsTGraphPayload.h"

#include <TFile.h>
#include <TGraph.h>

TGraphWriter::TGraphWriter(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  edm::VParameterSet cfgJobs = cfg.getParameter<edm::VParameterSet>("jobs");
  for ( edm::VParameterSet::const_iterator cfgJob = cfgJobs.begin();
	cfgJob != cfgJobs.end(); ++cfgJob ) {
    jobEntryType* job = new jobEntryType(*cfgJob);
    jobs_.push_back(job);
  }
}

TGraphWriter::~TGraphWriter()
{
  for ( std::vector<jobEntryType*>::iterator it = jobs_.begin();
	it != jobs_.end(); ++it ) {
    delete (*it);
  }
}

void TGraphWriter::analyze(const edm::Event&, const edm::EventSetup&)
{
  std::cout << "<TGraphWriter::analyze (moduleLabel = " << moduleLabel_ << ")>:" << std::endl;
  
  for ( std::vector<jobEntryType*>::iterator job = jobs_.begin();
	job != jobs_.end(); ++job ) {   
    TFile* inputFile = new TFile((*job)->inputFileName_.data());
    std::cout << "reading TGraph = " << (*job)->graphName_ << " from ROOT file = " << (*job)->inputFileName_ << "." << std::endl;
    const TGraph* graph = dynamic_cast<TGraph*>(inputFile->Get((*job)->graphName_.data()));
    delete inputFile;
    if ( !graph ) 
      throw cms::Exception("TGraphWriter") 
	<< " Failed to load TGraph = " << (*job)->graphName_.data() << " from file = " << (*job)->inputFileName_ << " !!\n";
    edm::Service<cond::service::PoolDBOutputService> dbService;
    if ( !dbService.isAvailable() ) 
      throw cms::Exception("TGraphWriter") 
	<< " Failed to access PoolDBOutputService !!\n";
    std::cout << " writing TGraph = " << (*job)->graphName_ << " to SQLlite file, record = " << (*job)->outputRecord_ << "." << std::endl;
    PhysicsTGraphPayload* graphPayload = new PhysicsTGraphPayload(*graph);
    delete graph;
    dbService->writeOne(graphPayload, dbService->beginOfTime(), (*job)->outputRecord_);
  }
  
  std::cout << "done." << std::endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TGraphWriter);
