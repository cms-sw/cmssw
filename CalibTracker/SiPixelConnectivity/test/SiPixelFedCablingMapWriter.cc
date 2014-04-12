// system include files
#include <memory>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CalibTracker/SiPixelConnectivity/interface/SiPixelFedCablingMapBuilder.h"

using namespace std;
using namespace edm;
using namespace sipixelobjects;

class SiPixelFedCablingMapWriter : public edm::EDAnalyzer {
 public:
  explicit SiPixelFedCablingMapWriter( const edm::ParameterSet& cfg);
  ~SiPixelFedCablingMapWriter();
  virtual void beginJob();
  virtual void endJob( );
  virtual void analyze(const edm::Event& , const edm::EventSetup&);
 private:
  SiPixelFedCablingTree * cabling;
  string record_;
  string pixelToFedAssociator_;
};

SiPixelFedCablingMapWriter::SiPixelFedCablingMapWriter( 
    const edm::ParameterSet& cfg ) 
  : 
    record_(cfg.getParameter<std::string>("record")), 
    pixelToFedAssociator_(cfg.getUntrackedParameter<std::string>("associator","PixelToFEDAssociateFromAscii")) 
{
  
  stringstream out;
  out << " HERE record:               " << record_ << endl;
  out << " HERE pixelToFedAssociator: " << pixelToFedAssociator_ << endl;
  LogInfo("initialisatino: ")<<out.str();


  ::putenv(const_cast<char*>(std::string("CORAL_AUTH_USER=me").c_str()));
  ::putenv(const_cast<char*>(std::string("CORAL_AUTH_PASSWORD=none").c_str())); 
}


SiPixelFedCablingMapWriter::~SiPixelFedCablingMapWriter(){
//  delete cabling;
}


void SiPixelFedCablingMapWriter::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  static int first(1); 
  if (1 == first) {
    first = 0; 
    std::cout << "-------HERE-----------" << endl;
    cabling = SiPixelFedCablingMapBuilder(pixelToFedAssociator_).produce(iSetup);
    std::cout << "-------HERE2-----------" << endl;
    edm::LogInfo("PRINTING MAP:") << cabling->print(3) << endl;
  }
}

void SiPixelFedCablingMapWriter::beginJob() {
}

void SiPixelFedCablingMapWriter::endJob( ) {
  SiPixelFedCablingMap * result = new SiPixelFedCablingMap(cabling);
  LogInfo("Now NEW writing to DB");
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    std::cout<<"db service unavailable"<<std::endl;
    return;
  } else { std::cout<<"OK"<<std::endl; }

  try {
    if( mydbservice->isNewTagRequest(record_) ){
      mydbservice->createNewIOV<SiPixelFedCablingMap>( result, 
						       mydbservice->beginOfTime(),
						       mydbservice->endOfTime(), 
						       record_);
    }else{
      mydbservice->appendSinceTime<SiPixelFedCablingMap>( 
          result, 
	  mydbservice->currentTime(), 
	  record_);
    }
  } 
  catch (std::exception &e) { LogError("std::exception:  ") << e.what(); }
  catch (...) { LogError("Unknown error caught "); }
  LogInfo("... all done, end");
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiPixelFedCablingMapWriter);
