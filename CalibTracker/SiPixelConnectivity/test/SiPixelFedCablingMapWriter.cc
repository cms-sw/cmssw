// system include files
#include <memory>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CalibTracker/SiPixelConnectivity/interface/SiPixelFedCablingMapBuilder.h"

using namespace std;
using namespace edm;
using namespace sipixelobjects;

class SiPixelFedCablingMapWriter : public edm::EDAnalyzer {
 public:
  explicit SiPixelFedCablingMapWriter( const edm::ParameterSet& cfg);
  ~SiPixelFedCablingMapWriter();
  virtual void beginJob( const edm::EventSetup& );
  virtual void endJob( );
  virtual void analyze(const edm::Event& , const edm::EventSetup& ){}
 private:
  SiPixelFedCablingMap * cabling;
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

  ::putenv("CORAL_AUTH_USER=me");
  ::putenv("CORAL_AUTH_PASSWORD=none"); 
}


SiPixelFedCablingMapWriter::~SiPixelFedCablingMapWriter(){
//  delete cabling;
}


void SiPixelFedCablingMapWriter::beginJob( const edm::EventSetup& iSetup ) {
   edm::LogInfo("BeginJob method ");
   cabling = SiPixelFedCablingMapBuilder(pixelToFedAssociator_).produce(iSetup);
   edm::LogInfo("PRINTING MAP:") << cabling->print(3) << endl;
   edm::LogInfo("BeginJob method .. end");
}

void SiPixelFedCablingMapWriter::endJob( ) {
  LogInfo("Now NEW writing to DB");
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    std::cout<<"db service unavailable"<<std::endl;
    return;
  } else { std::cout<<"OK"<<std::endl; }

  try {
    if( mydbservice->isNewTagRequest(record_) ){
      mydbservice->createNewIOV<SiPixelFedCablingMap>( cabling, mydbservice->endOfTime(), record_);
    }else{
      mydbservice->appendSinceTime<SiPixelFedCablingMap>( 
          cabling, mydbservice->currentTime(), record_);
    }
  } 
  catch (std::exception &e) { LogError("std::exception:  ") << e.what(); }
  catch (...) { LogError("Unknown error caught "); }
  LogInfo("... all done, end");
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiPixelFedCablingMapWriter);
