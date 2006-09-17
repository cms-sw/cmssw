// system include files
#include <memory>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/DBCommon/interface/DBWriter.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/ServiceLoader.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/IOVService/interface/IOV.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "SealKernel/Service.h"
#include "POOLCore/POOLContext.h"
#include "SealKernel/Context.h"

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
  virtual void analyze(const edm::Event& , const edm::EventSetup& ){}
 private:
  cond::ServiceLoader* loader;
  cond::DBSession* session;
  cond::DBWriter* writer;
  cond::DBWriter* iovwriter;
  cond::MetaData* metadataSvc;
  SiPixelFedCablingMap * cabling;
  string dbconnect_;
  string tag_;
  string catalog_;
  string pixelToFedAssociator_;
};

SiPixelFedCablingMapWriter::SiPixelFedCablingMapWriter( 
    const edm::ParameterSet& cfg ) 
  : dbconnect_(cfg.getParameter<std::string>("dbconnect")), 
    tag_(cfg.getParameter<std::string>("tag")), 
    catalog_(cfg.getUntrackedParameter<std::string>("catalog","")), 
    pixelToFedAssociator_(cfg.getUntrackedParameter<std::string>("associator","PixelToFEDAssociateFromAscii")) 
{
  
  stringstream out;
  out << " HERE dbconnect:            " << dbconnect_ << endl;
  out << " HERE tag:                  " << tag_ << endl;
  out << " HERE catalog:              " << catalog_ << endl;
  out << " HERE pixelToFedAssociator: " << pixelToFedAssociator_ << endl;
  LogInfo("initialisatino: ")<<out.str();

  ::putenv("CORAL_AUTH_USER=me");
  ::putenv("CORAL_AUTH_PASSWORD=none"); 
}


SiPixelFedCablingMapWriter::~SiPixelFedCablingMapWriter(){
  LogInfo("Now writing to DB");
  try {
    loader=new cond::ServiceLoader;
    loader->loadAuthenticationService( cond::Env );
    loader->loadMessageService( cond::Error );

    session=new cond::DBSession(dbconnect_);
    session->setCatalog(catalog_);
    session->connect(cond::ReadWriteCreate);

    writer   =new cond::DBWriter(*session, "SiPixelReadOutMap");
    iovwriter =new cond::DBWriter(*session, "IOV");
    session->startUpdateTransaction();

    cond::IOV* cabIOV= new cond::IOV; 
    int run = edm::IOVSyncValue::endOfTime().eventID().run();

    LogInfo("markWrite cabling...");
    string cabTok = writer->markWrite<SiPixelFedCablingMap>(cabling);  
    
    LogInfo( "Associate IOV...");
    cabIOV->iov.insert(std::make_pair(run, cabTok));
    
    LogInfo( "markWrite IOV...") ;
    string cabIOVTok = iovwriter->markWrite<cond::IOV>(cabIOV);  // ownership given
    LogInfo("Commit...");
    session->commit();  // pedIOV memory freed
    session->disconnect();
    LogInfo("Add MetaData... ");
    metadataSvc = new cond::MetaData(dbconnect_,*loader);
    metadataSvc->connect();
    metadataSvc->addMapping(tag_,cabIOVTok);
    metadataSvc->disconnect();
    LogInfo("... all done, end");
  }
  catch(const cond::Exception& e){
    LogError("cond::Exception: ") << e.what();
    if(loader) delete loader;
  } 
  catch (pool::Exception& e) {
   LogError("pool::Exception:  ")<< e.what();
    if(loader) delete loader;
  }
  catch (std::exception &e) {
    LogError("std::exception:  ") << e.what();
    if(loader) delete loader;
  }
  catch (...) {
    LogError("Unknown error caught ");
    if(loader) delete loader;
  }
  if(session) delete session;
  if (writer) delete writer;
  if(iovwriter) delete iovwriter;
  if (metadataSvc) delete metadataSvc;
  if(loader) delete loader;
}


// ------------ method called to produce the data  ------------
void SiPixelFedCablingMapWriter::beginJob( const edm::EventSetup& iSetup ) {
   edm::LogInfo("BeginJob method ");
   cabling = SiPixelFedCablingMapBuilder(pixelToFedAssociator_).produce(iSetup);
   edm::LogInfo("PRINTING MAP:") << cabling->print(4) << endl;
   edm::LogInfo("BeginJob method .. end");
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelFedCablingMapWriter)
