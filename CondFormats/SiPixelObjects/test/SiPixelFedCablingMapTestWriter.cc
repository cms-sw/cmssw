// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

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
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

using namespace std;
using namespace edm;

class SiPixelFedCablingMapTestWriter : public edm::EDAnalyzer {
 public:
  explicit SiPixelFedCablingMapTestWriter( const edm::ParameterSet& );
  ~SiPixelFedCablingMapTestWriter();
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
};


SiPixelFedCablingMapTestWriter::SiPixelFedCablingMapTestWriter( const edm::ParameterSet& iConfig ) : dbconnect_(iConfig.getParameter<std::string>("dbconnect")), tag_(iConfig.getParameter<std::string>("tag")), catalog_(iConfig.getUntrackedParameter<std::string>("catalog","")) {
  cout <<" HERE dbconnect: "
       << iConfig.getParameter<std::string>("dbconnect")<<endl;
  cout <<" HERE       tag: "
       <<iConfig.getParameter<std::string>("tag") << endl;
  cout <<" HERE   catalog: " 
       << iConfig.getUntrackedParameter<std::string>("catalog","") << endl;

  ::putenv("CORAL_AUTH_USER=konec");
  ::putenv("CORAL_AUTH_PASSWORD=konecPass"); 
}


SiPixelFedCablingMapTestWriter::~SiPixelFedCablingMapTestWriter(){
  cout<<"Now writing to DB"<<endl;
  try {
    loader=new cond::ServiceLoader;
    loader->loadAuthenticationService( cond::Env );
    loader->loadMessageService( cond::Error );

    session=new cond::DBSession(dbconnect_);
    session->setCatalog(catalog_);
    session->connect(cond::ReadWriteCreate);

    writer   =new cond::DBWriter(*session, "SiPixelFedCablingMap");
    iovwriter =new cond::DBWriter(*session, "IOV");
    session->startUpdateTransaction();

    cond::IOV* cabIOV= new cond::IOV; 
    int run = edm::IOVSyncValue::endOfTime().eventID().run();

    cout << "markWrite cabling..." << flush;
    string cabTok = writer->markWrite<SiPixelFedCablingMap>(cabling);  
    
    cout << "Associate IOV..." << flush;
    cabIOV->iov.insert(std::make_pair(run, cabTok));
    cout << "Done." << endl;
    
    cout << "markWrite IOV..." << flush;
    string cabIOVTok = iovwriter->markWrite<cond::IOV>(cabIOV);  // ownership given
    cout << "Commit..." << flush;
    //    writer->commitTransaction();  // pedIOV memory freed
    session->commit();  // pedIOV memory freed
    session->disconnect();
    cout << "Add MetaData... " << endl;
    metadataSvc = new cond::MetaData(dbconnect_,*loader);
    metadataSvc->connect();
    metadataSvc->addMapping(tag_,cabIOVTok);
    metadataSvc->disconnect();
    cout << "Done." << endl;
  }
  catch(const cond::Exception& e){
    std::cout<<"cond::Exception: " << e.what()<<std::endl;
    if(loader) delete loader;
  } 
  catch (pool::Exception& e) {
    cout << "pool::Exception:  " << e.what() << endl;
    if(loader) delete loader;
  }
  catch (std::exception &e) {
    cout << "std::exception:  " << e.what() << endl;
    if(loader) delete loader;
  }
  catch (...) {
    cout << "Unknown error caught" << endl;
    if(loader) delete loader;
  }
  if(session) delete session;
  if (writer) delete writer;
  if(iovwriter) delete iovwriter;
  if (metadataSvc) delete metadataSvc;
  if(loader) delete loader;
}


// ------------ method called to produce the data  ------------
void SiPixelFedCablingMapTestWriter::beginJob( const edm::EventSetup& iSetup ) {
   cout << "BeginJob method " << endl;
   cout<<"Building FED Cabling"<<endl;   
   cabling =  new SiPixelFedCablingMap("My map V-TEST");

   PixelROC r1(0,1,2,3,4);
   PixelROC r2(1,12,13,14,15);
   PixelFEDLink link(0);
   PixelFEDLink::ROCs rocs; rocs.push_back(r1); rocs.push_back(r2);
   PixelFEDLink::Connection  con = {0, "det_name", make_pair<int,int>(0,1) };
   link.add(con,rocs);

   PixelFEDCabling fed(0);
   fed.addLink(link);

   
   cabling->addFed(fed);

   
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelFedCablingMapTestWriter)
