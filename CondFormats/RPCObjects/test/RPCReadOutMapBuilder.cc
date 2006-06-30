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
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/RPCObjects/interface/DccSpec.h"
#include "CondFormats/RPCObjects/interface/TriggerBoardSpec.h"
#include "CondFormats/RPCObjects/interface/LinkConnSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"

using namespace std;
using namespace edm;

class RPCReadOutMapBuilder : public edm::EDAnalyzer {
 public:
  explicit RPCReadOutMapBuilder( const edm::ParameterSet& );
  ~RPCReadOutMapBuilder();
  virtual void beginJob( const edm::EventSetup& );
  virtual void analyze(const edm::Event& , const edm::EventSetup& ){}
 private:
  cond::ServiceLoader* loader;
  cond::DBSession* session;
  cond::DBWriter* writer;
  cond::DBWriter* iovwriter;
  cond::MetaData* metadataSvc;
  RPCReadOutMapping * cabling;
  string dbconnect_;
  string tag_;
  string catalog_;
};


RPCReadOutMapBuilder::RPCReadOutMapBuilder( const edm::ParameterSet& iConfig ) : dbconnect_(iConfig.getParameter<std::string>("dbconnect")), tag_(iConfig.getParameter<std::string>("tag")), catalog_(iConfig.getUntrackedParameter<std::string>("catalog","")) {
  cout <<" HERE dbconnect: "
       << iConfig.getParameter<std::string>("dbconnect")<<endl;
  cout <<" HERE       tag: "
       <<iConfig.getParameter<std::string>("tag") << endl;
  cout <<" HERE   catalog: " 
       << iConfig.getUntrackedParameter<std::string>("catalog","") << endl;

  ::putenv("CORAL_AUTH_USER=konec");
  ::putenv("CORAL_AUTH_PASSWORD=konecPass"); 
}


RPCReadOutMapBuilder::~RPCReadOutMapBuilder(){
  cout<<"Now writing to DB"<<endl;
  try {
    loader=new cond::ServiceLoader;
    loader->loadAuthenticationService( cond::Env );
    loader->loadMessageService( cond::Error );

    session=new cond::DBSession(dbconnect_);
    session->setCatalog(catalog_);
    session->connect(cond::ReadWriteCreate);

    writer   =new cond::DBWriter(*session, "RPCReadOutMapping");
    iovwriter =new cond::DBWriter(*session, "IOV");
    session->startUpdateTransaction();

    cond::IOV* cabIOV= new cond::IOV; 
    int run = edm::IOVSyncValue::endOfTime().eventID().run();

    cout << "markWrite cabling..." << flush;
    string cabTok = writer->markWrite<RPCReadOutMapping>(cabling);  
    
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
void RPCReadOutMapBuilder::beginJob( const edm::EventSetup& iSetup ) {
   cout << "BeginJob method " << endl;
   cout<<"Building RPC Cabling"<<endl;   
   cabling =  new RPCReadOutMapping("My map V-TEST");
   {
     DccSpec dcc(790);
     TriggerBoardSpec tb1(6);
     LinkConnSpec  lc(8); 
     ChamberLocationSpec chamber = {1,2,3,"Cos","+-","cos","cos","E"};
     LinkBoardSpec lb(true, 2, chamber);
     lc.add(lb);
     tb1.add(lc);
     dcc.add(tb1);

     TriggerBoardSpec tb2(2);
     dcc.add(tb2);
     cabling->add(dcc); 
   }
   {
     TriggerBoardSpec tb(12);
     DccSpec dcc(791);
     dcc.add(tb);
     cabling->add(dcc);
   }
   
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCReadOutMapBuilder)
