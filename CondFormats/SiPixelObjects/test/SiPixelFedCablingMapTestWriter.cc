// system include files
#include <memory>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class EventSetup;
class Event;

using namespace std;
using namespace edm;
using namespace sipixelobjects;

class SiPixelFedCablingMapTestWriter : public edm::EDAnalyzer {
 public:
  explicit SiPixelFedCablingMapTestWriter( const edm::ParameterSet& );
  ~SiPixelFedCablingMapTestWriter();
  virtual void beginJob( const edm::EventSetup& );
  virtual void endJob();
  virtual void analyze(const edm::Event& , const edm::EventSetup& ){}
 private:
  SiPixelFedCablingMap * cabling;
  string m_record;
};


SiPixelFedCablingMapTestWriter::SiPixelFedCablingMapTestWriter( const edm::ParameterSet& iConfig ) 
  : cabling(0),
    m_record(iConfig.getParameter<std::string>("record"))
{
  cout <<" HERE record: "<< m_record<<endl;
  ::putenv("CORAL_AUTH_USER=konec");
  ::putenv("CORAL_AUTH_PASSWORD=test"); 
}

void  SiPixelFedCablingMapTestWriter::endJob()
{
  cout<<"Now writing to DB"<<endl;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    cout<<"db service unavailable"<<endl;
    return;
  } else { cout<<"DB service OK"<<endl; }

  try {
    if( mydbservice->isNewTagRequest(m_record) ) {
      mydbservice->createNewIOV<SiPixelFedCablingMap>( 
          cabling, mydbservice->endOfTime(), m_record);
    } else {
      mydbservice->appendSinceTime<SiPixelFedCablingMap>(
          cabling, mydbservice->currentTime(), m_record);
    }
  }
  catch (std::exception &e) { cout <<"std::exception:  "<< e.what(); }
  catch (...) { cout << "Unknown error caught "<<endl; }
  cout<<"... all done, end"<<endl;

}

SiPixelFedCablingMapTestWriter::~SiPixelFedCablingMapTestWriter()
{
  cout <<"DTOR !" << endl;
}

// ------------ method called to produce the data  ------------
void SiPixelFedCablingMapTestWriter::beginJob( const edm::EventSetup& iSetup ) {
   cout << "BeginJob method " << endl;
   cout<<"Building FED Cabling"<<endl;   
   cabling =  new SiPixelFedCablingMap("My map V-TEST");
  

   PixelROC r1;
   PixelROC r2;

   PixelFEDLink link(0);
   PixelFEDLink::ROCs rocs; rocs.push_back(r1); rocs.push_back(r2);
   link.add(rocs);

   PixelFEDCabling fed(0);
   fed.addLink(link);
   cabling->addFed(fed);
   cout <<"PRINTING MAP:"<<endl<<cabling->print(3) << endl;

   
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiPixelFedCablingMapTestWriter);
