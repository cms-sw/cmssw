// system include files
#include <memory>

// user include files

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"


#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h" 
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"


#include "CondTools/SiStrip/plugins/SiStripDetVOffBuilder.h"

using namespace std;
using namespace cms;

SiStripDetVOffBuilder::SiStripDetVOffBuilder( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<bool>("printDebug",false)){}

SiStripDetVOffBuilder::~SiStripDetVOffBuilder(){}


void SiStripDetVOffBuilder::beginJob( const edm::EventSetup& iSetup ) {

  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get( pDD );
  edm::LogInfo("SiStripDetVOffBuilder") <<" There are "<<pDD->detUnits().size() <<" detectors"<<std::endl;
  
  for(TrackerGeometry::DetUnitContainer::const_iterator it = pDD->detUnits().begin(); it != pDD->detUnits().end(); it++){
  
    if( dynamic_cast<StripGeomDetUnit*>((*it))!=0){
      uint32_t detid=((*it)->geographicalId()).rawId();            
      const StripTopology& p = dynamic_cast<StripGeomDetUnit*>((*it))->specificTopology();
      unsigned short Nstrips = p.nstrips();
      if(Nstrips<1 || Nstrips>768 ) {
	edm::LogError("SiStripDetVOffBuilder")<<" Problem with Number of strips in detector.. "<< p.nstrips() <<" Exiting program"<<endl;
	exit(1);
      }
      detids.push_back(detid);
      if (printdebug_)
	edm::LogInfo("SiStripDetVOffBuilder")<< "detid " << detid;
    }
  }
}

void SiStripDetVOffBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup){

  unsigned int run=evt.id().run();

  edm::LogInfo("SiStripDetVOffBuilder") << "... creating dummy SiStripDetVOff Data for Run " << run << "\n " << std::endl;



  SiStripDetVOff* SiStripDetVOff_ = new SiStripDetVOff();

   // std::vector<uint32_t> TheDetIdHVVector;

    for(std::vector<uint32_t>::const_iterator it = detids.begin(); it != detids.end(); it++){
       
    //Generate HV and LV for each channel, if at least one of the two is off fill the value
    int hv=rand() % 20;
    int lv=rand() % 20;
    if( hv<=2 ) {
      edm::LogInfo("SiStripDetVOffBuilder") << "detid with HV off: " <<  *it << std::endl;
      SiStripDetVOff_->put( *it, false, true );
      // TheDetIdHVVector.push_back(*it);
    }
    if( lv<=2 ) {
      SiStripDetVOff_->put( *it, true, false );
      // TheDetIdHVVector.push_back(*it);
    }
  }



  // SiStripDetVOff_->put(TheDetIdHVVector);



  //End now write DetVOff data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  
  if( mydbservice.isAvailable() ){
    try{
      if( mydbservice->isNewTagRequest("SiStripDetVOffRcd") ){
	mydbservice->createNewIOV<SiStripDetVOff>(SiStripDetVOff_,mydbservice->endOfTime(),"SiStripDetVOffRcd");      
      } else {
	mydbservice->appendSinceTime<SiStripDetVOff>(SiStripDetVOff_,mydbservice->currentTime(),"SiStripDetVOffRcd");      
      }
    }catch(const cond::Exception& er){
      edm::LogError("SiStripDetVOffBuilder")<<er.what()<<std::endl;
    }catch(const std::exception& er){
      edm::LogError("SiStripDetVOffBuilder")<<"caught std::exception "<<er.what()<<std::endl;
    }catch(...){
      edm::LogError("SiStripDetVOffBuilder")<<"Funny error"<<std::endl;
    }
  }else{
    edm::LogError("SiStripDetVOffBuilder")<<"Service is unavailable"<<std::endl;
  }
}
