
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CalibTracker/SiStripLorentzAngle/interface/SiStripLorentzAngleDB.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

using namespace std;

  //Constructor

SiStripLorentzAngleDB::SiStripLorentzAngleDB(edm::ParameterSet const& conf) : 
  conf_(conf){}

  //BeginJob

void SiStripLorentzAngleDB::beginJob(const edm::EventSetup& c){
  
  appliedVoltage_   = conf_.getParameter<double>("AppliedVoltage");
  chargeMobility_   = conf_.getParameter<double>("ChargeMobility");
  temperature_      = conf_.getParameter<double>("Temperature");
  rhall_            = conf_.getParameter<double>("HoleRHAllParameter");
  holeBeta_         = conf_.getParameter<double>("HoleBeta");
  holeSaturationVelocity_ = conf_.getParameter<double>("HoleSaturationVelocity");
  
  edm::ESHandle<TrackerGeometry> pDD;
  c.get<TrackerDigiGeometryRecord>().get( pDD );
  edm::LogInfo("SiStripLorentzAngle") <<" There are "<<pDD->detUnits().size() <<" detectors"<<std::endl;
  float mulow = chargeMobility_*pow((temperature_/300.),-2.5);
  float vsat = holeSaturationVelocity_*pow((temperature_/300.),0.52);
  float beta = holeBeta_*pow((temperature_/300.),0.17);
  
  for(TrackerGeometry::DetUnitContainer::const_iterator it = pDD->detUnits().begin(); it != pDD->detUnits().end(); it++){
    
    if( dynamic_cast<StripGeomDetUnit*>((*it))!=0){
      uint32_t detid=((*it)->geographicalId()).rawId();

      float thickness=(*it)->specificSurface().bounds().thickness();


      float e = appliedVoltage_/thickness;
      float mu = ( mulow/(pow(double((1+pow((mulow*e/vsat),beta))),1./beta)));
      float hallMobility = 1.E-4*mu*rhall_;
      
      detid_la.push_back( pair<uint32_t,float>(detid,hallMobility) );
    }      
  } 
}
// Virtual destructor needed.

SiStripLorentzAngleDB::~SiStripLorentzAngleDB() {  }  

// Analyzer: Functions that gets called by framework every event

void SiStripLorentzAngleDB::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  
  unsigned int run=e.id().run();
  edm::LogInfo("SiStripLorentzAngle") << "... inserting SiStripLorentzAngle for Run " << run << "\n " << std::endl;
  
  SiStripLorentzAngle* LorentzAngle = new SiStripLorentzAngle();
  
  for(std::vector<std::pair<uint32_t, float> >::iterator it = detid_la.begin(); it != detid_la.end(); it++){
    //    edm::LogInfo("SiStripLorentzAngle") <<"DetId= "<<it->first<<endl;
    if ( ! LorentzAngle->putLorentzAngle(it->first, it->second) )
      edm::LogError("SiStripLorentzAngleDB")<<"[SiStripLorentzAngleDB::analyze] detid already exists"<<std::endl;
  }

  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( mydbservice.isAvailable() ){
    try{
      if( mydbservice->isNewTagRequest("SiStripLorentzAngleRcd") ){
	mydbservice->createNewIOV<SiStripLorentzAngle>(LorentzAngle,mydbservice->endOfTime(),"SiStripLorentzAngleRcd");
      } else {
	mydbservice->appendSinceTime<SiStripLorentzAngle>(LorentzAngle,mydbservice->currentTime(),"SiStripLorentzAngleRcd");
      }
    }catch(const cond::Exception& er){
      edm::LogError("SiStripLorentzAngleDB")<<er.what()<<std::endl;
    }catch(const std::exception& er){
      edm::LogError("SiStripLorentzAngleDB")<<"caught std::exception "<<er.what()<<std::endl;
    }catch(...){
      edm::LogError("SiStripLorentzAngleDB")<<"Funny error"<<std::endl;
    }
  }else{
    edm::LogError("SiStripLorentzAngleDB")<<"Service is unavailable"<<std::endl;
  }
}

void SiStripLorentzAngleDB::endJob(){

  //  std::vector<DetId>::iterator Iditer;
  
}
