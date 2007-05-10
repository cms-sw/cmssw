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
#include "CLHEP/Random/RandGauss.h"
using namespace std;

  //Constructor

SiStripLorentzAngleDB::SiStripLorentzAngleDB(edm::ParameterSet const& conf) : 
  conf_(conf){
  siStripLorentzAngleAlgorithm_=new SiStripLorentzAngleAlgorithm(conf);
}

  //BeginJob

void SiStripLorentzAngleDB::beginJob(const edm::EventSetup& c){
  
  appliedVoltage_   = conf_.getParameter<double>("AppliedVoltage");
  chargeMobility_   = conf_.getParameter<double>("ChargeMobility");
  temperature_      = conf_.getParameter<double>("Temperature");
  temperatureerror_      = conf_.getParameter<double>("TemperatureError");
  rhall_            = conf_.getParameter<double>("HoleRHAllParameter");
  holeBeta_         = conf_.getParameter<double>("HoleBeta");
  holeSaturationVelocity_ = conf_.getParameter<double>("HoleSaturationVelocity");
  
  //building histograms
  siStripLorentzAngleAlgorithm_->init(c);

  edm::ESHandle<TrackerGeometry> pDD;
  c.get<TrackerDigiGeometryRecord>().get( pDD );
  edm::LogInfo("SiStripLorentzAngle") <<" There are "<<pDD->detUnits().size() <<" detectors"<<std::endl;
  
  for(TrackerGeometry::DetUnitContainer::const_iterator it = pDD->detUnits().begin(); it != pDD->detUnits().end(); it++){
    
    if( dynamic_cast<StripGeomDetUnit*>((*it))!=0){
      uint32_t detid=((*it)->geographicalId()).rawId();

      double thickness=(*it)->specificSurface().bounds().thickness();
      float temperaturernd;
      if(temperatureerror_>0)temperaturernd=RandGauss::shoot(temperature_,temperatureerror_);
      else temperaturernd=temperature_;
      float mulow = chargeMobility_*pow((temperaturernd/300.),-2.5);
      float vsat = holeSaturationVelocity_*pow((temperaturernd/300.),0.52);
      float beta = holeBeta_*pow((temperaturernd/300.),0.17);
      float e = appliedVoltage_/thickness;
      float mu = ( mulow/(pow(double((1+pow((mulow*e/vsat),beta))),1./beta)));
      float hallMobility = 1.E-4*mu*rhall_;
      
      detid_la.push_back( pair<uint32_t,float>(detid,hallMobility) );
    }      
  } 

}
// Virtual destructor needed.

SiStripLorentzAngleDB::~SiStripLorentzAngleDB() {  
  if(siStripLorentzAngleAlgorithm_!=0){
    delete siStripLorentzAngleAlgorithm_;
    siStripLorentzAngleAlgorithm_=0;
  }
}  

// Analyzer: Functions that gets called by framework every event

void SiStripLorentzAngleDB::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  
  //fill histograms for each module
  siStripLorentzAngleAlgorithm_->run(e,es);
}

void SiStripLorentzAngleDB::endJob(){

  SiStripLorentzAngleAlgorithm::fitmap fits;
  SiStripLorentzAngle* LorentzAngle = new SiStripLorentzAngle();
  edm::LogInfo("SiStripLorentzAngle") <<"End job ";
  siStripLorentzAngleAlgorithm_->fit(fits);
  
  for(std::vector<std::pair<uint32_t, float> >::iterator it = detid_la.begin(); it != detid_la.end(); it++){

    SiStripLorentzAngleAlgorithm::fitmap::iterator thefit=fits.find(it->first);
    float langle=it->second;
    if(thefit!=fits.end()){
      if(thefit->second->p1>0&&thefit->second->p2>0)langle=thefit->second->p0;
    }
    if ( ! LorentzAngle->putLorentzAngle(it->first,langle) )
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
//  std::vector<DetId>::iterator Iditer;
  SiStripLorentzAngleAlgorithm::fitmap::iterator  fitpar;
  for( fitpar=fits.begin(); fitpar!=fits.end();++fitpar)delete fitpar->second;
}
