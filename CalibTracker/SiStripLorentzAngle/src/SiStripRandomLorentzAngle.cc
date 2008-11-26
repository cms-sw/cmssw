#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include "CalibTracker/SiStripLorentzAngle/interface/SiStripRandomLorentzAngle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CLHEP/Random/RandGauss.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
using namespace std;

  //Constructor

SiStripRandomLorentzAngle::SiStripRandomLorentzAngle(edm::ParameterSet const& conf) : ConditionDBWriter<SiStripLorentzAngle>(conf) , conf_(conf){}


void SiStripRandomLorentzAngle::algoBeginJob(const edm::EventSetup& c){
  
  appliedVoltage_   = conf_.getParameter<double>("AppliedVoltage");
  chargeMobility_   = conf_.getParameter<double>("ChargeMobility");
  temperature_      = conf_.getParameter<double>("Temperature");
  temperatureerror_      = conf_.getParameter<double>("TemperatureError");
  rhall_            = conf_.getParameter<double>("HoleRHAllParameter");
  holeBeta_         = conf_.getParameter<double>("HoleBeta");
  holeSaturationVelocity_ = conf_.getParameter<double>("HoleSaturationVelocity");
  

  edm::ESHandle<TrackerGeometry> pDD;
  c.get<TrackerDigiGeometryRecord>().get( pDD );
  edm::LogInfo("SiStripLorentzAngle") <<" There are "<<pDD->detUnits().size() <<" detectors"<<std::endl;
  
  for(TrackerGeometry::DetUnitContainer::const_iterator it = pDD->detUnits().begin(); it != pDD->detUnits().end(); it++){
    
    if( dynamic_cast<StripGeomDetUnit*>((*it))!=0){
    
      //float hallMobility = -1;
    
      uint32_t detid=((*it)->geographicalId()).rawId();

      double thickness=(*it)->specificSurface().bounds().thickness();
      thickness=thickness*10000;
      edm::LogInfo("SiStripLorentzAngle") <<" ### THICKNESS = "<<thickness<<std::endl;
      
      StripSubdetector subid(detid);
            
      /*if(subid.subdetId() == int (StripSubdetector::TIB)){
      hallMobility=0.01738;
      edm::LogInfo("SiStripLorentzAngle") <<" ### TIB HALL MOBILITY = "<<hallMobility<<std::endl;}
      if(subid.subdetId() == int (StripSubdetector::TOB)){
      hallMobility=0.02225;
      edm::LogInfo("SiStripLorentzAngle") <<" ### TOB HALL MOBILITY = "<<hallMobility<<std::endl;}
      if((subid.subdetId() == int (StripSubdetector::TID)) || (subid.subdetId() == int (StripSubdetector::TEC))){
      hallMobility=0.;
      edm::LogInfo("SiStripLorentzAngle") <<" ### TID/TEC HALL MOBILITY = "<<hallMobility<<std::endl;}*/
      
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

SiStripRandomLorentzAngle::~SiStripRandomLorentzAngle() {  
}  

// Analyzer: Functions that gets called by framework every event


SiStripLorentzAngle* SiStripRandomLorentzAngle::getNewObject(){

  SiStripLorentzAngle* LorentzAngle = new SiStripLorentzAngle();
  
  for(std::vector<std::pair<uint32_t, float> >::iterator it = detid_la.begin(); it != detid_la.end(); it++){
    
    float langle=it->second;
    if ( ! LorentzAngle->putLorentzAngle(it->first,langle) )
      edm::LogError("SiStripRandomLorentzAngle")<<"[SiStripRandomLorentzAngle::analyze] detid already exists"<<std::endl;
  }
  
  return LorentzAngle;
}
