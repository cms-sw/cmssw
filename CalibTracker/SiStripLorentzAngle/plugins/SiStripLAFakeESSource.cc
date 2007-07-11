#include "CalibTracker/SiStripLorentzAngle/plugins/SiStripLAFakeESSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"


#include <iostream>
#include<cmath>


SiStripLAFakeESSource::SiStripLAFakeESSource( const edm::ParameterSet& conf_ ) {

  edm::LogInfo("SiStripLAFakeESSource::SiStripLAFakeESSource");

  setWhatProduced( this );
  findingRecord<SiStripLorentzAngleRcd>();


  fp_ = conf_.getParameter<edm::FileInPath>("file");
  appliedVoltage_   = conf_.getParameter<double>("AppliedVoltage");
  chargeMobility_   = conf_.getParameter<double>("ChargeMobility");
  temperature_      = conf_.getParameter<double>("Temperature");
  temperatureerror_      = conf_.getParameter<double>("TemperatureError");
  rhall_            = conf_.getParameter<double>("HoleRHAllParameter");
  holeBeta_         = conf_.getParameter<double>("HoleBeta");
  holeSaturationVelocity_ = conf_.getParameter<double>("HoleSaturationVelocity");


}


std::auto_ptr<SiStripLorentzAngle> SiStripLAFakeESSource::produce( const SiStripLorentzAngleRcd& ) { 
  
  SiStripLorentzAngle * obj = new SiStripLorentzAngle();

  SiStripDetInfoFileReader reader(fp_.fullPath());
  //  SiStripDetInfoFileReader reader("");

  const std::vector<uint32_t> DetIds = reader.getAllDetIds();
  float mulow = chargeMobility_*std::pow((temperature_/300.),-2.5);
  float vsat = holeSaturationVelocity_*std::pow((temperature_/300.),0.52);
  float beta = holeBeta_*std::pow((temperature_/300.),0.17);
  
  for(std::vector<uint32_t>::const_iterator detit=DetIds.begin(); detit!=DetIds.end(); detit++){

    const float & thickness=reader.getThickness(*detit);
    float e = appliedVoltage_/thickness;
    float mu = ( mulow/(pow(double((1+pow((mulow*e/vsat),beta))),1./beta)));
    float hallMobility = 1.E-4*mu*rhall_;
    
    
    if ( ! obj->putLorentzAngle(*detit, hallMobility) )
      edm::LogError("SiStripLAFakeESSource::produce ")<<" detid already exists"<<std::endl;
    
  }
  

  return std::auto_ptr<SiStripLorentzAngle>(obj);


}


void SiStripLAFakeESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
						const edm::IOVSyncValue& iosv, 
						edm::ValidityInterval& oValidity ) {

  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;
  
}

