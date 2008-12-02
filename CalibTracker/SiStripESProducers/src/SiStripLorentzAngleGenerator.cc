#include "CalibTracker/SiStripESProducers/interface/SiStripLorentzAngleGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <boost/cstdint.hpp>
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include<cmath>

SiStripLorentzAngleGenerator::SiStripLorentzAngleGenerator(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg):
  SiStripCondObjBuilderBase<SiStripLorentzAngle>::SiStripCondObjBuilderBase(iConfig)
{
  edm::LogInfo("SiStripLorentzAngleGenerator") <<  "[SiStripLorentzAngleGenerator::SiStripLorentzAngleGenerator]";
}


SiStripLorentzAngleGenerator::~SiStripLorentzAngleGenerator() { 
  edm::LogInfo("SiStripLorentzAngleGenerator") <<  "[SiStripLorentzAngleGenerator::~SiStripLorentzAngleGenerator]";
}


void SiStripLorentzAngleGenerator::createObject(){
    
  obj_ = new SiStripLorentzAngle();


  edm::FileInPath fp_              = _pset.getParameter<edm::FileInPath>("file");
  double   appliedVoltage_         = _pset.getParameter<double>("AppliedVoltage");
  double   chargeMobility_         = _pset.getParameter<double>("ChargeMobility");
  double   temperature_            = _pset.getParameter<double>("Temperature");
  double   temperatureerror_       = _pset.getParameter<double>("TemperatureError");
  double   rhall_                  = _pset.getParameter<double>("HoleRHAllParameter");
  double   holeBeta_               = _pset.getParameter<double>("HoleBeta");
  double   holeSaturationVelocity_ = _pset.getParameter<double>("HoleSaturationVelocity");

  SiStripDetInfoFileReader reader(fp_.fullPath());

  const std::vector<uint32_t> DetIds = reader.getAllDetIds();
  float mulow = chargeMobility_*std::pow((temperature_/300.),-2.5);
  float vsat = holeSaturationVelocity_*std::pow((temperature_/300.),0.52);
  float beta = holeBeta_*std::pow((temperature_/300.),0.17);
  
  for(std::vector<uint32_t>::const_iterator detit=DetIds.begin(); detit!=DetIds.end(); detit++){

    const float & thickness=reader.getThickness(*detit);
    float e = appliedVoltage_/thickness;
    float mu = ( mulow/(pow(double((1+pow((mulow*e/vsat),beta))),1./beta)));
    float hallMobility = 1.E-4*mu*rhall_;
    
    
    if ( ! obj_->putLorentzAngle(*detit, hallMobility) )
      edm::LogError("SiStripLorentzAngleGenerator")<<" detid already exists"<<std::endl;
  }
}
