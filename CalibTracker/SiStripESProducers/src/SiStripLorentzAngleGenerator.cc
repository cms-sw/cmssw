#include "CalibTracker/SiStripESProducers/interface/SiStripLorentzAngleGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <boost/cstdint.hpp>
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "CLHEP/Random/RandGauss.h"

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


  edm::FileInPath fp_                 = _pset.getParameter<edm::FileInPath>("file");
  double   TIB_EstimatedValue         = _pset.getParameter<double>("TIB_EstimatedValue");
  double   TOB_EstimatedValue         = _pset.getParameter<double>("TOB_EstimatedValue");
  double   TIB_PerCent_Err            = _pset.getParameter<double>("TIB_PerCent_Err");
  double   TOB_PerCent_Err            = _pset.getParameter<double>("TOB_PerCent_Err");
  

  SiStripDetInfoFileReader reader(fp_.fullPath());
  
  const std::vector<uint32_t> DetIds = reader.getAllDetIds();
  double StdDev_TIB = (TIB_PerCent_Err/100)*TIB_EstimatedValue;
  double StdDev_TOB = (TOB_PerCent_Err/100)*TOB_EstimatedValue;
  
  
  for(std::vector<uint32_t>::const_iterator detit=DetIds.begin(); detit!=DetIds.end(); detit++){
    
    hallMobility = 0;
    StripSubdetector subid(*detit);
    
    if(subid.subdetId() == int (StripSubdetector::TIB)){
    if(StdDev_TIB>0)hallMobility=RandGauss::shoot(TIB_EstimatedValue,StdDev_TIB);
    else hallMobility = TIB_EstimatedValue;}
    
    if(subid.subdetId() == int (StripSubdetector::TOB)){
    if(StdDev_TOB>0)hallMobility=RandGauss::shoot(TOB_EstimatedValue,StdDev_TOB);
    else hallMobility = TOB_EstimatedValue;}
    
    if(subid.subdetId() == int (StripSubdetector::TID)){
    if(StdDev_TIB>0)hallMobility=RandGauss::shoot(TIB_EstimatedValue,StdDev_TIB);
    else hallMobility = TIB_EstimatedValue;}
    
    if(subid.subdetId() == int (StripSubdetector::TEC)){
    TECDetId TECid = TECDetId(*detit); 
    if(TECid.ringNumber()<5){
    if(StdDev_TIB>0)hallMobility=RandGauss::shoot(TIB_EstimatedValue,StdDev_TIB);
    else hallMobility = TIB_EstimatedValue;
    }else{
    if(StdDev_TOB>0)hallMobility=RandGauss::shoot(TOB_EstimatedValue,StdDev_TOB);
    else hallMobility = TOB_EstimatedValue;
    }
    }
      
    if ( ! obj_->putLorentzAngle(*detit, hallMobility) )
      edm::LogError("SiStripLorentzAngleGenerator")<<" detid already exists"<<std::endl;
  }
}
