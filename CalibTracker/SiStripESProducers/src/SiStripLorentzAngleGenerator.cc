#include "CalibTracker/SiStripESProducers/interface/SiStripLorentzAngleGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <boost/cstdint.hpp>
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandFlat.h"

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


  edm::FileInPath fp_                     = _pset.getParameter<edm::FileInPath>("file");
  vector<double> TIB_EstimatedValueMinMax = _pset.getParameter<vector<double> >("TIB_EstimatedValueMinMax");
  vector<double> TOB_EstimatedValueMinMax = _pset.getParameter<vector<double> >("TOB_EstimatedValueMinMax");
  double TIB_PerCent_Err                  = _pset.getParameter<double>("TIB_PerCent_Err");
  double TOB_PerCent_Err                  = _pset.getParameter<double>("TOB_PerCent_Err");

  if( TIB_EstimatedValueMinMax.size() > 2 || TIB_EstimatedValueMinMax.size() == 0 ||
      TOB_EstimatedValueMinMax.size() > 2 || TOB_EstimatedValueMinMax.size() == 0 ) {
    cout << "ERROR: provide exactly one or two values for TIB_EstimatedValueMinMax and TOB_EstimatedValueMinMax" << endl;
    cout << "TIB_EstimatedValueMinMax.size() = " << TIB_EstimatedValueMinMax.size() << endl;
    cout << "TOB_EstimatedValueMinMax.size() = " << TOB_EstimatedValueMinMax.size() << endl;
  }

  SiStripDetInfoFileReader reader(fp_.fullPath());

  double TIB_EstimatedValues[2] = {0., 0.};
  double TOB_EstimatedValues[2] = {0., 0.};
  bool uniformTIB = setEstimatedValues(TIB_EstimatedValueMinMax, TIB_EstimatedValues);
  bool uniformTOB = setEstimatedValues(TOB_EstimatedValueMinMax, TOB_EstimatedValues);

  const std::vector<uint32_t> DetIds = reader.getAllDetIds();
  double StdDev_TIB = (TIB_PerCent_Err/100)*TIB_EstimatedValues[0];
  double StdDev_TOB = (TOB_PerCent_Err/100)*TOB_EstimatedValues[0];

  for(std::vector<uint32_t>::const_iterator detit=DetIds.begin(); detit!=DetIds.end(); detit++) {

    hallMobility_ = 0.;
    StripSubdetector subid(*detit);

    if( subid.subdetId() == int(StripSubdetector::TIB) ) {
      setHallMobility(TIB_EstimatedValues, StdDev_TIB, uniformTIB);
    }
    else if( subid.subdetId() == int(StripSubdetector::TOB) ) {
      setHallMobility(TOB_EstimatedValues, StdDev_TOB, uniformTOB);
    }
    else if( subid.subdetId() == int(StripSubdetector::TID) ) {
      setHallMobility(TIB_EstimatedValues, StdDev_TIB, uniformTIB);
    }
    else if( subid.subdetId() == int(StripSubdetector::TEC) ) {
      TECDetId TECid = TECDetId(*detit); 
      if( TECid.ringNumber()<5 ) {
        setHallMobility(TIB_EstimatedValues, StdDev_TIB, uniformTIB);
      }
      else {
        setHallMobility(TOB_EstimatedValues, StdDev_TOB, uniformTOB);
      }
    }

    if ( !obj_->putLorentzAngle(*detit, hallMobility_) ) {
      edm::LogError("SiStripLorentzAngleGenerator")<<" detid already exists"<<std::endl;
    }
  }
}

bool SiStripLorentzAngleGenerator::setEstimatedValues(const vector<double> & estimatedValueMinMax, double * estimatedValues) const
{
  bool uniform = false;
  if( estimatedValueMinMax.size() == 1 || estimatedValueMinMax[0] == estimatedValueMinMax[1] ) {
    estimatedValues[0] = estimatedValueMinMax[0];
  }
  else {
    estimatedValues[0] = estimatedValueMinMax[0];
    estimatedValues[1] = estimatedValueMinMax[1];
    uniform = true;
  }
  return uniform;
}

void SiStripLorentzAngleGenerator::setHallMobility(const double * estimatedValue, const double & stdDev, const bool uniform)
{
  if( uniform ) hallMobility_ = RandFlat::shoot(estimatedValue[0], estimatedValue[1]);
  else if( stdDev > 0 ) hallMobility_ = RandGauss::shoot(estimatedValue[0], stdDev);
  else hallMobility_ = estimatedValue[0];
}
