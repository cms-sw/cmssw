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
#include<algorithm>
#include<numeric>

// Defined inside a nameless namespace so that it is local to this file
namespace {
  double computeSigma(const double & value, const double & perCentError) {
    return (perCentError/100)*value;
  }
}

SiStripLorentzAngleGenerator::SiStripLorentzAngleGenerator(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg):
  SiStripCondObjBuilderBase<SiStripLorentzAngle>::SiStripCondObjBuilderBase(iConfig)
{
  edm::LogInfo("SiStripLorentzAngleGenerator") <<  "[SiStripLorentzAngleGenerator::SiStripLorentzAngleGenerator]";
}


SiStripLorentzAngleGenerator::~SiStripLorentzAngleGenerator() { 
  edm::LogInfo("SiStripLorentzAngleGenerator") <<  "[SiStripLorentzAngleGenerator::~SiStripLorentzAngleGenerator]";
}

void SiStripLorentzAngleGenerator::setHallMobility(const double & meanMin, const double & meanMax, const double & sigma, const bool uniform) {
  if( uniform ) hallMobility_ = CLHEP::RandFlat::shoot(meanMin, meanMax);
  else if( sigma>0 ) hallMobility_ = CLHEP::RandGauss::shoot(meanMin, sigma);
  else hallMobility_ = meanMin;
}

void SiStripLorentzAngleGenerator::setUniform(const vector<double> & TIB_EstimatedValuesMin, const vector<double> & TIB_EstimatedValuesMax, vector<bool> & uniformTIB) {
  if( TIB_EstimatedValuesMax.size() != 0 ) {
    vector<double>::const_iterator min = TIB_EstimatedValuesMin.begin();
    vector<double>::const_iterator max = TIB_EstimatedValuesMax.begin();
    vector<bool>::iterator uniform = uniformTIB.begin();
    for( ; min != TIB_EstimatedValuesMin.end(); ++min, ++max ) {
      if( *min != *max ) *uniform = true;
    }
  }
}

void SiStripLorentzAngleGenerator::createObject() {

  obj_ = new SiStripLorentzAngle();


  edm::FileInPath fp_                 = _pset.getParameter<edm::FileInPath>("file");

  vector<double> TIB_EstimatedValuesMin(_pset.getParameter<vector<double> >("TIB_EstimatedValuesMin"));
  vector<double> TIB_EstimatedValuesMax(_pset.getParameter<vector<double> >("TIB_EstimatedValuesMax"));
  vector<double> TOB_EstimatedValuesMin(_pset.getParameter<vector<double> >("TOB_EstimatedValuesMin"));
  vector<double> TOB_EstimatedValuesMax(_pset.getParameter<vector<double> >("TOB_EstimatedValuesMax"));
  vector<double> TIB_PerCent_Errs(_pset.getParameter<vector<double> >("TIB_PerCent_Errs"));
  vector<double> TOB_PerCent_Errs(_pset.getParameter<vector<double> >("TOB_PerCent_Errs"));

  // If max values are passed they must be equal in number to the min values.
  if( TIB_EstimatedValuesMax.size() != 0 && (TIB_EstimatedValuesMin.size() != TIB_EstimatedValuesMax.size()) ||
      TOB_EstimatedValuesMax.size() != 0 && (TOB_EstimatedValuesMin.size() != TOB_EstimatedValuesMax.size()) ) {
    cout << "ERROR: size of min and max values is different" << endl;
    cout << "TIB_EstimatedValuesMin.size() = " << TIB_EstimatedValuesMin.size() << ", TIB_EstimatedValuesMax.size() " << TIB_EstimatedValuesMax.size() << endl;
    cout << "TOB_EstimatedValuesMin.size() = " << TOB_EstimatedValuesMin.size() << ", TOB_EstimatedValuesMax.size() " << TOB_EstimatedValuesMax.size() << endl;
  }
  vector<bool> uniformTIB(TIB_EstimatedValuesMin.size(), false);
  vector<bool> uniformTOB(TOB_EstimatedValuesMin.size(), false);

  setUniform(TIB_EstimatedValuesMin, TIB_EstimatedValuesMax, uniformTIB);
  setUniform(TOB_EstimatedValuesMin, TOB_EstimatedValuesMax, uniformTOB);
  
  SiStripDetInfoFileReader reader(fp_.fullPath());

  // Compute standard deviations
  vector<double> StdDevs_TIB(TIB_EstimatedValuesMin.size(), 0);
  vector<double> StdDevs_TOB(TOB_EstimatedValuesMin.size(), 0);
  transform(TIB_EstimatedValuesMin.begin(), TIB_EstimatedValuesMin.end(), TIB_PerCent_Errs.begin(), StdDevs_TIB.begin(), computeSigma);
  transform(TOB_EstimatedValuesMin.begin(), TOB_EstimatedValuesMin.end(), TOB_PerCent_Errs.begin(), StdDevs_TOB.begin(), computeSigma);

  // Compute mean values to be used with TID and TEC
  double TIBmeanValueMin = accumulate( TIB_EstimatedValuesMin.begin(), TIB_EstimatedValuesMin.end(), 0.)/double(TIB_EstimatedValuesMin.size());
  double TIBmeanValueMax = accumulate( TIB_EstimatedValuesMax.begin(), TIB_EstimatedValuesMax.end(), 0.)/double(TIB_EstimatedValuesMax.size());
  double TOBmeanValueMin = accumulate( TOB_EstimatedValuesMin.begin(), TOB_EstimatedValuesMin.end(), 0.)/double(TOB_EstimatedValuesMin.size());
  double TOBmeanValueMax = accumulate( TOB_EstimatedValuesMax.begin(), TOB_EstimatedValuesMax.end(), 0.)/double(TOB_EstimatedValuesMax.size());
  double TIBmeanPerCentError = accumulate( TIB_PerCent_Errs.begin(), TIB_PerCent_Errs.end(), 0.)/double(TIB_PerCent_Errs.size());
  double TOBmeanPerCentError = accumulate( TOB_PerCent_Errs.begin(), TOB_PerCent_Errs.end(), 0.)/double(TOB_PerCent_Errs.size());
  double TIBmeanStdDev = (TIBmeanPerCentError/100)*TIBmeanValueMin;
  double TOBmeanStdDev = (TOBmeanPerCentError/100)*TOBmeanValueMin;

  const std::vector<uint32_t> DetIds = reader.getAllDetIds();
  for(std::vector<uint32_t>::const_iterator detit=DetIds.begin(); detit!=DetIds.end(); detit++){
    
    hallMobility_ = 0;

    StripSubdetector subid(*detit);

    int layerId = 0;

    if(subid.subdetId() == int (StripSubdetector::TIB)) {
      TIBDetId theTIBDetId(*detit);
      layerId = theTIBDetId.layer() - 1;
      setHallMobility( TIB_EstimatedValuesMin[layerId], TIB_EstimatedValuesMax[layerId], StdDevs_TIB[layerId], uniformTIB[layerId] );
    }
    else if(subid.subdetId() == int (StripSubdetector::TOB)) {
      TOBDetId theTOBDetId(*detit);
      layerId = theTOBDetId.layer() - 1;
      setHallMobility( TOB_EstimatedValuesMin[layerId], TOB_EstimatedValuesMax[layerId], StdDevs_TOB[layerId], uniformTOB[layerId] );
    }
    else if(subid.subdetId() == int (StripSubdetector::TID)) {
      // ATTENTION: as of now the uniform generation for TID is decided by the setting for layer 0 of TIB
      setHallMobility( TIBmeanValueMin, TIBmeanValueMax, TIBmeanStdDev, uniformTIB[0] );
    }
    if(subid.subdetId() == int (StripSubdetector::TEC)){
      TECDetId TECid = TECDetId(*detit); 
      if(TECid.ringNumber()<5){
        // ATTENTION: as of now the uniform generation for TEC is decided by the setting for layer 0 of TIB
        setHallMobility( TIBmeanValueMin, TIBmeanValueMax, TIBmeanStdDev, uniformTIB[0] );
      }else{
        // ATTENTION: as of now the uniform generation for TEC is decided by the setting for layer 0 of TOB
        setHallMobility( TOBmeanValueMin, TOBmeanValueMax, TOBmeanStdDev, uniformTOB[0] );
      }
    }
      
    if ( ! obj_->putLorentzAngle(*detit, hallMobility_) ) {
      edm::LogError("SiStripLorentzAngleGenerator")<<" detid already exists"<<std::endl;
    }
  }
}
