#include "CalibTracker/SiStripESProducers/interface/SiStripLorentzAngleGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <boost/cstdint.hpp>
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
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
    SiStripDepCondObjBuilderBase<SiStripLorentzAngle,TrackerTopology>::SiStripDepCondObjBuilderBase(iConfig)
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

void SiStripLorentzAngleGenerator::setUniform(const std::vector<double> & estimatedValuesMin, const std::vector<double> & estimatedValuesMax, std::vector<bool> & uniform) {
  if( estimatedValuesMax.size() != 0 ) {
    std::vector<double>::const_iterator min = estimatedValuesMin.begin();
    std:: vector<double>::const_iterator max = estimatedValuesMax.begin();
    std::vector<bool>::iterator uniformIt = uniform.begin();
    for( ; min != estimatedValuesMin.end(); ++min, ++max, ++uniformIt ) {
      if( *min != *max ) *uniformIt = true;
    }
  }
}

SiStripLorentzAngle* SiStripLorentzAngleGenerator::createObject(const TrackerTopology* tTopo)
{
  SiStripLorentzAngle* obj = new SiStripLorentzAngle();

  edm::FileInPath fp_                 = _pset.getParameter<edm::FileInPath>("file");

  std::vector<double> TIB_EstimatedValuesMin(_pset.getParameter<std::vector<double> >("TIB_EstimatedValuesMin"));
  std::vector<double> TIB_EstimatedValuesMax(_pset.getParameter<std::vector<double> >("TIB_EstimatedValuesMax"));
  std::vector<double> TOB_EstimatedValuesMin(_pset.getParameter<std::vector<double> >("TOB_EstimatedValuesMin"));
  std::vector<double> TOB_EstimatedValuesMax(_pset.getParameter<std::vector<double> >("TOB_EstimatedValuesMax"));
  std::vector<double> TIB_PerCent_Errs(_pset.getParameter<std::vector<double> >("TIB_PerCent_Errs"));
  std::vector<double> TOB_PerCent_Errs(_pset.getParameter<std::vector<double> >("TOB_PerCent_Errs"));

  // If max values are passed they must be equal in number to the min values.
  if( (TIB_EstimatedValuesMax.size() != 0 && (TIB_EstimatedValuesMin.size() != TIB_EstimatedValuesMax.size())) ||
      (TOB_EstimatedValuesMax.size() != 0 && (TOB_EstimatedValuesMin.size() != TOB_EstimatedValuesMax.size())) ) {
    std::cout << "ERROR: size of min and max values is different" << std::endl;
    std::cout << "TIB_EstimatedValuesMin.size() = " << TIB_EstimatedValuesMin.size() << ", TIB_EstimatedValuesMax.size() " << TIB_EstimatedValuesMax.size() << std::endl;
    std::cout << "TOB_EstimatedValuesMin.size() = " << TOB_EstimatedValuesMin.size() << ", TOB_EstimatedValuesMax.size() " << TOB_EstimatedValuesMax.size() << std::endl;
  }
  std::vector<bool> uniformTIB(TIB_EstimatedValuesMin.size(), false);
  std::vector<bool> uniformTOB(TOB_EstimatedValuesMin.size(), false);

  setUniform(TIB_EstimatedValuesMin, TIB_EstimatedValuesMax, uniformTIB);
  setUniform(TOB_EstimatedValuesMin, TOB_EstimatedValuesMax, uniformTOB);
  
  SiStripDetInfoFileReader reader(fp_.fullPath());

  // Compute standard deviations
  std::vector<double> StdDevs_TIB(TIB_EstimatedValuesMin.size(), 0);
  std::vector<double> StdDevs_TOB(TOB_EstimatedValuesMin.size(), 0);
  transform(TIB_EstimatedValuesMin.begin(), TIB_EstimatedValuesMin.end(), TIB_PerCent_Errs.begin(), StdDevs_TIB.begin(), computeSigma);
  transform(TOB_EstimatedValuesMin.begin(), TOB_EstimatedValuesMin.end(), TOB_PerCent_Errs.begin(), StdDevs_TOB.begin(), computeSigma);

  // Compute mean values to be used with TID and TEC
  double TIBmeanValueMin = std::accumulate( TIB_EstimatedValuesMin.begin(), TIB_EstimatedValuesMin.end(), 0.)/double(TIB_EstimatedValuesMin.size());
  double TIBmeanValueMax = std::accumulate( TIB_EstimatedValuesMax.begin(), TIB_EstimatedValuesMax.end(), 0.)/double(TIB_EstimatedValuesMax.size());
  double TOBmeanValueMin = std::accumulate( TOB_EstimatedValuesMin.begin(), TOB_EstimatedValuesMin.end(), 0.)/double(TOB_EstimatedValuesMin.size());
  double TOBmeanValueMax = std::accumulate( TOB_EstimatedValuesMax.begin(), TOB_EstimatedValuesMax.end(), 0.)/double(TOB_EstimatedValuesMax.size());
  double TIBmeanPerCentError = std::accumulate( TIB_PerCent_Errs.begin(), TIB_PerCent_Errs.end(), 0.)/double(TIB_PerCent_Errs.size());
  double TOBmeanPerCentError = std::accumulate( TOB_PerCent_Errs.begin(), TOB_PerCent_Errs.end(), 0.)/double(TOB_PerCent_Errs.size());
  double TIBmeanStdDev = (TIBmeanPerCentError/100)*TIBmeanValueMin;
  double TOBmeanStdDev = (TOBmeanPerCentError/100)*TOBmeanValueMin;

  const std::vector<uint32_t> DetIds = reader.getAllDetIds();
  for(std::vector<uint32_t>::const_iterator detit=DetIds.begin(); detit!=DetIds.end(); detit++){
    const DetId detectorId=DetId(*detit);
    const int subDet = detectorId.subdetId();
    
    hallMobility_ = 0;

    int layerId = 0;

    if(subDet == int (StripSubdetector::TIB)) {
      layerId = tTopo->tibLayer(detectorId) -1;
      setHallMobility( TIB_EstimatedValuesMin[layerId], TIB_EstimatedValuesMax[layerId], StdDevs_TIB[layerId], uniformTIB[layerId] );
    }
    else if(subDet == int (StripSubdetector::TOB)) {
      layerId = tTopo->tobLayer(detectorId) -1;
      setHallMobility( TOB_EstimatedValuesMin[layerId], TOB_EstimatedValuesMax[layerId], StdDevs_TOB[layerId], uniformTOB[layerId] );
    }
    else if(subDet == int (StripSubdetector::TID)) {
      // ATTENTION: as of now the uniform generation for TID is decided by the setting for layer 0 of TIB
      setHallMobility( TIBmeanValueMin, TIBmeanValueMax, TIBmeanStdDev, uniformTIB[0] );
    }
    if(subDet == int (StripSubdetector::TEC)){
      if(tTopo->tecRing(detectorId)<5){
        // ATTENTION: as of now the uniform generation for TEC is decided by the setting for layer 0 of TIB
        setHallMobility( TIBmeanValueMin, TIBmeanValueMax, TIBmeanStdDev, uniformTIB[0] );
      }else{
        // ATTENTION: as of now the uniform generation for TEC is decided by the setting for layer 0 of TOB
        setHallMobility( TOBmeanValueMin, TOBmeanValueMax, TOBmeanStdDev, uniformTOB[0] );
      }
    }
      
    if ( ! obj->putLorentzAngle(*detit, hallMobility_) ) {
      edm::LogError("SiStripLorentzAngleGenerator")<<" detid already exists"<<std::endl;
    }
  }
  return obj;
}

