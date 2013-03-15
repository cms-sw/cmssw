#include "CalibTracker/SiStripESProducers/interface/SiStripBackPlaneCorrectionGenerator.h"
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

SiStripBackPlaneCorrectionGenerator::SiStripBackPlaneCorrectionGenerator(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg):
  SiStripCondObjBuilderBase<SiStripBackPlaneCorrection>::SiStripCondObjBuilderBase(iConfig)
{
  edm::LogInfo("SiStripBackPlaneCorrectionGenerator") <<  "[SiStripBackPlaneCorrectionGenerator::SiStripBackPlaneCorrectionGenerator]";
}


SiStripBackPlaneCorrectionGenerator::~SiStripBackPlaneCorrectionGenerator() { 
  edm::LogInfo("SiStripBackPlaneCorrectionGenerator") <<  "[SiStripBackPlaneCorrectionGenerator::~SiStripBackPlaneCorrectionGenerator]";
}

void SiStripBackPlaneCorrectionGenerator::createObject()
{
  obj_ = new SiStripBackPlaneCorrection();

  edm::FileInPath fp_                 = _pset.getParameter<edm::FileInPath>("file");
  std::vector<double> valuePerModuleGeometry(_pset.getParameter<std::vector<double> >("BackPlaneCorrection_PerModuleGeometry"));
  
  SiStripDetInfoFileReader reader(fp_.fullPath());
  const std::vector<uint32_t> DetIds = reader.getAllDetIds();
  for(std::vector<uint32_t>::const_iterator detit=DetIds.begin(); detit!=DetIds.end(); detit++){
    SiStripDetId SSdetId(*detit);
    unsigned int moduleGeometry = (SSdetId.moduleGeometry()-1);
    if(moduleGeometry>valuePerModuleGeometry.size())edm::LogError("SiStripBackPlaneCorrectionGenerator")<<" BackPlaneCorrection_PerModuleGeometry only contains "<< valuePerModuleGeometry.size() << "elements and module is out of range"<<std::endl;
    float value =     valuePerModuleGeometry[moduleGeometry];
  
    if (!obj_->putBackPlaneCorrection(*detit, value) ) {
      edm::LogError("SiStripBackPlaneCorrectionGenerator")<<" detid already exists"<<std::endl;
    }
  }
}
