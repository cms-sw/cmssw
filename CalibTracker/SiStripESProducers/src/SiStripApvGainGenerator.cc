#include "CalibTracker/SiStripESProducers/interface/SiStripApvGainGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <boost/cstdint.hpp>
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

SiStripApvGainGenerator::SiStripApvGainGenerator(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg):
  SiStripCondObjBuilderBase<SiStripApvGain>::SiStripCondObjBuilderBase(iConfig)
{
  edm::LogInfo("SiStripApvGainGenerator") <<  "[SiStripApvGainGenerator::SiStripApvGainGenerator]";
}

SiStripApvGainGenerator::~SiStripApvGainGenerator() { 
  edm::LogInfo("SiStripApvGainGenerator") <<  "[SiStripApvGainGenerator::~SiStripApvGainGenerator]";
}

void SiStripApvGainGenerator::createObject(){
    
  obj_ = new SiStripApvGain();

  std::string genMode = _pset.getParameter<std::string>("genMode");

  double meanGain_=_pset.getParameter<double>("MeanGain");
  double sigmaGain_=_pset.getParameter<double>("SigmaGain");
  double minimumPosValue_=_pset.getParameter<double>("MinPositiveGain");

  edm::FileInPath fp_ = _pset.getParameter<edm::FileInPath>("file");
  uint32_t  printdebug_ = _pset.getUntrackedParameter<uint32_t>("printDebug", 5);
  uint32_t count=0;

  SiStripDetInfoFileReader reader(fp_.fullPath());

  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > DetInfos  = reader.getAllData();
  float gainValue;
  for(std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator it = DetInfos.begin(); it != DetInfos.end(); it++){    
  
    std::vector<float> theSiStripVector;
    for(unsigned short j=0; j<it->second.nApvs; j++){
  
      if(genMode=="default")
	gainValue=meanGain_;
      else if (genMode=="gaussian") {
	gainValue = CLHEP::RandGauss::shoot(meanGain_, sigmaGain_);
	if(gainValue<=minimumPosValue_) gainValue=minimumPosValue_;
      }
      else {
        LogDebug("SiStripApvGain") << "ERROR: wrong genMode specifier : " << genMode << ", please select one of \"default\" or \"gaussian\"" << std::endl;
        exit(1);
      }
	
      if (count<printdebug_) {
	edm::LogInfo("SiStripApvGainGenerator") << "detid: " << it->first  << " Apv: " << j <<  " gain: " << gainValue  << std::endl; 	    
      }
      theSiStripVector.push_back(gainValue);
    }
    count++;
    SiStripApvGain::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if ( ! obj_->put(it->first,range) )
      edm::LogError("SiStripApvGainGenerator")<<" detid already exists"<<std::endl;
  }
}
