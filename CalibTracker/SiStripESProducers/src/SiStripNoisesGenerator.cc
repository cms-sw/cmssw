#include "CalibTracker/SiStripESProducers/interface/SiStripNoisesGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <boost/cstdint.hpp>
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

SiStripNoisesGenerator::SiStripNoisesGenerator(const edm::ParameterSet& iConfig,const edm::ActivityRegistry& aReg):
  SiStripCondObjBuilderBase<SiStripNoises>::SiStripCondObjBuilderBase(iConfig)
{
  edm::LogInfo("SiStripNoisesGenerator") <<  "[SiStripNoisesGenerator::SiStripNoisesGenerator]";
}


SiStripNoisesGenerator::~SiStripNoisesGenerator() { 
  edm::LogInfo("SiStripNoisesGenerator") <<  "[SiStripNoisesGenerator::~SiStripNoisesGenerator]";
}


void SiStripNoisesGenerator::createObject(){
    
  obj_ = new SiStripNoises();


  bool stripLengthMode_ = _pset.getParameter<bool>("StripLengthMode");
  
  //parameters for random noise generation. not used if Strip length mode is chosen
  double meanNoise_       = _pset.getParameter<double>("MeanNoise");       
  double sigmaNoise_	  = _pset.getParameter<double>("SigmaNoise");      
  double minimumPosValue_ = _pset.getParameter<double>("MinPositiveNoise");

  //parameters for strip length proportional noise generation. not used if random mode is chosen
  double noiseStripLengthLinearSlope_ = _pset.getParameter<double>("NoiseStripLengthSlope"); 
  double noiseStripLengthLinearQuote_ = _pset.getParameter<double>("NoiseStripLengthQuote"); 
  double electronsPerADC_	      = _pset.getParameter<double>("electronPerAdc");        

  uint32_t  printdebug_ = _pset.getUntrackedParameter<uint32_t>("printDebug", 5);
  uint32_t count=0;

  edm::FileInPath fp_ = _pset.getParameter<edm::FileInPath>("file");

  SiStripDetInfoFileReader reader(fp_.fullPath());
 
  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > DetInfos  = reader.getAllData();
  
  for(std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator it = DetInfos.begin(); it != DetInfos.end(); it++){    
    //Generate Noises for det detid
    SiStripNoises::InputVector theSiStripVector;   
    float noise;
    for(unsigned short j=0; j<128*it->second.nApvs; j++){
      if(stripLengthMode_){
	noise = ( noiseStripLengthLinearSlope_ * (it->second.stripLength) + noiseStripLengthLinearQuote_) / electronsPerADC_;
      }
      else{
	noise = RandGauss::shoot(meanNoise_,sigmaNoise_);
	if(noise<=minimumPosValue_) noise=minimumPosValue_;
      }
      if (count<printdebug_) {
	edm::LogInfo("SiStripNoisesDummyCalculator") << "detid: " << it->first  << " strip: " << j <<  " noise: " << noise     << " \t"   << std::endl; 	    
      }
      obj_->setData(noise,theSiStripVector);
    }
    count++;
    
    if ( ! obj_->put(it->first,theSiStripVector) )
      edm::LogError("SiStripNoisesFakeESSource::produce ")<<" detid already exists"<<std::endl;
  }
  
}
