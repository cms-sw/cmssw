#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripNoiseFakeESSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"


#include <iostream>


SiStripNoiseFakeESSource::SiStripNoiseFakeESSource( const edm::ParameterSet& pset ) : SiStripNoiseESSource(pset) {

  edm::LogInfo("SiStripNoiseFakeESSource::SiStripNoiseFakeESSource");

  noiseStripLengthLinearSlope_ = pset.getParameter<double>("NoiseStripLengthSlope");  
  noiseStripLengthLinearQuote_ = pset.getParameter<double>("NoiseStripLengthQuote");  
  electronsPerADC_ = pset.getParameter<double>("electronPerAdc");

  fp_ = pset.getParameter<edm::FileInPath>("file");

  printdebug_ = pset.getUntrackedParameter<bool>("printDebug", false);
 

  //  edm::LogInfo("SiStripNoiseFakeESSource::SiStripNoiseFakeESSource - exiting");

}


SiStripNoises * SiStripNoiseFakeESSource::makeNoise() { 
  
  SiStripNoises * obj = new SiStripNoises();

  SiStripDetInfoFileReader reader(fp_.fullPath());
  //  SiStripDetInfoFileReader reader("");

  const std::vector<uint32_t> DetIds = reader.getAllDetIds();

  bool firstdet=true;

  for(std::vector<uint32_t>::const_iterator detit=DetIds.begin(); detit!=DetIds.end(); detit++){

    SiStripNoises::InputVector theSiStripVector;

    const std::pair<unsigned short, double> ApvsAndStripLengths = reader.getNumberOfApvsAndStripLength(*detit);

    for(unsigned short j=0; j<ApvsAndStripLengths.first; j++){

      for(int strip=0; strip<128; ++strip){

	float noise = ( noiseStripLengthLinearSlope_ * (ApvsAndStripLengths.second) + noiseStripLengthLinearQuote_) / electronsPerADC_;
	
	
	if (printdebug_ && firstdet) {

	  edm::LogInfo("SiStripNoiseFakeESSource::makeNoise(): ") << "detid: " << *detit  << " strip: " << j*128+strip <<  " noise: " << noise     << " \t"   << std::endl; 	    


	}


	obj->setData(noise,theSiStripVector);

      }

    }	    
      
    firstdet=false;

    if ( ! obj->put(*detit, theSiStripVector) )
      edm::LogError("SiStripNoiseFakeESSource::produce ")<<" detid already exists"<<std::endl;

  }
  

  return obj;


}

