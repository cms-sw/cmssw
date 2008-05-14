#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripPedestalsFakeESSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"


#include <iostream>


SiStripPedestalsFakeESSource::SiStripPedestalsFakeESSource( const edm::ParameterSet& pset ) : SiStripPedestalsESSource(pset) {

  edm::LogInfo("SiStripPedestalsFakeESSource::SiStripPedestalsFakeESSource");

  PedestalValue_ = pset.getParameter<uint32_t>("PedestalsValue");  
  LowThValue_ = pset.getParameter<double>("LowThValue");  
  HighThValue_ = pset.getParameter<double>("HighThValue");

  fp_ = pset.getParameter<edm::FileInPath>("file");

  printdebug_ = pset.getUntrackedParameter<bool>("printDebug", false);
 
}


SiStripPedestals * SiStripPedestalsFakeESSource::makePedestals() { 
  
  SiStripPedestals * obj = new SiStripPedestals();

  SiStripDetInfoFileReader reader(fp_.fullPath());

  const std::vector<uint32_t> DetIds = reader.getAllDetIds();
  
  bool firstdet=true;

  for(std::vector<uint32_t>::const_iterator detit=DetIds.begin(); detit!=DetIds.end(); ++detit){
    
    SiStripPedestals::InputVector theSiStripVector;

    const std::pair<unsigned short, double> ApvsAndStripLengths = reader.getNumberOfApvsAndStripLength(*detit);

    for(unsigned short j=0; j<ApvsAndStripLengths.first; ++j){

      for(int strip=0; strip<128; ++strip){
	
	if (printdebug_ && firstdet) {

	  edm::LogInfo("SiStripPedestalsFakeESSource::makePedestals(): ") << "detid: " << *detit  << " strip: " << j*128+strip <<  " ped: " << PedestalValue_  <<  " LowThValue: " << LowThValue_  << " HighThValue: " << HighThValue_ << std::endl; 	    

	}

	obj->setData(PedestalValue_,theSiStripVector);
      }
      
    }	    
    
    firstdet=false;

    if ( ! obj->put(*detit, theSiStripVector) )
      edm::LogError("SiStripPedestalsFakeESSource::produce ")<<" detid already exists"<<std::endl;

  }
  

  return obj;
}

