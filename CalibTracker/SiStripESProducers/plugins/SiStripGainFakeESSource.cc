#include "CalibTracker/SiStripESProducers/plugins/SiStripGainFakeESSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"


#include <iostream>


SiStripGainFakeESSource::SiStripGainFakeESSource( const edm::ParameterSet& pset ) {

  edm::LogInfo("SiStripGainFakeESSource::SiStripGainFakeESSource");

  setWhatProduced( this );
  findingRecord<SiStripApvGainRcd>();


  fp_ = pset.getParameter<edm::FileInPath>("file");

  //  edm::LogInfo("SiStripGainFakeESSource::SiStripGainFakeESSource - exiting");

}


std::auto_ptr<SiStripApvGain> SiStripGainFakeESSource::produce( const SiStripApvGainRcd& ) { 
  
  SiStripApvGain * obj = new SiStripApvGain();

  SiStripDetInfoFileReader reader(fp_.fullPath());
  //  SiStripDetInfoFileReader reader("");

  const std::vector<uint32_t> DetIds = reader.getAllDetIds();

  for(std::vector<uint32_t>::const_iterator detit=DetIds.begin(); detit!=DetIds.end(); detit++){

    std::vector<float> theSiStripVector;

    const std::pair<unsigned short, double> ApvsAndStripLengths = reader.getNumberOfApvsAndStripLength(*detit);

    for(unsigned short j=0; j<ApvsAndStripLengths.first; j++){
      theSiStripVector.push_back(1);
    }
    
    
    SiStripApvGain::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if ( ! obj->put(*detit, range) )
      edm::LogError("SiStripGainFakeESSource::produce ")<<" detid already exists"<<std::endl;

  }
  

  return std::auto_ptr<SiStripApvGain>(obj);


}


void SiStripGainFakeESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
						const edm::IOVSyncValue& iosv, 
						edm::ValidityInterval& oValidity ) {

  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;
  
}

