#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripThresholdFakeESSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"


#include <iostream>


SiStripThresholdFakeESSource::SiStripThresholdFakeESSource( const edm::ParameterSet& pset ):
  fp_(pset.getParameter<edm::FileInPath>("file")),
  lTh_(pset.getParameter<double>("LowTh")),
  hTh_(pset.getParameter<double>("HighTh")){

  edm::LogInfo("SiStripThresholdFakeESSource::SiStripThresholdFakeESSource");

  setWhatProduced( this );
  findingRecord<SiStripThresholdRcd>();
}


std::auto_ptr<SiStripThreshold> SiStripThresholdFakeESSource::produce( const SiStripThresholdRcd& ) { 
  
  SiStripThreshold * obj = new SiStripThreshold();

  SiStripDetInfoFileReader reader(fp_.fullPath());
 
  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > DetInfos  = reader.getAllData();
  
  for(std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator it = DetInfos.begin(); it != DetInfos.end(); it++){    
    //Generate Thresholds for det detid
    SiStripThreshold::Container theSiStripVector;   
    uint16_t strip=0;
    float lTh = lTh_;
    float hTh = hTh_;

    obj->setData(strip,lTh,hTh,theSiStripVector);
    LogDebug("SiStripThresholdFakeESSource::produce") <<"detid: "  << it->first << " \t"
							  << "firstStrip: " << strip << " \t" << theSiStripVector.back().getFirstStrip() << " \t"
							  << "lTh: " << lTh       << " \t" << theSiStripVector.back().getLth() << " \t"
							  << "hTh: " << hTh       << " \t" << theSiStripVector.back().getHth() << " \t"
							  << "FirstStrip_and_Hth: " << theSiStripVector.back().FirstStrip_and_Hth << " \t"
							  << std::endl; 	    
    
    if ( ! obj->put(it->first,theSiStripVector) )
      edm::LogError("SiStripThresholdFakeESSource::produce ")<<" detid already exists"<<std::endl;
  }
  
  return std::auto_ptr<SiStripThreshold>(obj);

}


void SiStripThresholdFakeESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
						const edm::IOVSyncValue& iosv, 
						edm::ValidityInterval& oValidity ) {

  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;
  
}

