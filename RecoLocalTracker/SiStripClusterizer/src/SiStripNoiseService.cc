#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripNoiseService.h"

SiStripNoiseService::SiStripNoiseService(const edm::ParameterSet& conf):
  conf_(conf),
  UseCalibDataFromDB_(conf.getParameter<bool>("UseCalibDataFromDB")),
  //
  ElectronsPerADC_(conf.getParameter<double>("ElectronPerAdc")),
  ENC_(conf.getParameter<double>("EquivalentNoiseCharge300um")),
  BadStripProbability_(conf.getParameter<double>("BadStripProbability")){
  
  if (UseCalibDataFromDB_==false){	  
    edm::LogInfo("SiStripZeroSuppression")  << "[SiStripNoiseService::SiStripNoiseService] Using a Single Value for Pedestals, Noise, Low and High Threshold and good strip flags";
  } 
  else {
    edm::LogInfo("SiStripZeroSuppression")  << "[SiStripNoiseService::SiStripNoiseService] Using Calibration Data accessed from DB";
  }
  
  old_detID = 0;
  old_noise = -1.;
};


void SiStripNoiseService::setESObjects( const edm::EventSetup& es ) {
  if (UseCalibDataFromDB_==false) {
    es.get<TrackerDigiGeometryRecord>().get( tkgeom );
  } else {
    es.get<SiStripNoisesRcd>().get(noise);

    //Getting Cond Data
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&    
#ifdef DEBUG
    std::vector<uint32_t> detid;
    noise->getDetIds(detid);
    
    for (size_t id=0;id<detid.size();id++)
      {
	SiStripNoises::Range range=noise->getRange(detid[id]);
	
	int strip=0;
	for(int it=0;it<range.second-range.first;it++){
	  edm::LogInfo("SiStripNoiseService") << "[SiStripNoiseService::setESObjects]"  
					      << "detid " << detid[id] << " \t"
					      << " strip " << strip++ << " \t"
					      << noise->getNoise(it,range)     << " \t" 
					      << noise->getDisable(it,range)   << " \t" 
					      << std::endl; 	    
	}
      }
#endif
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  }
};


float SiStripNoiseService::getNoise(const uint32_t& detID,const uint16_t& strip)
{
  if (UseCalibDataFromDB_==false){	  
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    //Case of SingleValue of Noise for all strips of a Detector
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
   
    if (detID==old_detID)
      return old_noise;
    
    float noise = 0;
    //Access to geometry needs to know module Thickness
    const GeomDetUnit* it = tkgeom->idToDetUnit(DetId(detID));

    //FIXME throw exception
    if (dynamic_cast<const StripGeomDetUnit*>(it)==0) {
      edm::LogWarning("SiStripZeroSuppression") << "[SiStripNoiseService::getNoise] the detID " << detID << " doesn't seem to belong to Tracker" << std::endl; 
    }else{
      double moduleThickness = it->surface().bounds().thickness(); //thickness
      noise = ENC_*moduleThickness/(0.03)/ElectronsPerADC_;
    }
    return noise;

  } 
  else
    {
      //&&&&&&&&&&&&&&&&&&&&
      //Access from DB
      //&&&&&&&&&&&&&&&&&&&&
      if (detID!=old_detID){
	old_detID=detID;
	old_range=noise->getRange(detID);
      }
      return noise->getNoise(strip,old_range);
    }
}

bool SiStripNoiseService::getDisable(const uint32_t& detID,const uint16_t& strip)
{
  if (UseCalibDataFromDB_==false){	  
    //Case of SingleValue of Noise for all strips of a Detector
    return (RandFlat::shoot(1.) < BadStripProbability_ ? true:false);
  } 
  else
    {
      //&&&&&&&&&&&&&&&&&&&&
      //Access from DB
      //&&&&&&&&&&&&&&&&&&&&
      if (detID==old_detID)
	return noise->getDisable(strip,old_range);
      else{
	old_detID=detID;
	old_range=noise->getRange(detID);
	return noise->getDisable(strip,old_range);
      }
    }
}
