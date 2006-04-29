#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripNoiseService.h"

SiStripNoiseService::SiStripNoiseService(const edm::ParameterSet& conf):
  conf_(conf),
  userEnv_("CORAL_AUTH_USER=" + conf.getUntrackedParameter<std::string>("userEnv","me")),
  passwdEnv_("CORAL_AUTH_PASSWORD="+ conf.getUntrackedParameter<std::string>("passwdEnv","mypass")),
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
  
  ::putenv( const_cast<char*>( userEnv_.c_str() ) );
  ::putenv( const_cast<char*>( passwdEnv_.c_str() ) );
};

void SiStripNoiseService::configure( const edm::EventSetup& es ) {
  edm::LogInfo("SiStripZeroSuppression") << "[SiStripNoiseService::configure]";
  setESObjects(es);
  
  if (UseCalibDataFromDB_==false) {
    edm::LogInfo("SiStripZeroSuppression") << "[SiStripNoiseService::configure] There are "<<tkgeom->dets().size() <<" detectors instantiated in the geometry";  
  } else {
    edm::LogInfo("SiStripZeroSuppression") << "[SiStripNoiseService::configure] There are "<< noise->m_noises.size() <<" detector Noise descriptions";  
  }
}

void SiStripNoiseService::setESObjects( const edm::EventSetup& es ) {
  if (UseCalibDataFromDB_==false) {
    es.get<TrackerDigiGeometryRecord>().get( tkgeom );
  } else {
    es.get<SiStripNoisesRcd>().get(noise);

    //Getting Cond Data
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&    
#ifdef DEBUG
    SiStripNoiseMapIterator mapit = noise->m_noises.begin();
    for (;mapit!=noise->m_noises.end();mapit++)
      {
	unsigned int detid = (*mapit).first;
	const SiStripNoiseVector& theSiStripVector =  noise->getSiStripNoiseVector(detid);
	std::cout << "[SiStripNoiseService::setESObjects] detid " <<  detid << " # Strip " << theSiStripVector.size() << std::endl;
	int strip=0;
	for(SiStripNoiseVectorIterator iter=theSiStripVector.begin(); iter!=theSiStripVector.end(); iter++){
	  std::cout << "[SiStripNoiseService::setESObjects] strip " << strip++ << " =\t"
		    << iter->getNoise()     << " \t" 
		    << iter->getDisable()   << " \t" 
		    << std::endl; 	    
	} 
      }   
#endif
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  }
};


float SiStripNoiseService::getNoise(const uint32_t& detID,const uint32_t& strip) const
{
  if (UseCalibDataFromDB_==false){	  
    //Case of SingleValue of Noise for all strips of a Detector
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
    //Access from DB
      return (noise->getSiStripNoiseVector(detID))[strip].getNoise();
    }
}

bool SiStripNoiseService::getDisable(const uint32_t& detID,const uint32_t& strip) const
{
  if (UseCalibDataFromDB_==false){	  
    //Case of SingleValue of Noise for all strips of a Detector
    return (RandFlat::shoot(1.) < BadStripProbability_ ? true:false);
  } 
  else
    {
    //Access from DB
      return (noise->getSiStripNoiseVector(detID))[strip].getDisable();
    }
}
