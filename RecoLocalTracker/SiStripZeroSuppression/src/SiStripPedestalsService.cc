#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsService.h"

SiStripPedestalsService::SiStripPedestalsService(const edm::ParameterSet& conf):
  conf_(conf),
  userEnv_("CORAL_AUTH_USER=" + conf.getUntrackedParameter<std::string>("userEnv","me")),
  passwdEnv_("CORAL_AUTH_PASSWORD="+ conf.getUntrackedParameter<std::string>("passwdEnv","mypass")),
  UseCalibDataFromDB_(conf.getParameter<bool>("UseCalibDataFromDB")),
  //
  ElectronsPerADC_(conf.getParameter<double>("ElectronPerAdc")),
  ENC_(conf.getParameter<double>("EquivalentNoiseCharge300um")),
  BadStripProbability_(conf.getParameter<double>("BadStripProbability")),
  PedestalValue_(conf.getParameter<uint32_t>("PedestalValue")),
  LTh_(conf.getParameter<double>("LowThValue")),
  HTh_(conf.getParameter<double>("HighThValue")){
  
  if (UseCalibDataFromDB_==false){	  
    std::cout << "Using a Single Value for Pedestals, Noise, Low and High Threshold and good strip flags" << std::endl;
  } 
  else {
    std::cout << "Using Calibration Data accessed from DB" << std::endl;
  }
  
  ::putenv( const_cast<char*>( userEnv_.c_str() ) );
  ::putenv( const_cast<char*>( passwdEnv_.c_str() ) );
};


void SiStripPedestalsService::configure( const edm::EventSetup& es ) {
    edm::LogInfo("SiStripPedestalsService") << "configure ";

    //Getting Geometry
    es.get<TrackerDigiGeometryRecord>().get( tkgeom );
    edm::LogInfo("SiStripPedestalsService") << "There are "<<tkgeom->dets().size() <<" detectors";
    

    if (UseCalibDataFromDB_==true){
      es.get<SiStripPedestalsRcd>().get(ped);
      //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
      // FIXME
      // Debug: show noise for DetIDs
      SiStripPedestalsMapIterator mapit = (*ped).m_pedestals.begin();
      for (;mapit!=(*ped).m_pedestals.end();mapit++)
	{
	  unsigned int detid = (*mapit).first;
	  std::cout << "detid " <<  detid << "# Strip " << (*mapit).second.size()<<std::endl;
	  SiStripPedestalsVector theSiStripVector =  (*mapit).second;     
	  //const SiStripPedestalsVector theSiStripVector =  (*ped).getSiStripPedestalsVector(detid);
	
	  int strip=0;
	  for(SiStripPedestalsVectorIterator iter=theSiStripVector.begin(); iter!=theSiStripVector.end(); iter++){
	    
	    std::cout << " strip " << strip++ << " =\t"
		      << iter->getPed()       << " \t" 
		      << iter->getNoise()     << " \t" 
		      << iter->getLowTh()     << " \t" 
		      << iter->getHighTh()    << " \t" 
		      << iter->getDisable()   << " \t" 
		      << std::endl; 	    
	  } 
	}   
      //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    }
};

int16_t SiStripPedestalsService::getPedestal(const uint32_t& detID,const uint32_t& strip) const
{
  if (UseCalibDataFromDB_==false){	  
    //Case of SingleValue of Pedestals for all strips of a Detector
    return (int16_t) PedestalValue_;
  } 
  else
    {
    //Access from DB
      return ped->getSiStripPedestalsVector(detID)[strip].getPed();
    }
}

float SiStripPedestalsService::getNoise(const uint32_t& detID,const uint32_t& strip)
{
  if (UseCalibDataFromDB_==false){	  
    //Case of SingleValue of Pedestals for all strips of a Detector
    float noise = 0;
    //Access to geometry needs to know module Thickness
    const GeomDetUnit* it = tkgeom->idToDetUnit(DetId(detID));
    //FIXME throw exception
    if (dynamic_cast<const StripGeomDetUnit*>(it)==0) {
      std::cout<< "WARNING: the detID " << detID << " doesn't seem to belong to Tracker" << std::endl; 
    }else{
      double moduleThickness = it->surface().bounds().thickness(); //thickness
      noise = ENC_*moduleThickness/(0.03)/ElectronsPerADC_;
    }
    return noise;
  } 
  else
    {
    //Access from DB
      return ped->getSiStripPedestalsVector(detID)[strip].getNoise();
    }
}

float SiStripPedestalsService::getLowTh(const uint32_t& detID,const uint32_t& strip)
{
  if (UseCalibDataFromDB_==false){	  
    //Case of SingleValue of Pedestals for all strips of a Detector
    return LTh_;
  } 
  else
    {
    //Access from DB
      return ped->getSiStripPedestalsVector(detID)[strip].getLowTh();
    }
}

float SiStripPedestalsService::getHighTh(const uint32_t& detID,const uint32_t& strip)
{
  if (UseCalibDataFromDB_==false){	  
    //Case of SingleValue of Pedestals for all strips of a Detector
    return HTh_;
  } 
  else
    {
    //Case of Noise and BadStrip flags access from DB
      return ped->getSiStripPedestalsVector(detID)[strip].getHighTh();
    }
}
