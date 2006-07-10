#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsService.h"

SiStripPedestalsService::SiStripPedestalsService(const edm::ParameterSet& conf):
  conf_(conf),
  UseCalibDataFromDB_(conf.getParameter<bool>("UseCalibDataFromDB")),
  //
  ElectronsPerADC_(conf.getParameter<double>("ElectronPerAdc")),
  ENC_(conf.getParameter<double>("EquivalentNoiseCharge300um")),
  BadStripProbability_(conf.getParameter<double>("BadStripProbability")),
  PedestalValue_(conf.getParameter<uint32_t>("PedestalValue")),
  LTh_(conf.getParameter<double>("LowThValue")),
  HTh_(conf.getParameter<double>("HighThValue")){
  
  if (UseCalibDataFromDB_==false){	  
    edm::LogInfo("SiStripZeroSuppression")  << "[SiStripPedestalsService::SiStripPedestalsService] Using a Single Value for Pedestals, Noise, Low and High Threshold and good strip flags";
  } 
  else {
    edm::LogInfo("SiStripZeroSuppression")  << "[SiStripPedestalsService::SiStripPedestalsService] Using Calibration Data accessed from DB";
  }
  
  old_detID = 0;
  old_range = SiStripPedestals::Range((SiStripPedestals::ContainerIterator)0,(SiStripPedestals::ContainerIterator)0);
  old_noise = -1.;

};

void SiStripPedestalsService::setESObjects( const edm::EventSetup& es ) {
  if (UseCalibDataFromDB_==true) {
    es.get<SiStripPedestalsRcd>().get(ped);

    //Getting Cond Data
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&    
#ifdef DEBUG
    std::vector<uint32_t> detid;
    ped->getDetIds(detid);
    
    for (size_t id=0;id<detid.size();id++)
      {
	SiStripPedestals::Range range=ped->getRange(detid[id]);
	
	int strip=0;
	for(int it=0;it<range.second-range.first;it++){
	  edm::LogInfo("SiStripPedestalservice") << "[SiStripPedestalservice::setESObjects]"  
						 << "detid " << detid[id] << " \t"
						 << " strip " << strip++ << " \t"
						 << ped->getPed   (it,range)   << " \t" 
						 << ped->getLowTh (it,range)   << " \t" 
						 << ped->getHighTh(it,range)   << " \t" 
						 << std::endl; 	    
	} 
      }
#endif
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  }
};

int16_t SiStripPedestalsService::getPedestal(const uint32_t& detID,const uint16_t& strip)
{
  if (UseCalibDataFromDB_==false){	  
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    //Case of SingleValue of Pedestals for all strips of a Detector
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    return (int16_t) PedestalValue_;
  } 
  else
    {
      //&&&&&&&&&&&&&&&&&&&&
      //Access from DB
      //&&&&&&&&&&&&&&&&&&&&
      if (detID!=old_detID){
	old_detID=detID;
	old_range=ped->getRange(detID);
      }
      return (int16_t) ped->getPed(strip,old_range);
    }
}


float SiStripPedestalsService::getLowTh(const uint32_t& detID,const uint16_t& strip)
{
  if (UseCalibDataFromDB_==false){	  
    //Case of SingleValue of Pedestals for all strips of a Detector
    return LTh_;
  } 
  else
    {
      //&&&&&&&&&&&&&&&&&&&&
      //Access from DB
      //&&&&&&&&&&&&&&&&&&&&
      if (detID!=old_detID){
	old_detID=detID;
	old_range=ped->getRange(detID);
      }
      return ped->getLowTh(strip,old_range);
    }
}

float SiStripPedestalsService::getHighTh(const uint32_t& detID,const uint16_t& strip)
{
  if (UseCalibDataFromDB_==false){	  
    //Case of SingleValue of Pedestals for all strips of a Detector
    return HTh_;
  } 
  else
    {
      //&&&&&&&&&&&&&&&&&&&&
      //Access from DB
      //&&&&&&&&&&&&&&&&&&&&
      if (detID!=old_detID){
	old_detID=detID;
	old_range=ped->getRange(detID);
      }
      return ped->getHighTh(strip,old_range);
    }
}
