#include "EventFilter/EcalRawToDigi/interface/DCCDataUnpacker.h"
#include "EventFilter/EcalRawToDigi/interface/DCCEBEventBlock.h"
#include "EventFilter/EcalRawToDigi/interface/DCCEEEventBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "EventFilter/EcalRawToDigi/interface/EcalElectronicsMapper.h"
#include <set>

bool DCCDataUnpacker::silentMode_ = false;

DCCDataUnpacker::DCCDataUnpacker( 
  EcalElectronicsMapper * mapper, bool hU, bool srpU, bool tccU, bool feU , bool memU, bool syncCheck, bool feIdCheck, bool forceToKeepFRdata
){ 
  electronicsMapper_ = mapper;
  ebEventBlock_   = new DCCEBEventBlock(this,mapper,hU,srpU,tccU,feU,memU,forceToKeepFRdata);
  eeEventBlock_   = new DCCEEEventBlock(this,mapper,hU,srpU,tccU,feU,memU,forceToKeepFRdata);
  if(syncCheck){
    ebEventBlock_->enableSyncChecks();  
    eeEventBlock_->enableSyncChecks();
  }
  if(feIdCheck){
    ebEventBlock_->enableFeIdChecks();  
    eeEventBlock_->enableFeIdChecks();
  }
}


void DCCDataUnpacker::unpack(const uint64_t* buffer, size_t bufferSize, unsigned int smId, unsigned int fedId){
  //buffer is pointer to binary data
  //See if this fed is on EB or in EE

  if(smId>9&&smId<46){ 
    
    currentEvent_      = ebEventBlock_;
    ebEventBlock_    ->updateCollectors();
    ebEventBlock_    ->unpack(buffer,bufferSize,fedId); 
	 
  }
  else{               
   
    currentEvent_     = eeEventBlock_;
    eeEventBlock_    ->updateCollectors();
    eeEventBlock_    ->unpack(buffer,bufferSize,fedId); 

  }
    
}

DCCDataUnpacker::~DCCDataUnpacker(){
  delete ebEventBlock_;
  delete eeEventBlock_;
}

uint16_t DCCDataUnpacker::getChannelStatus(const DetId& id) const
{
  // return code for situation of missing channel record
  // equal to "non responding isolated channel (dead of type other)":
  //   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideEcalRecoLocalReco#Treatment_of_problematic_channel
  // TODO: think on a better way how to cover this case
  const uint16_t NO_DATA = 11;
  
  if (chdb_ == 0) {
    edm::LogError("IncorrectMapping")
      << "ECAL channel status database do not initialized";
    return NO_DATA;
  }
  
  EcalChannelStatus::const_iterator pCh = chdb_->find(id);
  
  if (pCh != chdb_->end()) {
    return pCh->getStatusCode();
  }
  else {
    edm::LogError("IncorrectMapping")
      << "No channel status record found for detit = " << id.rawId();
    return NO_DATA;
  }
}

uint16_t DCCDataUnpacker::getChannelValue(const DetId& id) const
{
  return getChannelStatus(id) & 0x1F;
}

uint16_t DCCDataUnpacker::getChannelValue(const int fed, const int ccu, const int strip, const int xtal) const
{
  // conversion FED ID [601 - 654] -> DCC ID [1 - 54]
  const int dcc = electronicsMapper_->getSMId(fed);
  
  // convert (dcc, ccu, strip, xtal) -> DetId
  const EcalElectronicsId eid(dcc, ccu, strip, xtal);
  const DetId id = electronicsMapper_->mapping()->getDetId(eid);
  
  return getChannelStatus(id) & 0x1F;
}

uint16_t DCCDataUnpacker::getCCUValue(const int fed, const int ccu) const
{
  // get list of crystals (DetId) which correspond to given CCU
  // (return empty list for MEM channels [CCU > 68])
  const int dcc = electronicsMapper_->getSMId(fed);
  const std::vector<DetId> xtals =
    (ccu <= 68) ?
      electronicsMapper_->mapping()->dccTowerConstituents(dcc, ccu) :
      std::vector<DetId>();
  
  // collect set of status codes of given CCU
  std::set<uint16_t> set;
  for (size_t i = 0; i < xtals.size(); ++i) {
    const uint16_t val = getChannelValue(xtals[i]);
    set.insert(val);
  }
  
  // if all crystals in CCU have the same status
  // then this status is treated as CCU status
  if (set.size() == 1) return *set.begin();
  
  // if there are several or no statuses:
  return 0;
}
