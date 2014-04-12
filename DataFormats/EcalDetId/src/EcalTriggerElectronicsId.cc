#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <ostream>


EcalTriggerElectronicsId::EcalTriggerElectronicsId() {
  EcalTriggerElectronicsId_=0xFFFFFFFFu;
}

EcalTriggerElectronicsId::EcalTriggerElectronicsId(uint32_t id) {
  EcalTriggerElectronicsId_=id;
}

EcalTriggerElectronicsId::EcalTriggerElectronicsId(int tccid, int ttid, int pseudostripid, int channelid){
  if ( (tccid < MIN_TCCID) || (tccid > MAX_TCCID) ||  
       (ttid < MIN_TTID) || (ttid > MAX_TTID) ||  
       (pseudostripid < MIN_PSEUDOSTRIPID) || (pseudostripid > MAX_PSEUDOSTRIPID) ||  
       (channelid < MIN_CHANNELID) || (channelid > MAX_CHANNELID) )
    throw cms::Exception("InvalidDetId") << "EcalTriggerElectronicsId:  Cannot create object.  Indexes out of bounds.";
  EcalTriggerElectronicsId_= (channelid&0x7) | ( (pseudostripid&0x7) << 3) | ( (ttid&0x7F) << 6) | ((tccid&0x7F) << 13);
}

int EcalTriggerElectronicsId::zside() const {
        int tcc = tccId();
        if ( (tcc >= MIN_TCCID_EEM && tcc <= MAX_TCCID_EEM)) return -1;
        if ( (tcc >= MIN_TCCID_EBM && tcc <= MAX_TCCID_EBM)) return -1;
        if ( (tcc >= MIN_TCCID_EEP && tcc <= MAX_TCCID_EEP)) return +1;
        if ( (tcc >= MIN_TCCID_EBP && tcc <= MAX_TCCID_EBP)) return +1;
        return 0;
}


EcalSubdetector EcalTriggerElectronicsId::subdet() const {
	int tcc = tccId();
	if ( (tcc >= MIN_TCCID_EBM && tcc <= MAX_TCCID_EBM) ||
	     (tcc >= MIN_TCCID_EBP && tcc <= MAX_TCCID_EBP) ) return EcalBarrel;
	else return EcalEndcap;
}

std::ostream& operator<<(std::ostream& os,const EcalTriggerElectronicsId& id) 
{
  return os << id.tccId() << ',' << id.ttId()  << ',' << id.pseudoStripId() << ',' << id.channelId() ;
}

