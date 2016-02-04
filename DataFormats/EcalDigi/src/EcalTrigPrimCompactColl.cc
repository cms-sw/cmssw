#include "DataFormats/EcalDigi/interface/EcalTrigPrimCompactColl.h"

void EcalTrigPrimCompactColl::toEcalTrigPrimDigiCollection(EcalTrigPrimDigiCollection& dest) const{
  const int nTtEtaBins = 56;
  const int nTtPhiBins = 72;
  EcalTrigPrimDigiCollection tpColl;
  tpColl.reserve(nTtEtaBins*nTtPhiBins);
  
  for(int zside = -1; zside <= 1; zside +=2){
    for(int iabseta = 1; iabseta <= nTtEtaBins/2; ++iabseta){
      EcalSubdetector subdet = (iabseta <= 17) ? EcalBarrel : EcalEndcap;
      for(int iphi = 1; iphi <= 72; ++iphi){
	const EcalTrigTowerDetId ttId(zside, subdet, iabseta, iphi, EcalTrigTowerDetId::SUBDETIJMODE);
	EcalTriggerPrimitiveDigi tp(ttId);
	const int rawTp = raw(ttId.ieta(), ttId.iphi());
	const EcalTriggerPrimitiveSample tps(rawTp);
	tp.setSize(1);
	tp.setSample(0, tps);
	tpColl.push_back(tp);
      }
    }
  }
  tpColl.sort();
  dest.swap(tpColl);
}
