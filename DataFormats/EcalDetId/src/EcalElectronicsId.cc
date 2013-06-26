#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <ostream>

EcalElectronicsId::EcalElectronicsId() {
  EcalElectronicsId_=0xFFFFFFFFu;
}

EcalElectronicsId::EcalElectronicsId(uint32_t id) {
  EcalElectronicsId_=id;
}

EcalElectronicsId::EcalElectronicsId(int dccid, int towerid, int stripid, int xtalid){
  if ( (dccid < MIN_DCCID) || (dccid > MAX_DCCID) ||  
       (towerid < MIN_TOWERID) || (towerid > MAX_TOWERID) ||  
       (stripid < MIN_STRIPID) || (stripid > MAX_STRIPID) ||  
       (xtalid < MIN_XTALID) || (xtalid > MAX_XTALID) )
    throw cms::Exception("InvalidDetId") << "EcalElectronicsId:  Cannot create object.  Indexes out of bounds. Dcc tower strip xtal " << dccid << " " << towerid << " " << stripid << " " << xtalid << ".";
  EcalElectronicsId_= (xtalid&0x7) | ( (stripid&0x7) << 3) | ( (towerid&0x7F) << 6) | ((dccid&0x7F) << 13);
}


EcalSubdetector EcalElectronicsId::subdet() const {
	int dcc = dccId();
	if ( (dcc >= MIN_DCCID_EBM && dcc <= MAX_DCCID_EBM) ||
	     (dcc >= MIN_DCCID_EBP && dcc <= MAX_DCCID_EBP) ) return EcalBarrel;
	else return EcalEndcap;
}

int EcalElectronicsId::zside() const {
	int dcc = dccId();
	if ( (dcc >= MIN_DCCID_EEM && dcc <= MAX_DCCID_EEM)) return -1;
        if ( (dcc >= MIN_DCCID_EBM && dcc <= MAX_DCCID_EBM)) return -1;
        if ( (dcc >= MIN_DCCID_EEP && dcc <= MAX_DCCID_EEP)) return +1;
        if ( (dcc >= MIN_DCCID_EBP && dcc <= MAX_DCCID_EBP)) return +1;
	return 0;
}



static int EEQuadrant(int dcc, int dcc_channel) {
        // Q1 = EE+N or EE-F, Top
        // Q2 = EE+F or EE-N, Top
        // Q3 = EE+F or EE-N, Bottom
        // Q4 = EE+N or EE-F, Bottom
        //  (local notation)
        // in Q1-Q3 and in Q2-Q4, the relation between strip#, channel# and xtal_id
        // is the same
 int q=-1;
  if ( (dcc == EcalElectronicsId::DCC_EEP + 1) || ( dcc == EcalElectronicsId::DCC_EEP + 2) ||
       (dcc == EcalElectronicsId::DCC_EEP && dcc_channel <= EcalElectronicsId::kDCCChannelBoundary) ||
       (dcc == EcalElectronicsId::DCC_EEM + 3) || ( dcc == EcalElectronicsId::DCC_EEM + 4) ||
       (dcc == EcalElectronicsId::DCC_EEM + 5 && dcc_channel <= EcalElectronicsId::kDCCChannelBoundary) ) q=1;
  else if ( (dcc == EcalElectronicsId::DCC_EEP + 3) || ( dcc == EcalElectronicsId::DCC_EEP + 4) ||
       (dcc == EcalElectronicsId::DCC_EEP+5 && dcc_channel <= EcalElectronicsId::kDCCChannelBoundary) ||
       (dcc == EcalElectronicsId::DCC_EEM && dcc_channel <= EcalElectronicsId::kDCCChannelBoundary) ||
       (dcc == EcalElectronicsId::DCC_EEM + 1) || ( dcc == EcalElectronicsId::DCC_EEM + 2) ) q=2;
  else if ( (dcc == EcalElectronicsId::DCC_EEP + 6) ||
       (dcc == EcalElectronicsId::DCC_EEP + 5 && dcc_channel > EcalElectronicsId::kDCCChannelBoundary) ||
       (dcc == EcalElectronicsId::DCC_EEP + 7 && dcc_channel > EcalElectronicsId::kDCCChannelBoundary) ||
       (dcc == EcalElectronicsId::DCC_EEM && dcc_channel > EcalElectronicsId::kDCCChannelBoundary) ||
       (dcc == EcalElectronicsId::DCC_EEM + 8) ||
       (dcc == EcalElectronicsId::DCC_EEM + 7 && dcc_channel > EcalElectronicsId::kDCCChannelBoundary)) q=3;
  else
       q=4;
  return q;
}

int EcalElectronicsId::channelId() const {
 int dcc = dccId() ;
 int dcc_channel = towerId();
 int quadrant = EEQuadrant(dcc, dcc_channel);
 int strip = stripId();
 int xtal = xtalId();
 int channel;
 if (quadrant ==1 || quadrant== 3) channel = 5*(strip-1) + xtal;
 else channel = 5*(xtal-1) + strip;
 return channel;
}

/*
int EcalElectronicsId::stripId() {
 int dcc = dccId() ;
 int dcc_channel = towerId();
 int quadrant = EEQuadrant(dcc, dcc_channel);
 int xtal = channelId();
 int strip;
 if (quadrant ==1 || quadrant== 3) strip = (xtal-1)/5 +1;
 else strip = (xtal-1) % 5 +1;
 return strip;
}

int EcalElectronicsId::XtalInStripId() {
 int dcc = dccId() ;
 int dcc_channel = towerId();
 int quadrant = EEQuadrant(dcc, dcc_channel);
 int xtal = channelId();
 int id;
 if (quadrant ==1 || quadrant== 3) id = (xtal-1)%5 + 1;
 else id = (xtal-1)/5 +1;
 return id;
}
*/



std::ostream& operator<<(std::ostream& os,const EcalElectronicsId& id) 
{
  return os << id.dccId() << ',' << id.towerId() << ',' << id.stripId() << ',' << id.xtalId() ;
}

