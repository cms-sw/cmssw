#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include <iostream>

using namespace reco;
using namespace std;


void CaloID::setDetector(Detectors theDetector, bool value) {
  
  //  cout<<"CaloID::setDetector "<<theDetector<<" "<<(1<<theDetector)<<endl;
  if(value)
    detectors_ = detectors_ | (1<<theDetector);
  else 
    detectors_ = detectors_ ^ (1<<theDetector);

  // cout<<detectors_<<endl;
}



bool CaloID::detector(Detectors theDetector) const {

  return (detectors_>>theDetector) & 1;
}


CaloID::Detectors  CaloID::detector() const {
  if( ! isSingleDetector() ) return DET_NONE;
  
  int pos =  leastSignificantBitPosition( detectors_ );
  
  CaloID::Detectors det = static_cast<CaloID::Detectors>(pos);
							       
  return det;
}



int CaloID::leastSignificantBitPosition(unsigned n) const {
    if (n == 0)
      return -1;
 
    int pos = 31;

    if (n & 0x000000000000FFFFLL) { pos -= 16; } else { n >>= 16; }
    if (n & 0x00000000000000FFLL) { pos -=  8; } else { n >>=  8; }
    if (n & 0x000000000000000FLL) { pos -=  4; } else { n >>=  4; }
    if (n & 0x0000000000000003LL) { pos -=  2; } else { n >>=  2; }
    if (n & 0x0000000000000001LL) { pos -=  1; }
    return pos;
}


std::ostream& reco::operator<<(std::ostream& out,
			       const CaloID& id) {
  if(!out) return out;

  out<<"CaloID: "<<id.detectors();
  return out;
}
