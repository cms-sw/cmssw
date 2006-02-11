#include "DataFormats/SiStripDetId/interface/TIBDetId.h"

TIBDetId::TIBDetId() : DetId(){
}
TIBDetId::TIBDetId(uint32_t rawid) : DetId(rawid){
}
TIBDetId::TIBDetId(const DetId& id) : DetId(id.rawId()){
}
  

std::ostream& operator<<(std::ostream& os,const TIBDetId& id) {
  return os << "(TIB " 
    //	     << id.layer() << ',' 
    //	     << id.strng() << ',' 
    //	     << id.det() << ',' 
    //	     << id.ster() <<')';
	   <<')';
}
  
