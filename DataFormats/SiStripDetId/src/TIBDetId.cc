#include "DataFormats/SiStripDetId/interface/TIBDetId.h"

TIBDetId::TIBDetId() : SiStripDetId(){
}
TIBDetId::TIBDetId(uint32_t rawid) : SiStripDetId(rawid){
}
TIBDetId::TIBDetId(const DetId& id) : SiStripDetId(id.rawId()){
}
  

std::ostream& operator<<(std::ostream& os,const TIBDetId& id) {
  return os << "(TIB " 
    //	     << id.layer() << ',' 
    //	     << id.strng() << ',' 
    //	     << id.det() << ',' 
    //	     << id.ster() <<')';
	   <<')';
}
  
