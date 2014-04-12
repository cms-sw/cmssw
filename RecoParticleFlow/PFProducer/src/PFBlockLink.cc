#include "RecoParticleFlow/PFProducer/interface/PFBlockLink.h"

#include <iomanip>

using namespace std;

std::ostream& operator<<(std::ostream& out, 
			 const PFBlockLink& l) {
  if(!out) return out;  

  out<<setiosflags(ios::fixed);
  
  out<<"link : "
     <<" 0x"<<std::hex<<l.type_<<std::dec<<"\t";

  out<<setiosflags(ios::right);
  out<<setw(10)<<l.dist_
     <<" "<<l.element1_<<" "<<l.element2_;
  
  out<<resetiosflags(ios::right|ios::fixed);
  
  return out;  
}
