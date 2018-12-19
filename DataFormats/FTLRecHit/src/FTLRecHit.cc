#include "DataFormats/FTLRecHit/interface/FTLRecHit.h"
#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"
#include <cassert>
#include <cmath>
#include <limits>

namespace {
  constexpr float timereso_max = 10000;
}

FTLRecHit::FTLRecHit() : id_(DetId()), energy_(-1.f), time_(-1.f), timeError_(-1.f), row_(0), column_(0), flagBits_(std::numeric_limits<unsigned char>::max()) 
{}

FTLRecHit::FTLRecHit(const DetId& id, float energy, float time, float timeError, uint32_t flagBits) :
  id_(id), energy_(energy), time_(time), timeError_(timeError), row_(0), column_(0), flagBits_(flagBits) 
{}

FTLRecHit::FTLRecHit(const DetId& id, uint8_t row, uint8_t column, float energy, 
		     float time, float timeError, uint32_t flagBits) :
  id_(id), energy_(energy), time_(time), timeError_(timeError), row_(row), column_(column), flagBits_(flagBits)
{}

bool FTLRecHit::isTimeValid() const {
  if(timeError() < 0)
    return false;
  else
    return true;
}

bool FTLRecHit::isTimeErrorValid() const {
  if(!isTimeValid())
    return false;
  if(timeError() >= timereso_max)
    return false;
  
  return true;
}

/// check if one of the flags in a set is true
bool FTLRecHit::checkFlags(const std::vector<int>&  flagsvec ) const {
  
  for (std::vector<int>::const_iterator flagPtr = flagsvec.begin(); 
       flagPtr!= flagsvec.end(); ++flagPtr) { // check if one of the flags is up
    if (checkFlag(*flagPtr)) return true;    
  }
  return false;
}


std::ostream& operator<<(std::ostream& s, const FTLRecHit& hit) {
  if (hit.detid().det() == DetId::Forward && hit.detid().subdetId() == FastTime) 
    return s << FastTimeDetId(hit.detid()) << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  else
    return s << "FTLRecHit undefined subdetector" ;
}
