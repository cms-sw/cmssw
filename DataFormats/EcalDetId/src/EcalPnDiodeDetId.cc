#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include <stdexcept>

EcalPnDiodeDetId::EcalPnDiodeDetId() {
}
  
  
EcalPnDiodeDetId::EcalPnDiodeDetId(uint32_t rawid) : DetId(rawid) {
}
  
EcalPnDiodeDetId::EcalPnDiodeDetId(int EcalSubDetectorId, int DCCId, int PnId) : DetId(Ecal,EcalLaserPnDiode) {
  id_|= ((((EcalSubDetectorId==EcalBarrel)?(0):(1))<<10) | 
	((DCCId&0x3F)<<4) |
	(PnId&0xF));
}
  
EcalPnDiodeDetId::EcalPnDiodeDetId(const DetId& gen) {
  if (gen.det()!=Ecal || gen.subdetId()!=EcalTriggerTower) {
    throw new std::exception();
  }
  id_=gen.rawId();
}
  
EcalPnDiodeDetId& EcalPnDiodeDetId::operator=(const DetId& gen) {
  if (gen.det()!=Ecal || gen.subdetId()!=EcalLaserPnDiode) {
    throw new std::exception();
  }
  id_=gen.rawId();
  return *this;
}

int EcalPnDiodeDetId::hashedIndex() const
{
  throw(std::runtime_error("EcalPnDiodeDetId: Method not yet implemented"));
}    

std::ostream& operator<<(std::ostream& s,const EcalPnDiodeDetId& id) {
  return s << "(EcalPnDiode " << id.iEcalSubDetectorId() << ',' << id.iDCCId() << ',' << id.iPnId() << ')';
}

