#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

HcalCondObjectContainerBase::HcalCondObjectContainerBase(const HcalTopology* topo) : packedIndexVersion_(0), topo_(topo) { 
  if (topo_) packedIndexVersion_=topo_->topoVersion();
}

void HcalCondObjectContainerBase::setTopo(const HcalTopology* topo) const {
  if (topo && !topo->denseIdConsistent(packedIndexVersion_)) {
    edm::LogError("HCAL") << std::string("Inconsistent dense packing between current topology (") << topo->topoVersion() << ") and calibration object (" << packedIndexVersion_ << ")";
  }
  topo_=topo;
}
void HcalCondObjectContainerBase::setTopo(const HcalTopology* topo) {
  if (topo && !topo->denseIdConsistent(packedIndexVersion_)) {
    edm::LogError("HCAL") << "Inconsistent dense packing between current topology (" << topo->topoVersion() << ") and calibration object (" << packedIndexVersion_ << ")";
  }
  topo_=topo;  
}

unsigned int HcalCondObjectContainerBase::indexFor(DetId fId) const { 
  unsigned int retval=0xFFFFFFFFu;
  if (!topo_) {
    edm::LogError("HCAL") << "Topology pointer not set, HCAL conditions non-functional";
    throw cms::Exception("Topology pointer not set, HCAL conditions non-functional");
    return retval;
  }

  if (fId.det()==DetId::Hcal) {
    switch (HcalSubdetector(fId.subdetId())) {
    case(HcalBarrel) : retval=topo_->detId2denseIdHB(fId); break;
    case(HcalEndcap) : retval=topo_->detId2denseIdHE(fId); break;
    case(HcalOuter) : retval=topo_->detId2denseIdHO(fId); break;
    case(HcalForward) : retval=topo_->detId2denseIdHF(fId); break;
    case(HcalTriggerTower) : retval=topo_->detId2denseIdHT(fId); break;
    case(HcalOther) : if (extractOther(fId)==HcalCalibration)
	retval=topo_->detId2denseIdCALIB(fId);
      break; 
    default: break;
    }
  } else if (fId.det()==DetId::Calo) {
    if (fId.subdetId()==HcalCastorDetId::SubdetectorId) {
      // the historical packing from HcalGeneric is different from HcalCastorDetId, so we clone the old packing here.
      HcalCastorDetId tid(fId); 
      int zside = tid.zside();
      int sector = tid.sector();
      int module = tid.module();
      static const int CASTORhalf=224;

      int index = 14*(sector-1) + (module-1);
      if (zside == -1) index += CASTORhalf;

      retval=(unsigned int)(index);
    } else if (fId.subdetId()==HcalZDCDetId::SubdetectorId) {
      HcalZDCDetId direct(fId);
      // THIS IS A HORRIBLE HACK because there were _two_ dense indices for ZDC differing in their handling of +/-z
      HcalZDCDetId swapZ(direct.section(),direct.zside()<0,direct.channel());
      retval=swapZ.denseIndex();
    }
  }
  return retval;
}

unsigned int HcalCondObjectContainerBase::sizeFor(DetId fId) const {
  unsigned int retval=0;

  if (!topo_) {
    edm::LogError("HCAL") << "Topology pointer not set, HCAL conditions non-functional";
    throw cms::Exception("Topology pointer not set, HCAL conditions non-functional");
    return retval;
  }

  if (fId.det()==DetId::Hcal) {
    switch (HcalSubdetector(fId.subdetId())) {
    case(HcalBarrel) : retval=topo_->getHBSize(); break;
    case(HcalEndcap) : retval=topo_->getHESize(); break;
    case(HcalOuter) : retval=topo_->getHOSize(); break;
    case(HcalForward) : retval=topo_->getHFSize(); break;
    case(HcalTriggerTower) : retval=topo_->getHTSize(); break;
    case(HcalOther) : if (extractOther(fId)==HcalCalibration) retval=topo_->getCALIBSize();
      break; 
    default: break;
    }
  } else if (fId.det()==DetId::Calo) {
    if (fId.subdetId()==HcalCastorDetId::SubdetectorId) {
      retval=HcalCastorDetId::kSizeForDenseIndexing;
    } else if (fId.subdetId()==HcalZDCDetId::SubdetectorId) {
      retval=HcalZDCDetId::kSizeForDenseIndexing;
    }
  }
  return retval;
}

std::string HcalCondObjectContainerBase::textForId(const DetId& id) const {
  std::ostringstream os;
  os << std::hex << "(0x" << id.rawId() << ") " << std::dec;

  if (id.det()==DetId::Hcal) {
    switch (HcalSubdetector(id.subdetId())) {
    case(HcalBarrel) : 
    case(HcalEndcap) : 
    case(HcalOuter) : 
    case(HcalForward) : os << HcalDetId(id); break;
    case(HcalTriggerTower) : os << HcalTrigTowerDetId(id); break;
    case(HcalOther) : 
      if (extractOther(id)==HcalCalibration) os << HcalCalibDetId(id);
      break; 
    default: break;
    }
  } else if (id.det()==DetId::Calo) {
    if (id.subdetId()==HcalCastorDetId::SubdetectorId) {
      os << HcalCastorDetId(id);
    } else if (id.subdetId()==HcalZDCDetId::SubdetectorId) {
      os << HcalZDCDetId(id);
    }
  }  
  return os.str();
}
