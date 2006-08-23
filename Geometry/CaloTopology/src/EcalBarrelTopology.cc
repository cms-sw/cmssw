#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"


EBDetId EcalBarrelTopology::incrementIeta(const EBDetId& id) const {
  try 
    {
      if (!(*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalBarrel)->present(id))
	return EBDetId(0);
      
      EBDetId nextPoint;
      if (id.ieta()==-1) 
	  nextPoint=EBDetId (1,id.iphi());
      else 
	nextPoint=EBDetId(id.ieta()+1,id.iphi());
      if ((*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalBarrel)->present(nextPoint))
	return nextPoint;
      else
	return EBDetId(0);
    } 
  catch ( cms::Exception &e ) 
    { 
      return EBDetId(0);
    }
}

EBDetId EcalBarrelTopology::decrementIeta(const EBDetId& id) const {

  try 
    {
      if (!(*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalBarrel)->present(id))
	return EBDetId(0);
      
      EBDetId nextPoint;
      if (id.ieta()==1) 
	nextPoint=EBDetId(-1,id.iphi());
      else 
	nextPoint=EBDetId(id.ieta()-1,id.iphi());
      
      if ((*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalBarrel)->present(nextPoint))
	return nextPoint;
      else
	return EBDetId(0);
    } 
  catch ( cms::Exception &e ) 
    { 
      return EBDetId(0);
    }

}

EBDetId EcalBarrelTopology::incrementIphi(const EBDetId& id) const {

  try 
    {
      if (!(*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalBarrel)->present(id))
	  return EBDetId(0);
      
      EBDetId nextPoint;

      if (id.iphi()==EBDetId::MAX_IPHI) 
	 nextPoint=EBDetId(id.ieta(),EBDetId::MIN_IPHI);
      else
	nextPoint=EBDetId(id.ieta(),id.iphi()+1);
      
      
      if ((*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalBarrel)->present(nextPoint))
	return nextPoint;
      else
	return EBDetId(0);
    } 
  catch ( cms::Exception &e ) 
    { 
      return EBDetId(0);
    }

}

EBDetId EcalBarrelTopology::decrementIphi(const EBDetId& id) const {

  try 
    {
      if (!(*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalBarrel)->present(id))
	  return EBDetId(0);

      EBDetId nextPoint;

      if (id.iphi()==EBDetId::MIN_IPHI) 
	nextPoint=EBDetId(id.ieta(),EBDetId::MAX_IPHI);
      else
	nextPoint=EBDetId(id.ieta(),id.iphi()-1);

      if ((*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalBarrel)->present(nextPoint))
	return nextPoint;
      else
	return EBDetId(0);
    } 
  catch ( cms::Exception &e ) 
    { 
      return EBDetId(0);
    }

}
