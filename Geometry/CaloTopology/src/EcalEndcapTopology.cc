#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"


EEDetId EcalEndcapTopology::incrementIy(const EEDetId& id) const {
  try 
    {
      if (!(*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalEndcap)->present(id))
	{
	  return EEDetId(0);
	}
      EEDetId nextPoint;
      nextPoint=EEDetId(id.ix(),id.iy()+1,id.zside());
      if ((*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalEndcap)->present(nextPoint))
	return nextPoint;
      else
	return EEDetId(0);
    } 
  catch ( std::runtime_error &e ) 
    { 
      return EEDetId(0);
    }
}

EEDetId EcalEndcapTopology::decrementIy(const EEDetId& id) const {
  try 
    {
      if (!(*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalEndcap)->present(id))
	{
	  return EEDetId(0);
	}
      EEDetId nextPoint;
      nextPoint=EEDetId(id.ix(),id.iy()-1,id.zside());
      if ((*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalEndcap)->present(nextPoint))
	return nextPoint;
      else
	return EEDetId(0);
    } 
  catch ( std::runtime_error &e ) 
    { 
      return EEDetId(0);
    }
}

EEDetId EcalEndcapTopology::incrementIx(const EEDetId& id) const {
  try 
    {
      if (!(*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalEndcap)->present(id))
	{
	  return EEDetId(0);
	}
      
      EEDetId nextPoint;
      nextPoint=EEDetId(id.ix()+1,id.iy(),id.zside());
      if ((*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalEndcap)->present(nextPoint))
	return nextPoint;
      else
	return EEDetId(0);
    } 
  catch ( std::runtime_error &e ) 
    { 
      return EEDetId(0);
    }
}

EEDetId EcalEndcapTopology::decrementIx(const EEDetId& id) const {
  try 
    {
      if (!(*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalEndcap)->present(id))
	{
	  return EEDetId(0);
	}
      
      EEDetId nextPoint;
      nextPoint=EEDetId(id.ix()-1,id.iy(),id.zside());
      if ((*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalEndcap)->present(nextPoint))
	return nextPoint;
      else
	return EEDetId(0);
    } 
  catch ( std::runtime_error &e ) 
    { 
      return EEDetId(0);
    }
}
