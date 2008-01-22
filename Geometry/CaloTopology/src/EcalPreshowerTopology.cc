#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include <stdexcept>

ESDetId EcalPreshowerTopology::incrementIy(const ESDetId& id) const {
  if (!(*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalPreshower)->present(id))
    {
      return ESDetId(0);
    }

  ESDetId nextPoint;
  //Strips orientend along x direction for plane 2
  if (id.plane() == 2)
    {
      if (id.strip() < 32 )
	{
	  //Incrementing just strip number
	  if (ESDetId::validDetId(id.strip()+1,id.six(),id.siy(),id.plane(),id.zside()))
	    nextPoint=ESDetId(id.strip()+1,id.six(),id.siy(),id.plane(),id.zside());
	  else
	    return ESDetId(0); 
	}
      else
	{
	  //Changing wafer
	  if (ESDetId::validDetId(1,id.six(),id.siy()+1,id.plane(),id.zside()))
	    nextPoint=ESDetId(1,id.six(),id.siy()+1,id.plane(),id.zside());
	  else
	    return ESDetId(0);
	}
    }
  //Strips orientend along y direction for plane 1
  else if (id.plane() == 1)
    {
      //Changing wafer
      if (ESDetId::validDetId(id.strip(),id.six(),id.siy()+1,id.plane(),id.zside()))
	nextPoint=ESDetId(id.strip(),id.six(),id.siy()+1,id.plane(),id.zside());
      else
	return ESDetId(0);
    }
  else
    return ESDetId(0);
  
  if ((*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalPreshower)->present(nextPoint))
    return nextPoint;
  else
    return ESDetId(0);
} 


ESDetId EcalPreshowerTopology::decrementIy(const ESDetId& id) const {
  if (!(*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalPreshower)->present(id))
    {
      return ESDetId(0);
    }
  ESDetId nextPoint;
  //Strips orientend along x direction for plane 2
  if (id.plane() == 2)
    {
      if (id.strip() >1 )
	{
	  //Decrementing just strip number
	  if (ESDetId::validDetId(id.strip()-1,id.six(),id.siy(),id.plane(),id.zside()))
	    nextPoint=ESDetId(id.strip()-1,id.six(),id.siy(),id.plane(),id.zside());
	  else
	    return ESDetId(0);
	}
      else
	{
	  //Changing wafer
	  if (ESDetId::validDetId(32,id.six(),id.siy()-1,id.plane(),id.zside()))
	    nextPoint=ESDetId(32,id.six(),id.siy()-1,id.plane(),id.zside());
	  else
	    return ESDetId(0);
	}
    }
  //Strips orientend along y direction for plane 1
  else if (id.plane() == 1)
    {
      //Changing wafer
      if (ESDetId::validDetId(id.strip(),id.six(),id.siy()-1,id.plane(),id.zside()))
	nextPoint=ESDetId(id.strip(),id.six(),id.siy()-1,id.plane(),id.zside());
      else
	return ESDetId(0);
    }
  else
    return ESDetId(0);

  if ((*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalPreshower)->present(nextPoint))
    return nextPoint;
  else
    return ESDetId(0);
} 


ESDetId EcalPreshowerTopology::incrementIx(const ESDetId& id) const {

  if (!(*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalPreshower)->present(id))
    {
      return ESDetId(0);
    }      
  ESDetId nextPoint;

  //Strips orientend along x direction for plane 2
  if (id.plane() == 2)
    {
      //Changing wafer
      if (ESDetId::validDetId(id.strip(),id.six()+1,id.siy(),id.plane(),id.zside()))
	nextPoint=ESDetId(id.strip(),id.six()+1,id.siy(),id.plane(),id.zside());
      else
        return ESDetId(0);
    }
  //Strips orientend along y direction for plane 1
  else if (id.plane() == 1)
    {
      if (id.strip() < 32 )
	{
	//Incrementing just strip number
	  if (ESDetId::validDetId(id.strip()+1,id.six(),id.siy(),id.plane(),id.zside())) 
	    nextPoint=ESDetId(id.strip()+1,id.six(),id.siy(),id.plane(),id.zside());
	  else
	    return ESDetId(0);
	}
      else
	{
	//Changing wafer
	  if (ESDetId::validDetId(1,id.six()+1,id.siy(),id.plane(),id.zside()))
	    nextPoint=ESDetId(1,id.six()+1,id.siy(),id.plane(),id.zside());
	  else
	    return ESDetId(0);
	}
    }
  else
    return ESDetId(0);

  if ((*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalPreshower)->present(nextPoint))
    return nextPoint;
  else
    return ESDetId(0);
} 


ESDetId EcalPreshowerTopology::decrementIx(const ESDetId& id) const {

  if (!(*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalPreshower)->present(id))
    {
      return ESDetId(0);
    }      
  ESDetId nextPoint;
  //Strips orientend along x direction for plane 2
  if (id.plane() == 2)
    {
      //Changing wafer
      if (ESDetId::validDetId(id.strip(),id.six()-1,id.siy(),id.plane(),id.zside()))
	nextPoint=ESDetId(id.strip(),id.six()-1,id.siy(),id.plane(),id.zside());
      else
	return ESDetId(0);
    }
  //Strips orientend along y direction for plane 1
  else if (id.plane() == 1)
    {
      if (id.strip() > 1 )
	{
	  //Decrementing just strip number
	  if (ESDetId::validDetId(id.strip()-1,id.six(),id.siy(),id.plane(),id.zside()))
	    nextPoint=ESDetId(id.strip()-1,id.six(),id.siy(),id.plane(),id.zside());
	  else
	    return ESDetId(0);
	}
      else
	{
	  //Changing wafer
	  if (ESDetId::validDetId(32,id.six()-1,id.siy(),id.plane(),id.zside()))
	    nextPoint=ESDetId(32,id.six()-1,id.siy(),id.plane(),id.zside());
	  else
            return ESDetId(0);
	}
    }
  else
    return ESDetId(0);
  
  if ((*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalPreshower)->present(nextPoint))
    return nextPoint;
  else
    return ESDetId(0);
} 


ESDetId EcalPreshowerTopology::incrementIz(const ESDetId& id) const {

  if (!(*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalPreshower)->present(id))
    {
      return ESDetId(0);
    }      
  ESDetId nextPoint;
  if (ESDetId::validDetId(id.strip(),id.six(),id.siy(),id.plane()+1,id.zside()))
    nextPoint=ESDetId(id.strip(),id.six(),id.siy(),id.plane()+1,id.zside());
  else
    return ESDetId(0);

  if ((*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalPreshower)->present(nextPoint))
    return nextPoint;
  else
    return ESDetId(0);
} 



ESDetId EcalPreshowerTopology::decrementIz(const ESDetId& id) const {

  if (!(*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalPreshower)->present(id))
    {
      return ESDetId(0);
    }      
  ESDetId nextPoint;
  if (ESDetId::validDetId(id.strip(),id.six(),id.siy(),id.plane()-1,id.zside()))
    nextPoint=ESDetId(id.strip(),id.six(),id.siy(),id.plane()-1,id.zside());
  else
    return ESDetId(0);
  
  if ((*theGeom_).getSubdetectorGeometry(DetId::Ecal,EcalPreshower)->present(nextPoint))
    return nextPoint;
  else
    return ESDetId(0);
} 







