#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/MuonNumbering/interface/CSCNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include <iostream>

//#define LOCAL_DEBUG

CSCNumberingScheme::CSCNumberingScheme( const MuonDDDConstants& muonConstants ) {
  initMe(muonConstants);
}

CSCNumberingScheme::CSCNumberingScheme( const DDCompactView& cpv )
{
  MuonDDDConstants muonConstants(cpv);
  initMe(muonConstants);
}

void CSCNumberingScheme::initMe (  const MuonDDDConstants& muonConstants ) {
  int theLevelPart=muonConstants.getValue("level");
  theRegionLevel=muonConstants.getValue("me_region")/theLevelPart;
  theStationLevel=muonConstants.getValue("me_station")/theLevelPart;
  theSubringLevel=muonConstants.getValue("me_subring")/theLevelPart;
  theSectorLevel=muonConstants.getValue("me_sector")/theLevelPart;
  theLayerLevel=muonConstants.getValue("me_layer")/theLevelPart;
  theRingLevel=muonConstants.getValue("me_ring")/theLevelPart;
#ifdef LOCAL_DEBUG
  std::cout << "theRegionLevel " << theRegionLevel <<std::endl;
  std::cout << "theStationLevel " << theStationLevel <<std::endl;
  std::cout << "theSubringLevel " << theSubringLevel <<std::endl;
  std::cout << "theSectorLevel " << theSectorLevel <<std::endl;
  std::cout << "theLayerLevel " << theLayerLevel <<std::endl;
  std::cout << "theRingLevel " << theRingLevel <<std::endl;
#endif
}

int CSCNumberingScheme::baseNumberToUnitNumber(const MuonBaseNumber num){

#ifdef LOCAL_DEBUG
  std::cout << "CSCNumbering "<<num.getLevels()<<std::endl;
  for (int level=1;level<=num.getLevels();level++) {
    std::cout << level << " " << num.getSuperNo(level)
	 << " " << num.getBaseNo(level) << std::endl;
  }
#endif

  if (num.getLevels()!=theRingLevel) {
    std::cout << "CSCNS::BNToUN "
	 << "BaseNumber has " << num.getLevels() << " levels,"
	 << "need "<<theRingLevel<<std::endl;
    return 0;
  }

  int fwbw_id=0;
  int station_id=0;
  int ring_id=0;
  int subring_id=0;
  int sector_id=0;
  int layer_id=0;


  //decode 6 significant endcap levels
  
  for (int level=1;level<=theRingLevel;level++) {

    //decode
    if (level==theRegionLevel) {
      const int copyno=num.getBaseNo(level);
      fwbw_id=copyno+1;

    } else if (level==theStationLevel) {
      const int station_tag = num.getSuperNo(level);      
      station_id=station_tag;
      
    } else if (level==theSubringLevel) {
      const int copyno=num.getBaseNo(level);
      subring_id=copyno+1;

    } else if (level==theSectorLevel) {
      const int copyno=num.getBaseNo(level);
      sector_id=copyno+1;

    } else if (level==theLayerLevel) {
      const int copyno=num.getBaseNo(level);
      layer_id=copyno+1;

    } else if (level==theRingLevel) {
      const int ring_tag = num.getSuperNo(level);      
      ring_id=ring_tag;
    }
  }      

  // check validity 
    
  if ((fwbw_id < 1) || (fwbw_id > 2)) {
    std::cout << "CSCNumberingScheme: ";
    std::cout << "forward/backward id out of range: ";
    std::cout << fwbw_id <<std::endl;
  }
    
  if ((station_id < 1) || (station_id > 4)) {
    std::cout << "CSCNumberingScheme: ";
    std::cout << "station id out of range: ";
    std::cout << station_id <<std::endl;
  }
    
  if ((ring_id < 1) || (ring_id > 4)) {
    std::cout << "CSCNumberingScheme: ";
    std::cout << "ring id out of range: ";
    std::cout << ring_id <<std::endl;
  }
    
  if ((subring_id < 1) || (subring_id > 2)) {
    std::cout << "CSCNumberingScheme: ";
    std::cout << "subring id out of range: ";
    std::cout << subring_id <<std::endl;
  }
    
  if ((sector_id < 1) || (sector_id > 36)) {
    std::cout << "CSCNumberingScheme: ";
    std::cout << "sector id out of range: ";
    std::cout << sector_id <<std::endl;
  }
    
  if ((layer_id < 1) || (layer_id > 6)) {
    std::cout << "CSCNumberingScheme: ";
    std::cout << "layer id out of range: ";
    std::cout << layer_id <<std::endl;
  }
    
  // find appropriate chamber label
    
  int chamber_id=chamberIndex(station_id, ring_id,
			      subring_id, sector_id);
    
  // convert into raw id of appropriate DetId
    
  int intIndex=CSCDetId::rawIdMaker(fwbw_id, station_id, ring_id,
		     chamber_id, layer_id);

#ifdef LOCAL_DEBUG
  std::cout << "CSCNumberingScheme : ";
  std::cout << " fw/bw " <<  fwbw_id;
  std::cout << " station " <<  station_id;
  std::cout << " ring " <<  ring_id;
  std::cout << " subring " <<  subring_id;
  std::cout << " chamber " <<  chamber_id;
  std::cout << " sector " <<  sector_id;
  std::cout << " layer " <<  layer_id;
  std::cout << std::endl;
#endif

  return intIndex;
}

int CSCNumberingScheme::chamberIndex(int station_id,  
           int ring_id, int subring_id, int sector_id) const {

  //@@ FIXME Nov-2005 THIS NO LONGER MATCHES HARDWARE
  //@@ CHAMBER ZERO IS ROTATED BY 20 DEG (I THINK!)

  int chamber_id=0;

  // chamber label is related to sector_id but we need to
  // adjust to real hardware labelling

  if (ring_id == 3) {
    chamber_id=sector_id;
  } else {
    if (subring_id == 1) {
      chamber_id=2*sector_id-1;
    } else { 
      chamber_id=2*sector_id;
    }
  }

//   if (station_id == 1) {
//     if (ring_id == 3) {
//       chamber_id--;
//       if (chamber_id < 1) chamber_id=36;
//     }
//   } else {
//     if (ring_id == 1) {
//       chamber_id++;
//       if (chamber_id > 18) chamber_id=1;
//     }
//   }

  return chamber_id;

}
