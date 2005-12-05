/** 
$Author: ratnikov
$Date: 2005/10/20 05:18:37 $
$Revision: 1.2 $
*/

#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalMapping/interface/HcalMapping.h"
#include <iostream>

const HcalDetId HcalMapping::lookup(HcalElectronicsId el) const {
  return HcalDetId (mMap->eId2chId (el (), true));
}

const HcalElectronicsId HcalMapping::lookup(HcalDetId id) const {
  return HcalElectronicsId (mMap->chId2eId (id (), false));
}

const HcalTrigTowerDetId HcalMapping::lookupTrigger(HcalElectronicsId el) const {
  return HcalTrigTowerDetId (mMap->eId2tId (el (), true));
}

const HcalElectronicsId HcalMapping::lookupTrigger(HcalTrigTowerDetId id) const {
  return HcalElectronicsId (mMap->tId2eId (id (), false));
}

bool HcalMapping::subdetectorPresent(HcalSubdetector det, int dccid) const {
  std::cerr << "HcalMapping::subdetectorPresent-> not efficient implementation. Get rid of using this function" << std::endl;
  std::vector <unsigned long> allChannels = mMap->allElectronicsId ();
  int i = allChannels.size ();
  while (--i >= 0) {
    HcalElectronicsId eId (allChannels [i]);
    if (eId.dccid () == dccid) {
      HcalDetId id (mMap->eId2chId (eId (), true));
      if (id.subdet() == det) return true;
    }
  }
  return false;
}

std::vector <HcalElectronicsId> HcalMapping::allElectronicsId () const {
  std::vector <HcalElectronicsId> result;
  std::vector <unsigned long> ids = mMap->allElectronicsId ();
  for (unsigned i = 0; i < ids.size (); i++) result.push_back (HcalElectronicsId (ids [i]));
  return result;
}

std::vector <HcalDetId> HcalMapping::allDetectorId () const {
  std::vector <HcalDetId> result;
  std::vector <unsigned long> ids = mMap->allDetectorId ();
  for (unsigned i = 0; i < ids.size (); i++) result.push_back (HcalDetId (ids [i]));
  return result;
}

std::vector <HcalTrigTowerDetId> HcalMapping::allTriggerId () const {
  std::vector <HcalTrigTowerDetId> result;
  std::vector <unsigned long> ids = mMap->allTriggerId ();
  for (unsigned i = 0; i < ids.size (); i++) result.push_back (HcalTrigTowerDetId (ids [i]));
  return result;
}
