/* -*- C++ -*- */
#ifndef HcalMapping_h_included
#define HcalMapping_h_included 1

#include <vector>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"

class HcalElectronicsMap;


/** \class HcalMapping
 *  Map between electronics id and logical ids for use in Hcal Unpacking
 *
 *  $Date: 2005/10/10 14:28:17 $
 *  $Revision: 1.3 $
 *  $Author: J. Mans - Minnesota
 */
class HcalMapping {
public:
  HcalMapping(const HcalElectronicsMap* fMap) : mMap (fMap) {}

  /** \brief lookup the logical detid associated with the given electronics id
      \return Null item if no such mapping
  */
  const HcalDetId lookup(HcalElectronicsId) const;
  /** \brief lookup the electronics detid associated with the given logical id
      \return Null item if no such mapping
  */
  const HcalElectronicsId lookup(HcalDetId) const;
  /** \brief lookup the trigger logical detid associated with the given electronics id
      \return Null item if no such mapping
  */
  const HcalTrigTowerDetId lookupTrigger(HcalElectronicsId) const;
  /** \brief lookup the electronics detid associated with the given trigger logical id
      \return Null item if no such mapping
  */
  const HcalElectronicsId lookupTrigger(HcalTrigTowerDetId) const;
/** \brief Test if this subdetector is present in this dccid */
  bool subdetectorPresent(HcalSubdetector det, int dccid) const;
  /** \brief vector of all available IDs */
  std::vector <HcalElectronicsId> allElectronicsId () const;
  /** \brief vector of all available IDs */
  std::vector <HcalDetId> allDetectorId () const;
  /** \brief vector of all available IDs */
  std::vector <HcalTrigTowerDetId> allTriggerId () const;

private:
  const HcalElectronicsMap* mMap;
};

#endif // HcalMapping_h_included
