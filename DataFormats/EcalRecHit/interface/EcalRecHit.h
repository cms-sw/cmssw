#ifndef DATAFORMATS_ECALRECHIT_H
#define DATAFORMATS_ECALRECHIT_H 1

#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

/** \class EcalRecHit
 *  
 * $id: $
 * \author P. Meridiani INFN Roma1
 */

class EcalRecHit : public CaloRecHit {
public:
  typedef DetId key_type;

  static const int kRECOVERED = 9999;

  EcalRecHit();
  EcalRecHit(const DetId& id, float energy, float time);
  /// get the id
  // For the moment not returning a specific id for subdetector
  DetId id() const { return DetId(detid());}
  bool isRecovered() const;
};

std::ostream& operator<<(std::ostream& s, const EcalRecHit& hit);

#endif
