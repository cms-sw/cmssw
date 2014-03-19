#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cassert>
#include <math.h>

std::ostream& operator<<(std::ostream& s, const EcalRecHit& hit) {
  if (hit.detid().det() == DetId::Ecal && hit.detid().subdetId() == EcalBarrel) 
    return s << EBDetId(hit.detid()) << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  else if (hit.detid().det() == DetId::Ecal && hit.detid().subdetId() == EcalEndcap) 
    return s << EEDetId(hit.detid()) << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  else if (hit.detid().det() == DetId::Ecal && hit.detid().subdetId() == EcalPreshower) 
    return s << ESDetId(hit.detid()) << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  else
    return s << "EcalRecHit undefined subdetector" ;
}


