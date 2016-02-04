//
// Toyoko Orimoto (Caltech), 10 July 2007
//

#ifndef ECALLASERCORRECTION_ECALLASERDBRECORD_H
#define ECALLASERCORRECTION_ECALLASERDBRECORD_H

#include "boost/mpl/vector.hpp"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
// #include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"

// class EcalLaserDbRecord : public edm::eventsetup::EventSetupRecordImplementation<EcalLaserDbRecord> {};

class EcalLaserDbRecord : public edm::eventsetup::DependentRecordImplementation <EcalLaserDbRecord,  
  boost::mpl::vector<EcalLaserAlphasRcd, EcalLaserAPDPNRatiosRefRcd, EcalLaserAPDPNRatiosRcd> > {}; 

#endif /* ECALLASERCORRECTION_ECALLASERDBRECORD_H */

