#ifndef RecoLocalCalo_HcalRecAlgos_HcalChannelPropertiesRecord_h_
#define RecoLocalCalo_HcalRecAlgos_HcalChannelPropertiesRecord_h_

#include "boost/mpl/vector.hpp"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

class HcalDbRecord;
class CaloGeometryRecord;
class HcalSeverityLevelComputerRcd;

class HcalChannelPropertiesRecord
    : public edm::eventsetup::DependentRecordImplementation<
          HcalChannelPropertiesRecord,
          boost::mpl::vector<CaloGeometryRecord, HcalDbRecord, HcalSeverityLevelComputerRcd> > {};

#endif  // RecoLocalCalo_HcalRecAlgos_HcalChannelPropertiesRecord_h_
