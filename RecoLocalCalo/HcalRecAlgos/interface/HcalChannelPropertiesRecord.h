#ifndef RecoLocalCalo_HcalRecAlgos_HcalChannelPropertiesRecord_h_
#define RecoLocalCalo_HcalRecAlgos_HcalChannelPropertiesRecord_h_

#include "FWCore/Framework/interface/DependentRecordImplementation.h"

class HcalDbRecord;
class CaloGeometryRecord;
class HcalSeverityLevelComputerRcd;
class HcalChannelPropertiesAuxRecord;

class HcalChannelPropertiesRecord
    : public edm::eventsetup::DependentRecordImplementation<
          HcalChannelPropertiesRecord,
          edm::mpl::
              Vector<CaloGeometryRecord, HcalDbRecord, HcalSeverityLevelComputerRcd, HcalChannelPropertiesAuxRecord> > {
};

#endif  // RecoLocalCalo_HcalRecAlgos_HcalChannelPropertiesRecord_h_
