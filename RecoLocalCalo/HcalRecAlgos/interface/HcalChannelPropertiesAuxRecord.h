#ifndef RecoLocalCalo_HcalRecAlgos_HcalChannelPropertiesAuxRecord_h_
#define RecoLocalCalo_HcalRecAlgos_HcalChannelPropertiesAuxRecord_h_

#include "FWCore/Utilities/interface/mplVector.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

class HcalRecNumberingRecord;
class HcalRecoParamsRcd;

class HcalChannelPropertiesAuxRecord : public edm::eventsetup::DependentRecordImplementation<
                                           HcalChannelPropertiesAuxRecord,
                                           edm::mpl::Vector<HcalRecNumberingRecord, HcalRecoParamsRcd> > {};

#endif  // RecoLocalCalo_HcalRecAlgos_HcalChannelPropertiesAuxRecord_h_
