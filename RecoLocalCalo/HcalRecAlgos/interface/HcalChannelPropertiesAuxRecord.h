#ifndef RecoLocalCalo_HcalRecAlgos_HcalChannelPropertiesAuxRecord_h_
#define RecoLocalCalo_HcalRecAlgos_HcalChannelPropertiesAuxRecord_h_

#include "boost/mpl/vector.hpp"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

class HcalRecNumberingRecord;
class HcalRecoParamsRcd;

class HcalChannelPropertiesAuxRecord : public edm::eventsetup::DependentRecordImplementation<
                                           HcalChannelPropertiesAuxRecord,
                                           boost::mpl::vector<HcalRecNumberingRecord, HcalRecoParamsRcd> > {};

#endif  // RecoLocalCalo_HcalRecAlgos_HcalChannelPropertiesAuxRecord_h_
