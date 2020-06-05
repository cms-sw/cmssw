#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalCombinedRecord_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalCombinedRecord_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"

template <typename... Sources>
class HcalCombinedRecord : public edm::eventsetup::DependentRecordImplementation<HcalCombinedRecord<Sources...>,
                                                                                 boost::mpl::vector<Sources...>> {
public:
  using DependencyRecords = std::tuple<Sources...>;
};

#endif  // RecoLocalCalo_HcalRecAlgos_interface_HcalCombinedRecord_h
