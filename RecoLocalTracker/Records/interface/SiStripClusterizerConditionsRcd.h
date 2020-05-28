#ifndef RecoLocalTracker_Records_SiStripClusterizerConditionsRcd_h
#define RecoLocalTracker_Records_SiStripClusterizerConditionsRcd_h
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "boost/mpl/vector.hpp"

#include "CalibTracker/Records/interface/SiStripDependentRecords.h"

class SiStripClusterizerConditionsRcd : public edm::eventsetup::DependentRecordImplementation<
                                            SiStripClusterizerConditionsRcd,
                                            boost::mpl::vector<SiStripGainRcd, SiStripNoisesRcd, SiStripQualityRcd>> {};

#endif  // RecoLocalTracker_Records_SiStripClusterizerConditionsRcd_h
