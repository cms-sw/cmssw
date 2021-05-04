#ifndef RecoLocalTracker_Records_SiStripClusterizerConditionsRcd_h
#define RecoLocalTracker_Records_SiStripClusterizerConditionsRcd_h
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"

#include "CalibTracker/Records/interface/SiStripDependentRecords.h"

class SiStripClusterizerConditionsRcd : public edm::eventsetup::DependentRecordImplementation<
                                            SiStripClusterizerConditionsRcd,
                                            edm::mpl::Vector<SiStripGainRcd, SiStripNoisesRcd, SiStripQualityRcd>> {};

#endif  // RecoLocalTracker_Records_SiStripClusterizerConditionsRcd_h
