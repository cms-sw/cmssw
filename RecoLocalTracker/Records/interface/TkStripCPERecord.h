#ifndef RecoLocalTracker_Records_TkStripCPERecord_h
#define RecoLocalTracker_Records_TkStripCPERecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include <boost/mp11/list.hpp>

class TkStripCPERecord
    : public edm::eventsetup::DependentRecordImplementation<TkStripCPERecord,
                                                            boost::mp11::mp_list<TrackerDigiGeometryRecord,
                                                                                 IdealMagneticFieldRecord,
                                                                                 SiStripLorentzAngleDepRcd,
                                                                                 SiStripBackPlaneCorrectionDepRcd,
                                                                                 SiStripConfObjectRcd,
                                                                                 SiStripLatencyRcd,
                                                                                 SiStripNoisesRcd,
                                                                                 SiStripApvGainRcd,
                                                                                 SiStripBadChannelRcd> > {};

#endif
