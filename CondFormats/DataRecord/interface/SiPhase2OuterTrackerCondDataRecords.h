#ifndef CondFormats_SiPhase2OuterTrackerCondDataRecords_h
#define CondFormats_SiPhase2OuterTrackerCondDataRecords_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

/*Record associated to SiPhase2OuterTrackerLorentzAngle Object: the SimRcd is used in simulation only*/
class SiPhase2OuterTrackerLorentzAngleRcd
    : public edm::eventsetup::DependentRecordImplementation<SiPhase2OuterTrackerLorentzAngleRcd,
                                                            edm::mpl::Vector<TrackerTopologyRcd> > {};
class SiPhase2OuterTrackerLorentzAngleSimRcd
    : public edm::eventsetup::DependentRecordImplementation<SiPhase2OuterTrackerLorentzAngleSimRcd,
                                                            edm::mpl::Vector<TrackerTopologyRcd> > {};
#endif
