#ifndef HeterogeneousCore_AlpakaTest_interface_AlpakaTestRecords_h
#define HeterogeneousCore_AlpakaTest_interface_AlpakaTestRecords_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

class AlpakaESTestRecordA : public edm::eventsetup::EventSetupRecordImplementation<AlpakaESTestRecordA> {};

class AlpakaESTestRecordB : public edm::eventsetup::EventSetupRecordImplementation<AlpakaESTestRecordB> {};

class AlpakaESTestRecordC : public edm::eventsetup::EventSetupRecordImplementation<AlpakaESTestRecordC> {};

class AlpakaESTestRecordD
    : public edm::eventsetup::DependentRecordImplementation<AlpakaESTestRecordD,
                                                            edm::mpl::Vector<AlpakaESTestRecordA, AlpakaESTestRecordB>> {
};

#endif
