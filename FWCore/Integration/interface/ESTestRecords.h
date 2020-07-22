#ifndef FWCore_Integration_ESTestRecords_h
#define FWCore_Integration_ESTestRecords_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

class ESTestRecordA : public edm::eventsetup::EventSetupRecordImplementation<ESTestRecordA> {
public:
  static constexpr bool allowConcurrentIOVs_ = false;
};

class ESTestRecordC : public edm::eventsetup::EventSetupRecordImplementation<ESTestRecordC> {};

class ESTestRecordF : public edm::eventsetup::EventSetupRecordImplementation<ESTestRecordF> {};

class ESTestRecordG : public edm::eventsetup::EventSetupRecordImplementation<ESTestRecordG> {};

class ESTestRecordH : public edm::eventsetup::EventSetupRecordImplementation<ESTestRecordH> {};

class ESTestRecordE : public edm::eventsetup::EventSetupRecordImplementation<ESTestRecordE> {};

class ESTestRecordD
    : public edm::eventsetup::
          DependentRecordImplementation<ESTestRecordD, edm::mpl::Vector<ESTestRecordF, ESTestRecordG, ESTestRecordH> > {
};

class ESTestRecordB
    : public edm::eventsetup::
          DependentRecordImplementation<ESTestRecordB, edm::mpl::Vector<ESTestRecordC, ESTestRecordD, ESTestRecordE> > {
};

class ESTestRecordZ : public edm::eventsetup::EventSetupRecordImplementation<ESTestRecordZ> {};

class ESTestRecordK : public edm::eventsetup::EventSetupRecordImplementation<ESTestRecordK> {};

class ESTestRecordI
    : public edm::eventsetup::DependentRecordImplementation<ESTestRecordI, edm::mpl::Vector<ESTestRecordK> > {};

class ESTestRecordJ
    : public edm::eventsetup::DependentRecordImplementation<ESTestRecordJ, edm::mpl::Vector<ESTestRecordK> > {};

#endif
