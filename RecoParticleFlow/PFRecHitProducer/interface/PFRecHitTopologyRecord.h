#ifndef RecoParticleFlow_PFRecHitProducer_interface_PFRecHitTopologyRecord_h
#define RecoParticleFlow_PFRecHitProducer_interface_PFRecHitTopologyRecord_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class PFRecHitHCALTopologyRecord : public edm::eventsetup::DependentRecordImplementation<
                                       PFRecHitHCALTopologyRecord,
                                       edm::mpl::Vector<HcalRecNumberingRecord, CaloGeometryRecord>> {};

class PFRecHitECALTopologyRecord
    : public edm::eventsetup::DependentRecordImplementation<PFRecHitECALTopologyRecord,
                                                            edm::mpl::Vector<CaloGeometryRecord>> {};

#endif  // RecoParticleFlow_PFRecHitProducer_interface_PFRecHitTopologyRecord_h
