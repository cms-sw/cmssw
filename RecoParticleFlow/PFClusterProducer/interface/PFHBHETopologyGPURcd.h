// RecoParticleFlow/PFClusterProducer/interface/HBHETopologyGPURcd.h 
#ifndef RecoParticleFlow_PFClusterProducer_PFHBHETopologyGPURcd_h
#define RecoParticleFlow_PFClusterProducer_PFHBHETopologyGPURcd_h

//#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTRcd.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

class PFHBHETopologyGPURcd
//: public edm::eventsetup::EventSetupRecordImplementation<HBHETopologyGPURcd> {};
: public edm::eventsetup::DependentRecordImplementation<PFHBHETopologyGPURcd,
edm::mpl::Vector<HcalRecNumberingRecord, CaloGeometryRecord>> {};

#endif  // RecoParticleFlow_PFClusterProducer_PFHBHETopologyGPURcd_h
