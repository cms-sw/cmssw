#ifndef FastSimulation_CalorimeterProperties_CalorimetryConsumer_h
#define FastSimulation_CalorimeterProperties_CalorimetryConsumer_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/HcalCommonData/interface/HcalSimulationConstants.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

struct CalorimetryConsumer {
  CalorimetryConsumer(edm::ConsumesCollector&& iC);

  edm::ESGetToken<HepPDT::ParticleDataTable, edm::DefaultRecord> particleDataTableESToken;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryESToken;
  edm::ESGetToken<CaloTopology, CaloTopologyRecord> caloTopologyESToken;
  edm::ESGetToken<HcalDDDSimConstants, HcalSimNumberingRecord> hcalDDDSimConstantsESToken;
  edm::ESGetToken<HcalSimulationConstants, HcalSimNumberingRecord> hcalSimulationConstantsESToken;
};

#endif
