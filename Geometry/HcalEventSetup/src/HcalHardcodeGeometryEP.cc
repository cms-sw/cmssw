// -*- C++ -*-
//
// Package:    HcalHardcodeGeometryEP
// Class:      HcalHardcodeGeometryEP
//
/**\class HcalHardcodeGeometryEP HcalHardcodeGeometryEP.h tmp/HcalHardcodeGeometryEP/interface/HcalHardcodeGeometryEP.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
//
//

#include "Geometry/HcalEventSetup/interface/HcalHardcodeGeometryEP.h"

#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/HcalGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalTowerAlgo/interface/HcalFlexiHardcodeGeometryLoader.h"
#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class HcalTopology;

HcalHardcodeGeometryEP::HcalHardcodeGeometryEP(const edm::ParameterSet& ps) {
  useOld_ = ps.getParameter<bool>("UseOldLoader");
  //the following line is needed to tell the framework what
  // data is being produced
  auto cc = setWhatProduced(this, &HcalHardcodeGeometryEP::produceAligned, edm::es::Label(HcalGeometry::producerTag()));
  if (not useOld_) {
    consToken_ = cc.consumesFrom<HcalDDDRecConstants, HcalRecNumberingRecord>(edm::ESInputTag{});
  }
  topologyToken_ = cc.consumesFrom<HcalTopology, HcalRecNumberingRecord>(edm::ESInputTag{});
}

HcalHardcodeGeometryEP::ReturnType HcalHardcodeGeometryEP::produceAligned(const HcalGeometryRecord& iRecord) {
  edm::LogInfo("HCAL") << "Using default HCAL topology";
  const auto& topology = iRecord.get(topologyToken_);
  if (useOld_) {
    HcalHardcodeGeometryLoader loader;
    return ReturnType(loader.load(topology));
  } else {
    const auto& cons = iRecord.get(consToken_);
    HcalFlexiHardcodeGeometryLoader loader;
    return ReturnType(loader.load(topology, cons));
  }
}
