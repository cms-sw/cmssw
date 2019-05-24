// -*- C++ -*-
//
// Package:    HcalDDDGeometryEP
// Class:      HcalDDDGeometryEP
//
/**\class HcalDDDGeometryEP HcalDDDGeometryEP.h tmp/HcalDDDGeometryEP/interface/HcalDDDGeometryEP.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Thu Oct 20 11:35:27 CDT 2006
//
//

#include "Geometry/HcalEventSetup/interface/HcalDDDGeometryEP.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalDDDGeometryEP::HcalDDDGeometryEP(const edm::ParameterSet& ps) {
  //the following line is needed to tell the framework what
  // data is being produced
  auto cc = setWhatProduced(this, &HcalDDDGeometryEP::produceAligned, edm::es::Label("HCAL"));
  consToken_ = cc.consumesFrom<HcalDDDRecConstants, HcalRecNumberingRecord>(edm::ESInputTag{});
  topologyToken_ = cc.consumesFrom<HcalTopology, HcalRecNumberingRecord>(edm::ESInputTag{});
}

// ------------ method called to produce the data  ------------
HcalDDDGeometryEP::ReturnType HcalDDDGeometryEP::produceAligned(const HcalGeometryRecord& iRecord) {
  edm::LogInfo("HCAL") << "Using default HCAL topology";
  const auto& cons = iRecord.get(consToken_);
  const auto& topology = iRecord.get(topologyToken_);

  HcalDDDGeometryLoader loader(&cons);

  return ReturnType(loader.load(topology));
}
