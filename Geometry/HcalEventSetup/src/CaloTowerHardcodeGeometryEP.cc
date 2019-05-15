// -*- C++ -*-
//
// Package:    CaloTowerHardcodeGeometryEP
// Class:      CaloTowerHardcodeGeometryEP
//
/**\class CaloTowerHardcodeGeometryEP CaloTowerHardcodeGeometryEP.h tmp/CaloTowerHardcodeGeometryEP/interface/CaloTowerHardcodeGeometryEP.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
//
//

#include "Geometry/HcalEventSetup/src/CaloTowerHardcodeGeometryEP.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"

CaloTowerHardcodeGeometryEP::CaloTowerHardcodeGeometryEP(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
  auto cc = setWhatProduced(this,
                            &CaloTowerHardcodeGeometryEP::produce,
                            edm::es::Label("TOWER"));
  cttopoToken_ = cc.consumesFrom<CaloTowerTopology, HcalRecNumberingRecord>(edm::ESInputTag{});
  hcaltopoToken_ = cc.consumesFrom<HcalTopology, HcalRecNumberingRecord>(edm::ESInputTag{});
  consToken_ = cc.consumesFrom<HcalDDDRecConstants, HcalRecNumberingRecord>(edm::ESInputTag{});
}

// ------------ method called to produce the data  ------------
CaloTowerHardcodeGeometryEP::ReturnType
CaloTowerHardcodeGeometryEP::produce(const CaloTowerGeometryRecord& iRecord) {
  const auto& cttopo = iRecord.get(cttopoToken_);
  const auto& hcaltopo = iRecord.get(hcaltopoToken_);
  const auto& cons = iRecord.get(consToken_);

  return std::unique_ptr<CaloSubdetectorGeometry>( loader_.load( &cttopo, &hcaltopo, &cons ));
}
