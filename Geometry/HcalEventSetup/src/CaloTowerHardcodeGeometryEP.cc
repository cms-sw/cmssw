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
   setWhatProduced(this,
                   &CaloTowerHardcodeGeometryEP::produce,
		   edm::es::Label("TOWER"));

  loader_=new CaloTowerHardcodeGeometryLoader(); /// TODO : allow override of Topology.
}

CaloTowerHardcodeGeometryEP::~CaloTowerHardcodeGeometryEP() {
  delete loader_;
}

// ------------ method called to produce the data  ------------
CaloTowerHardcodeGeometryEP::ReturnType
CaloTowerHardcodeGeometryEP::produce(const CaloTowerGeometryRecord& iRecord) {
  edm::ESHandle<CaloTowerTopology> cttopo;
  iRecord.getRecord<HcalRecNumberingRecord>().get( cttopo );
  edm::ESHandle<HcalTopology> hcaltopo;
  iRecord.getRecord<HcalRecNumberingRecord>().get( hcaltopo );
  edm::ESHandle<HcalDDDRecConstants> pHRNDC;
  iRecord.getRecord<HcalRecNumberingRecord>().get( pHRNDC );

  return std::unique_ptr<CaloSubdetectorGeometry>( loader_->load( &*cttopo, &*hcaltopo, &*pHRNDC ));
}
