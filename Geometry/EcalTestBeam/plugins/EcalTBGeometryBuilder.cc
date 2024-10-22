// -*- C++ -*-
//
// Package:    EcalTBGeometryBuilder
// Class:      EcalTBGeometryBuilder
//
/**\class EcalTBGeometryBuilder EcalTBGeometryBuilder.h tmp/EcalTBGeometryBuilder/interface/EcalTBGeometryBuilder.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
//
//

// user include files
#include "Geometry/EcalTestBeam/plugins/EcalTBGeometryBuilder.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
EcalTBGeometryBuilder::EcalTBGeometryBuilder(const edm::ParameterSet& iConfig) {
  //the following line is needed to tell the framework what
  // data is being produced
  auto cc = setWhatProduced(this);

  barrelToken_ = cc.consumes<CaloSubdetectorGeometry>(edm::ESInputTag{"", "EcalBarrel"});
  hodoscopeToken_ = cc.consumes<CaloSubdetectorGeometry>(edm::ESInputTag{"", "EcalLaserPnDiode"});

  //now do what ever other initialization is needed
}

EcalTBGeometryBuilder::~EcalTBGeometryBuilder() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
EcalTBGeometryBuilder::ReturnType EcalTBGeometryBuilder::produce(const IdealGeometryRecord& iRecord) {
  auto pCaloGeom = std::make_unique<CaloGeometry>();

  // TODO: Look for ECAL parts
  if (auto pG = iRecord.getHandle(barrelToken_)) {
    pCaloGeom->setSubdetGeometry(DetId::Ecal, EcalBarrel, pG.product());
  } else {
    edm::LogWarning("MissingInput") << "No Ecal Barrel Geometry found";
  }
  if (auto pG = iRecord.getHandle(hodoscopeToken_)) {
    pCaloGeom->setSubdetGeometry(DetId::Ecal, EcalLaserPnDiode, pG.product());
  } else {
    edm::LogWarning("MissingInput") << "No Ecal TB Hodoscope Geometry found";
  }

  return pCaloGeom;
}
