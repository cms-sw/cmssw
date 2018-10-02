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

HcalDDDGeometryEP::HcalDDDGeometryEP(const edm::ParameterSet& ps ) :
  m_applyAlignment(ps.getUntrackedParameter<bool>("applyAlignment", false) ) {

  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced( this,
		   &HcalDDDGeometryEP::produceAligned,
		   edm::es::Label("HCAL"));
}

// ------------ method called to produce the data  ------------
HcalDDDGeometryEP::ReturnType
HcalDDDGeometryEP::produceIdeal(const HcalRecNumberingRecord& iRecord) {

  edm::LogInfo("HCAL") << "Using default HCAL topology" ;
  edm::ESHandle<HcalDDDRecConstants> hcons;
  iRecord.get( hcons ) ;

  edm::ESHandle<HcalTopology> topology ;
  iRecord.get( topology ) ;

  HcalDDDGeometryLoader loader(&(*hcons));

  return ReturnType(loader.load(*topology));
}

HcalDDDGeometryEP::ReturnType
HcalDDDGeometryEP::produceAligned(const HcalGeometryRecord& iRecord) {

  const HcalRecNumberingRecord& idealRecord = iRecord.getRecord<HcalRecNumberingRecord>();
  return produceIdeal (idealRecord);
}
