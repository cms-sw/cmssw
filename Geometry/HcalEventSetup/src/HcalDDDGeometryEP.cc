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
//#define DebugLog
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

HcalDDDGeometryEP::HcalDDDGeometryEP(const edm::ParameterSet& ps ) :
  m_loader ( nullptr ) ,
  m_applyAlignment(ps.getUntrackedParameter<bool>("applyAlignment", false) ) {

  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced( this,
		   &HcalDDDGeometryEP::produceAligned,
		   dependsOn( &HcalDDDGeometryEP::idealRecordCallBack ),
		   "HCAL");
}

HcalDDDGeometryEP::~HcalDDDGeometryEP() { 
  if (m_loader) delete m_loader ;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
HcalDDDGeometryEP::ReturnType
HcalDDDGeometryEP::produceIdeal(const HcalRecNumberingRecord& iRecord) {

  edm::LogInfo("HCAL") << "Using default HCAL topology" ;
  edm::ESHandle<HcalDDDRecConstants> hcons;
  iRecord.get( hcons ) ;

  edm::ESHandle<HcalTopology> topology ;
  iRecord.get( topology ) ;

  assert( nullptr == m_loader ) ;
  m_loader = new HcalDDDGeometryLoader(&(*hcons)); 
#ifdef DebugLog
  LogDebug("HCalGeom")<<"HcalDDDGeometryEP:Initialize HcalDDDGeometryLoader";
#endif
  ReturnType pC ( m_loader->load(*topology) ) ;

  return pC;
}

HcalDDDGeometryEP::ReturnType
HcalDDDGeometryEP::produceAligned(const HcalGeometryRecord& iRecord) {

  const HcalRecNumberingRecord& idealRecord = iRecord.getRecord<HcalRecNumberingRecord>();
  return produceIdeal (idealRecord);
}
