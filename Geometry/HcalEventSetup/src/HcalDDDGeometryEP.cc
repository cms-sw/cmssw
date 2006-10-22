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
// $Id: HcalDDDGeometryEP.cc,v 1.0 2006/10/20 18:24:43 sunanda Exp $
//
//

#include "Geometry/HcalEventSetup/interface/HcalDDDGeometryEP.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HcalDDDGeometryEP::HcalDDDGeometryEP(const edm::ParameterSet& ) : loader_(0) {

  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this,"HCAL");
}


HcalDDDGeometryEP::~HcalDDDGeometryEP() { 
  if (loader_) delete loader_;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
HcalDDDGeometryEP::ReturnType
HcalDDDGeometryEP::produce(const IdealGeometryRecord& iRecord) {

  //now do what ever other initialization is needed
  if (loader_==0) {
    edm::ESHandle<DDCompactView> pDD;
    iRecord.get(pDD);
    loader_= new HcalDDDGeometryLoader(*pDD); 
    LogDebug("HCalGeom")<<"HcalDDDGeometryEP:Initialize HcalDDDGeometryLoader";
  }

  std::auto_ptr<CaloSubdetectorGeometry> pCaloSubdetectorGeometry(loader_->load());

  return pCaloSubdetectorGeometry ;
}


