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
// $Id: CaloTowerHardcodeGeometryEP.cc,v 1.6 2012/08/27 15:20:41 yana Exp $
//
//

#include "Geometry/HcalEventSetup/src/CaloTowerHardcodeGeometryEP.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CaloTowerHardcodeGeometryEP::CaloTowerHardcodeGeometryEP(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this,"TOWER");

   //now do what ever other initialization is needed
   loader_=new CaloTowerHardcodeGeometryLoader(); /// TODO : allow override of Topology.
}


CaloTowerHardcodeGeometryEP::~CaloTowerHardcodeGeometryEP()
{ 
  delete loader_;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CaloTowerHardcodeGeometryEP::ReturnType
CaloTowerHardcodeGeometryEP::produce(const CaloTowerGeometryRecord& iRecord)
{
  edm::ESHandle<HcalTopology> hcalTopology;
  iRecord.getRecord<IdealGeometryRecord>().get( hcalTopology );
  
  std::auto_ptr<CaloSubdetectorGeometry> pCaloSubdetectorGeometry( loader_->load( &*hcalTopology ));

  return pCaloSubdetectorGeometry ;
}


