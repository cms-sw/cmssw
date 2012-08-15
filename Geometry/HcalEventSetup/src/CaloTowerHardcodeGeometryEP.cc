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
// $Id: CaloTowerHardcodeGeometryEP.cc,v 1.4 2011/09/27 12:06:26 yana Exp $
//
//

#include "Geometry/HcalEventSetup/src/CaloTowerHardcodeGeometryEP.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"

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
    : m_hcalTopoConsts( iConfig.getParameter<edm::ParameterSet>( "hcalTopologyConstants" ))
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
CaloTowerHardcodeGeometryEP::produce(const CaloTowerGeometryRecord& /*iRecord*/)
{
   std::auto_ptr<CaloSubdetectorGeometry> pCaloSubdetectorGeometry(loader_->load( new HcalTopology((HcalTopology::Mode) StringToEnumValue<HcalTopology::Mode>(m_hcalTopoConsts.getParameter<std::string>("mode")),
												   m_hcalTopoConsts.getParameter<int>("maxDepthHB"),
												   m_hcalTopoConsts.getParameter<int>("maxDepthHE")))) ;

   return pCaloSubdetectorGeometry ;
}


