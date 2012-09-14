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
// $Id: CaloTowerHardcodeGeometryEP.cc,v 1.5 2012/08/15 15:00:40 yana Exp $
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
    : m_pSet( iConfig )
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
  HcalTopologyMode::Mode mode = HcalTopologyMode::LHC;
  int maxDepthHB = 2;
  int maxDepthHE = 3;
  if( m_pSet.exists( "hcalTopologyConstants" ))
  {
    const edm::ParameterSet hcalTopoConsts( m_pSet.getParameter<edm::ParameterSet>( "hcalTopologyConstants" ));
    StringToEnumParser<HcalTopologyMode::Mode> parser;
    mode = (HcalTopologyMode::Mode) parser.parseString(hcalTopoConsts.getParameter<std::string>("mode"));
    maxDepthHB = hcalTopoConsts.getParameter<int>("maxDepthHB");
    maxDepthHE = hcalTopoConsts.getParameter<int>("maxDepthHE");
  }
    
  std::auto_ptr<CaloSubdetectorGeometry> pCaloSubdetectorGeometry( loader_->load( new HcalTopology( mode, maxDepthHB, maxDepthHE )));

  return pCaloSubdetectorGeometry ;
}


