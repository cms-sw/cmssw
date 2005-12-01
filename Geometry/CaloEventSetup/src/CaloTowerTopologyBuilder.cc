// -*- C++ -*-
//
// Package:    CaloTowerTopologyBuilder
// Class:      CaloTowerTopologyBuilder
// 
/**\class CaloTowerTopologyBuilder CaloTowerTopologyBuilder.h tmp/CaloTowerTopologyBuilder/interface/CaloTowerTopologyBuilder.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
// $Id: CaloTowerTopologyBuilder.cc,v 1.4 2005/11/02 07:55:24 meridian Exp $
//
//


// user include files
#include "Geometry/CaloEventSetup/src/CaloTowerTopologyBuilder.h"

//
// constructors and destructor
//
CaloTowerTopologyBuilder::CaloTowerTopologyBuilder(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


CaloTowerTopologyBuilder::~CaloTowerTopologyBuilder()
{ 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CaloTowerTopologyBuilder::ReturnType
CaloTowerTopologyBuilder::produce(const IdealGeometryRecord& iRecord)
{
   using namespace edm::es;
   std::auto_ptr<CaloTowerTopology> prod(new CaloTowerTopology());

   
   return prod;
}
