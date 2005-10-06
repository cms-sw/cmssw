// -*- C++ -*-
//
// Package:    HcalHardcodeGeometryEP
// Class:      HcalHardcodeGeometryEP
// 
/**\class HcalHardcodeGeometryEP HcalHardcodeGeometryEP.h tmp/HcalHardcodeGeometryEP/interface/HcalHardcodeGeometryEP.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
// $Id: HcalHardcodeGeometryEP.cc,v 1.2 2005/10/04 17:46:29 mansj Exp $
//
//

#include "Geometry/HcalEventSetup/src/HcalHardcodeGeometryEP.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HcalHardcodeGeometryEP::HcalHardcodeGeometryEP(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this,"HCAL");

   //now do what ever other initialization is needed
   loader_=new HcalHardcodeGeometryLoader(); /// TODO : allow override of Topology.
}


HcalHardcodeGeometryEP::~HcalHardcodeGeometryEP()
{ 
  delete loader_;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
HcalHardcodeGeometryEP::ReturnType
HcalHardcodeGeometryEP::produce(const IdealGeometryRecord& iRecord)
{
   using namespace edm::es;
   std::auto_ptr<CaloSubdetectorGeometry> pCaloSubdetectorGeometry(loader_->load()) ;

   return pCaloSubdetectorGeometry ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalHardcodeGeometryEP)
