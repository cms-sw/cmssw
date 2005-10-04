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
// $Id$
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"

//
// class decleration
//

class HcalHardcodeGeometryEP : public edm::eventsetup::ESProducer {
   public:
      HcalHardcodeGeometryEP(const edm::ParameterSet&);
      ~HcalHardcodeGeometryEP();

      typedef std::auto_ptr<CaloSubdetectorGeometry> ReturnType;

      ReturnType produce(const IdealGeometryRecord&);
private:
      // ----------member data ---------------------------
  HcalHardcodeGeometryLoader* loader_;
};

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
