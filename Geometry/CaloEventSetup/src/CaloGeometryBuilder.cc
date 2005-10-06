// -*- C++ -*-
//
// Package:    CaloGeometryBuilder
// Class:      CaloGeometryBuilder
// 
/**\class CaloGeometryBuilder CaloGeometryBuilder.h tmp/CaloGeometryBuilder/interface/CaloGeometryBuilder.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
// $Id: CaloGeometryBuilder.cc,v 1.2 2005/10/04 17:46:29 mansj Exp $
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
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

//
// class decleration
//

class CaloGeometryBuilder : public edm::ESProducer {
   public:
  CaloGeometryBuilder(const edm::ParameterSet&);
  ~CaloGeometryBuilder();

  typedef std::auto_ptr<CaloGeometry> ReturnType;

  ReturnType produce(const IdealGeometryRecord&);
private:
      // ----------member data ---------------------------
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
CaloGeometryBuilder::CaloGeometryBuilder(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


CaloGeometryBuilder::~CaloGeometryBuilder()
{ 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CaloGeometryBuilder::ReturnType
CaloGeometryBuilder::produce(const IdealGeometryRecord& iRecord)
{
   using namespace edm::es;
   edm::ESHandle<CaloSubdetectorGeometry> pG;

   std::auto_ptr<CaloGeometry> pCaloGeom(new CaloGeometry());

   // look for HCAL parts

   // assume 'HCAL' for all of HCAL.  
   // TODO: Eventually change to looking for "HO" and "HF" separately and fallback to HCAL
   iRecord.get("HCAL", pG); 
   pCaloGeom->setSubdetGeometry(DetId::Hcal,HcalBarrel,pG.product());
   pCaloGeom->setSubdetGeometry(DetId::Hcal,HcalEndcap,pG.product());
   pCaloGeom->setSubdetGeometry(DetId::Hcal,HcalOuter,pG.product());
   pCaloGeom->setSubdetGeometry(DetId::Hcal,HcalForward,pG.product());
   
   // TODO: Look for ECAL parts

   // look for TOWER parts
   iRecord.get("TOWER",pG);
   pCaloGeom->setSubdetGeometry(DetId::Calo,1,pG.product());
   
   return pCaloGeom;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(CaloGeometryBuilder)
