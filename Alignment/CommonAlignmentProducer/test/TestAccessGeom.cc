// -*- C++ -*-
//
// Package:    TestAccessGeom
// Class:      TestAccessGeom
// 
/**\class TestAccessGeom Alignment/CommonAlignmentProducer/test/TestAccessGeom.cc

 Description: <one line class summary>

 Implementation:
 Module accessing tracking geometries for tracker, DT and CSC
*/
//
// Original Author:  Gero Flucke
//         Created:  Sat Feb 16 20:56:04 CET 2008
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"  
#include "Geometry/DTGeometry/interface/DTGeometry.h"  
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"  

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"

//
// class declaration
//

class TestAccessGeom : public edm::EDAnalyzer {
   public:
      explicit TestAccessGeom(const edm::ParameterSet&);
      ~TestAccessGeom();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

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
TestAccessGeom::TestAccessGeom(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

}


TestAccessGeom::~TestAccessGeom()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TestAccessGeom::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   edm::ESHandle<TrackerGeometry> tkGeomHandle;
   iSetup.get<TrackerDigiGeometryRecord>().get(tkGeomHandle);

   edm::ESHandle<DTGeometry> dtGeomHandle;
   iSetup.get<MuonGeometryRecord>().get(dtGeomHandle);

   edm::ESHandle<CSCGeometry> cscGeomHandle;
   iSetup.get<MuonGeometryRecord>().get(cscGeomHandle);

   edm::LogInfo("Test") << "Succesfully accessed Tracker-, DT- and CSC-geometry";
}


// ------------ method called once each job just before starting event loop  ------------
void 
TestAccessGeom::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TestAccessGeom::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestAccessGeom);
