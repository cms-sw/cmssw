// -*- C++ -*-
//
// Package:    GeometricDetAnalyzer
// Class:      GeometricDetAnalyzer
// 
/**\class GeometricDetAnalyzer GeometricDetAnalyzer.cc test/GeometricDetAnalyzer/src/GeometricDetAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tommaso Boccali
//         Created:  Tue Jul 26 08:47:57 CEST 2005
// $Id: GeometricDetAnalyzer.cc,v 1.3 2005/10/18 19:50:54 fambrogl Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerDebugNavigator.h"


//
//
// class decleration
//

class GeometricDetAnalyzer : public edm::EDAnalyzer {
   public:
      explicit GeometricDetAnalyzer( const edm::ParameterSet& );
      ~GeometricDetAnalyzer();


      virtual void analyze( const edm::Event&, const edm::EventSetup& );
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
GeometricDetAnalyzer::GeometricDetAnalyzer( const edm::ParameterSet& iConfig )
{
   //now do what ever initialization is needed

}


GeometricDetAnalyzer::~GeometricDetAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GeometricDetAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   edm::LogInfo("GeometricDetAnalyzer")<< "Here I am ";
   //
   // get the GeometricDet
   //
   edm::ESHandle<GeometricDet> pDD;
   iSetup.get<IdealGeometryRecord>().get( pDD );     
   edm::LogInfo("GeometricDetAnalyzer")<< " Top node is  "<<&(*pDD);   
   edm::LogInfo("GeometricDetAnalyzer")<< " And Contains  Daughters: "<<(*pDD).deepComponents().size();   
   CmsTrackerDebugNavigator nav;
   nav.dump(&(*pDD));
}

//define this as a plug-in
DEFINE_FWK_MODULE(GeometricDetAnalyzer)
