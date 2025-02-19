// -*- C++ -*-
//
// Package:    L1CaloGeometryDump
// Class:      L1CaloGeometryDump
// 
/**\class L1CaloGeometryDump L1CaloGeometryDump.cc L1TriggerConfig/L1CaloGeometryDump/src/L1CaloGeometryDump.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Mon Sep 28 22:17:24 CEST 2009
// $Id: L1CaloGeometryDump.cc,v 1.1 2009/09/28 23:01:25 wsun Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

//
// class decleration
//

class L1CaloGeometryDump : public edm::EDAnalyzer {
   public:
      explicit L1CaloGeometryDump(const edm::ParameterSet&);
      ~L1CaloGeometryDump();


   private:
      virtual void beginJob() ;
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
L1CaloGeometryDump::L1CaloGeometryDump(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

}


L1CaloGeometryDump::~L1CaloGeometryDump()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1CaloGeometryDump::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   ESHandle< L1CaloGeometry > pGeom ;
   iSetup.get< L1CaloGeometryRecord >().get( pGeom ) ;

   LogDebug( "L1CaloGeometryDump" ) << *pGeom << std::endl ;
}


// ------------ method called once each job just before starting event loop  ------------
void 
L1CaloGeometryDump::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1CaloGeometryDump::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1CaloGeometryDump);
