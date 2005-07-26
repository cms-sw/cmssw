// -*- C++ -*-
//
// Package:    PerfectGeometryAnalyzer
// Class:      PerfectGeometryAnalyzer
// 
/**\class PerfectGeometryAnalyzer PerfectGeometryAnalyzer.cc test/PerfectGeometryAnalyzer/src/PerfectGeometryAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tommaso Boccali
//         Created:  Tue Jul 26 08:47:57 CEST 2005
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
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/DDCore/interface/DDCompactView.h"
#include "DetectorDescription/DDCore/interface/DDExpandedView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
//
// class decleration
//

class PerfectGeometryAnalyzer : public edm::EDAnalyzer {
   public:
      explicit PerfectGeometryAnalyzer( const edm::ParameterSet& );
      ~PerfectGeometryAnalyzer();


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
PerfectGeometryAnalyzer::PerfectGeometryAnalyzer( const edm::ParameterSet& iConfig )
{
   //now do what ever initialization is needed

}


PerfectGeometryAnalyzer::~PerfectGeometryAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PerfectGeometryAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   std::cout << "Here I am " << std::endl;
   //
   // get the DDCompactView
   //
   edm::eventsetup::ESHandle<DDCompactView> pDD;
   iSetup.get<IdealGeometryRecord>().get( pDD );     
   DDExpandedView ex(*pDD);
   ex.firstChild();
   std::cout << " Top node is a "<< ex.logicalPart() << std::endl;
   
}

//define this as a plug-in
DEFINE_FWK_MODULE(PerfectGeometryAnalyzer)
