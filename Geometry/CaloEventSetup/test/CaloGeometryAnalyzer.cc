// -*- C++ -*-
//
// Package:    CaloGeometryAnalyzer
// Class:      CaloGeometryAnalyzer
// 
/**\class CaloGeometryAnalyzer CaloGeometryAnalyzer.cc test/CaloGeometryAnalyzer/src/CaloGeometryAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
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
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

//
// class decleration
//

class CaloGeometryAnalyzer : public edm::EDAnalyzer {
   public:
      explicit CaloGeometryAnalyzer( const edm::ParameterSet& );
      ~CaloGeometryAnalyzer();


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
CaloGeometryAnalyzer::CaloGeometryAnalyzer( const edm::ParameterSet& iConfig )
{
   //now do what ever initialization is needed

}


CaloGeometryAnalyzer::~CaloGeometryAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CaloGeometryAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   std::cout << "Here I am " << std::endl;
   //
   // get the hcal geometry
   //
   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<IdealGeometryRecord>().get(pG);     
      
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloGeometryAnalyzer)
