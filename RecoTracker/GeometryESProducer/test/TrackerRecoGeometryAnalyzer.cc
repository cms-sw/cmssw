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

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

//
//
// class decleration
//

class TrackerRecoGeometryAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TrackerRecoGeometryAnalyzer( const edm::ParameterSet& );
      ~TrackerRecoGeometryAnalyzer();


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
TrackerRecoGeometryAnalyzer::TrackerRecoGeometryAnalyzer( const edm::ParameterSet& iConfig )
{
   //now do what ever initialization is needed

}


TrackerRecoGeometryAnalyzer::~TrackerRecoGeometryAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TrackerRecoGeometryAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   std::cout << "Here I am " << std::endl;
   //
   // get the GeometricSearchDet
   //
   edm::ESHandle<GeometricSearchTracker> trak;
   iSetup.get<TrackerRecoGeometryRecord>().get( trak );     
   std::cout <<" AFTER  " <<std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackerRecoGeometryAnalyzer)
 
 
