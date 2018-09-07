#include <memory>
/** \file MTDNavigationTest
 *
 *  \author L. Gray
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMTD/Navigation/interface/MTDNavigationPrinter.h"
#include "RecoMTD/Navigation/interface/MTDNavigationSchool.h"
#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"
#include "RecoMTD/Navigation/interface/MTDNavigationPrinter.h"
#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class MTDNavigationTest : public edm::EDAnalyzer {
   public:
      explicit MTDNavigationTest( const edm::ParameterSet& );
      ~MTDNavigationTest();

      virtual void analyze( const edm::Event&, const edm::EventSetup& );
   private:
};

// constructor

MTDNavigationTest::MTDNavigationTest( const edm::ParameterSet& iConfig )
{
      std::cout<<"Muon Navigation Printer Begin:"<<std::endl;
}


MTDNavigationTest::~MTDNavigationTest()
{
       std::cout<<"Muon Navigation Printer End. "<<std::endl;
}


void
MTDNavigationTest::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   //choose ONE and ONLY one to be true
   bool testMuon = true;
//   bool testMuonTk = true;
   //
   // get Geometry
   //
   edm::ESHandle<MTDDetLayerGeometry> mtd;
   iSetup.get<MTDRecoGeometryRecord>().get(mtd);     
   const MTDDetLayerGeometry * mm(&(*mtd));

   if ( testMuon ) {
      MTDNavigationSchool school(mm);
      MTDNavigationPrinter* printer = new MTDNavigationPrinter(mm, school );
      delete printer;
   }
/*
   if ( testMuonTk ) {
     edm::ESHandle<GeometricSearchTracker> tracker;
     iSetup.get<TrackerRecoGeometryRecord>().get(tracker);

     edm::ESHandle<MagneticField> theMF;
     iSetup.get<IdealMagneticFieldRecord>().get(theMF);

     const GeometricSearchTracker * tt(&(*tracker));
     const MagneticField * field(&(*theMF));

     MuonTkNavigationSchool school(mm,tt,field);
     MTDNavigationPrinter* printer = new MTDNavigationPrinter(mm, tt);
     delete printer;
  }
*/
}

//define this as a plug-in
DEFINE_FWK_MODULE(MTDNavigationTest);

