#include <memory>
/** \file MuonNavigationTest
 *
 *  $Date: 2006/10/24 14:43:59 $
 *  $Revision: 1.5 $
 *  \author Chang Liu
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/Navigation/interface/MuonNavigationPrinter.h"
#include "RecoMuon/Navigation/interface/MuonNavigationSchool.h"
//#include "RecoMuon/Navigation/interface/MuonTkNavigationSchool.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "RecoMuon/Navigation/interface/MuonNavigationPrinter.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
//#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
//#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class MuonNavigationTest : public edm::EDAnalyzer {
   public:
      explicit MuonNavigationTest( const edm::ParameterSet& );
      ~MuonNavigationTest();

      virtual void analyze( const edm::Event&, const edm::EventSetup& );
   private:
};

// constructor

MuonNavigationTest::MuonNavigationTest( const edm::ParameterSet& iConfig )
{
      std::cout<<"Muon Navigation Printer Begin:"<<std::endl;
}


MuonNavigationTest::~MuonNavigationTest()
{
       std::cout<<"Muon Navigation Printer End. "<<std::endl;
}


void
MuonNavigationTest::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   //choose ONE and ONLY one to be true
   bool testMuon = true;
//   bool testMuonTk = true;
   //
   // get Geometry
   //
   edm::ESHandle<MuonDetLayerGeometry> muon;
   iSetup.get<MuonRecoGeometryRecord>().get(muon);     
   const MuonDetLayerGeometry * mm(&(*muon));

   if ( testMuon ) {
      MuonNavigationSchool school(mm);
      NavigationSetter setter(school);
      MuonNavigationPrinter* printer = new MuonNavigationPrinter(mm);
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
     NavigationSetter setter(school);
     MuonNavigationPrinter* printer = new MuonNavigationPrinter(mm, tt);
     delete printer;
  }
*/
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonNavigationTest);

