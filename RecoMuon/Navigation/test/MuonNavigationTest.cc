#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/Navigation/interface/MuonNavigationPrinter.h"
#include "RecoMuon/Navigation/interface/MuonNavigationSchool.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "RecoMuon/Navigation/interface/MuonNavigationPrinter.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"


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

   std::cout << "Print out DetLayers: " << std::endl;
   //
   // get the GeometricSearchDet
   //
   edm::ESHandle<MuonDetLayerGeometry> muon;
   iSetup.get<MuonRecoGeometryRecord>().get(muon);     
   
   const MuonDetLayerGeometry & mm(*muon);
   
   MuonNavigationSchool school(&mm);
   NavigationSetter setter(school);
   MuonNavigationPrinter* printer = new MuonNavigationPrinter(&mm);
   delete printer;
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonNavigationTest)

