#include <memory>
/** \file MuonNavigationTest
 *
 *  \author Chang Liu
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/Navigation/interface/MuonNavigationPrinter.h"
#include "RecoMuon/Navigation/interface/MuonNavigationSchool.h"
//#include "RecoMuon/Navigation/interface/MuonTkNavigationSchool.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "RecoMuon/Navigation/interface/MuonNavigationPrinter.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
//#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
//#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class MuonNavigationTest : public edm::one::EDAnalyzer<> {
public:
  explicit MuonNavigationTest(const edm::ParameterSet&);
  ~MuonNavigationTest() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::ESGetToken<MuonDetLayerGeometry, MuonRecoGeometryRecord> geomToken_;
};

// constructor

MuonNavigationTest::MuonNavigationTest(const edm::ParameterSet& iConfig) {
  geomToken_ = esConsumes();
  std::cout << "Muon Navigation Printer Begin:" << std::endl;
}

MuonNavigationTest::~MuonNavigationTest() { std::cout << "Muon Navigation Printer End. " << std::endl; }

void MuonNavigationTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  //choose ONE and ONLY one to be true
  bool testMuon = true;
  //   bool testMuonTk = true;
  //
  // get Geometry
  //
  const MuonDetLayerGeometry& mm = iSetup.getData(geomToken_);

  if (testMuon) {
    MuonNavigationSchool school(&mm);
    MuonNavigationPrinter* printer = new MuonNavigationPrinter(&mm, school);
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
     MuonNavigationPrinter* printer = new MuonNavigationPrinter(mm, tt);
     delete printer;
  }
*/
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonNavigationTest);
