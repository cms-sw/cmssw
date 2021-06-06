// -*- C++ -*-
//
// Package:    Geometry/TrackerGeometryBuilder
// Class:      PixelTopologyMapTest
//
/**\class PixelTopologyMapTest PixelTopologyMapTest.cc Geometry/TrackerGeometryBuilder/test/PixelTopologyMapTest.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Musich
//         Created:  Wed, 31 Mar 2021 11:01:16 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyMap.h"

//
// class declaration
//

class PixelTopologyMapTest : public edm::global::EDAnalyzer<> {
public:
  explicit PixelTopologyMapTest(const edm::ParameterSet&);
  ~PixelTopologyMapTest() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const& iSetup) const override;

  // ----------member data ---------------------------
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomEsToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
};

//
// constructors and destructor
//
PixelTopologyMapTest::PixelTopologyMapTest(const edm::ParameterSet& iConfig)
    : geomEsToken_(esConsumes()), topoToken_(esConsumes()) {}

//
// member functions
//

// ------------ method called for each event  ------------
void PixelTopologyMapTest::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const& iSetup) const {
  using namespace edm;

  // get the Tracker geometry from event setup
  const TrackerGeometry* pDD = &iSetup.getData(geomEsToken_);
  const TrackerTopology* tTopo = &iSetup.getData(topoToken_);

  if ((pDD->isThere(GeomDetEnumerators::P2PXB)) || (pDD->isThere(GeomDetEnumerators::P2PXEC))) {
    edm::LogPrint("PixelTopologyMapTest") << "===== Testing Phase-2 Pixel Tracker geometry =====" << std::endl;
  } else if ((pDD->isThere(GeomDetEnumerators::P1PXB)) || (pDD->isThere(GeomDetEnumerators::P1PXEC))) {
    edm::LogPrint("PixelTopologyMapTest") << "===== Testing Phase-1 Pixel Tracker geometry =====" << std::endl;
  } else {
    edm::LogPrint("PixelTopologyMapTest") << "===== Testing Phase-0 Pixel Tracker geometry =====" << std::endl;
  }

  PixelTopologyMap PTMap = PixelTopologyMap(pDD, tTopo);
  edm::LogPrint("PixelTopologyMapTest") << PTMap << std::endl;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void PixelTopologyMapTest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PixelTopologyMapTest);
