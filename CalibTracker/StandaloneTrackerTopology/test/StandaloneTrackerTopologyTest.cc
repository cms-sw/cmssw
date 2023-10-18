// -*- C++ -*-
//
// Package:    CalibTracker/StandaloneTrackerTopology
// Class:      StandaloneTrackerTopologyTest
//
/**\class StandaloneTrackerTopologyTest StandaloneTrackerTopologyTest.cc CalibTracker/StandaloneTrackerTopology/test/StandaloneTrackerTopologyTest.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Musich
//         Created:  Wed, 18 Oct 2023 10:00:00 GMT
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
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

//
// class declaration
//

class StandaloneTrackerTopologyTest : public edm::global::EDAnalyzer<> {
public:
  explicit StandaloneTrackerTopologyTest(const edm::ParameterSet&);
  ~StandaloneTrackerTopologyTest() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const& iSetup) const override;
  void testTopology(const TrackerGeometry* pDD,
                    const TrackerTopology* tTopoFromDB,
                    const TrackerTopology* tTopoStandalone) const;

  // ----------member data ---------------------------
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomEsToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
};

//
// constructors and destructor
//
StandaloneTrackerTopologyTest::StandaloneTrackerTopologyTest(const edm::ParameterSet& iConfig)
    : geomEsToken_(esConsumes()), topoToken_(esConsumes()) {}

//
// member functions
//
void StandaloneTrackerTopologyTest::testTopology(const TrackerGeometry* pDD,
                                                 const TrackerTopology* tTopoFromDB,
                                                 const TrackerTopology* tTopoStandalone) const {
  // test Barrel Pixel
  for (auto det : pDD->detsPXB()) {
    const PixelGeomDetUnit* pixelDet = dynamic_cast<const PixelGeomDetUnit*>(det);
    const auto& layerA = tTopoFromDB->pxbLayer(pixelDet->geographicalId());
    const auto& ladderA = tTopoFromDB->pxbLadder(pixelDet->geographicalId());
    const auto& moduleA = tTopoFromDB->pxbModule(pixelDet->geographicalId());

    const auto& layerB = tTopoStandalone->pxbLayer(pixelDet->geographicalId());
    const auto& ladderB = tTopoStandalone->pxbLadder(pixelDet->geographicalId());
    const auto& moduleB = tTopoStandalone->pxbModule(pixelDet->geographicalId());

    if (layerA != layerB || ladderA != ladderB || moduleA != moduleB) {
      throw cms::Exception("ConfigurationMismatch")
          << "PXB: Topology from DB doesn't match with Topology from XML file!"
          << " DetId: " << pixelDet->geographicalId() << " layerA: " << layerA << " layerB: " << layerB
          << " ladderA :" << ladderA << " ladderB :" << ladderB << " moduleA :" << moduleA << " moduleB :" << moduleB
          << std::endl;
    }
  }

  // test Pixel Endcaps
  for (auto det : pDD->detsPXF()) {
    const PixelGeomDetUnit* pixelDet = dynamic_cast<const PixelGeomDetUnit*>(det);
    const auto& diskA = tTopoFromDB->pxfDisk(pixelDet->geographicalId());
    const auto& bladeA = tTopoFromDB->pxfBlade(pixelDet->geographicalId());
    const auto& moduleA = tTopoFromDB->pxfModule(pixelDet->geographicalId());

    const auto& diskB = tTopoStandalone->pxfDisk(pixelDet->geographicalId());
    const auto& bladeB = tTopoStandalone->pxfBlade(pixelDet->geographicalId());
    const auto& moduleB = tTopoStandalone->pxfModule(pixelDet->geographicalId());

    if (diskA != diskB || bladeA != bladeB || moduleA != moduleB) {
      throw cms::Exception("ConfigurationMismatch")
          << "PXF: Topology from DB doesn't match with Topology from XML file"
          << " DetId: " << pixelDet->geographicalId() << " diskA: " << diskA << " diskB: " << diskB
          << " bladeA :" << bladeA << " bladeB :" << bladeB << " moduleA :" << moduleA << " moduleB :" << moduleB
          << std::endl;
    }
  }

  // test inner barrel
  for (auto det : pDD->detsTIB()) {
    const GeomDetUnit* Det = dynamic_cast<const GeomDetUnit*>(det);
    const auto& sideA = tTopoFromDB->tibSide(Det->geographicalId());
    const auto& layerA = tTopoFromDB->tibLayer(Det->geographicalId());
    const auto& moduleA = tTopoFromDB->tibModule(Det->geographicalId());

    const auto& sideB = tTopoStandalone->tibSide(Det->geographicalId());
    const auto& layerB = tTopoStandalone->tibLayer(Det->geographicalId());
    const auto& moduleB = tTopoStandalone->tibModule(Det->geographicalId());

    if (sideA != sideB || layerA != layerB || moduleA != moduleB) {
      throw cms::Exception("ConfigurationMismatch")
          << "TIB: Topology from DB doesn't match with Topology from XML file"
          << " DetId: " << Det->geographicalId() << " sideA: " << sideA << " sideB: " << sideB << " layerA :" << layerA
          << " layerB :" << layerB << " moduleA :" << moduleA << " moduleB :" << moduleB << std::endl;
    }
  }

  // test inner disks
  for (auto det : pDD->detsTID()) {
    const GeomDetUnit* Det = dynamic_cast<const GeomDetUnit*>(det);
    const auto& sideA = tTopoFromDB->tidSide(Det->geographicalId());
    const auto& wheelA = tTopoFromDB->tidWheel(Det->geographicalId());
    const auto& moduleA = tTopoFromDB->tidModule(Det->geographicalId());

    const auto& sideB = tTopoStandalone->tidSide(Det->geographicalId());
    const auto& wheelB = tTopoStandalone->tidWheel(Det->geographicalId());
    const auto& moduleB = tTopoStandalone->tidModule(Det->geographicalId());

    if (sideA != sideB || wheelA != wheelB || moduleA != moduleB) {
      throw cms::Exception("ConfigurationMismatch")
          << "TID: Topology from DB doesn't match with Topology from XML file"
          << " DetId: " << Det->geographicalId() << " sideA: " << sideA << " sideB: " << sideB << " wheelA :" << wheelA
          << " wheelB :" << wheelB << " moduleA :" << moduleA << " moduleB :" << moduleB << std::endl;
    }
  }

  // test outer barrel
  for (auto det : pDD->detsTOB()) {
    const GeomDetUnit* Det = dynamic_cast<const GeomDetUnit*>(det);

    const auto& layerA = tTopoFromDB->tobLayer(Det->geographicalId());
    const auto& rodA = tTopoFromDB->tobRod(Det->geographicalId());
    const auto& moduleA = tTopoFromDB->tobModule(Det->geographicalId());

    const auto& layerB = tTopoStandalone->tobLayer(Det->geographicalId());
    const auto& rodB = tTopoStandalone->tobRod(Det->geographicalId());
    const auto& moduleB = tTopoStandalone->tobModule(Det->geographicalId());

    if (layerA != layerB || rodA != rodB || moduleA != moduleB) {
      throw cms::Exception("ConfigurationMismatch")
          << "TOB: Topology from DB doesn't match with Topology from XML file"
          << " DetId: " << Det->geographicalId() << " layerA :" << layerA << " layerB :" << layerB << " rodA: " << rodA
          << " rodB: " << rodB << " moduleA :" << moduleA << " moduleB :" << moduleB << std::endl;
    }
  }

  // test outer disks
  for (auto det : pDD->detsTEC()) {
    const GeomDetUnit* Det = dynamic_cast<const GeomDetUnit*>(det);
    const auto& sideA = tTopoFromDB->tecSide(Det->geographicalId());
    const auto& wheelA = tTopoFromDB->tecWheel(Det->geographicalId());
    const auto& moduleA = tTopoFromDB->tecModule(Det->geographicalId());

    const auto& sideB = tTopoStandalone->tecSide(Det->geographicalId());
    const auto& wheelB = tTopoStandalone->tecWheel(Det->geographicalId());
    const auto& moduleB = tTopoStandalone->tecModule(Det->geographicalId());

    if (sideA != sideB || wheelA != wheelB || moduleA != moduleB) {
      throw cms::Exception("ConfigurationMismatch")
          << "TEC: Topology from DB doesn't match with Topology from XML file"
          << " DetId: " << Det->geographicalId() << " sideA: " << sideA << " sideB: " << sideB << " wheelA :" << wheelA
          << " wheelB :" << wheelB << " moduleA :" << moduleA << " moduleB :" << moduleB << std::endl;
    }
  }
}

// ------------ method called for each event  ------------
void StandaloneTrackerTopologyTest::analyze(edm::StreamID,
                                            edm::Event const& iEvent,
                                            edm::EventSetup const& iSetup) const {
  using namespace edm;

  // get the Tracker geometry from event setup
  const TrackerGeometry* pDD = &iSetup.getData(geomEsToken_);
  const TrackerTopology* tTopo = &iSetup.getData(topoToken_);

  const char* pathToTopologyXML;
  if ((pDD->isThere(GeomDetEnumerators::P2PXB)) || (pDD->isThere(GeomDetEnumerators::P2PXEC))) {
    edm::LogPrint("StandaloneTrackerTopologyTest") << "===== Testing Phase-2 Pixel Tracker geometry =====" << std::endl;
    pathToTopologyXML = "Geometry/TrackerCommonData/data/PhaseII/trackerParameters.xml";
  } else if ((pDD->isThere(GeomDetEnumerators::P1PXB)) || (pDD->isThere(GeomDetEnumerators::P1PXEC))) {
    edm::LogPrint("StandaloneTrackerTopologyTest") << "===== Testing Phase-1 Pixel Tracker geometry =====" << std::endl;
    pathToTopologyXML = "Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml";
  } else {
    edm::LogPrint("StandaloneTrackerTopologyTest") << "===== Testing Phase-0 Pixel Tracker geometry =====" << std::endl;
    pathToTopologyXML = "Geometry/TrackerCommonData/data/trackerParameters.xml";
  }

  edm::LogPrint("StandaloneTrackerTopologyTest") << "parameters file" << pathToTopologyXML << std::endl;

  TrackerTopology tTopoStandalone =
      StandaloneTrackerTopology::fromTrackerParametersXMLFile(edm::FileInPath(pathToTopologyXML).fullPath());

  // the actual test
  testTopology(pDD, tTopo, &tTopoStandalone);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void StandaloneTrackerTopologyTest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(StandaloneTrackerTopologyTest);
