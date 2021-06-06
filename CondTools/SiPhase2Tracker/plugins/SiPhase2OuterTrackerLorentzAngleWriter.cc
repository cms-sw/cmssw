// -*- C++ -*-
//
// Package:  CondTools/SiPhase2Tracker
// Class:    SiPhase2OuterTrackerLorentzAngleWriter
//
/**\class SiPhase2OuterTrackerLorentzAngleWriter SiPhase2OuterTrackerLorentzAngleWriter.cc CondTools/SiPhase2Tracker/plugins/SiPhase2OuterTrackerLorentzAngleWriter.cc

 Description: Put the values of the Lorentz angle for Phase-2 Outer Tracker detId in as database file

 Implementation:
     [Notes on implementation]
     
*/
//
// Original Author:  Marco Musich
//         Created:  Mon, 18 May 2020 18:00:00 GMT
//
//

// system include files
#include <memory>
#include <iostream>
#include <string>
#include <unordered_map>

// user include files
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/SiPhase2OuterTrackerLorentzAngleRcd.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/SiPhase2OuterTrackerLorentzAngle.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

//
// class declaration
//

class SiPhase2OuterTrackerLorentzAngleWriter : public edm::one::EDAnalyzer<> {
public:
  explicit SiPhase2OuterTrackerLorentzAngleWriter(const edm::ParameterSet&);
  ~SiPhase2OuterTrackerLorentzAngleWriter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  std::string m_record;
  std::string m_tag;
  float m_value;
};

//
// constructors and destructor
//
SiPhase2OuterTrackerLorentzAngleWriter::SiPhase2OuterTrackerLorentzAngleWriter(const edm::ParameterSet& iConfig)
    : topoToken_(esConsumes()),
      geomToken_(esConsumes()),
      m_record(iConfig.getParameter<std::string>("record")),
      m_tag(iConfig.getParameter<std::string>("tag")),
      m_value(iConfig.getParameter<double>("value")) {}

SiPhase2OuterTrackerLorentzAngleWriter::~SiPhase2OuterTrackerLorentzAngleWriter() {
  edm::LogInfo("SiPhase2OuterTrackerLorentzAngleWriter")
      << "SiPhase2OuterTrackerLorentzAngleWriter::~SiPhase2OuterTrackerLorentzAngleWriter" << std::endl;
}

//
// member functions
//

// ------------ method called for each event  ------------
void SiPhase2OuterTrackerLorentzAngleWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using Phase2TrackerGeomDetUnit = PixelGeomDetUnit;
  using namespace edm;
  edm::LogInfo("SiPhase2OuterTrackerLorentzAngleWriter")
      << "SiPhase2OuterTrackerLorentzAngleWriter::analyze " << std::endl;

  // Database services (write)
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (!mydbservice.isAvailable()) {
    edm::LogWarning("SiPhase2OuterTrackerLorentzAngleWriter") << "Service is unavailable" << std::endl;
    return;
  }

  std::string tag = mydbservice->tag(m_record);
  unsigned int irun = iEvent.id().run();
  edm::LogVerbatim("SiPhase2OuterTrackerLorentzAngleWriter") << "Writing on tag : " << tag << std::endl;
  edm::LogVerbatim("SiPhase2OuterTrackerLorentzAngleWriter") << "Usinng as IOV run : " << irun << std::endl;

  // map to be filled
  std::unordered_map<unsigned int, float> detsLAtoDB;

  // Retrieve tracker topology from geometry
  const TrackerTopology* const tTopo = &iSetup.getData(topoToken_);

  // Retrieve old style tracker geometry from geometry
  const TrackerGeometry* pDD = &iSetup.getData(geomToken_);
  edm::LogInfo("SiPhase2OuterTrackerLorentzAngleWriter")
      << " There are " << pDD->detUnits().size() << " modules in this geometry." << std::endl;

  for (auto const& det_u : pDD->detUnits()) {
    const DetId detid = det_u->geographicalId();
    uint32_t rawId = detid.rawId();
    int subid = detid.subdetId();
    if (detid.det() == DetId::Detector::Tracker) {
      const Phase2TrackerGeomDetUnit* pixdet = dynamic_cast<const Phase2TrackerGeomDetUnit*>(det_u);
      assert(pixdet);
      LogDebug("SiPhase2OuterTrackerLorentzAngleWriter") << rawId << " is a " << subid << " det" << std::endl;
      if (subid == StripSubdetector::TOB || subid == StripSubdetector::TID) {
        LogDebug("SiPhase2OuterTrackerLorentzAngleWriter")
            << "subdetector ID:" << subid << " layer:" << tTopo->layer(detid) << std::endl;
        detsLAtoDB[rawId] = m_value;
      }
    }
  }

  edm::LogInfo("SiPhase2OuterTrackerLorentzAngleWriter")
      << " There are " << detsLAtoDB.size() << " OT Lorentz Angle values assigned" << std::endl;

  // SiStripLorentzAngle object

  auto lorentzAngle = std::make_unique<SiPhase2OuterTrackerLorentzAngle>();
  lorentzAngle->putLorentzAngles(detsLAtoDB);
  edm::LogInfo("SiPhase2OuterTrackerLorentzAngleWriter") << "currentTime " << mydbservice->currentTime() << std::endl;
  mydbservice->writeOne(lorentzAngle.get(), mydbservice->currentTime(), m_record);
}

// ------------ method called once each job just before starting event loop  ------------
void SiPhase2OuterTrackerLorentzAngleWriter::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void SiPhase2OuterTrackerLorentzAngleWriter::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiPhase2OuterTrackerLorentzAngleWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("record", "SiPhase2OuterTrackerLorentzAngleRcd");
  desc.add<std::string>("tag", "SiPhase2OuterTrackerLorentzAngle");
  desc.add<double>("value", 0.07);
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPhase2OuterTrackerLorentzAngleWriter);
