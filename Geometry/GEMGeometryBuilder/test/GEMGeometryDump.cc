// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMChamber.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

//STL headers
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>

class GEMGeometryDump : public edm::one::EDAnalyzer<> {
public:
  explicit GEMGeometryDump(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  const bool verbose_;
  const edm::ESGetToken<GEMGeometry, MuonGeometryRecord> tokGeom_;
  const GEMGeometry* gemGeometry_;
};

GEMGeometryDump::GEMGeometryDump(const edm::ParameterSet& iC)
    : verbose_(iC.getParameter<bool>("verbose")),
      tokGeom_{esConsumes<GEMGeometry, MuonGeometryRecord>(edm::ESInputTag{})} {}

void GEMGeometryDump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("verbose", false);
  descriptions.add("gemGeometryDump", desc);
}

void GEMGeometryDump::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  gemGeometry_ = &eventSetup.getData(tokGeom_);

  auto const& regions = gemGeometry_->regions();
  edm::LogVerbatim("GEMGeometry") << "GEMGeometry found with " << regions.size() << " regions\n";
  for (unsigned int k1 = 0; k1 < regions.size(); ++k1) {
    edm::LogVerbatim("GEMGeometry") << "\nRegion " << k1 << ":" << regions[k1]->region() << " with "
                                    << regions[k1]->nStations() << " staions";
    auto const& stations = regions[k1]->stations();

    for (unsigned int k2 = 0; k2 < stations.size(); ++k2) {
      edm::LogVerbatim("GEMGeometry") << "\nStation " << k2 << ":" << stations[k2]->station() << ":"
                                      << stations[k2]->getName() << " with " << stations[k2]->nRings() << " rings";
      auto const& rings = stations[k2]->rings();

      for (unsigned int k3 = 0; k3 < rings.size(); ++k3) {
        edm::LogVerbatim("GEMGeometry") << "\nRing " << k3 << ":" << rings[k3]->ring() << " with "
                                        << rings[k3]->nSuperChambers() << " superChambers";
        auto const& superChambers = rings[k3]->superChambers();

        for (unsigned int k4 = 0; k4 < superChambers.size(); ++k4) {
          edm::LogVerbatim("GEMGeometry") << "\nSuperChamber " << k4 << ":" << superChambers[k4]->id() << " with "
                                          << superChambers[k4]->nChambers() << " chambers";
          auto const& chambers = superChambers[k4]->chambers();

          for (unsigned int k5 = 0; k5 < chambers.size(); ++k5) {
            edm::LogVerbatim("GEMGeometry") << "\nChamber " << k5 << ":" << chambers[k5]->id() << " with "
                                            << chambers[k5]->nEtaPartitions() << " etaPartitions";
            auto const& etaPartitions = chambers[k5]->etaPartitions();

            for (unsigned int k6 = 0; k6 < etaPartitions.size(); ++k6) {
              edm::LogVerbatim("GEMGeometry")
                  << "\nEtaPartition " << k6 << ":" << etaPartitions[k6]->id() << etaPartitions[k6]->type().name()
                  << " with " << etaPartitions[k6]->nstrips() << " strips of pitch " << etaPartitions[k6]->pitch()
                  << " and " << etaPartitions[k6]->npads() << " pads of pitch " << etaPartitions[k6]->padPitch();
              if (verbose_) {
                for (int k = 0; k < etaPartitions[k6]->nstrips(); ++k)
                  edm::LogVerbatim("GEMGeometry") << "Strip[" << k << "] " << etaPartitions[k6]->centreOfStrip(k);
                for (int k = 0; k < etaPartitions[k6]->npads(); ++k)
                  edm::LogVerbatim("GEMGeometry") << "Pad[" << k << "] " << etaPartitions[k6]->centreOfPad(k);
              }
            }
          }
        }
      }
    }
  }
}

DEFINE_FWK_MODULE(GEMGeometryDump);
