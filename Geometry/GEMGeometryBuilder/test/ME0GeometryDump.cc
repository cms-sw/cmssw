// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/GEMGeometry/interface/ME0Chamber.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

//STL headers
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>

class ME0GeometryDump : public edm::one::EDAnalyzer<> {
public:
  explicit ME0GeometryDump(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  const bool verbose_;
  const edm::ESGetToken<ME0Geometry, MuonGeometryRecord> tokGeom_;
  const ME0Geometry* me0Geometry_;
};

ME0GeometryDump::ME0GeometryDump(const edm::ParameterSet& iC)
    : verbose_(iC.getParameter<bool>("verbose")),
      tokGeom_{esConsumes<ME0Geometry, MuonGeometryRecord>(edm::ESInputTag{})} {}

void ME0GeometryDump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("verbose", false);
  descriptions.add("me0GeometryDump", desc);
}

void ME0GeometryDump::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  me0Geometry_ = &eventSetup.getData(tokGeom_);

  auto const& chambers = me0Geometry_->chambers();
  edm::LogVerbatim("ME0Geometry") << "ME0Geometry found with " << chambers.size() << " chambers\n";
  for (unsigned int k1 = 0; k1 < chambers.size(); ++k1) {
    edm::LogVerbatim("ME0Geometry") << "\nChamber " << k1 << ":" << chambers[k1]->id() << " with "
                                    << chambers[k1]->nLayers() << " layers";
    auto const& layers = chambers[k1]->layers();

    for (unsigned int k2 = 0; k2 < layers.size(); ++k2) {
      edm::LogVerbatim("ME0Geometry") << "\nLayer " << k2 << ":" << layers[k2]->id() << " with "
                                      << layers[k2]->nEtaPartitions() << " etaPartitions";
      auto const& etaPartitions = layers[k2]->etaPartitions();

      for (unsigned int k3 = 0; k3 < etaPartitions.size(); ++k3) {
        edm::LogVerbatim("ME0Geometry") << "\nEtaPartition " << k3 << ":" << etaPartitions[k3]->id()
                                        << etaPartitions[k3]->type().name() << " with " << etaPartitions[k3]->nstrips()
                                        << " strips of pitch " << std::setprecision(4) << etaPartitions[k3]->pitch()
                                        << " and " << etaPartitions[k3]->npads() << " pads of pitch "
                                        << std::setprecision(4) << etaPartitions[k3]->padPitch();
        if (verbose_) {
          for (int k = 0; k < etaPartitions[k3]->nstrips(); ++k)
            edm::LogVerbatim("ME0Geometry")
                << "Strip[" << k << "] " << std::setprecision(4) << etaPartitions[k3]->centreOfStrip(k);
          for (int k = 0; k < etaPartitions[k3]->npads(); ++k)
            edm::LogVerbatim("ME0Geometry")
                << "Pad[" << k << "] " << std::setprecision(4) << etaPartitions[k3]->centreOfPad(k);
        }
      }
    }
  }
}

DEFINE_FWK_MODULE(ME0GeometryDump);
