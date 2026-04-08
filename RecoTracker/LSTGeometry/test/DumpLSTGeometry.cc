#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoTracker/LSTGeometry/interface/Common.h"
#include "RecoTracker/LSTGeometry/interface/Geometry.h"
#include "RecoTracker/LSTGeometry/interface/IO.h"

class DumpLSTGeometry : public edm::one::EDAnalyzer<> {
public:
  explicit DumpLSTGeometry(const edm::ParameterSet& config);

private:
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;

  double ptCut_;
  std::string ptCutStr_;
  std::string outputDirectory_;
  bool binaryOutput_;

  edm::ESGetToken<lstgeometry::Geometry, TrackerRecoGeometryRecord> lstGeoToken_;
};

DumpLSTGeometry::DumpLSTGeometry(const edm::ParameterSet& config)
    : ptCut_(config.getParameter<double>("ptCut")),
      ptCutStr_(lst::floatToStr(ptCut_, 1)),
      outputDirectory_(config.getUntrackedParameter<std::string>("outputDirectory", "data/")),
      binaryOutput_(config.getUntrackedParameter<bool>("outputAsBinary", true)),
      lstGeoToken_{esConsumes(edm::ESInputTag("", ptCutStr_))} {}

void DumpLSTGeometry::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const& lstg = iSetup.getData(lstGeoToken_);

  lstgeometry::writeSensorCentroids(lstg.sensors, outputDirectory_ + "sensor_centroids", binaryOutput_);
  lstgeometry::writeSlopes(
      lstg.barrel_slopes, lstg.sensors, outputDirectory_ + "tilted_barrel_orientation", binaryOutput_);
  lstgeometry::writeSlopes(lstg.endcap_slopes, lstg.sensors, outputDirectory_ + "endcap_orientation", binaryOutput_);
  lstgeometry::writePixelMaps(lstg.pixel_map, outputDirectory_ + "pixelmap/pLS_map", binaryOutput_);
  lstgeometry::writeModuleConnections(
      lstg.module_map, outputDirectory_ + "module_connection_tracing_merged", binaryOutput_);

  edm::LogInfo("DumpLSTGeometry") << "Geometry data was successfully dumped." << std::endl;
}

DEFINE_FWK_MODULE(DumpLSTGeometry);
