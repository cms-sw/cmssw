#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTrackerBuilder.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>
#include <string>

class TrackerMTDRecoGeometryESProducer : public edm::ESProducer {
public:
  TrackerMTDRecoGeometryESProducer(const edm::ParameterSet &p);

  std::unique_ptr<GeometricSearchTracker> produce(const TrackerRecoGeometryRecord &);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopToken_;
  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeomToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdTopToken_;
  bool usePhase2Stacks_;
};

using namespace edm;

TrackerMTDRecoGeometryESProducer::TrackerMTDRecoGeometryESProducer(const edm::ParameterSet &p)
    : usePhase2Stacks_(p.getParameter<bool>("usePhase2Stacks")) {
  auto c = setWhatProduced(this);

  tTopToken_ = c.consumes();
  geomToken_ = c.consumes(edm::ESInputTag("", p.getUntrackedParameter<std::string>("trackerGeometryLabel")));
  mtdgeomToken_ = c.consumes();
  mtdTopToken_ = c.consumes();
}

std::unique_ptr<GeometricSearchTracker> TrackerMTDRecoGeometryESProducer::produce(
    const TrackerRecoGeometryRecord &iRecord) {
  TrackerGeometry const &tG = iRecord.get(geomToken_);
  MTDGeometry const &mG = iRecord.get(mtdgeomToken_);

  GeometricSearchTrackerBuilder builder;
  return std::unique_ptr<GeometricSearchTracker>(
      builder.build(tG.trackerDet(), &tG, &iRecord.get(tTopToken_), &mG, &iRecord.get(mtdTopToken_), usePhase2Stacks_));
}

void TrackerMTDRecoGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("usePhase2Stacks", false);
  desc.addUntracked<std::string>("trackerGeometryLabel", "");
  descriptions.addDefault(desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerMTDRecoGeometryESProducer);
