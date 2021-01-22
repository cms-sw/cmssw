#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTrackerBuilder.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>
#include <string>

class TrackerRecoGeometryESProducer : public edm::ESProducer {
public:
  TrackerRecoGeometryESProducer(const edm::ParameterSet &p);

  std::unique_ptr<GeometricSearchTracker> produce(const TrackerRecoGeometryRecord &);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopToken_;
  bool usePhase2Stacks_;
};

using namespace edm;

TrackerRecoGeometryESProducer::TrackerRecoGeometryESProducer(const edm::ParameterSet &p)
    : usePhase2Stacks_(p.getParameter<bool>("usePhase2Stacks")) {
  auto c = setWhatProduced(this);

  // 08-Oct-2007 - Patrick Janot
  // Allow several reco geometries to be created, corresponding to the labelled
  // TrackerDigiGeometry's - that must created beforehand. Useful to handle an
  // aligned and a misaligned geometry in the same job.
  // The default parameter ("") makes this change transparent to the user
  // See FastSimulation/Configuration/data/ for examples of cfi's.
  tTopToken_ = c.consumes();
  geomToken_ = c.consumes(edm::ESInputTag("", p.getUntrackedParameter<std::string>("trackerGeometryLabel")));
}

std::unique_ptr<GeometricSearchTracker> TrackerRecoGeometryESProducer::produce(
    const TrackerRecoGeometryRecord &iRecord) {
  TrackerGeometry const &tG = iRecord.get(geomToken_);

  GeometricSearchTrackerBuilder builder;
  return std::unique_ptr<GeometricSearchTracker>(
      builder.build(tG.trackerDet(), &tG, &iRecord.get(tTopToken_), usePhase2Stacks_));
}

void TrackerRecoGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("usePhase2Stacks", false);
  desc.addUntracked<std::string>("trackerGeometryLabel", "");
  descriptions.addDefault(desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerRecoGeometryESProducer);
