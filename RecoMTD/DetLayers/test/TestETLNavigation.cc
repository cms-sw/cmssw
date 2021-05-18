#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"

#include "RecoMTD/DetLayers/interface/MTDSectorForwardDoubleLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetSector.h"

#include <DataFormats/ForwardDetId/interface/ETLDetId.h>

#include <sstream>

using namespace std;
using namespace edm;

class TestETLNavigation : public EDAnalyzer {
public:
  TestETLNavigation(const ParameterSet& pset);

  void analyze(const Event& ev, const EventSetup& es) override;

private:
  const edm::ESInputTag tag_;
  edm::ESGetToken<MTDDetLayerGeometry, MTDRecoGeometryRecord> geomToken_;
};

TestETLNavigation::TestETLNavigation(const ParameterSet& iConfig) : tag_(edm::ESInputTag{"", ""}) {
  geomToken_ = esConsumes<MTDDetLayerGeometry, MTDRecoGeometryRecord>(tag_);
}

void TestETLNavigation::analyze(const Event& ev, const EventSetup& es) {
  auto geo = es.getTransientHandle(geomToken_);

  const vector<const DetLayer*>& layers = geo->allETLLayers();

  // dump of ETL layers structure

  LogVerbatim("MTDLayerDump") << "\n\nTest of ETL navigation \n\n";

  for (const auto& ilay : layers) {
    const MTDSectorForwardDoubleLayer* layer = static_cast<const MTDSectorForwardDoubleLayer*>(ilay);

    LogVerbatim("MTDLayerDump") << std::fixed << "\nETL layer " << std::setw(4) << layer->subDetector()
                                << " at z = " << std::setw(14) << layer->surface().position().z()
                                << " sectors = " << std::setw(14) << layer->sectors().size()
                                << " dets = " << std::setw(14) << layer->basicComponents().size()
                                << " front dets = " << std::setw(14) << layer->frontLayer()->basicComponents().size()
                                << " back dets = " << std::setw(14) << layer->backLayer()->basicComponents().size();

    unsigned int isectInd(0);
    for (const auto& isector : layer->sectors()) {
      isectInd++;
      LogVerbatim("MTDLayerDump") << std::fixed << "\nSector " << std::setw(4) << isectInd << "\n" << (*isector);
      unsigned int imodInd(0);
      for (const auto& imod : isector->basicComponents()) {
        imodInd++;
        ETLDetId modId(imod->geographicalId().rawId());
        LogVerbatim("MTDLayerDump") << std::fixed << std::setw(5) << imodInd << " ETLDetId " << modId.rawId()
                                    << " side = " << std::setw(4) << modId.mtdSide()
                                    << " Disc/Side/Sector = " << std::setw(4) << modId.nDisc() << " " << std::setw(4)
                                    << modId.discSide() << " " << std::setw(4) << modId.sector()
                                    << " mod/type = " << std::setw(4) << modId.module() << " " << std::setw(4)
                                    << modId.modType() << " pos = " << std::setprecision(4) << imod->position();
        for (int iside = -1; iside <= 1; iside += 2) {
          size_t idetNew = isector->hshift(modId, iside);
          if (idetNew >= isector->basicComponents().size()) {
            LogVerbatim("MTDLayerDump") << "...............hshift= " << std::fixed << std::setw(2) << iside
                                        << " out of range";
          } else {
            ETLDetId newId(isector->basicComponents()[idetNew]->geographicalId().rawId());
            LogVerbatim("MTDLayerDump") << std::fixed << "...............hshift= " << std::setw(2) << iside
                                        << " side = " << std::setw(4) << newId.mtdSide()
                                        << " Disc/Side/Sector = " << std::setw(4) << newId.nDisc() << " "
                                        << std::setw(4) << newId.discSide() << " " << std::setw(4) << newId.sector()
                                        << " mod/type = " << std::setw(4) << newId.module() << " " << std::setw(4)
                                        << newId.modType() << " pos = " << std::setprecision(4)
                                        << isector->basicComponents()[idetNew]->position();
          }
        }
        for (int iside = -1; iside <= 1; iside += 2) {
          size_t closest(isector->basicComponents().size());
          size_t idetNew = isector->vshift(modId, iside, closest);
          if (idetNew >= isector->basicComponents().size()) {
            LogVerbatim("MTDLayerDump") << "...............vshift= " << std::fixed << std::setw(2) << iside
                                        << " out of range";
            if (closest < isector->basicComponents().size()) {
              ETLDetId newId(isector->basicComponents()[closest]->geographicalId().rawId());
              LogVerbatim("MTDLayerDump")
                  << std::fixed << ".......closest.vshift= " << std::setw(2) << iside << " side = " << std::setw(4)
                  << newId.mtdSide() << " Disc/Side/Sector = " << std::setw(4) << newId.nDisc() << " " << std::setw(4)
                  << newId.discSide() << " " << std::setw(4) << newId.sector() << " mod/type = " << std::setw(4)
                  << newId.module() << " " << std::setw(4) << newId.modType() << " pos = " << std::setprecision(4)
                  << isector->basicComponents()[closest]->position();
            }
          } else {
            ETLDetId newId(isector->basicComponents()[idetNew]->geographicalId().rawId());
            LogVerbatim("MTDLayerDump") << std::fixed << "...............vshift= " << std::setw(2) << iside
                                        << " side = " << std::setw(4) << newId.mtdSide()
                                        << " Disc/Side/Sector = " << std::setw(4) << newId.nDisc() << " "
                                        << std::setw(4) << newId.discSide() << " " << std::setw(4) << newId.sector()
                                        << " mod/type = " << std::setw(4) << newId.module() << " " << std::setw(4)
                                        << newId.modType() << " pos = " << std::setprecision(4)
                                        << isector->basicComponents()[idetNew]->position();
          }
        }
      }
    }
  }
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(TestETLNavigation);
