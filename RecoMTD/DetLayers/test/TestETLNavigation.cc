#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"

#include "RecoMTD/DetLayers/interface/MTDSectorForwardDoubleLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetSector.h"

#include <DataFormats/ForwardDetId/interface/ETLDetId.h>

#include <sstream>

#include "DataFormats/Math/interface/Rounding.h"

using namespace std;
using namespace edm;
using namespace cms_rounding;

class TestETLNavigation : public global::EDAnalyzer<> {
public:
  TestETLNavigation(const ParameterSet& pset);

  void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

private:
  inline std::string fround(const double in, const size_t prec) const {
    std::stringstream ss;
    ss << std::setprecision(prec) << std::fixed << std::setw(14) << roundIfNear0(in);
    return ss.str();
  }

  inline std::string fvecround(const GlobalPoint vecin, const size_t prec) const {
    std::stringstream ss;
    ss << std::setprecision(prec) << std::fixed << std::setw(14) << roundVecIfNear0(vecin);
    return ss.str();
  }

  const edm::ESInputTag tag_;
  edm::ESGetToken<MTDDetLayerGeometry, MTDRecoGeometryRecord> geomToken_;
};

TestETLNavigation::TestETLNavigation(const ParameterSet& iConfig) : tag_(edm::ESInputTag{"", ""}) {
  geomToken_ = esConsumes<MTDDetLayerGeometry, MTDRecoGeometryRecord>(tag_);
}

void TestETLNavigation::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const& es) const {
  auto geo = es.getTransientHandle(geomToken_);

  const vector<const DetLayer*>& layers = geo->allETLLayers();

  // dump of ETL layers structure

  LogVerbatim("MTDLayerDumpFull") << "\n\nTest of ETL navigation \n\n";
  LogVerbatim("MTDLayerDump") << "\n\nTest of ETL navigation \n\n";

  for (const auto& ilay : layers) {
    const MTDSectorForwardDoubleLayer* layer = static_cast<const MTDSectorForwardDoubleLayer*>(ilay);

    LogVerbatim("MTDLayerDumpFull") << std::fixed << "\nETL layer " << std::setw(4) << layer->subDetector()
                                    << " at z = " << fround(layer->surface().position().z(), 4)
                                    << " sectors = " << std::setw(14) << layer->sectors().size()
                                    << " dets = " << std::setw(14) << layer->basicComponents().size()
                                    << " front dets = " << std::setw(14)
                                    << layer->frontLayer()->basicComponents().size() << " back dets = " << std::setw(14)
                                    << layer->backLayer()->basicComponents().size();
    LogVerbatim("MTDLayerDump") << std::fixed << "\nETL layer " << std::setw(4) << layer->subDetector()
                                << " at z = " << fround(layer->surface().position().z(), 2)
                                << " sectors = " << std::setw(14) << layer->sectors().size()
                                << " dets = " << std::setw(14) << layer->basicComponents().size()
                                << " front dets = " << std::setw(14) << layer->frontLayer()->basicComponents().size()
                                << " back dets = " << std::setw(14) << layer->backLayer()->basicComponents().size();

    unsigned int isectInd(0);
    for (const auto& isector : layer->sectors()) {
      isectInd++;
      LogVerbatim("MTDLayerDumpFull") << std::fixed << "\nSector " << std::setw(4) << isectInd << "\n" << (*isector);
      LogVerbatim("MTDLayerDump") << std::fixed << "\nSector " << std::setw(4) << isectInd << "\n" << (*isector);
      unsigned int imodInd(0);
      for (const auto& imod : isector->basicComponents()) {
        imodInd++;
        ETLDetId modId(imod->geographicalId().rawId());
        LogVerbatim("MTDLayerDumpFull") << std::fixed << std::setw(5) << imodInd << " ETLDetId " << modId.rawId()
                                        << " side = " << std::setw(4) << modId.mtdSide()
                                        << " Disc/Side/Sector = " << std::setw(4) << modId.nDisc() << " "
                                        << std::setw(4) << modId.discSide() << " " << std::setw(4) << modId.sector()
                                        << " mod/type/sens = " << std::setw(4) << modId.module() << " " << std::setw(4)
                                        << modId.modType() << std::setw(4) << modId.sensor()
                                        << " pos = " << fvecround(imod->position(), 4);
        LogVerbatim("MTDLayerDump") << std::fixed << std::setw(5) << imodInd << " ETLDetId " << modId.rawId()
                                    << " side = " << std::setw(4) << modId.mtdSide()
                                    << " Disc/Side/Sector = " << std::setw(4) << modId.nDisc() << " " << std::setw(4)
                                    << modId.discSide() << " " << std::setw(4) << modId.sector()
                                    << " mod/type/sens = " << std::setw(4) << modId.module() << " " << std::setw(4)
                                    << modId.modType() << std::setw(4) << modId.sensor()
                                    << " pos = " << fvecround(imod->position(), 2);
        for (int iside = -1; iside <= 1; iside += 2) {
          size_t idetNew = isector->hshift(modId, iside);
          if (idetNew >= isector->basicComponents().size()) {
            LogVerbatim("MTDLayerDumpFull")
                << "...............hshift= " << std::fixed << std::setw(2) << iside << " out of range";
            LogVerbatim("MTDLayerDump") << "...............hshift= " << std::fixed << std::setw(2) << iside
                                        << " out of range";
          } else {
            ETLDetId newId(isector->basicComponents()[idetNew]->geographicalId().rawId());
            LogVerbatim("MTDLayerDumpFull")
                << std::fixed << "...............hshift= " << std::setw(2) << iside << " side = " << std::setw(4)
                << newId.mtdSide() << " Disc/Side/Sector = " << std::setw(4) << newId.nDisc() << " " << std::setw(4)
                << newId.discSide() << " " << std::setw(4) << newId.sector() << " mod/type/sens = " << std::setw(4)
                << newId.module() << " " << std::setw(4) << newId.modType() << std::setw(4) << newId.sensor()
                << " pos = " << fvecround(isector->basicComponents()[idetNew]->position(), 4);
            LogVerbatim("MTDLayerDump") << std::fixed << "...............hshift= " << std::setw(2) << iside
                                        << " side = " << std::setw(4) << newId.mtdSide()
                                        << " Disc/Side/Sector = " << std::setw(4) << newId.nDisc() << " "
                                        << std::setw(4) << newId.discSide() << " " << std::setw(4) << newId.sector()
                                        << " mod/type/sens = " << std::setw(4) << newId.module() << " " << std::setw(4)
                                        << newId.modType() << std::setw(4) << newId.sensor()
                                        << " pos = " << fvecround(isector->basicComponents()[idetNew]->position(), 2);
          }
        }
        for (int iside = -1; iside <= 1; iside += 2) {
          size_t closest(isector->basicComponents().size());
          size_t idetNew = isector->vshift(modId, iside, closest);
          if (idetNew >= isector->basicComponents().size()) {
            LogVerbatim("MTDLayerDumpFull")
                << "...............vshift= " << std::fixed << std::setw(2) << iside << " out of range";
            LogVerbatim("MTDLayerDump") << "...............vshift= " << std::fixed << std::setw(2) << iside
                                        << " out of range";
            if (closest < isector->basicComponents().size()) {
              ETLDetId newId(isector->basicComponents()[closest]->geographicalId().rawId());
              LogVerbatim("MTDLayerDumpFull")
                  << std::fixed << ".......closest.vshift= " << std::setw(2) << iside << " side = " << std::setw(4)
                  << newId.mtdSide() << " Disc/Side/Sector = " << std::setw(4) << newId.nDisc() << " " << std::setw(4)
                  << newId.discSide() << " " << std::setw(4) << newId.sector() << " mod/type/sens = " << std::setw(4)
                  << newId.module() << " " << std::setw(4) << newId.modType() << std::setw(4) << newId.sensor()
                  << " pos = " << fvecround(isector->basicComponents()[closest]->position(), 4);
              LogVerbatim("MTDLayerDump")
                  << std::fixed << ".......closest.vshift= " << std::setw(2) << iside << " side = " << std::setw(4)
                  << newId.mtdSide() << " Disc/Side/Sector = " << std::setw(4) << newId.nDisc() << " " << std::setw(4)
                  << newId.discSide() << " " << std::setw(4) << newId.sector() << " mod/type/sens = " << std::setw(4)
                  << newId.module() << " " << std::setw(4) << newId.modType() << std::setw(4) << newId.sensor()
                  << " pos = " << fvecround(isector->basicComponents()[closest]->position(), 2);
            }
          } else {
            ETLDetId newId(isector->basicComponents()[idetNew]->geographicalId().rawId());
            LogVerbatim("MTDLayerDumpFull")
                << std::fixed << "...............vshift= " << std::setw(2) << iside << " side = " << std::setw(4)
                << newId.mtdSide() << " Disc/Side/Sector = " << std::setw(4) << newId.nDisc() << " " << std::setw(4)
                << newId.discSide() << " " << std::setw(4) << newId.sector() << " mod/type/sens = " << std::setw(4)
                << newId.module() << " " << std::setw(4) << newId.modType() << std::setw(4) << newId.sensor()
                << " pos = " << fvecround(isector->basicComponents()[idetNew]->position(), 4);
            LogVerbatim("MTDLayerDump") << std::fixed << "...............vshift= " << std::setw(2) << iside
                                        << " side = " << std::setw(4) << newId.mtdSide()
                                        << " Disc/Side/Sector = " << std::setw(4) << newId.nDisc() << " "
                                        << std::setw(4) << newId.discSide() << " " << std::setw(4) << newId.sector()
                                        << " mod/type/sens = " << std::setw(4) << newId.module() << " " << std::setw(4)
                                        << newId.modType() << std::setw(4) << newId.sensor()
                                        << " pos = " << fvecround(isector->basicComponents()[idetNew]->position(), 2);
          }
        }
      }
    }
  }
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(TestETLNavigation);
