/// Stand-alone test of bad strip channels testing via CSCRecoConditions
/// Tim Cox - 07-Oct-2014 - now using MessageLogger

#include <string>
#include <vector>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "CalibMuon/CSCCalibration/interface/CSCIndexerBase.h"
#include "CalibMuon/CSCCalibration/interface/CSCIndexerRecord.h"
#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperBase.h"
#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperRecord.h"

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>

#include <RecoLocalMuon/CSCRecHitD/src/CSCRecoConditions.h>

class CSCRecoBadChannelsAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit CSCRecoBadChannelsAnalyzer(edm::ParameterSet const& ps)
      : dashedLineWidth_(80),
        dashedLine_(std::string(dashedLineWidth_, '-')),
        myName_("CSCRecoBadChannelsAnalyzer"),
        readBadChannels_(ps.getParameter<bool>("readBadChannels")),
        recoConditions_(new CSCRecoConditions(ps, consumesCollector())),
        indexerToken_(esConsumes<CSCIndexerBase, CSCIndexerRecord>()),
        mapperToken_(esConsumes<CSCChannelMapperBase, CSCChannelMapperRecord>()),
        geometryToken_(esConsumes<CSCGeometry, MuonGeometryRecord>()) {}

  ~CSCRecoBadChannelsAnalyzer() override { delete recoConditions_; }
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  /// did we request reading bad channel info from db?
  bool readBadChannels() const { return readBadChannels_; }
  const std::string& myName() { return myName_; }

private:
  const int dashedLineWidth_;
  const std::string dashedLine_;
  const std::string myName_;

  bool readBadChannels_;
  CSCRecoConditions* recoConditions_;
  edm::ESGetToken<CSCIndexerBase, CSCIndexerRecord> indexerToken_;
  edm::ESGetToken<CSCChannelMapperBase, CSCChannelMapperRecord> mapperToken_;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> geometryToken_;
};

void CSCRecoBadChannelsAnalyzer::analyze(const edm::Event& ev, const edm::EventSetup& evsetup) {
  using namespace edm::eventsetup;

  edm::LogVerbatim("CSCBadChannels") << myName() << "::analyze running...";
  edm::LogVerbatim("CSCBadChannels") << "start " << dashedLine_;

  edm::LogVerbatim("CSCBadChannels") << "RUN# " << ev.id().run();
  edm::LogVerbatim("CSCBadChannels") << "EVENT# " << ev.id().event();

  edm::ESHandle<CSCIndexerBase> theIndexer = evsetup.getHandle(indexerToken_);

  edm::LogVerbatim("CSCBadChannels") << myName() << "::analyze sees indexer " << theIndexer->name()
                                     << " in Event Setup";

  edm::ESHandle<CSCChannelMapperBase> theMapper = evsetup.getHandle(mapperToken_);

  edm::LogVerbatim("CSCBadChannels") << myName() << "::analyze sees mapper " << theMapper->name() << " in Event Setup";

  edm::ESHandle<CSCGeometry> theGeometry = evsetup.getHandle(geometryToken_);

  edm::LogVerbatim("CSCBadChannels") << " Geometry node for CSCGeom is  " << &(*theGeometry);
  edm::LogVerbatim("CSCBadChannels") << " There are " << theGeometry->dets().size() << " dets";
  edm::LogVerbatim("CSCBadChannels") << " There are " << theGeometry->detTypes().size() << " types"
                                     << "\n";

  // INITIALIZE CSCConditions
  recoConditions_->initializeEvent(evsetup);

  // HERE NEED TO ITERATE OVER ALL CSCDetId

  edm::LogVerbatim("CSCBadChannels") << myName() << ": Begin iteration over geometry...";

  const CSCGeometry::LayerContainer& vecOfLayers = theGeometry->layers();
  edm::LogVerbatim("CSCBadChannels") << "There are " << vecOfLayers.size() << " layers";

  edm::LogVerbatim("CSCBadChannels") << dashedLine_;

  int ibadchannels = 0;  // COUNT OF BAD STRIP CHANNELS
  int ibadlayers = 0;    //COUNT OF LAYERS WITH BAD STRIP CHANNELS

  for (auto it = vecOfLayers.begin(); it != vecOfLayers.end(); ++it) {
    const CSCLayer* layer = *it;

    if (layer) {
      CSCDetId id = layer->id();
      int nstrips = layer->geometry()->numberOfStrips();
      edm::LogVerbatim("CSCBadChannels") << "Layer " << id << " has " << nstrips << " strips";

      // GET BAD CHANNELS FOR THIS LAYER

      recoConditions_->fillBadChannelWords(id);

      // SEARCH FOR BAD STRIP CHANNELS IN THIS LAYER - GEOMETRIC STRIP INPUT!!

      bool layerhasbadchannels = false;
      for (short is = 1; is <= nstrips; ++is) {
        if (recoConditions_->badStrip(id, is, nstrips)) {
          ++ibadchannels;
          layerhasbadchannels = true;
          edm::LogVerbatim("CSCBadChannels") << id << " strip " << is << " is bad";
        }
      }

      for (short is = 1; is <= nstrips; ++is) {
        if (recoConditions_->nearBadStrip(id, is, nstrips)) {
          edm::LogVerbatim("CSCBadChannels") << id << " strip " << is << " is a neighbor of a bad strip";
        }
      }

      if (layerhasbadchannels)
        ++ibadlayers;

    } else {
      edm::LogVerbatim("CSCBadChannels") << "WEIRD ERROR: a null CSCLayer* was seen";
    }
  }

  edm::LogVerbatim("CSCBadChannels") << "No. of layers with bad strip channels = " << ibadlayers;
  edm::LogVerbatim("CSCBadChannels") << "No. of bad strip channels seen = " << ibadchannels;

  edm::LogVerbatim("CSCBadChannels") << dashedLine_ << " end";
}

DEFINE_FWK_MODULE(CSCRecoBadChannelsAnalyzer);
