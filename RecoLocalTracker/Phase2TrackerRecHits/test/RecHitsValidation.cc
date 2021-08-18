#include <iostream>
#include <map>
#include <vector>
#include <algorithm>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include <TH2F.h>
#include <TH1F.h>

struct RecHitHistos {
  // use TH1D instead of TH1F to avoid stauration at 2^31
  // above this increments with +1 don't work for float, need double

  TH1D* numberRecHits[3];
  TH1D* clusterSize[3];

  TH2F* globalPosXY[3][5];
  TH2F* localPosXY[3][5];

  TH1F* deltaX[3][5];
  TH1F* deltaY[3][5];
  TH1F* deltaX_P[3][5];
  TH1F* deltaY_P[3][5];

  TH1F* pullX[3][5];
  TH1F* pullY[3][5];
  TH1F* pullX_P[3][5];
  TH1F* pullY_P[3][5];

  TH2F* deltaX_eta[3][5];
  TH2F* deltaY_eta[3][5];
  TH2F* deltaX_eta_P[3][5];
  TH2F* deltaY_eta_P[3][5];

  TH2F* pullX_eta[3][5];
  TH2F* pullY_eta[3][5];
  TH2F* pullX_eta_P[3][5];
  TH2F* pullY_eta_P[3][5];

  TH1D* primarySimHits[3];
  TH1D* otherSimHits[3];
};

class Phase2TrackerRecHitsValidation : public edm::EDAnalyzer {
public:
  typedef std::map<unsigned int, std::vector<PSimHit> > SimHitsMap;
  typedef std::map<unsigned int, SimTrack> SimTracksMap;

  explicit Phase2TrackerRecHitsValidation(const edm::ParameterSet&);
  ~Phase2TrackerRecHitsValidation();
  void beginJob() override;
  void endJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  std::map<unsigned int, RecHitHistos>::iterator createLayerHistograms(unsigned int);
  std::vector<unsigned int> getSimTrackId(const edm::Handle<edm::DetSetVector<PixelDigiSimLink> >&,
                                          const DetId&,
                                          unsigned int);

  edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> tokenRecHits_;
  edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> tokenClusters_;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink> > tokenLinks_;
  edm::EDGetTokenT<edm::PSimHitContainer> tokenSimHitsB_;
  edm::EDGetTokenT<edm::PSimHitContainer> tokenSimHitsE_;
  edm::EDGetTokenT<edm::SimTrackContainer> tokenSimTracks_;

  bool catECasRings_;
  double simtrackminpt_;
  bool makeEtaPlots_;
  double mineta_;
  double maxeta_;

  TH2F* trackerLayout_;
  TH2F* trackerLayoutXY_;
  TH2F* trackerLayoutXYBar_;
  TH2F* trackerLayoutXYEC_;

  std::map<unsigned int, RecHitHistos> histograms_;
};

Phase2TrackerRecHitsValidation::Phase2TrackerRecHitsValidation(const edm::ParameterSet& conf)
    : tokenRecHits_(consumes<Phase2TrackerRecHit1DCollectionNew>(conf.getParameter<edm::InputTag>("src"))),
      tokenClusters_(consumes<Phase2TrackerCluster1DCollectionNew>(conf.getParameter<edm::InputTag>("clusters"))),
      tokenLinks_(consumes<edm::DetSetVector<PixelDigiSimLink> >(conf.getParameter<edm::InputTag>("links"))),
      tokenSimHitsB_(consumes<edm::PSimHitContainer>(conf.getParameter<edm::InputTag>("simhitsbarrel"))),
      tokenSimHitsE_(consumes<edm::PSimHitContainer>(conf.getParameter<edm::InputTag>("simhitsendcap"))),
      tokenSimTracks_(consumes<edm::SimTrackContainer>(conf.getParameter<edm::InputTag>("simtracks"))),
      catECasRings_(conf.getParameter<bool>("ECasRings")),
      simtrackminpt_(conf.getParameter<double>("SimTrackMinPt")),
      makeEtaPlots_(conf.getParameter<bool>("MakeEtaPlots")),
      mineta_(conf.getParameter<double>("MinEta")),
      maxeta_(conf.getParameter<double>("MaxEta")) {}

Phase2TrackerRecHitsValidation::~Phase2TrackerRecHitsValidation() {}

void Phase2TrackerRecHitsValidation::beginJob() {
  edm::Service<TFileService> fs;
  fs->file().cd("/");
  TFileDirectory td = fs->mkdir("Common");
  // Create common histograms
  trackerLayout_ = td.make<TH2F>("RVsZ", "R vs. z position", 6000, -300.0, 300.0, 1200, 0.0, 120.0);
  trackerLayoutXY_ = td.make<TH2F>("XVsY", "x vs. y position", 2400, -120.0, 120.0, 2400, -120.0, 120.0);
  trackerLayoutXYBar_ = td.make<TH2F>("XVsYBar", "x vs. y position", 2400, -120.0, 120.0, 2400, -120.0, 120.0);
  trackerLayoutXYEC_ = td.make<TH2F>("XVsYEC", "x vs. y position", 2400, -120.0, 120.0, 2400, -120.0, 120.0);
}

void Phase2TrackerRecHitsValidation::endJob() {}

void Phase2TrackerRecHitsValidation::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  /*
     * Get the needed objects
     */

  // Get the RecHits
  edm::Handle<Phase2TrackerRecHit1DCollectionNew> rechits;
  event.getByToken(tokenRecHits_, rechits);

  // Get the Clusters
  edm::Handle<Phase2TrackerCluster1DCollectionNew> clusters;
  event.getByToken(tokenClusters_, clusters);

  // Get the PixelDigiSimLinks
  edm::Handle<edm::DetSetVector<PixelDigiSimLink> > pixelSimLinks;
  event.getByToken(tokenLinks_, pixelSimLinks);

  // Get the SimHits
  edm::Handle<edm::PSimHitContainer> simHitsRaw[2];
  event.getByToken(tokenSimHitsB_, simHitsRaw[0]);
  event.getByToken(tokenSimHitsE_, simHitsRaw[1]);

  // Get the SimTracks
  edm::Handle<edm::SimTrackContainer> simTracksRaw;
  event.getByToken(tokenSimTracks_, simTracksRaw);

  // Get the geometry
  edm::ESHandle<TrackerGeometry> geomHandle;
  eventSetup.get<TrackerDigiGeometryRecord>().get(geomHandle);
  const TrackerGeometry* tkGeom = &(*geomHandle);

  edm::ESHandle<TrackerTopology> tTopoHandle;
  eventSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* tTopo = tTopoHandle.product();

  /*
     * Rearrange the simTracks
     */

  // Rearrange the simTracks for ease of use <simTrackID, simTrack>
  SimTracksMap simTracks;
  for (edm::SimTrackContainer::const_iterator simTrackIt(simTracksRaw->begin()); simTrackIt != simTracksRaw->end();
       ++simTrackIt) {
    if (simTrackIt->momentum().pt() > simtrackminpt_) {
      simTracks.insert(std::pair<unsigned int, SimTrack>(simTrackIt->trackId(), *simTrackIt));
    }
  }

  /*
     * Validation   
     */

  // Number of rechits
  std::map<unsigned int, unsigned int> nRecHits[3];
  std::map<unsigned int, unsigned int> nPrimarySimHits[3];
  std::map<unsigned int, unsigned int> nOtherSimHits[3];

  // Loop over modules
  for (Phase2TrackerRecHit1DCollectionNew::const_iterator DSViter = rechits->begin(); DSViter != rechits->end();
       ++DSViter) {
    // Get the detector unit's id
    unsigned int rawid(DSViter->detId());
    DetId detId(rawid);
    unsigned int layer = (tTopo->side(detId) != 0) * 1000;  // don't split up endcap sides
    if (!layer) {
      layer += tTopo->layer(detId);
    } else {
      layer += (catECasRings_ ? tTopo->tidRing(detId) * 10 : tTopo->layer(detId));
    }

    // determine the detector we are in
    TrackerGeometry::ModuleType mType = tkGeom->getDetectorType(detId);
    unsigned int det = 0;
    if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
      det = 1;
    } else if (mType == TrackerGeometry::ModuleType::Ph2PSS || mType == TrackerGeometry::ModuleType::Ph2SS) {
      det = 2;
    } else {
      std::cout << "UNKNOWN DETECTOR TYPE!" << std::endl;
    }

    // Get the geomdet
    const GeomDetUnit* geomDetUnit(tkGeom->idToDetUnit(detId));
    if (!geomDetUnit)
      continue;

    // initialize the nhit counters if they don't exist for this layer
    auto nhitit(nRecHits[det].find(layer));
    if (nhitit == nRecHits[det].end()) {
      nRecHits[det].emplace(layer, 0);
      nPrimarySimHits[det].emplace(layer, 0);
      nOtherSimHits[det].emplace(layer, 0);
    }

    // Create histograms if they do not yet exist for this layer
    std::map<unsigned int, RecHitHistos>::iterator histogramLayer(histograms_.find(layer));
    if (histogramLayer == histograms_.end())
      histogramLayer = createLayerHistograms(layer);

    // Loop over the rechits in the detector unit
    for (edmNew::DetSet<Phase2TrackerRecHit1D>::const_iterator rechitIt = DSViter->begin(); rechitIt != DSViter->end();
         ++rechitIt) {
      // determine the position
      LocalPoint localPosClu = rechitIt->localPosition();
      Global3DPoint globalPosClu = geomDetUnit->surface().toGlobal(localPosClu);

      // restrict eta range
      float eta = globalPosClu.eta();
      if (fabs(eta) < mineta_ || fabs(eta) > maxeta_)
        continue;

      // Get the cluster from the rechit
      const Phase2TrackerCluster1D* clustIt = &*rechitIt->cluster();

      // Get all the simTracks that form the cluster
      std::vector<unsigned int> clusterSimTrackIds;
      for (unsigned int i(0); i < clustIt->size(); ++i) {
        unsigned int channel(Phase2TrackerDigi::pixelToChannel(clustIt->firstRow() + i, clustIt->column()));
        std::vector<unsigned int> simTrackIds(getSimTrackId(pixelSimLinks, detId, channel));
        for (unsigned int i = 0; i < simTrackIds.size(); ++i) {
          bool add = true;
          for (unsigned int j = 0; j < clusterSimTrackIds.size(); ++j) {
            // only save simtrackids that are not present yet
            if (simTrackIds.at(i) == clusterSimTrackIds.at(j))
              add = false;
          }
          if (add)
            clusterSimTrackIds.push_back(simTrackIds.at(i));
        }
      }

      // find the closest simhit
      // this is needed because otherwise you get cases with simhits and clusters being swapped
      // when there are more than 1 cluster with common simtrackids
      const PSimHit* simhit = 0;  // bad naming to avoid changing code below. This is the closest simhit in x
      float minx = 10000;
      for (unsigned int simhitidx = 0; simhitidx < 2; ++simhitidx) {  // loop over both barrel and endcap hits
        for (edm::PSimHitContainer::const_iterator simhitIt(simHitsRaw[simhitidx]->begin());
             simhitIt != simHitsRaw[simhitidx]->end();
             ++simhitIt) {
          if (rawid == simhitIt->detUnitId()) {
            //std::cout << "=== " << rawid << " " << &*simhitIt << " " << simhitIt->trackId() << " " << simhitIt->localPosition().x() << " " << simhitIt->localPosition().y() << std::endl;
            auto it = std::lower_bound(clusterSimTrackIds.begin(), clusterSimTrackIds.end(), simhitIt->trackId());
            if (it != clusterSimTrackIds.end() && *it == simhitIt->trackId()) {
              if (!simhit || fabs(simhitIt->localPosition().x() - localPosClu.x()) < minx) {
                minx = fabs(simhitIt->localPosition().x() - localPosClu.x());
                simhit = &*simhitIt;
              }
            }
          }
        }
      }
      if (!simhit)
        continue;

      // only look at simhits from highpT tracks
      std::map<unsigned int, SimTrack>::const_iterator simTrackIt(simTracks.find(simhit->trackId()));
      if (simTrackIt == simTracks.end())
        continue;

      /*
             * Rechit related variables
             */

      ++(nRecHits[det].at(layer));
      ++(nOtherSimHits[det].at(layer));

      // cluster size
      unsigned int nch = rechitIt->cluster()->size();
      histogramLayer->second.clusterSize[det]->Fill(nch);
      if (nch > 4)
        nch = 4;  // collapse 4 or more strips to 4

      // Fill the position histograms
      trackerLayout_->Fill(globalPosClu.z(), globalPosClu.perp());
      trackerLayoutXY_->Fill(globalPosClu.x(), globalPosClu.y());
      if (layer < 1000) {
        trackerLayoutXYBar_->Fill(globalPosClu.x(), globalPosClu.y());
      } else {
        trackerLayoutXYEC_->Fill(globalPosClu.x(), globalPosClu.y());
      }

      histogramLayer->second.localPosXY[det][0]->Fill(localPosClu.x(), localPosClu.y());
      histogramLayer->second.localPosXY[det][nch]->Fill(localPosClu.x(), localPosClu.y());
      if (layer < 1000) {
        histogramLayer->second.globalPosXY[det][0]->Fill(globalPosClu.z(), globalPosClu.perp());
        histogramLayer->second.globalPosXY[det][nch]->Fill(globalPosClu.z(), globalPosClu.perp());
      } else {
        histogramLayer->second.globalPosXY[det][0]->Fill(globalPosClu.x(), globalPosClu.y());
        histogramLayer->second.globalPosXY[det][nch]->Fill(globalPosClu.x(), globalPosClu.y());
      }

      // now get the position of the closest hit
      Local3DPoint localPosHit(simhit->localPosition());

      // and fill bias and pull histograms
      histogramLayer->second.deltaX[det][0]->Fill(localPosClu.x() - localPosHit.x());
      histogramLayer->second.deltaX[det][nch]->Fill(localPosClu.x() - localPosHit.x());
      histogramLayer->second.deltaY[det][0]->Fill(localPosClu.y() - localPosHit.y());
      histogramLayer->second.deltaY[det][nch]->Fill(localPosClu.y() - localPosHit.y());
      if (rechitIt->localPositionError().xx() && rechitIt->localPositionError().yy()) {
        histogramLayer->second.pullX[det][0]->Fill((localPosClu.x() - localPosHit.x()) /
                                                   sqrt(rechitIt->localPositionError().xx()));
        histogramLayer->second.pullX[det][nch]->Fill((localPosClu.x() - localPosHit.x()) /
                                                     sqrt(rechitIt->localPositionError().xx()));
        histogramLayer->second.pullY[det][0]->Fill((localPosClu.y() - localPosHit.y()) /
                                                   sqrt(rechitIt->localPositionError().yy()));
        histogramLayer->second.pullY[det][nch]->Fill((localPosClu.y() - localPosHit.y()) /
                                                     sqrt(rechitIt->localPositionError().yy()));
      }
      if (makeEtaPlots_) {
        histogramLayer->second.deltaX_eta[det][0]->Fill(eta, localPosClu.x() - localPosHit.x());
        histogramLayer->second.deltaX_eta[det][nch]->Fill(eta, localPosClu.x() - localPosHit.x());
        histogramLayer->second.deltaY_eta[det][0]->Fill(eta, localPosClu.y() - localPosHit.y());
        histogramLayer->second.deltaY_eta[det][nch]->Fill(eta, localPosClu.y() - localPosHit.y());
        if (rechitIt->localPositionError().xx() && rechitIt->localPositionError().yy()) {
          histogramLayer->second.pullX_eta[det][0]->Fill(
              eta, (localPosClu.x() - localPosHit.x()) / sqrt(rechitIt->localPositionError().xx()));
          histogramLayer->second.pullX_eta[det][nch]->Fill(
              eta, (localPosClu.x() - localPosHit.x()) / sqrt(rechitIt->localPositionError().xx()));
          histogramLayer->second.pullY_eta[det][0]->Fill(
              eta, (localPosClu.y() - localPosHit.y()) / sqrt(rechitIt->localPositionError().yy()));
          histogramLayer->second.pullY_eta[det][nch]->Fill(
              eta, (localPosClu.y() - localPosHit.y()) / sqrt(rechitIt->localPositionError().yy()));
        }
      }

      // fill histos for primary particles only
      unsigned int procT(simhit->processType());
      if (simTrackIt->second.vertIndex() == 0 and
          (procT == 2 || procT == 7 || procT == 9 || procT == 11 || procT == 13 || procT == 15)) {
        ++(nPrimarySimHits[det].at(layer));
        --(nOtherSimHits[det].at(layer));  // avoid double counting
        histogramLayer->second.deltaX_P[det][0]->Fill(localPosClu.x() - localPosHit.x());
        histogramLayer->second.deltaX_P[det][nch]->Fill(localPosClu.x() - localPosHit.x());
        histogramLayer->second.deltaY_P[det][0]->Fill(localPosClu.y() - localPosHit.y());
        histogramLayer->second.deltaY_P[det][nch]->Fill(localPosClu.y() - localPosHit.y());
        if (rechitIt->localPositionError().xx() && rechitIt->localPositionError().yy()) {
          histogramLayer->second.pullX_P[det][0]->Fill((localPosClu.x() - localPosHit.x()) /
                                                       sqrt(rechitIt->localPositionError().xx()));
          histogramLayer->second.pullX_P[det][nch]->Fill((localPosClu.x() - localPosHit.x()) /
                                                         sqrt(rechitIt->localPositionError().xx()));
          histogramLayer->second.pullY_P[det][0]->Fill((localPosClu.y() - localPosHit.y()) /
                                                       sqrt(rechitIt->localPositionError().yy()));
          histogramLayer->second.pullY_P[det][nch]->Fill((localPosClu.y() - localPosHit.y()) /
                                                         sqrt(rechitIt->localPositionError().yy()));
        }
        if (makeEtaPlots_) {
          histogramLayer->second.deltaX_eta_P[det][0]->Fill(eta, localPosClu.x() - localPosHit.x());
          histogramLayer->second.deltaX_eta_P[det][nch]->Fill(eta, localPosClu.x() - localPosHit.x());
          histogramLayer->second.deltaY_eta_P[det][0]->Fill(eta, localPosClu.y() - localPosHit.y());
          histogramLayer->second.deltaY_eta_P[det][nch]->Fill(eta, localPosClu.y() - localPosHit.y());
          if (rechitIt->localPositionError().xx() && rechitIt->localPositionError().yy()) {
            histogramLayer->second.pullX_eta_P[det][0]->Fill(
                eta, (localPosClu.x() - localPosHit.x()) / sqrt(rechitIt->localPositionError().xx()));
            histogramLayer->second.pullX_eta_P[det][nch]->Fill(
                eta, (localPosClu.x() - localPosHit.x()) / sqrt(rechitIt->localPositionError().xx()));
            histogramLayer->second.pullY_eta_P[det][0]->Fill(
                eta, (localPosClu.y() - localPosHit.y()) / sqrt(rechitIt->localPositionError().yy()));
            histogramLayer->second.pullY_eta_P[det][nch]->Fill(
                eta, (localPosClu.y() - localPosHit.y()) / sqrt(rechitIt->localPositionError().yy()));
          }
        }
      }
    }
  }

  // fill the counter histos per layer
  for (unsigned int det = 1; det < 3; ++det) {
    for (auto it : nRecHits[det]) {
      auto histogramLayer(histograms_.find(it.first));
      if (histogramLayer == histograms_.end())
        std::cout << "*** SL *** No histogram for an existing counter! This should not happen!" << std::endl;
      histogramLayer->second.numberRecHits[det]->Fill(it.second);
    }
    for (auto it : nPrimarySimHits[det]) {
      auto histogramLayer(histograms_.find(it.first));
      if (histogramLayer == histograms_.end())
        std::cout << "*** SL *** No histogram for an existing counter! This should not happen!" << std::endl;
      histogramLayer->second.primarySimHits[det]->Fill(it.second);
    }
    for (auto it : nOtherSimHits[det]) {
      auto histogramLayer(histograms_.find(it.first));
      if (histogramLayer == histograms_.end())
        std::cout << "*** SL *** No histogram for an existing counter! This should not happen!" << std::endl;
      histogramLayer->second.otherSimHits[det]->Fill(it.second);
    }
  }
}

// Create the histograms
std::map<unsigned int, RecHitHistos>::iterator Phase2TrackerRecHitsValidation::createLayerHistograms(unsigned int ival) {
  std::ostringstream fname1, fname2;

  edm::Service<TFileService> fs;
  fs->file().cd("/");

  std::string tag;
  unsigned int id;
  if (ival < 1000) {
    id = ival;
    fname1 << "Barrel";
    fname2 << "Layer_" << id;
    tag = "_layer_";
  } else {
    int side = ival / 1000;
    id = ival - side * 1000;
    if (ival > 10) {
      id /= 10;
      //        fname1 << "EndCap_Side_" << side;
      fname1 << "EndCap";
      fname2 << "Ring_" << id;
      tag = "_ring_";
    } else {
      id = ival;
      //        fname1 << "EndCap_Side_" << side;
      fname1 << "EndCap";
      fname2 << "Disk_" << id;
      tag = "_disk_";
    }
  }

  TFileDirectory td1 = fs->mkdir(fname1.str().c_str());
  TFileDirectory td = td1.mkdir(fname2.str().c_str());

  RecHitHistos local_histos;

  std::ostringstream histoName;

  /*
     * Number of rechits
     */

  histoName.str("");
  histoName << "Number_RecHits_Pixel" << tag.c_str() << id;
  local_histos.numberRecHits[1] = td.make<TH1D>(histoName.str().c_str(), histoName.str().c_str(), 50, 0., 0.);
  histoName.str("");
  histoName << "Number_RecHits_Strip" << tag.c_str() << id;
  local_histos.numberRecHits[2] = td.make<TH1D>(histoName.str().c_str(), histoName.str().c_str(), 50, 0., 0.);

  /*
     * Cluster size
     */

  histoName.str("");
  histoName << "Cluster_Size_Pixel" << tag.c_str() << id;
  local_histos.clusterSize[1] = td.make<TH1D>(histoName.str().c_str(), histoName.str().c_str(), 21, -0.5, 20.5);
  histoName.str("");
  histoName << "Cluster_Size_Strip" << tag.c_str() << id;
  local_histos.clusterSize[2] = td.make<TH1D>(histoName.str().c_str(), histoName.str().c_str(), 21, -0.5, 20.5);

  /*
     * Local and Global positions
     */

  for (int cls = 0; cls < 5; ++cls) {
    std::string clsstr = "";
    if (cls > 0)
      clsstr = "_ClS_" + std::to_string(cls);

    histoName.str("");
    histoName << "Local_Position_XY_Pixel" << tag.c_str() << id << clsstr.c_str();
    local_histos.localPosXY[1][cls] =
        td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 500, 0., 0., 500, 0., 0.);

    histoName.str("");
    histoName << "Local_Position_XY_Strip" << tag.c_str() << id << clsstr.c_str();
    local_histos.localPosXY[2][cls] =
        td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 500, 0., 0., 500, 0., 0.);

    histoName.str("");
    histoName << "Global_Position_XY_Pixel" << tag.c_str() << id << clsstr.c_str();
    local_histos.globalPosXY[1][cls] =
        td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 500, -120., 120., 500, -120., 120.);

    histoName.str("");
    histoName << "Global_Position_XY_Strip" << tag.c_str() << id << clsstr.c_str();
    local_histos.globalPosXY[2][cls] =
        td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 500, -120., 120., 500, -120., 120.);

    /*
       * Delta positions with SimHits
       */

    histoName.str("");
    histoName << "Delta_X_Pixel" << tag.c_str() << id << clsstr.c_str();
    local_histos.deltaX[1][cls] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, -0.02, 0.02);

    histoName.str("");
    histoName << "Delta_X_Strip" << tag.c_str() << id << clsstr.c_str();
    local_histos.deltaX[2][cls] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, -0.02, 0.02);

    histoName.str("");
    histoName << "Delta_Y_Pixel" << tag.c_str() << id << clsstr.c_str();
    local_histos.deltaY[1][cls] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, -0.2, 0.2);

    histoName.str("");
    histoName << "Delta_Y_Strip" << tag.c_str() << id << clsstr.c_str();
    local_histos.deltaY[2][cls] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, -3, 3);

    if (makeEtaPlots_) {
      histoName.str("");
      histoName << "Delta_X_vs_Eta_Pixel" << tag.c_str() << id << clsstr.c_str();
      local_histos.deltaX_eta[1][cls] =
          td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 50, -2.5, 2.5, 100, -0.02, 0.02);

      histoName.str("");
      histoName << "Delta_X_vs_Eta_Strip" << tag.c_str() << id << clsstr.c_str();
      local_histos.deltaX_eta[2][cls] =
          td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 50, -2.5, 2.5, 100, -0.02, 0.02);

      histoName.str("");
      histoName << "Delta_Y_vs_Eta_Pixel" << tag.c_str() << id << clsstr.c_str();
      local_histos.deltaY_eta[1][cls] =
          td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 50, -2.5, 2.5, 100, -0.2, 0.2);

      histoName.str("");
      histoName << "Delta_Y_vs_Eta_Strip" << tag.c_str() << id << clsstr.c_str();
      local_histos.deltaY_eta[2][cls] =
          td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 50, -2.5, 2.5, 100, -3, 3);
    }

    /*
       * Pulls
       */

    histoName.str("");
    histoName << "Pull_X_Pixel" << tag.c_str() << id << clsstr.c_str();
    local_histos.pullX[1][cls] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, -3., 3.);

    histoName.str("");
    histoName << "Pull_X_Strip" << tag.c_str() << id << clsstr.c_str();
    local_histos.pullX[2][cls] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, -3., 3.);

    histoName.str("");
    histoName << "Pull_Y_Pixel" << tag.c_str() << id << clsstr.c_str();
    local_histos.pullY[1][cls] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, -3., 3.);

    histoName.str("");
    histoName << "Pull_Y_Strip" << tag.c_str() << id << clsstr.c_str();
    local_histos.pullY[2][cls] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, -3., 3.);

    if (makeEtaPlots_) {
      histoName.str("");
      histoName << "Pull_X_Eta_Pixel" << tag.c_str() << id << clsstr.c_str();
      local_histos.pullX_eta[1][cls] =
          td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 50, -2.5, 2.5, 100, -3., 3.);

      histoName.str("");
      histoName << "Pull_X_Eta_Strip" << tag.c_str() << id << clsstr.c_str();
      local_histos.pullX_eta[2][cls] =
          td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 50, -2.5, 2.5, 100, -3., 3.);

      histoName.str("");
      histoName << "Pull_Y_Eta_Pixel" << tag.c_str() << id << clsstr.c_str();
      local_histos.pullY_eta[1][cls] =
          td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 50, -2.5, 2.5, 100, -3., 3.);

      histoName.str("");
      histoName << "Pull_Y_Eta_Strip" << tag.c_str() << id << clsstr.c_str();
      local_histos.pullY_eta[2][cls] =
          td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 50, -2.5, 2.5, 100, -3., 3.);
    }

    /*
       * Delta position with simHits for primary tracks only
       */

    histoName.str("");
    histoName << "Delta_X_Pixel_P" << tag.c_str() << id << clsstr.c_str();
    local_histos.deltaX_P[1][cls] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, -0.02, 0.02);

    histoName.str("");
    histoName << "Delta_X_Strip_P" << tag.c_str() << id << clsstr.c_str();
    local_histos.deltaX_P[2][cls] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, -0.02, 0.02);

    histoName.str("");
    histoName << "Delta_Y_Pixel_P" << tag.c_str() << id << clsstr.c_str();
    local_histos.deltaY_P[1][cls] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, -0.2, 0.2);

    histoName.str("");
    histoName << "Delta_Y_Strip_P" << tag.c_str() << id << clsstr.c_str();
    local_histos.deltaY_P[2][cls] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, -3., 3.);

    if (makeEtaPlots_) {
      histoName.str("");
      histoName << "Delta_X_vs_Eta_Pixel_P" << tag.c_str() << id << clsstr.c_str();
      local_histos.deltaX_eta_P[1][cls] =
          td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 50, -2.5, 2.5, 100, -0.02, 0.02);

      histoName.str("");
      histoName << "Delta_X_vs_Eta_Strip_P" << tag.c_str() << id << clsstr.c_str();
      local_histos.deltaX_eta_P[2][cls] =
          td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 50, -2.5, 2.5, 100, -0.02, 0.02);

      histoName.str("");
      histoName << "Delta_Y_vs_Eta_Pixel_P" << tag.c_str() << id << clsstr.c_str();
      local_histos.deltaY_eta_P[1][cls] =
          td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 50, -2.5, 2.5, 100, -0.2, 0.2);

      histoName.str("");
      histoName << "Delta_Y_vs_Eta_Strip_P" << tag.c_str() << id << clsstr.c_str();
      local_histos.deltaY_eta_P[2][cls] =
          td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 50, -2.5, 2.5, 100, -3, 3);
    }

    /*
       * Pulls for primary tracks only
       */

    histoName.str("");
    histoName << "Pull_X_Pixel_P" << tag.c_str() << id << clsstr.c_str();
    local_histos.pullX_P[1][cls] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, -3., 3.);

    histoName.str("");
    histoName << "Pull_X_Strip_P" << tag.c_str() << id << clsstr.c_str();
    local_histos.pullX_P[2][cls] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, -3., 3.);

    histoName.str("");
    histoName << "Pull_Y_Pixel_P" << tag.c_str() << id << clsstr.c_str();
    local_histos.pullY_P[1][cls] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, -3., 3.);

    histoName.str("");
    histoName << "Pull_Y_Strip_P" << tag.c_str() << id << clsstr.c_str();
    local_histos.pullY_P[2][cls] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, -3., 3.);

    if (makeEtaPlots_) {
      histoName.str("");
      histoName << "Pull_X_Eta_Pixel_P" << tag.c_str() << id << clsstr.c_str();
      local_histos.pullX_eta_P[1][cls] =
          td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 50, -2.5, 2.5, 100, -3., 3.);

      histoName.str("");
      histoName << "Pull_X_Eta_Strip_P" << tag.c_str() << id << clsstr.c_str();
      local_histos.pullX_eta_P[2][cls] =
          td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 50, -2.5, 2.5, 100, -3., 3.);

      histoName.str("");
      histoName << "Pull_Y_Eta_Pixel_P" << tag.c_str() << id << clsstr.c_str();
      local_histos.pullY_eta_P[1][cls] =
          td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 50, -2.5, 2.5, 100, -3., 3.);

      histoName.str("");
      histoName << "Pull_Y_Eta_Strip_P" << tag.c_str() << id << clsstr.c_str();
      local_histos.pullY_eta_P[2][cls] =
          td.make<TH2F>(histoName.str().c_str(), histoName.str().c_str(), 50, -2.5, 2.5, 100, -3., 3.);
    }
  }

  /*
     * Information on the Digis per cluster
     */

  histoName.str("");
  histoName << "Primary_Digis_Pixel" << tag.c_str() << id;
  local_histos.primarySimHits[1] = td.make<TH1D>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);
  histoName.str("");
  histoName << "Primary_Digis_Strip" << tag.c_str() << id;
  local_histos.primarySimHits[2] = td.make<TH1D>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Other_Digis_Pixel" << tag.c_str() << id;
  local_histos.otherSimHits[1] = td.make<TH1D>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);
  histoName.str("");
  histoName << "Other_Digis_Strip" << tag.c_str() << id;
  local_histos.otherSimHits[2] = td.make<TH1D>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  /*
     * End
     */

  std::pair<std::map<unsigned int, RecHitHistos>::iterator, bool> insertedIt(
      histograms_.insert(std::make_pair(ival, local_histos)));
  fs->file().cd("/");

  return insertedIt.first;
}

std::vector<unsigned int> Phase2TrackerRecHitsValidation::getSimTrackId(
    const edm::Handle<edm::DetSetVector<PixelDigiSimLink> >& pixelSimLinks, const DetId& detId, unsigned int channel) {
  std::vector<unsigned int> retvec;
  edm::DetSetVector<PixelDigiSimLink>::const_iterator DSViter(pixelSimLinks->find(detId));
  if (DSViter == pixelSimLinks->end())
    return retvec;
  for (edm::DetSet<PixelDigiSimLink>::const_iterator it = DSViter->data.begin(); it != DSViter->data.end(); ++it) {
    if (channel == it->channel()) {
      retvec.push_back(it->SimTrackId());
    }
  }
  return retvec;
}

DEFINE_FWK_MODULE(Phase2TrackerRecHitsValidation);
