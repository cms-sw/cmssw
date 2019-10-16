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
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include <TGraph.h>
#include <TH1F.h>
#include <THStack.h>
#include <TTree.h>

struct ClusterHistos {
  THStack* numberClustersMixed;
  TH1F* numberClusterPixel;
  TH1F* numberClusterStrip;

  THStack* clustersSizeMixed;
  TH1F* clusterSizePixel;
  TH1F* clusterSizeStrip;

  TGraph* globalPosXY[3];
  TGraph* localPosXY[3];

  TH1F* deltaXClusterSimHits[3];
  TH1F* deltaYClusterSimHits[3];

  TH1F* deltaXClusterSimHits_P[3];
  TH1F* deltaYClusterSimHits_P[3];

  TH1F* digiEfficiency[3];

  TH1F* primarySimHits;
  TH1F* otherSimHits;
};

class Phase2TrackerClusterizerValidationTGraph : public edm::EDAnalyzer {
public:
  typedef std::map<unsigned int, std::vector<PSimHit> > SimHitsMap;
  typedef std::map<unsigned int, SimTrack> SimTracksMap;

  explicit Phase2TrackerClusterizerValidationTGraph(const edm::ParameterSet&);
  ~Phase2TrackerClusterizerValidationTGraph();
  void beginJob();
  void endJob();
  void analyze(const edm::Event&, const edm::EventSetup&);

private:
  std::map<unsigned int, ClusterHistos>::iterator createLayerHistograms(unsigned int);
  unsigned int getLayerNumber(const DetId&, const TrackerTopology*);
  unsigned int getModuleNumber(const DetId&, const TrackerTopology*);
  unsigned int getSimTrackId(const edm::Handle<edm::DetSetVector<PixelDigiSimLink> >&, const DetId&, unsigned int);

  edm::EDGetTokenT<edmNew::DetSetVector<Phase2TrackerCluster1D> > srcClu_;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink> > siphase2OTSimLinksToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> simHitsToken_;
  edm::EDGetTokenT<edm::SimTrackContainer> simTracksToken_;
  edm::EDGetTokenT<edm::SimVertexContainer> simVerticesToken_;
  const TrackerGeometry* tkGeom;
  const TrackerTopology* tkTopo;

  TTree* tree;
  TGraph* trackerLayout_[3];
  TGraph* trackerLayoutXY_[3];
  TGraph* trackerLayoutXYBar_;
  TGraph* trackerLayoutXYEC_;

  std::map<unsigned int, ClusterHistos> histograms_;
};

Phase2TrackerClusterizerValidationTGraph::Phase2TrackerClusterizerValidationTGraph(const edm::ParameterSet& conf) {
  srcClu_ =
      consumes<edmNew::DetSetVector<Phase2TrackerCluster1D> >(edm::InputTag(conf.getParameter<std::string>("src")));
  siphase2OTSimLinksToken_ = consumes<edm::DetSetVector<PixelDigiSimLink> >(conf.getParameter<edm::InputTag>("links"));
  simHitsToken_ = consumes<edm::PSimHitContainer>(edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"));
  simTracksToken_ = consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"));
  simVerticesToken_ = consumes<edm::SimVertexContainer>(edm::InputTag("g4SimHits"));
}

Phase2TrackerClusterizerValidationTGraph::~Phase2TrackerClusterizerValidationTGraph() {}

void Phase2TrackerClusterizerValidationTGraph::beginJob() {
  edm::Service<TFileService> fs;
  fs->file().cd("/");
  TFileDirectory td = fs->mkdir("Common");
  //Create common ntuple
  tree = td.make<TTree>("Phase2TrackerClusters", "Phase2TrackerClusters");
  // Create common histograms
  trackerLayout_[0] = td.make<TGraph>();  //"RVsZ_Mixed", "R vs. z position", 6000, -300.0, 300.0, 1200, 0.0, 120.0);
  trackerLayout_[0]->SetName("RVsZ_Mixed");
  trackerLayout_[1] = td.make<TGraph>();  //"RVsZ_Pixel", "R vs. z position", 6000, -300.0, 300.0, 1200, 0.0, 120.0);
  trackerLayout_[1]->SetName("RVsZ_Pixel");
  trackerLayout_[2] = td.make<TGraph>();  //"RVsZ_Strip", "R vs. z position", 6000, -300.0, 300.0, 1200, 0.0, 120.0);
  trackerLayout_[2]->SetName("RVsZ_Strip");
  trackerLayoutXY_[0] =
      td.make<TGraph>();  //"XVsY_Mixed", "x vs. y position", 2400, -120.0, 120.0, 2400, -120.0, 120.0);
  trackerLayoutXY_[0]->SetName("YVsX_Mixed");
  trackerLayoutXY_[1] =
      td.make<TGraph>();  //"XVsY_Pixel", "x vs. y position", 2400, -120.0, 120.0, 2400, -120.0, 120.0);
  trackerLayoutXY_[1]->SetName("YVsX_Pixel");
  trackerLayoutXY_[2] =
      td.make<TGraph>();  //"XVsY_Strip", "x vs. y position", 2400, -120.0, 120.0, 2400, -120.0, 120.0);
  trackerLayoutXY_[2]->SetName("YVsX_Strip");
  trackerLayoutXYBar_ = td.make<TGraph>();  //"XVsYBar", "x vs. y position", 2400, -120.0, 120.0, 2400, -120.0, 120.0);
  trackerLayoutXYBar_->SetName("YVsXBar");
  trackerLayoutXYEC_ = td.make<TGraph>();  //"XVsYEC", "x vs. y position", 2400, -120.0, 120.0, 2400, -120.0, 120.0);
  trackerLayoutXYEC_->SetName("YVsXEC");
}

void Phase2TrackerClusterizerValidationTGraph::endJob() {}

void Phase2TrackerClusterizerValidationTGraph::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  // Get the needed objects

  // Get the clusters
  edm::Handle<Phase2TrackerCluster1DCollectionNew> clusters;
  event.getByToken(srcClu_, clusters);

  // Get the Phase2 DigiSimLink
  edm::Handle<edm::DetSetVector<PixelDigiSimLink> > siphase2SimLinks;
  event.getByToken(siphase2OTSimLinksToken_, siphase2SimLinks);

  // Get the SimHits
  edm::Handle<edm::PSimHitContainer> simHitsRaw;
  event.getByToken(simHitsToken_, simHitsRaw);
  //  edm::Handle< edm::PSimHitContainer > simHitsRawEndcap;
  //  event.getByLabel("g4SimHits", "TrackerHitsPixelEndcapLowTof", simHitsRawEndcap);

  // Get the SimTracks
  edm::Handle<edm::SimTrackContainer> simTracksRaw;
  event.getByToken(simTracksToken_, simTracksRaw);

  // Get the SimVertex
  edm::Handle<edm::SimVertexContainer> simVertices;
  event.getByToken(simVerticesToken_, simVertices);

  // Get the geometry
  edm::ESHandle<TrackerGeometry> geomHandle;
  eventSetup.get<TrackerDigiGeometryRecord>().get(geomHandle);
  tkGeom = &(*geomHandle);
  edm::ESHandle<TrackerTopology> tTopoHandle;
  eventSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  tkTopo = tTopoHandle.product();

  //set up for tree
  int layer_number;
  //int track_id;
  int module_id;
  int module_number;
  int module_type;  //1: pixel, 2: strip
  float x_global, y_global, z_global;
  float x_local, y_local, z_local;

  tree->Branch("layer_number", &layer_number, "layer_number/I");
  //tree -> Branch("track_id",&track_id,"track_id/I");
  tree->Branch("module_id", &module_id, "module_id/I");
  tree->Branch("module_type", &module_type, "module_type/I");
  tree->Branch("module_number", &module_number, "module_number/I");
  tree->Branch("x_global", &x_global, "x_global/F");
  tree->Branch("y_global", &y_global, "y_global/F");
  tree->Branch("z_global", &z_global, "z_global/F");
  tree->Branch("x_local", &x_local, "x_local/F");
  tree->Branch("y_local", &y_local, "y_local/F");
  tree->Branch("z_local", &z_local, "z_local/F");

  // Rearrange the simTracks for ease of use <simTrackID, simTrack>
  SimTracksMap simTracks;
  for (edm::SimTrackContainer::const_iterator simTrackIt(simTracksRaw->begin()); simTrackIt != simTracksRaw->end();
       ++simTrackIt)
    simTracks.insert(std::pair<unsigned int, SimTrack>(simTrackIt->trackId(), *simTrackIt));

  // Rearrange the simHits by detUnit

  // Rearrange the simHits for ease of use
  SimHitsMap simHitsDetUnit;
  SimHitsMap simHitsTrackId;
  for (edm::PSimHitContainer::const_iterator simHitIt(simHitsRaw->begin()); simHitIt != simHitsRaw->end(); ++simHitIt) {
    SimHitsMap::iterator simHitsDetUnitIt(simHitsDetUnit.find(simHitIt->detUnitId()));
    if (simHitsDetUnitIt == simHitsDetUnit.end()) {
      std::pair<SimHitsMap::iterator, bool> newIt(simHitsDetUnit.insert(
          std::pair<unsigned int, std::vector<PSimHit> >(simHitIt->detUnitId(), std::vector<PSimHit>())));
      simHitsDetUnitIt = newIt.first;
    }
    simHitsDetUnitIt->second.push_back(*simHitIt);
    SimHitsMap::iterator simHitsTrackIdIt(simHitsTrackId.find(simHitIt->trackId()));
    if (simHitsTrackIdIt == simHitsTrackId.end()) {
      std::pair<SimHitsMap::iterator, bool> newIt(simHitsTrackId.insert(
          std::pair<unsigned int, std::vector<PSimHit> >(simHitIt->trackId(), std::vector<PSimHit>())));
      simHitsTrackIdIt = newIt.first;
    }
    simHitsTrackIdIt->second.push_back(*simHitIt);
  }

  // ValidationTGraph
  unsigned int nClustersTot(0), nClustersPixelTot(0), nClustersStripTot(0);

  // Loop over modules
  for (Phase2TrackerCluster1DCollectionNew::const_iterator DSViter = clusters->begin(); DSViter != clusters->end();
       ++DSViter) {
    // Get the detector unit's id
    unsigned int rawid(DSViter->detId());
    module_id = rawid;
    DetId detId(rawid);

    layer_number = getLayerNumber(detId, tkTopo);
    module_number = getModuleNumber(detId, tkTopo);
    unsigned int layer(getLayerNumber(detId, tkTopo));

    // Get the geometry of the tracker
    const GeomDetUnit* geomDetUnit(tkGeom->idToDetUnit(detId));
    const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(geomDetUnit);
    const PixelTopology& topol = theGeomDet->specificTopology();

    if (!geomDetUnit)
      break;

    // Create histograms for the layer if they do not yet exist
    std::map<unsigned int, ClusterHistos>::iterator histogramLayer(histograms_.find(layer));
    if (histogramLayer == histograms_.end())
      histogramLayer = createLayerHistograms(layer);

    // Number of clusters
    unsigned int nClustersPixel(0), nClustersStrip(0);

    // Loop over the clusters in the detector unit
    for (edmNew::DetSet<Phase2TrackerCluster1D>::const_iterator clustIt = DSViter->begin(); clustIt != DSViter->end();
         ++clustIt) {
      // Cluster related variables
      MeasurementPoint mpClu(clustIt->center(), clustIt->column() + 0.5);
      Local3DPoint localPosClu = geomDetUnit->topology().localPosition(mpClu);
      x_local = localPosClu.x();
      y_local = localPosClu.y();
      z_local = localPosClu.z();

      Global3DPoint globalPosClu = geomDetUnit->surface().toGlobal(localPosClu);
      x_global = globalPosClu.x();
      y_global = globalPosClu.y();
      z_global = globalPosClu.z();

      // Fill the position histograms
      trackerLayout_[0]->SetPoint(nClustersTot, globalPosClu.z(), globalPosClu.perp());
      trackerLayoutXY_[0]->SetPoint(nClustersTot, globalPosClu.x(), globalPosClu.y());

      if (layer < 100)
        trackerLayoutXYBar_->SetPoint(nClustersTot, globalPosClu.x(), globalPosClu.y());
      else
        trackerLayoutXYEC_->SetPoint(nClustersTot, globalPosClu.x(), globalPosClu.y());

      histogramLayer->second.localPosXY[0]->SetPoint(nClustersTot, localPosClu.x(), localPosClu.y());
      histogramLayer->second.globalPosXY[0]->SetPoint(nClustersTot, globalPosClu.x(), globalPosClu.y());

      // Pixel module
      if (topol.ncolumns() == 32) {
        module_type = 1;
        trackerLayout_[1]->SetPoint(nClustersPixelTot, globalPosClu.z(), globalPosClu.perp());
        trackerLayoutXY_[1]->SetPoint(nClustersPixelTot, globalPosClu.x(), globalPosClu.y());

        histogramLayer->second.localPosXY[1]->SetPoint(nClustersPixelTot, localPosClu.x(), localPosClu.y());
        histogramLayer->second.globalPosXY[1]->SetPoint(nClustersPixelTot, globalPosClu.x(), globalPosClu.y());
        histogramLayer->second.clusterSizePixel->Fill(clustIt->size());
        ++nClustersPixel;
        ++nClustersPixelTot;
      }
      // Strip module
      else if (topol.ncolumns() == 2) {
        module_type = 2;
        trackerLayout_[2]->SetPoint(nClustersStripTot, globalPosClu.z(), globalPosClu.perp());
        trackerLayoutXY_[2]->SetPoint(nClustersStripTot, globalPosClu.x(), globalPosClu.y());

        histogramLayer->second.localPosXY[2]->SetPoint(nClustersStripTot, localPosClu.x(), localPosClu.y());
        histogramLayer->second.globalPosXY[2]->SetPoint(nClustersStripTot, globalPosClu.x(), globalPosClu.y());
        histogramLayer->second.clusterSizeStrip->Fill(clustIt->size());
        ++nClustersStrip;
        ++nClustersStripTot;
      }

      // * Digis related variables

      std::vector<unsigned int> clusterSimTrackIds;

      // Get all the simTracks that form the cluster
      for (unsigned int i(0); i < clustIt->size(); ++i) {
        unsigned int channel(PixelDigi::pixelToChannel(
            clustIt->firstRow() + i,
            clustIt
                ->column()));  // Here we have to use the old pixelToChannel function (not Phase2TrackerDigi but PixelDigi), change this when using new Digis
        unsigned int simTrackId(getSimTrackId(siphase2SimLinks, detId, channel));
        clusterSimTrackIds.push_back(simTrackId);
      }

      /*
             // SimHits related variables



            unsigned int primarySimHits(0);
            unsigned int otherSimHits(0);

            for (edm::PSimHitContainer::const_iterator hitIt(simHitsRaw->begin()); hitIt != simHitsRaw->end(); ++hitIt) {
                if (rawid == hitIt->detUnitId() and std::find(clusterSimTrackIds.begin(), clusterSimTrackIds.end(), hitIt->trackId()) != clusterSimTrackIds.end()) {
                    Local3DPoint localPosHit(hitIt->localPosition());

                    histogramLayer->second.deltaXClusterSimHits[0]->Fill(localPosClu.x() - localPosHit.x());
                    histogramLayer->second.deltaYClusterSimHits[0]->Fill(localPosClu.y() - localPosHit.y());

                    // Pixel module
                    if (topol.ncolumns() == 32) {
                        histogramLayer->second.deltaXClusterSimHits[1]->Fill(localPosClu.x() - localPosHit.x());
                        histogramLayer->second.deltaYClusterSimHits[1]->Fill(localPosClu.y() - localPosHit.y());
                    }
                    // Strip module
                    else if (topol.ncolumns() == 2) {
                        histogramLayer->second.deltaXClusterSimHits[2]->Fill(localPosClu.x() - localPosHit.x());
                        histogramLayer->second.deltaYClusterSimHits[2]->Fill(localPosClu.y() - localPosHit.y());
                    }

                    ++otherSimHits;

                    std::map< unsigned int, SimTrack >::const_iterator simTrackIt(simTracks.find(hitIt->trackId()));
                    if (simTrackIt == simTracks.end()) continue;

                    // Primary particles only
                    unsigned int processType(hitIt->processType());
                    if (simTrackIt->second.vertIndex() == 0 and (processType == 2 || processType == 7 || processType == 9 || processType == 11 || processType == 13 || processType == 15)) {
                        histogramLayer->second.deltaXClusterSimHits_P[0]->Fill(localPosClu.x() - localPosHit.x());
                        histogramLayer->second.deltaYClusterSimHits_P[0]->Fill(localPosClu.y() - localPosHit.y());

                        // Pixel module
                        if (topol.ncolumns() == 32) {
                            histogramLayer->second.deltaXClusterSimHits_P[1]->Fill(localPosClu.x() - localPosHit.x());
                            histogramLayer->second.deltaYClusterSimHits_P[1]->Fill(localPosClu.y() - localPosHit.y());
                        }
                        // Strip module
                        else if (topol.ncolumns() == 2) {
                            histogramLayer->second.deltaXClusterSimHits_P[2]->Fill(localPosClu.x() - localPosHit.x());
                            histogramLayer->second.deltaYClusterSimHits_P[2]->Fill(localPosClu.y() - localPosHit.y());
                        }

                        ++primarySimHits;
                    }
                }
            }

            otherSimHits -= primarySimHits;

            histogramLayer->second.primarySimHits->Fill(primarySimHits);
            histogramLayer->second.otherSimHits->Fill(otherSimHits);
*/
    }

    if (nClustersPixel)
      histogramLayer->second.numberClusterPixel->Fill(nClustersPixel);
    if (nClustersStrip)
      histogramLayer->second.numberClusterStrip->Fill(nClustersStrip);
    nClustersTot++;
    tree->Fill();
  }
}

// Create the histograms
std::map<unsigned int, ClusterHistos>::iterator Phase2TrackerClusterizerValidationTGraph::createLayerHistograms(
    unsigned int ival) {
  std::ostringstream fname1, fname2;

  edm::Service<TFileService> fs;
  fs->file().cd("/");

  std::string tag;
  unsigned int id;
  if (ival < 100) {
    id = ival;
    fname1 << "Barrel";
    fname2 << "Layer_" << id;
    tag = "_layer_";
  } else {
    int side = ival / 100;
    id = ival - side * 100;
    fname1 << "EndCap_Side_" << side;
    fname2 << "Disc_" << id;
    tag = "_disc_";
  }

  TFileDirectory td1 = fs->mkdir(fname1.str().c_str());
  TFileDirectory td = td1.mkdir(fname2.str().c_str());

  ClusterHistos local_histos;

  std::ostringstream histoName;

  /*
     * Number of clusters
     */

  histoName.str("");
  histoName << "Number_Clusters_Pixel" << tag.c_str() << id;
  local_histos.numberClusterPixel = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 20, 0., 20.);
  local_histos.numberClusterPixel->SetFillColor(kAzure + 7);

  histoName.str("");
  histoName << "Number_Clusters_Strip" << tag.c_str() << id;
  local_histos.numberClusterStrip = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 20, 0., 20.);
  local_histos.numberClusterStrip->SetFillColor(kOrange - 3);

  histoName.str("");
  histoName << "Number_Clusters_Mixed" << tag.c_str() << id;
  local_histos.numberClustersMixed = td.make<THStack>(histoName.str().c_str(), histoName.str().c_str());
  local_histos.numberClustersMixed->Add(local_histos.numberClusterPixel);
  local_histos.numberClustersMixed->Add(local_histos.numberClusterStrip);

  /*
     * Cluster size
     */

  histoName.str("");
  histoName << "Cluster_Size_Pixel" << tag.c_str() << id;
  local_histos.clusterSizePixel = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 10, 0., 10.);
  local_histos.clusterSizePixel->SetFillColor(kAzure + 7);

  histoName.str("");
  histoName << "Cluster_Size_Strip" << tag.c_str() << id;
  local_histos.clusterSizeStrip = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 10, 0., 10.);
  local_histos.clusterSizeStrip->SetFillColor(kOrange - 3);

  histoName.str("");
  histoName << "Cluster_Size_Mixed" << tag.c_str() << id;
  local_histos.clustersSizeMixed = td.make<THStack>(histoName.str().c_str(), histoName.str().c_str());
  local_histos.clustersSizeMixed->Add(local_histos.clusterSizePixel);
  local_histos.clustersSizeMixed->Add(local_histos.clusterSizeStrip);

  /*
     * Local and Global positions
     */

  histoName.str("");
  histoName << "Local_Position_XY_Mixed" << tag.c_str() << id;
  local_histos.localPosXY[0] =
      td.make<TGraph>();  //histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0., 2000, 0., 0.);
  local_histos.localPosXY[0]->SetName(histoName.str().c_str());

  histoName.str("");
  histoName << "Local_Position_XY_Pixel" << tag.c_str() << id;
  local_histos.localPosXY[1] =
      td.make<TGraph>();  //histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0., 2000, 0., 0.);
  local_histos.localPosXY[1]->SetName(histoName.str().c_str());

  histoName.str("");
  histoName << "Local_Position_XY_Strip" << tag.c_str() << id;
  local_histos.localPosXY[2] =
      td.make<TGraph>();  //histoName.str().c_str(), histoName.str().c_str(), 2000, 0., 0., 2000, 0., 0.);
  local_histos.localPosXY[2]->SetName(histoName.str().c_str());

  histoName.str("");
  histoName << "Global_Position_XY_Mixed" << tag.c_str() << id;
  local_histos.globalPosXY[0] =
      td.make<TGraph>();  //histoName.str().c_str(), histoName.str().c_str(), 2400, -120.0, 120.0, 2400, -120.0, 120.0);
  local_histos.globalPosXY[0]->SetName(histoName.str().c_str());

  histoName.str("");
  histoName << "Global_Position_XY_Pixel" << tag.c_str() << id;
  local_histos.globalPosXY[1] =
      td.make<TGraph>();  //histoName.str().c_str(), histoName.str().c_str(), 2400, -120.0, 120.0, 2400, -120.0, 120.0);
  local_histos.globalPosXY[1]->SetName(histoName.str().c_str());

  histoName.str("");
  histoName << "Global_Position_XY_Strip" << tag.c_str() << id;
  local_histos.globalPosXY[2] =
      td.make<TGraph>();  //histoName.str().c_str(), histoName.str().c_str(), 2400, -120.0, 120.0, 2400, -120.0, 120.0);
  local_histos.globalPosXY[2]->SetName(histoName.str().c_str());

  /*
     * Delta positions with SimHits
     */

  histoName.str("");
  histoName << "Delta_X_Cluster_SimHits_Mixed" << tag.c_str() << id;
  local_histos.deltaXClusterSimHits[0] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_X_Cluster_SimHits_Pixel" << tag.c_str() << id;
  local_histos.deltaXClusterSimHits[1] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_X_Cluster_SimHits_Strip" << tag.c_str() << id;
  local_histos.deltaXClusterSimHits[2] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_Y_Cluster_SimHits_Mixed" << tag.c_str() << id;
  local_histos.deltaYClusterSimHits[0] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_Y_Cluster_SimHits_Pixel" << tag.c_str() << id;
  local_histos.deltaYClusterSimHits[1] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_Y_Cluster_SimHits_Strip" << tag.c_str() << id;
  local_histos.deltaYClusterSimHits[2] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  /*
     * Delta position with simHits for primary tracks only
     */

  histoName.str("");
  histoName << "Delta_X_Cluster_SimHits_Mixed_P" << tag.c_str() << id;
  local_histos.deltaXClusterSimHits_P[0] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_X_Cluster_SimHits_Pixel_P" << tag.c_str() << id;
  local_histos.deltaXClusterSimHits_P[1] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_X_Cluster_SimHits_Strip_P" << tag.c_str() << id;
  local_histos.deltaXClusterSimHits_P[2] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_Y_Cluster_SimHits_Mixed_P" << tag.c_str() << id;
  local_histos.deltaYClusterSimHits_P[0] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_Y_Cluster_SimHits_Pixel_P" << tag.c_str() << id;
  local_histos.deltaYClusterSimHits_P[1] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_Y_Cluster_SimHits_Strip_P" << tag.c_str() << id;
  local_histos.deltaYClusterSimHits_P[2] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  /*
     * Information on the Digis per cluster
     */

  histoName.str("");
  histoName << "Primary_Digis" << tag.c_str() << id;
  local_histos.primarySimHits = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 10, 0., 10.);

  histoName.str("");
  histoName << "Other_Digis" << tag.c_str() << id;
  local_histos.otherSimHits = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 10, 0., 10.);

  /*
     * End
     */

  std::pair<std::map<unsigned int, ClusterHistos>::iterator, bool> insertedIt(
      histograms_.insert(std::make_pair(ival, local_histos)));
  fs->file().cd("/");

  return insertedIt.first;
}

unsigned int Phase2TrackerClusterizerValidationTGraph::getLayerNumber(const DetId& detid, const TrackerTopology* topo) {
  if (detid.det() == DetId::Tracker) {
    if (detid.subdetId() == PixelSubdetector::PixelBarrel)
      return (topo->pxbLayer(detid));
    else if (detid.subdetId() == PixelSubdetector::PixelEndcap)
      return (100 * topo->pxfSide(detid) + topo->pxfDisk(detid));
    else
      return 999;
  }
  return 999;
}

unsigned int Phase2TrackerClusterizerValidationTGraph::getModuleNumber(const DetId& detid,
                                                                       const TrackerTopology* topo) {
  if (detid.det() == DetId::Tracker) {
    if (detid.subdetId() == PixelSubdetector::PixelBarrel) {
      return (topo->pxbModule(detid));
    } else if (detid.subdetId() == PixelSubdetector::PixelEndcap) {
      return (topo->pxfModule(detid));
    } else
      return 999;
  }
  return 999;
}

unsigned int Phase2TrackerClusterizerValidationTGraph::getSimTrackId(
    const edm::Handle<edm::DetSetVector<PixelDigiSimLink> >& siphase2SimLinks,
    const DetId& detId,
    unsigned int channel) {
  edm::DetSetVector<PixelDigiSimLink>::const_iterator DSViter(siphase2SimLinks->find(detId));
  if (DSViter == siphase2SimLinks->end())
    return 0;
  for (edm::DetSet<PixelDigiSimLink>::const_iterator it = DSViter->data.begin(); it != DSViter->data.end(); ++it) {
    if (channel == it->channel())
      return it->SimTrackId();
  }
  return 0;
}

DEFINE_FWK_MODULE(Phase2TrackerClusterizerValidationTGraph);
