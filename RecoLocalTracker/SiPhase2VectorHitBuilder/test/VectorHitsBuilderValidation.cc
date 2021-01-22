#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"
#include "RecoLocalTracker/SiPhase2VectorHitBuilder/test/VectorHitsBuilderValidation.h"
#include "Geometry/CommonDetUnit/interface/StackGeomDet.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

VectorHitsBuilderValidation::VectorHitsBuilderValidation(const edm::ParameterSet& conf)
    : cpeTag_(conf.getParameter<edm::ESInputTag>("CPE")) {
  srcClu_ =
      consumes<edmNew::DetSetVector<Phase2TrackerCluster1D> >(edm::InputTag(conf.getParameter<std::string>("src")));
  VHacc_ = consumes<VectorHitCollection>(edm::InputTag(conf.getParameter<edm::InputTag>("VH_acc")));
  VHrej_ = consumes<VectorHitCollection>(edm::InputTag(conf.getParameter<edm::InputTag>("VH_rej")));
  siphase2OTSimLinksToken_ = consumes<edm::DetSetVector<PixelDigiSimLink> >(conf.getParameter<edm::InputTag>("links"));
  simHitsToken_ = consumes<edm::PSimHitContainer>(edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"));
  simTracksToken_ = consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"));
  simVerticesToken_ = consumes<edm::SimVertexContainer>(edm::InputTag("g4SimHits"));
  trackingParticleToken_ =
      consumes<TrackingParticleCollection>(conf.getParameter<edm::InputTag>("trackingParticleSrc"));
}

VectorHitsBuilderValidation::~VectorHitsBuilderValidation() {}

void VectorHitsBuilderValidation::beginJob() {
  edm::Service<TFileService> fs;
  fs->file().cd("/");
  TFileDirectory td = fs->mkdir("Common");

  //Create common ntuple
  tree_ = td.make<TTree>("VectorHits", "VectorHits");

  // Create common graphs
  TFileDirectory tdGloPos = td.mkdir("GlobalPositions");
  trackerLayoutRZ_[0] = tdGloPos.make<TGraph>();
  trackerLayoutRZ_[0]->SetName("RVsZ_Mixed");
  trackerLayoutRZ_[1] = tdGloPos.make<TGraph>();
  trackerLayoutRZ_[1]->SetName("RVsZ_Pixel");
  trackerLayoutRZ_[2] = tdGloPos.make<TGraph>();
  trackerLayoutRZ_[2]->SetName("RVsZ_Strip");
  trackerLayoutXY_[0] = tdGloPos.make<TGraph>();
  trackerLayoutXY_[0]->SetName("YVsX_Mixed");
  trackerLayoutXY_[1] = tdGloPos.make<TGraph>();
  trackerLayoutXY_[1]->SetName("YVsX_Pixel");
  trackerLayoutXY_[2] = tdGloPos.make<TGraph>();
  trackerLayoutXY_[2]->SetName("YVsX_Strip");
  trackerLayoutXYBar_ = tdGloPos.make<TGraph>();
  trackerLayoutXYBar_->SetName("YVsXBar");
  trackerLayoutXYEC_ = tdGloPos.make<TGraph>();
  trackerLayoutXYEC_->SetName("YVsXEC");

  TFileDirectory tdLocPos = td.mkdir("LocalPositions");
  localPosXvsDeltaX_[0] = tdLocPos.make<TGraph>();
  localPosXvsDeltaX_[0]->SetName("localPosXvsDeltaX_Mixed");
  localPosXvsDeltaX_[1] = tdLocPos.make<TGraph>();
  localPosXvsDeltaX_[1]->SetName("localPosXvsDeltaX_Pixel");
  localPosXvsDeltaX_[2] = tdLocPos.make<TGraph>();
  localPosXvsDeltaX_[2]->SetName("localPosXvsDeltaX_Strip");
  localPosYvsDeltaY_[0] = tdLocPos.make<TGraph>();
  localPosYvsDeltaY_[0]->SetName("localPosYvsDeltaY_Mixed");
  localPosYvsDeltaY_[1] = tdLocPos.make<TGraph>();
  localPosYvsDeltaY_[1]->SetName("localPosYvsDeltaY_Pixel");
  localPosYvsDeltaY_[2] = tdLocPos.make<TGraph>();
  localPosYvsDeltaY_[2]->SetName("localPosYvsDeltaY_Strip");

  //drawing VHs arrows
  TFileDirectory tdArr = td.mkdir("Directions");

  TFileDirectory tdWid = td.mkdir("CombinatorialStudies");
  ParallaxCorrectionRZ_ =
      tdWid.make<TH2D>("ParallaxCorrectionRZ", "ParallaxCorrectionRZ", 100, 0., 300., 100., 0., 120.);
  ParallaxCorrectionRZ_->SetName("ParallaxCorrectionFactor");
  VHaccLayer_ = tdWid.make<TH1F>("VHacceptedLayer", "VHacceptedLayer", 250, 0., 250.);
  VHaccLayer_->SetName("VHaccepted");
  VHrejLayer_ = tdWid.make<TH1F>("VHrejectedLayer", "VHrejectedLayer", 250, 0., 250.);
  VHrejLayer_->SetName("VHrejected");
  VHaccTrueLayer_ = tdWid.make<TH1F>("VHaccTrueLayer", "VHaccTrueLayer", 250, 0., 250.);
  VHaccTrueLayer_->SetName("VHaccepted_true");
  VHrejTrueLayer_ = tdWid.make<TH1F>("VHrejTrueLayer", "VHrejTrueLayer", 250, 0., 250.);
  VHrejTrueLayer_->SetName("VHrejected_true");
  VHaccTrue_signal_Layer_ = tdWid.make<TH1F>("VHaccTrueSignalLayer", "VHaccTrueSignalLayer", 250, 0., 250.);
  VHaccTrue_signal_Layer_->SetName("VHaccepted_true_signal");
  VHrejTrue_signal_Layer_ = tdWid.make<TH1F>("VHrejTrueSignalLayer", "VHrejTrueSignalLayer", 250, 0., 250.);
  VHrejTrue_signal_Layer_->SetName("VHrejected_true_signal");
}

void VectorHitsBuilderValidation::endJob() {}

void VectorHitsBuilderValidation::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  // Get the needed objects

  // Get the clusters
  edm::Handle<Phase2TrackerCluster1DCollectionNew> clusters;
  event.getByToken(srcClu_, clusters);

  // Get the vector hits
  edm::Handle<VectorHitCollection> vhsAcc;
  event.getByToken(VHacc_, vhsAcc);

  edm::Handle<VectorHitCollection> vhsRej;
  event.getByToken(VHrej_, vhsRej);

  // load the cpe via the eventsetup
  edm::ESHandle<ClusterParameterEstimator<Phase2TrackerCluster1D> > cpeHandle;
  eventSetup.get<TkPhase2OTCPERecord>().get(cpeTag_, cpeHandle);
  cpe_ = cpeHandle.product();

  // Get the Phase2 DigiSimLink
  edm::Handle<edm::DetSetVector<PixelDigiSimLink> > siphase2SimLinks;
  event.getByToken(siphase2OTSimLinksToken_, siphase2SimLinks);

  // Get the SimHits
  edm::Handle<edm::PSimHitContainer> simHitsRaw;
  event.getByToken(simHitsToken_, simHitsRaw);

  // Get the SimTracks
  edm::Handle<edm::SimTrackContainer> simTracksRaw;
  event.getByToken(simTracksToken_, simTracksRaw);

  // Get the SimVertex
  edm::Handle<edm::SimVertexContainer> simVertices;
  event.getByToken(simVerticesToken_, simVertices);

  // Get the geometry
  edm::ESHandle<TrackerGeometry> geomHandle;
  eventSetup.get<TrackerDigiGeometryRecord>().get(geomHandle);
  tkGeom_ = &(*geomHandle);

  // Get the Topology
  edm::ESHandle<TrackerTopology> tTopoHandle;
  eventSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  tkTopo_ = tTopoHandle.product();

  edm::ESHandle<MagneticField> magFieldHandle;
  eventSetup.get<IdealMagneticFieldRecord>().get(magFieldHandle);
  magField_ = magFieldHandle.product();

  //Tracking Particle collection
  edm::Handle<TrackingParticleCollection> TPCollectionH;
  event.getByToken(trackingParticleToken_, TPCollectionH);

  auto clusterTPList = std::make_unique<ClusterTPAssociation>(TPCollectionH);
  std::map<std::pair<size_t, EncodedEventId>, TrackingParticleRef> mapping;

  for (TrackingParticleCollection::size_type itp = 0; itp < TPCollectionH.product()->size(); ++itp) {
    TrackingParticleRef trackingParticle(TPCollectionH, itp);
    EncodedEventId eid(trackingParticle->eventId());
    for (std::vector<SimTrack>::const_iterator itrk = trackingParticle->g4Track_begin();
         itrk != trackingParticle->g4Track_end();
         ++itrk) {
      std::pair<uint32_t, EncodedEventId> trkid(itrk->trackId(), eid);
      LogTrace("VectorHitsBuilderValidation")
          << "creating map for id: " << trkid.first << " with tp: " << trackingParticle.key();
      mapping.insert(std::make_pair(trkid, trackingParticle));
    }
  }

  //set up for tree
  int eventNum;
  int layer;
  int module_id;
  int module_number;
  int module_type;  //1: pixel, 2: strip
  int VHacc = 0.0;
  int VHrej = 0.0;
  int vh_isTrue;

  float x_global, y_global, z_global;
  float vh_x_local, vh_y_local;
  float vh_x_le, vh_y_le;
  float curvature, phi;
  float QOverPT, QOverP;
  float chi2;

  int low_tp_id, upp_tp_id;
  float vh_sim_trackPt;
  float sim_x_local, sim_y_local;
  float sim_x_global, sim_y_global, sim_z_global;
  float low_x_global, low_y_global, low_z_global;
  float upp_x_global, upp_y_global, upp_z_global;
  float low_xx_global_err, low_yy_global_err, low_zz_global_err;
  float low_xy_global_err, low_zx_global_err, low_zy_global_err;
  float upp_xx_global_err, upp_yy_global_err, upp_zz_global_err;
  float upp_xy_global_err, upp_zx_global_err, upp_zy_global_err;
  float deltaXVHSimHits, deltaYVHSimHits;
  int multiplicity;
  float width, deltaXlocal;
  unsigned int processType(99);

  tree_->Branch("event", &eventNum, "eventNum/I");
  tree_->Branch("accepted", &VHacc, "VHacc/I");
  tree_->Branch("rejected", &VHrej, "VHrej/I");
  tree_->Branch("layer", &layer, "layer/I");
  tree_->Branch("module_id", &module_id, "module_id/I");
  tree_->Branch("module_type", &module_type, "module_type/I");
  tree_->Branch("module_number", &module_number, "module_number/I");
  tree_->Branch("vh_isTrue", &vh_isTrue, "vh_isTrue/I");
  tree_->Branch("x_global", &x_global, "x_global/F");
  tree_->Branch("y_global", &y_global, "y_global/F");
  tree_->Branch("z_global", &z_global, "z_global/F");
  tree_->Branch("vh_x_local", &vh_x_local, "vh_x_local/F");
  tree_->Branch("vh_y_local", &vh_y_local, "vh_y_local/F");
  tree_->Branch("vh_x_lError", &vh_x_le, "vh_x_le/F");
  tree_->Branch("vh_y_lError", &vh_y_le, "vh_y_le/F");
  tree_->Branch("curvature", &curvature, "curvature/F");
  tree_->Branch("chi2", &chi2, "chi2/F");
  tree_->Branch("phi", &phi, "phi/F");
  tree_->Branch("QOverP", &QOverP, "QOverP/F");
  tree_->Branch("QOverPT", &QOverPT, "QOverPT/F");
  tree_->Branch("low_tp_id", &low_tp_id, "low_tp_id/I");
  tree_->Branch("upp_tp_id", &upp_tp_id, "upp_tp_id/I");
  tree_->Branch("vh_sim_trackPt", &vh_sim_trackPt, "vh_sim_trackPt/F");
  tree_->Branch("sim_x_local", &sim_x_local, "sim_x_local/F");
  tree_->Branch("sim_y_local", &sim_y_local, "sim_y_local/F");
  tree_->Branch("sim_x_global", &sim_x_global, "sim_x_global/F");
  tree_->Branch("sim_y_global", &sim_y_global, "sim_y_global/F");
  tree_->Branch("sim_z_global", &sim_z_global, "sim_z_global/F");
  tree_->Branch("low_x_global", &low_x_global, "low_x_global/F");
  tree_->Branch("low_y_global", &low_y_global, "low_y_global/F");
  tree_->Branch("low_z_global", &low_z_global, "low_z_global/F");
  tree_->Branch("low_xx_global_err", &low_xx_global_err, "low_xx_global_err/F");
  tree_->Branch("low_yy_global_err", &low_yy_global_err, "low_yy_global_err/F");
  tree_->Branch("low_zz_global_err", &low_zz_global_err, "low_zz_global_err/F");
  tree_->Branch("low_xy_global_err", &low_xy_global_err, "low_xy_global_err/F");
  tree_->Branch("low_zx_global_err", &low_zx_global_err, "low_zx_global_err/F");
  tree_->Branch("low_zy_global_err", &low_zy_global_err, "low_zy_global_err/F");
  tree_->Branch("upp_x_global", &upp_x_global, "upp_x_global/F");
  tree_->Branch("upp_y_global", &upp_y_global, "upp_y_global/F");
  tree_->Branch("upp_z_global", &upp_z_global, "upp_z_global/F");
  tree_->Branch("upp_xx_global_err", &upp_xx_global_err, "upp_xx_global_err/F");
  tree_->Branch("upp_yy_global_err", &upp_yy_global_err, "upp_yy_global_err/F");
  tree_->Branch("upp_zz_global_err", &upp_zz_global_err, "upp_zz_global_err/F");
  tree_->Branch("upp_xy_global_err", &upp_xy_global_err, "upp_xy_global_err/F");
  tree_->Branch("upp_zx_global_err", &upp_zx_global_err, "upp_zx_global_err/F");
  tree_->Branch("upp_zy_global_err", &upp_zy_global_err, "upp_zy_global_err/F");
  tree_->Branch("deltaXVHSimHits", &deltaXVHSimHits, "deltaXVHSimHits/F");
  tree_->Branch("deltaYVHSimHits", &deltaYVHSimHits, "deltaYVHSimHits/F");
  tree_->Branch("multiplicity", &multiplicity, "multiplicity/I");
  tree_->Branch("width", &width, "width/F");
  tree_->Branch("deltaXlocal", &deltaXlocal, "deltaXlocal/F");
  tree_->Branch("processType", &processType, "processType/i");

  // Rearrange the simTracks for ease of use <simTrackID, simTrack>
  SimTracksMap simTracks;
  for (const auto& simTrackIt : *simTracksRaw.product())
    simTracks.emplace(std::pair<unsigned int, SimTrack>(simTrackIt.trackId(), simTrackIt));

  // Rearrange the simHits by detUnit for ease of use
  SimHitsMap simHitsDetUnit;
  SimHitsMap simHitsTrackId;
  for (const auto& simHitIt : *simHitsRaw.product()) {
    SimHitsMap::iterator simHitsDetUnitIt(simHitsDetUnit.find(simHitIt.detUnitId()));
    if (simHitsDetUnitIt == simHitsDetUnit.end()) {
      std::pair<SimHitsMap::iterator, bool> newIt(simHitsDetUnit.insert(
          std::pair<unsigned int, std::vector<PSimHit> >(simHitIt.detUnitId(), std::vector<PSimHit>())));
      simHitsDetUnitIt = newIt.first;
    }
    simHitsDetUnitIt->second.push_back(simHitIt);

    SimHitsMap::iterator simHitsTrackIdIt(simHitsTrackId.find(simHitIt.trackId()));
    if (simHitsTrackIdIt == simHitsTrackId.end()) {
      std::pair<SimHitsMap::iterator, bool> newIt(simHitsTrackId.insert(
          std::pair<unsigned int, std::vector<PSimHit> >(simHitIt.trackId(), std::vector<PSimHit>())));
      simHitsTrackIdIt = newIt.first;
    }
    simHitsTrackIdIt->second.push_back(simHitIt);
  }

  //Printout outer tracker clusters in the event
  for (const auto& DSViter : *clusters) {
    unsigned int rawid(DSViter.detId());
    DetId detId(rawid);
    const GeomDetUnit* geomDetUnit(tkGeom_->idToDetUnit(detId));
    const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(geomDetUnit);
    for (const auto& clustIt : DSViter) {
      auto&& lparams = cpe_->localParameters(clustIt, *theGeomDet);
      Global3DPoint gparams = theGeomDet->surface().toGlobal(lparams.first);
      LogTrace("VectorHitsBuilderValidation") << "phase2 OT clusters: " << gparams << " DetId: " << rawid;
    }
  }

  for (const auto& DSViter : *vhsAcc) {
    for (const auto& vhIt : DSViter) {
      LogTrace("VectorHitsBuilderValidation") << "accepted VH: " << vhIt;
    }
  }
  for (const auto& DSViter : *vhsRej) {
    for (const auto& vhIt : DSViter) {
      LogTrace("VectorHitsBuilderValidation") << "rejected VH: " << vhIt;
    }
  }
  // Validation
  eventNum = event.id().event();

  unsigned int nVHsTot(0), nVHsPSTot(0), nVHs2STot(0);
  std::vector<Global3DPoint> glVHs;
  std::vector<Global3DVector> dirVHs;
  std::vector<int> detIds;

  // Loop over modules
  for (const auto DSViter : *vhsAcc) {
    // Get the detector unit's id
    unsigned int rawid(DSViter.detId());
    module_id = rawid;
    DetId detId(rawid);

    module_number = getModuleNumber(detId);
    layer = getLayerNumber(detId);

    LogDebug("VectorHitsBuilderValidation") << "Layer: " << layer << "  det id" << rawid << std::endl;

    // Get the geometry of the tracker
    const GeomDet* geomDet(tkGeom_->idToDet(detId));
    if (!geomDet)
      break;

    // Create histograms for the layer if they do not yet exist
    std::map<unsigned int, VHHistos>::iterator histogramLayer(histograms_.find(layer));
    if (histogramLayer == histograms_.end())
      histogramLayer = createLayerHistograms(layer);
    // Number of clusters
    unsigned int nVHsPS(0), nVHs2S(0);

    LogDebug("VectorHitsBuilderValidation") << "DSViter size: " << DSViter.size();

    // Loop over the vhs in the detector unit
    for (const auto& vhIt : DSViter) {
      // vh variables
      if (vhIt.isValid()) {
        LogDebug("VectorHitsBuilderValidation") << " vh analyzing ...";
        chi2 = vhIt.chi2();
        LogTrace("VectorHitsBuilderValidation") << "VH chi2 " << chi2 << std::endl;

        Local3DPoint localPosVH = vhIt.localPosition();
        vh_x_local = localPosVH.x();
        vh_y_local = localPosVH.y();
        LogTrace("VectorHitsBuilderValidation") << "local VH position " << localPosVH << std::endl;

        LocalError localErrVH = vhIt.localPositionError();
        vh_x_le = localErrVH.xx();
        vh_y_le = localErrVH.yy();
        LogTrace("VectorHitsBuilderValidation") << "local VH error " << localErrVH << std::endl;

        Global3DPoint globalPosVH = geomDet->surface().toGlobal(localPosVH);
        x_global = globalPosVH.x();
        y_global = globalPosVH.y();
        z_global = globalPosVH.z();
        glVHs.push_back(globalPosVH);
        LogTrace("VectorHitsBuilderValidation") << " global VH position " << globalPosVH << std::endl;

        Local3DVector localDirVH = vhIt.localDirection();
        LogTrace("VectorHitsBuilderValidation") << "local VH direction " << localDirVH << std::endl;

        VectorHit vh = vhIt;
        Global3DVector globalDirVH = vh.globalDirectionVH();
        dirVHs.push_back(globalDirVH);
        LogTrace("VectorHitsBuilderValidation") << "global VH direction " << globalDirVH << std::endl;

        // Fill the position histograms
        trackerLayoutRZ_[0]->SetPoint(nVHsTot, globalPosVH.z(), globalPosVH.perp());
        trackerLayoutXY_[0]->SetPoint(nVHsTot, globalPosVH.x(), globalPosVH.y());

        if (layer < 100)
          trackerLayoutXYBar_->SetPoint(nVHsTot, globalPosVH.x(), globalPosVH.y());
        else
          trackerLayoutXYEC_->SetPoint(nVHsTot, globalPosVH.x(), globalPosVH.y());

        histogramLayer->second.localPosXY[0]->SetPoint(nVHsTot, vh_x_local, vh_y_local);
        histogramLayer->second.globalPosXY[0]->SetPoint(nVHsTot, globalPosVH.x(), globalPosVH.y());

        localPosXvsDeltaX_[0]->SetPoint(nVHsTot, vh_x_local, localDirVH.x());
        localPosYvsDeltaY_[0]->SetPoint(nVHsTot, vh_y_local, localDirVH.y());

        // Pixel module
        const StackGeomDet* stackDet = dynamic_cast<const StackGeomDet*>(geomDet);
        const PixelGeomDetUnit* geomDetLower = dynamic_cast<const PixelGeomDetUnit*>(stackDet->lowerDet());
        DetId lowerDetId = stackDet->lowerDet()->geographicalId();
        DetId upperDetId = stackDet->upperDet()->geographicalId();

        TrackerGeometry::ModuleType mType = tkGeom_->getDetectorType(lowerDetId);
        module_type = 0;
        if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
          module_type = 1;
          trackerLayoutRZ_[1]->SetPoint(nVHsPSTot, globalPosVH.z(), globalPosVH.perp());
          trackerLayoutXY_[1]->SetPoint(nVHsPSTot, globalPosVH.x(), globalPosVH.y());

          histogramLayer->second.localPosXY[1]->SetPoint(nVHsPSTot, vh_x_local, vh_y_local);
          histogramLayer->second.globalPosXY[1]->SetPoint(nVHsPSTot, globalPosVH.x(), globalPosVH.y());

          localPosXvsDeltaX_[1]->SetPoint(nVHsPSTot, vh_x_local, localDirVH.x());
          localPosYvsDeltaY_[1]->SetPoint(nVHsPSTot, vh_y_local, localDirVH.y());

          ++nVHsPS;
          ++nVHsPSTot;
        }

        // Strip module
        else if (mType == TrackerGeometry::ModuleType::Ph2SS) {
          module_type = 2;
          trackerLayoutRZ_[2]->SetPoint(nVHs2STot, globalPosVH.z(), globalPosVH.perp());
          trackerLayoutXY_[2]->SetPoint(nVHs2STot, globalPosVH.x(), globalPosVH.y());

          histogramLayer->second.localPosXY[2]->SetPoint(nVHs2STot, vh_x_local, vh_y_local);
          histogramLayer->second.globalPosXY[2]->SetPoint(nVHs2STot, globalPosVH.x(), globalPosVH.y());

          localPosXvsDeltaX_[2]->SetPoint(nVHs2STot, vh_x_local, localDirVH.x());
          localPosYvsDeltaY_[2]->SetPoint(nVHs2STot, vh_y_local, localDirVH.y());

          ++nVHs2S;
          ++nVHs2STot;
        } else if (mType == TrackerGeometry::ModuleType::Ph2PSS) {
          edm::LogError("VectorHitsBuilderValidation") << "module type " << module_type << " should never happen!";
        }
        LogTrace("VectorHitsBuilderValidation") << "module type " << module_type << std::endl;

        // get the geomDetUnit of the clusters
        low_x_global = vhIt.lowerGlobalPos().x();
        low_y_global = vhIt.lowerGlobalPos().y();
        low_z_global = vhIt.lowerGlobalPos().z();
        upp_x_global = vhIt.upperGlobalPos().x();
        upp_y_global = vhIt.upperGlobalPos().y();
        upp_z_global = vhIt.upperGlobalPos().z();

        low_xx_global_err = vhIt.lowerGlobalPosErr().cxx();
        low_yy_global_err = vhIt.lowerGlobalPosErr().cyy();
        low_zz_global_err = vhIt.lowerGlobalPosErr().czz();
        low_xy_global_err = vhIt.lowerGlobalPosErr().cyx();
        low_zx_global_err = vhIt.lowerGlobalPosErr().czx();
        low_zy_global_err = vhIt.lowerGlobalPosErr().czy();

        upp_xx_global_err = vhIt.upperGlobalPosErr().cxx();
        upp_yy_global_err = vhIt.upperGlobalPosErr().cyy();
        upp_zz_global_err = vhIt.upperGlobalPosErr().czz();
        upp_xy_global_err = vhIt.upperGlobalPosErr().cyx();
        upp_zx_global_err = vhIt.upperGlobalPosErr().czx();
        upp_zy_global_err = vhIt.upperGlobalPosErr().czy();

        LogDebug("VectorHitsBuilderValidation") << "print Clusters into the VH:" << std::endl;
        printCluster(geomDetLower, vhIt.lowerClusterRef());
        LogTrace("VectorHitsBuilderValidation") << "\t global pos lower " << vhIt.lowerGlobalPos() << std::endl;
        LogTrace("VectorHitsBuilderValidation")
            << "\t global posErr lower " << vhIt.lowerGlobalPosErr().cxx() << std::endl;
        const GeomDetUnit* geomDetUpper = stackDet->upperDet();
        printCluster(geomDetUpper, vhIt.upperClusterRef());
        LogTrace("VectorHitsBuilderValidation") << "\t global pos upper " << vhIt.upperGlobalPos() << std::endl;

        //comparison with SIM hits
        LogDebug("VectorHitsBuilderValidation") << "comparison Clusters with sim hits ... " << std::endl;
        std::vector<unsigned int> clusterSimTrackIds;
        std::vector<unsigned int> clusterSimTrackIdsUpp;
        std::set<std::pair<uint32_t, EncodedEventId> > simTkIds;
        const GeomDetUnit* geomDetUnit_low(tkGeom_->idToDetUnit(lowerDetId));
        LogTrace("VectorHitsBuilderValidation") << " lowerDetID : " << lowerDetId.rawId();
        const GeomDetUnit* geomDetUnit_upp(tkGeom_->idToDetUnit(upperDetId));
        LogTrace("VectorHitsBuilderValidation") << " upperDetID : " << upperDetId.rawId();

        for (unsigned int istr(0); istr < (*(vhIt.lowerClusterRef().cluster_phase2OT())).size(); ++istr) {
          uint32_t channel =
              Phase2TrackerDigi::pixelToChannel((*(vhIt.lowerClusterRef().cluster_phase2OT())).firstRow() + istr,
                                                (*(vhIt.lowerClusterRef().cluster_phase2OT())).column());
          unsigned int LowerSimTrackId(getSimTrackId(siphase2SimLinks, lowerDetId, channel));
          std::vector<std::pair<uint32_t, EncodedEventId> > trkid(
              getSimTrackIds(siphase2SimLinks, lowerDetId, channel));
          if (trkid.size() == 0)
            continue;
          clusterSimTrackIds.push_back(LowerSimTrackId);
          simTkIds.insert(trkid.begin(), trkid.end());
          LogTrace("VectorHitsBuilderValidation") << "LowerSimTrackId " << LowerSimTrackId << std::endl;
        }
        // In the case of PU, we need the TPs to find the proper SimTrackID
        for (const auto& iset : simTkIds) {
          auto ipos = mapping.find(iset);
          if (ipos != mapping.end()) {
            LogTrace("VectorHitsBuilderValidation") << "lower cluster in detid: " << lowerDetId.rawId()
                                                    << " from tp: " << ipos->second.key() << " " << iset.first;
            LogTrace("VectorHitsBuilderValidation") << "with pt(): " << (*ipos->second).pt();
            low_tp_id = ipos->second.key();
            vh_sim_trackPt = (*ipos->second).pt();
          }
        }

        simTkIds.clear();
        for (unsigned int istr(0); istr < (*(vhIt.upperClusterRef().cluster_phase2OT())).size(); ++istr) {
          uint32_t channel =
              Phase2TrackerDigi::pixelToChannel((*(vhIt.upperClusterRef().cluster_phase2OT())).firstRow() + istr,
                                                (*(vhIt.upperClusterRef().cluster_phase2OT())).column());
          unsigned int UpperSimTrackId(getSimTrackId(siphase2SimLinks, upperDetId, channel));
          std::vector<std::pair<uint32_t, EncodedEventId> > trkid(
              getSimTrackIds(siphase2SimLinks, upperDetId, channel));
          if (trkid.size() == 0)
            continue;
          clusterSimTrackIdsUpp.push_back(UpperSimTrackId);
          simTkIds.insert(trkid.begin(), trkid.end());
          LogTrace("VectorHitsBuilderValidation") << "UpperSimTrackId " << UpperSimTrackId << std::endl;
        }
        // In the case of PU, we need the TPs to find the proper SimTrackID
        for (const auto& iset : simTkIds) {
          auto ipos = mapping.find(iset);
          if (ipos != mapping.end()) {
            LogTrace("VectorHitsBuilderValidation")
                << "upper cluster in detid: " << upperDetId.rawId() << " from tp: " << ipos->second.key() << " "
                << iset.first << std::endl;
            upp_tp_id = ipos->second.key();
          }
        }
        //compute if the vhits is 'true' or 'false' and save sim pT
        std::pair<bool, uint32_t> istrue = isTrue(vhIt, siphase2SimLinks, detId);
        vh_isTrue = 0;
        if (istrue.first) {
          vh_isTrue = 1;
        }

        // loop over all simHits
        unsigned int totalSimHits(0);
        unsigned int primarySimHits(0);
        unsigned int otherSimHits(0);

        for (const auto& hitIt : *simHitsRaw) {
          if (hitIt.detUnitId() == geomDetLower->geographicalId()) {
            //check clusters track id compatibility
            if (std::find(clusterSimTrackIds.begin(), clusterSimTrackIds.end(), hitIt.trackId()) !=
                clusterSimTrackIds.end()) {
              Local3DPoint localPosHit(hitIt.localPosition());
              sim_x_local = localPosHit.x();
              sim_y_local = localPosHit.y();

              deltaXVHSimHits = vh_x_local - sim_x_local;
              deltaYVHSimHits = vh_y_local - sim_y_local;

              Global3DPoint globalPosHit = geomDetLower->surface().toGlobal(localPosHit);
              sim_x_global = globalPosHit.x();
              sim_y_global = globalPosHit.y();
              sim_z_global = globalPosHit.z();

              histogramLayer->second.deltaXVHSimHits[0]->Fill(vh_x_local - sim_x_local);
              histogramLayer->second.deltaYVHSimHits[0]->Fill(vh_y_local - sim_y_local);

              // Pixel module
              if (layer == 1 || layer == 2 || layer == 3) {
                histogramLayer->second.deltaXVHSimHits[1]->Fill(vh_x_local - sim_x_local);
                histogramLayer->second.deltaYVHSimHits[1]->Fill(vh_y_local - sim_y_local);
              }
              // Strip module
              else if (layer == 4 || layer == 5 || layer == 6) {
                histogramLayer->second.deltaXVHSimHits[2]->Fill(vh_x_local - sim_x_local);
                histogramLayer->second.deltaYVHSimHits[2]->Fill(vh_y_local - sim_y_local);
              }

              ++totalSimHits;

              std::map<unsigned int, SimTrack>::const_iterator simTrackIt(simTracks.find(hitIt.trackId()));
              if (simTrackIt == simTracks.end())
                continue;

              // Primary particles only
              processType = hitIt.processType();

              if (simTrackIt->second.vertIndex() == 0 and
                  (processType == 2 || processType == 7 || processType == 9 || processType == 11 || processType == 13 ||
                   processType == 15)) {
                histogramLayer->second.deltaXVHSimHits_P[0]->Fill(vh_x_local - sim_x_local);
                histogramLayer->second.deltaYVHSimHits_P[0]->Fill(vh_y_local - sim_y_local);

                // Pixel module
                if (layer == 1 || layer == 2 || layer == 3) {
                  histogramLayer->second.deltaXVHSimHits_P[1]->Fill(vh_x_local - sim_x_local);
                  histogramLayer->second.deltaYVHSimHits_P[1]->Fill(vh_y_local - sim_y_local);
                }
                // Strip module
                else if (layer == 4 || layer == 5 || layer == 6) {
                  histogramLayer->second.deltaXVHSimHits_P[2]->Fill(vh_x_local - sim_x_local);
                  histogramLayer->second.deltaYVHSimHits_P[2]->Fill(vh_y_local - sim_y_local);
                }

                ++primarySimHits;
              }

              otherSimHits = totalSimHits - primarySimHits;

              histogramLayer->second.totalSimHits->Fill(totalSimHits);
              histogramLayer->second.primarySimHits->Fill(primarySimHits);
              histogramLayer->second.otherSimHits->Fill(otherSimHits);
            }
          }
        }  // loop simhits

        nVHsTot++;

        //******************************
        //combinatorial studies : not filling if more than 1 VH has been produced
        //******************************
        multiplicity = DSViter.size();
        if (DSViter.size() > 1) {
          LogTrace("VectorHitsBuilderValidation") << " not filling if more than 1 VH has been produced";
          width = -100;
          deltaXlocal = -100;
          tree_->Fill();
          continue;
        }

        //curvature
        GlobalPoint center(0.0, 0.0, 0.0);
        curvature = vh.curvature();
        phi = vh.phi();
        QOverPT = vh.transverseMomentum(magField_->inTesla(center).z());
        QOverP = vh.momentum(magField_->inTesla(center).z());
        histogramLayer->second.curvature->Fill(curvature);

        //stub width

        auto&& lparamsUpp = cpe_->localParameters(*vhIt.upperClusterRef().cluster_phase2OT(), *geomDetUnit_upp);
        LogTrace("VectorHitsBuilderValidation") << " upper local pos (in its system of reference):" << lparamsUpp.first;
        Global3DPoint gparamsUpp = geomDetUnit_upp->surface().toGlobal(lparamsUpp.first);
        LogTrace("VectorHitsBuilderValidation") << " upper global pos :" << gparamsUpp;
        Local3DPoint lparamsUppInLow = geomDetUnit_low->surface().toLocal(gparamsUpp);
        LogTrace("VectorHitsBuilderValidation") << " upper local pos (in low system of reference):" << lparamsUppInLow;
        auto&& lparamsLow = cpe_->localParameters(*vhIt.lowerClusterRef().cluster_phase2OT(), *geomDetUnit_low);
        LogTrace("VectorHitsBuilderValidation") << " lower local pos (in its system of reference):" << lparamsLow.first;
        Global3DPoint gparamsLow = geomDetUnit_low->surface().toGlobal(lparamsLow.first);
        LogTrace("VectorHitsBuilderValidation") << " lower global pos :" << gparamsLow;

        deltaXlocal = lparamsUppInLow.x() - lparamsLow.first.x();
        histogramLayer->second.deltaXlocal->Fill(deltaXlocal);
        LogTrace("VectorHitsBuilderValidation") << " deltaXlocal : " << deltaXlocal;

        double parallCorr = 0.0;

        Global3DPoint origin(0, 0, 0);
        GlobalVector gV = gparamsLow - origin;
        LocalVector lV = geomDetUnit_low->surface().toLocal(gV);
        LocalVector lV_norm = lV / lV.z();
        parallCorr = lV_norm.x() * lparamsUppInLow.z();
        LogTrace("VectorHitsBuilderValidation") << " parallalex correction:" << parallCorr;

        double lpos_upp_corr = 0.0;
        double lpos_low_corr = 0.0;
        if (lparamsUpp.first.x() > lparamsLow.first.x()) {
          if (lparamsUpp.first.x() > 0) {
            lpos_low_corr = lparamsLow.first.x();
            lpos_upp_corr = lparamsUpp.first.x() - fabs(parallCorr);
          }
          if (lparamsUpp.first.x() < 0) {
            lpos_low_corr = lparamsLow.first.x() + fabs(parallCorr);
            lpos_upp_corr = lparamsUpp.first.x();
          }
        } else if (lparamsUpp.first.x() < lparamsLow.first.x()) {
          if (lparamsUpp.first.x() > 0) {
            lpos_low_corr = lparamsLow.first.x() - fabs(parallCorr);
            lpos_upp_corr = lparamsUpp.first.x();
          }
          if (lparamsUpp.first.x() < 0) {
            lpos_low_corr = lparamsLow.first.x();
            lpos_upp_corr = lparamsUpp.first.x() + fabs(parallCorr);
          }
        } else {
          if (lparamsUpp.first.x() > 0) {
            lpos_upp_corr = lparamsUpp.first.x() - fabs(parallCorr);
            lpos_low_corr = lparamsLow.first.x();
          }
          if (lparamsUpp.first.x() < 0) {
            lpos_upp_corr = lparamsUpp.first.x() + fabs(parallCorr);
            lpos_low_corr = lparamsLow.first.x();
          }
        }

        LogDebug("VectorHitsBuilderValidation") << " \t local pos upper corrected (x):" << lpos_upp_corr << std::endl;
        LogDebug("VectorHitsBuilderValidation") << " \t local pos lower corrected (x):" << lpos_low_corr << std::endl;

        width = lpos_low_corr - lpos_upp_corr;
        histogramLayer->second.width->Fill(width);
        LogTrace("VectorHitsBuilderValidation") << " width:" << width;

        tree_->Fill();

      }  // vh valid

    }  // loop vhs

    if (nVHsPS)
      histogramLayer->second.numberVHsPS->Fill(nVHsPS);
    if (nVHs2S)
      histogramLayer->second.numberVHs2S->Fill(nVHs2S);
    LogTrace("VectorHitsBuilderValidation")
        << "nVHsPS for this layer : " << nVHsPS << ", nVHs2S for this layer : " << nVHs2S << std::endl;
  }

  CreateVHsXYGraph(glVHs, dirVHs);
  CreateVHsRZGraph(glVHs, dirVHs);

  int VHaccTrue = 0.0;
  int VHaccFalse = 0.0;
  int VHrejTrue = 0.0;
  int VHrejFalse = 0.0;
  int VHaccTrue_signal = 0.0;
  int VHrejTrue_signal = 0.0;

  // Loop over modules
  for (const auto& DSViter : *vhsAcc) {
    unsigned int rawid(DSViter.detId());
    DetId detId(rawid);
    int layerAcc = getLayerNumber(detId);
    LogTrace("VectorHitsBuilderValidation") << "acc Layer: " << layerAcc << "  det id" << rawid << std::endl;
    for (const auto& vhIt : DSViter) {
      if (vhIt.isValid()) {
        VHaccLayer_->Fill(layerAcc);
        VHacc++;

        //compute if the vhits is 'true' or 'false'
        std::pair<bool, uint32_t> istrue = isTrue(vhIt, siphase2SimLinks, detId);
        if (istrue.first) {
          LogTrace("VectorHitsBuilderValidation") << "this vectorhit is a 'true' vhit.";
          VHaccTrueLayer_->Fill(layerAcc);
          VHaccTrue++;

          //saving info of 'signal' track
          std::map<unsigned int, SimTrack>::const_iterator simTrackIt(simTracks.find(istrue.second));
          if (simTrackIt == simTracks.end())
            continue;
          LogTrace("VectorHitsBuilderValidation") << "this vectorhit is associated with SimTrackId: " << istrue.second;
          LogTrace("VectorHitsBuilderValidation") << "the SimTrack has pt: " << simTrackIt->second.momentum().pt();
          if (simTrackIt->second.momentum().pt() > 1) {
            VHaccTrue_signal_Layer_->Fill(layerAcc);
            LogTrace("VectorHitsBuilderValidation") << "the vectorhit belongs to signal";
            VHaccTrue_signal++;
          }

        } else {
          LogTrace("VectorHitsBuilderValidation") << "this vectorhit is a 'false' vhit.";
          VHaccFalse++;
        }
      }
    }
  }

  for (const auto& DSViter : *vhsRej) {
    unsigned int rawid(DSViter.detId());
    DetId detId(rawid);
    int layerRej = getLayerNumber(detId);
    LogTrace("VectorHitsBuilderValidation") << "rej Layer: " << layerRej << "  det id" << rawid << std::endl;
    for (const auto& vhIt : DSViter) {
      VHrejLayer_->Fill(layerRej);
      VHrej++;

      //compute if the vhits is 'true' or 'false'
      std::pair<bool, uint32_t> istrue = isTrue(vhIt, siphase2SimLinks, detId);
      if (istrue.first) {
        LogTrace("VectorHitsBuilderValidation") << "this vectorhit is a 'true' vhit.";
        VHrejTrueLayer_->Fill(layerRej);
        VHrejTrue++;

        //saving info of 'signal' track
        std::map<unsigned int, SimTrack>::const_iterator simTrackIt(simTracks.find(istrue.second));
        if (simTrackIt == simTracks.end())
          continue;
        LogTrace("VectorHitsBuilderValidation") << "this vectorhit is associated with SimTrackId: " << istrue.second;
        LogTrace("VectorHitsBuilderValidation") << "the SimTrack has pt: " << simTrackIt->second.momentum().pt();
        if (simTrackIt->second.momentum().pt() > 1) {
          VHrejTrue_signal_Layer_->Fill(layerRej);
          LogTrace("VectorHitsBuilderValidation") << "the vectorhit belongs to signal";
          VHrejTrue_signal++;
        }

      } else {
        LogTrace("VectorHitsBuilderValidation") << "this vectorhit is a 'false' vhit.";
        VHrejFalse++;
      }
    }
  }

  int VHtot = VHacc + VHrej;
  LogTrace("VectorHitsBuilderValidation")
      << "VH total: " << VHtot << " with " << VHacc << " VHs accepted and " << VHrej << " VHs rejected.";
  LogTrace("VectorHitsBuilderValidation")
      << "of the VH accepted, there are " << VHaccTrue << " true and " << VHaccFalse << " false.";
  LogTrace("VectorHitsBuilderValidation")
      << "of the VH rejected, there are " << VHrejTrue << " true and " << VHrejFalse << " false.";
  LogTrace("VectorHitsBuilderValidation")
      << "of the true VH    , there are " << VHaccTrue_signal << " accepted belonging to signal and "
      << VHrejTrue_signal << " rejected belonging to signal.";

  //    CreateWindowCorrGraph();
}

// Check if the vector hit is true (both clusters are formed from the same SimTrack
std::pair<bool, uint32_t> VectorHitsBuilderValidation::isTrue(
    const VectorHit vh, const edm::Handle<edm::DetSetVector<PixelDigiSimLink> >& siphase2SimLinks, DetId& detId) const {
  const GeomDet* geomDet(tkGeom_->idToDet(detId));
  const StackGeomDet* stackDet = dynamic_cast<const StackGeomDet*>(geomDet);
  const GeomDetUnit* geomDetLower = stackDet->lowerDet();
  const GeomDetUnit* geomDetUpper = stackDet->upperDet();

  std::vector<unsigned int> lowClusterSimTrackIds;

  for (unsigned int istr(0); istr < (*(vh.lowerClusterRef().cluster_phase2OT())).size(); ++istr) {
    uint32_t channel = Phase2TrackerDigi::pixelToChannel((*(vh.lowerClusterRef().cluster_phase2OT())).firstRow() + istr,
                                                         (*(vh.lowerClusterRef().cluster_phase2OT())).column());
    DetId detIdCluster = geomDetLower->geographicalId();
    unsigned int simTrackId(getSimTrackId(siphase2SimLinks, detIdCluster, channel));
    LogTrace("VectorHitsBuilderValidation") << "LowerSimTrackId " << simTrackId << std::endl;
    std::vector<std::pair<uint32_t, EncodedEventId> > trkid(getSimTrackIds(siphase2SimLinks, detIdCluster, channel));
    if (trkid.size() == 0)
      continue;
    lowClusterSimTrackIds.push_back(simTrackId);
  }

  std::vector<unsigned int>::iterator it_simTrackUpper;

  for (unsigned int istr(0); istr < (*(vh.upperClusterRef().cluster_phase2OT())).size(); ++istr) {
    uint32_t channel = Phase2TrackerDigi::pixelToChannel((*(vh.upperClusterRef().cluster_phase2OT())).firstRow() + istr,
                                                         (*(vh.upperClusterRef().cluster_phase2OT())).column());
    DetId detIdCluster = geomDetUpper->geographicalId();
    unsigned int simTrackId(getSimTrackId(siphase2SimLinks, detIdCluster, channel));
    LogTrace("VectorHitsBuilderValidation") << "UpperSimTrackId " << simTrackId << std::endl;
    std::vector<std::pair<uint32_t, EncodedEventId> > trkid(getSimTrackIds(siphase2SimLinks, detIdCluster, channel));
    if (trkid.size() == 0)
      continue;
    it_simTrackUpper = std::find(lowClusterSimTrackIds.begin(), lowClusterSimTrackIds.end(), simTrackId);
    if (it_simTrackUpper != lowClusterSimTrackIds.end()) {
      LogTrace("VectorHitsBuilderValidation") << " UpperSimTrackId found in lowClusterSimTrackIds ";
      return std::make_pair(true, simTrackId);
    }
  }
  return std::make_pair(false, 0);
}

// Create the histograms
std::map<unsigned int, VHHistos>::iterator VectorHitsBuilderValidation::createLayerHistograms(unsigned int ival) {
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

  VHHistos local_histos;

  std::ostringstream histoName;

  /*
     * Number of clusters
     */

  histoName.str("");
  histoName << "Number_VHs_PS" << tag.c_str() << id;
  local_histos.numberVHsPS = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 20, 0., 20.);
  local_histos.numberVHsPS->SetFillColor(kAzure + 7);

  histoName.str("");
  histoName << "Number_VHs_2S" << tag.c_str() << id;
  local_histos.numberVHs2S = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 20, 0., 20.);
  local_histos.numberVHs2S->SetFillColor(kOrange - 3);

  histoName.str("");
  histoName << "Number_VHs_Mixed" << tag.c_str() << id;
  local_histos.numberVHsMixed = td.make<THStack>(histoName.str().c_str(), histoName.str().c_str());
  local_histos.numberVHsMixed->Add(local_histos.numberVHsPS);
  local_histos.numberVHsMixed->Add(local_histos.numberVHs2S);

  /*
     * Local and Global positions
     */

  histoName.str("");
  histoName << "Local_Position_XY_Mixed" << tag.c_str() << id;
  local_histos.localPosXY[0] = td.make<TGraph>();
  local_histos.localPosXY[0]->SetName(histoName.str().c_str());

  histoName.str("");
  histoName << "Local_Position_XY_PS" << tag.c_str() << id;
  local_histos.localPosXY[1] = td.make<TGraph>();
  local_histos.localPosXY[1]->SetName(histoName.str().c_str());

  histoName.str("");
  histoName << "Local_Position_XY_2S" << tag.c_str() << id;
  local_histos.localPosXY[2] = td.make<TGraph>();
  local_histos.localPosXY[2]->SetName(histoName.str().c_str());

  histoName.str("");
  histoName << "Global_Position_XY_Mixed" << tag.c_str() << id;
  local_histos.globalPosXY[0] = td.make<TGraph>();
  local_histos.globalPosXY[0]->SetName(histoName.str().c_str());

  histoName.str("");
  histoName << "Global_Position_XY_PS" << tag.c_str() << id;
  local_histos.globalPosXY[1] = td.make<TGraph>();
  local_histos.globalPosXY[1]->SetName(histoName.str().c_str());

  histoName.str("");
  histoName << "Global_Position_XY_2S" << tag.c_str() << id;
  local_histos.globalPosXY[2] = td.make<TGraph>();
  local_histos.globalPosXY[2]->SetName(histoName.str().c_str());

  /*
     * Delta positions with SimHits
     */

  histoName.str("");
  histoName << "Delta_X_VH_SimHits_Mixed" << tag.c_str() << id;
  local_histos.deltaXVHSimHits[0] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_X_VH_SimHits_PS" << tag.c_str() << id;
  local_histos.deltaXVHSimHits[1] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_X_VH_SimHits_2S" << tag.c_str() << id;
  local_histos.deltaXVHSimHits[2] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_Y_VH_SimHits_Mixed" << tag.c_str() << id;
  local_histos.deltaYVHSimHits[0] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_Y_VH_SimHits_PS" << tag.c_str() << id;
  local_histos.deltaYVHSimHits[1] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_Y_VH_SimHits_2S" << tag.c_str() << id;
  local_histos.deltaYVHSimHits[2] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  /*
     * Delta position with simHits for primary tracks only
     */

  histoName.str("");
  histoName << "Delta_X_VH_SimHits_Mixed_P" << tag.c_str() << id;
  local_histos.deltaXVHSimHits_P[0] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_X_VH_SimHits_PS_P" << tag.c_str() << id;
  local_histos.deltaXVHSimHits_P[1] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_X_VH_SimHits_2S_P" << tag.c_str() << id;
  local_histos.deltaXVHSimHits_P[2] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_Y_VH_SimHits_Mixed_P" << tag.c_str() << id;
  local_histos.deltaYVHSimHits_P[0] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_Y_VH_SimHits_PS_P" << tag.c_str() << id;
  local_histos.deltaYVHSimHits_P[1] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  histoName.str("");
  histoName << "Delta_Y_VH_SimHits_2S_P" << tag.c_str() << id;
  local_histos.deltaYVHSimHits_P[2] = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 100, 0., 0.);

  /*
     * Information on the Digis per cluster
     */

  histoName.str("");
  histoName << "Total_Digis" << tag.c_str() << id;
  local_histos.totalSimHits = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 10, 0., 10.);

  histoName.str("");
  histoName << "Primary_Digis" << tag.c_str() << id;
  local_histos.primarySimHits = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 10, 0., 10.);

  histoName.str("");
  histoName << "Other_Digis" << tag.c_str() << id;
  local_histos.otherSimHits = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 10, 0., 10.);

  /*
     * Study on the clusters combinatorial problem
     */

  histoName.str("");
  histoName << "DeltaXlocal_clusters" << tag.c_str() << id;
  local_histos.deltaXlocal = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 200, -0.4, 0.4);
  histoName.str("");
  histoName << "Width" << tag.c_str() << id;
  local_histos.width = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 200, -0.4, 0.4);
  histoName.str("");
  histoName << "Curvature" << tag.c_str() << id;
  local_histos.curvature = td.make<TH1F>(histoName.str().c_str(), histoName.str().c_str(), 200, -0.4, 0.4);

  std::pair<std::map<unsigned int, VHHistos>::iterator, bool> insertedIt(
      histograms_.insert(std::make_pair(ival, local_histos)));
  fs->file().cd("/");

  return insertedIt.first;
}

void VectorHitsBuilderValidation::CreateVHsXYGraph(const std::vector<Global3DPoint> glVHs,
                                                   const std::vector<Global3DVector> dirVHs) {
  if (glVHs.size() != dirVHs.size()) {
    std::cout << "Cannot fullfil the graphs for this event. Return." << std::endl;
    return;
  }

  // opening canvas and drawing XY TGraph

  for (unsigned int nVH = 0; nVH < glVHs.size(); nVH++) {
    //same r
    if ((fabs(dirVHs.at(nVH).x()) < 10e-5) && (fabs(dirVHs.at(nVH).y()) < 10e-5)) {
      continue;

    } else {
    }
  }

  return;
}

void VectorHitsBuilderValidation::CreateVHsRZGraph(const std::vector<Global3DPoint> glVHs,
                                                   const std::vector<Global3DVector> dirVHs) {
  if (glVHs.size() != dirVHs.size()) {
    std::cout << "Cannot fullfil the graphs for this event. Return." << std::endl;
    return;
  }

  return;
}

void VectorHitsBuilderValidation::CreateWindowCorrGraph() {
  //FIXME: This function is not working properly, yet.

  //return if we are not using Phase2 OT
  if (!tkGeom_->isThere(GeomDetEnumerators::P2OTB) && !tkGeom_->isThere(GeomDetEnumerators::P2OTEC))
    return;

  for (auto det : tkGeom_->detsTOB()) {
    ParallaxCorrectionRZ_->Fill(det->position().z(), det->position().perp(), 5.);
  }
  for (auto det : tkGeom_->detsTID()) {
    ParallaxCorrectionRZ_->Fill(det->position().z(), det->position().perp(), 10.);
  }
  ParallaxCorrectionRZ_->Fill(0., 0., 5.);
  return;
}

unsigned int VectorHitsBuilderValidation::getLayerNumber(const DetId& detid) {
  if (detid.det() == DetId::Tracker) {
    if (detid.subdetId() == StripSubdetector::TOB)
      return (tkTopo_->layer(detid));
    else if (detid.subdetId() == StripSubdetector::TID)
      return (100 * tkTopo_->side(detid) + tkTopo_->layer(detid));
    else
      return 999;
  }
  return 999;
}

unsigned int VectorHitsBuilderValidation::getModuleNumber(const DetId& detid) { return (tkTopo_->module(detid)); }

std::vector<std::pair<uint32_t, EncodedEventId> > VectorHitsBuilderValidation::getSimTrackIds(
    const edm::Handle<edm::DetSetVector<PixelDigiSimLink> >& simLinks, const DetId& detId, uint32_t channel) const {
  std::vector<std::pair<uint32_t, EncodedEventId> > simTrkId;
  auto isearch = simLinks->find(detId);
  if (isearch != simLinks->end()) {
    // Loop over DigiSimLink in this det unit
    edm::DetSet<PixelDigiSimLink> link_detset = (*isearch);
    for (const auto& it : link_detset.data) {
      if (channel == it.channel())
        simTrkId.push_back(std::make_pair(it.SimTrackId(), it.eventId()));
    }
  }
  return simTrkId;
}

unsigned int VectorHitsBuilderValidation::getSimTrackId(
    const edm::Handle<edm::DetSetVector<PixelDigiSimLink> >& pixelSimLinks,
    const DetId& detId,
    unsigned int channel) const {
  edm::DetSetVector<PixelDigiSimLink>::const_iterator DSViter(pixelSimLinks->find(detId));
  if (DSViter == pixelSimLinks->end())
    return 0;
  for (const auto& it : DSViter->data) {
    if (channel == it.channel())
      return it.SimTrackId();
  }
  return 0;
}

void VectorHitsBuilderValidation::printCluster(const GeomDetUnit* geomDetUnit, const OmniClusterRef cluster) {
  if (!geomDetUnit)
    return;

  const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(geomDetUnit);
  const PixelTopology& topol = theGeomDet->specificTopology();

  unsigned int layer = getLayerNumber(geomDetUnit->geographicalId());
  unsigned int module = getModuleNumber(geomDetUnit->geographicalId());
  LogTrace("VectorHitsBuilderValidation") << "Layer:" << layer << std::endl;
  if (topol.ncolumns() == 32)
    LogTrace("VectorHitsBuilderValidation") << "Pixel cluster with detId:" << geomDetUnit->geographicalId().rawId()
                                            << "(module:" << module << ") " << std::endl;
  else if (topol.ncolumns() == 2)
    LogTrace("VectorHitsBuilderValidation") << "Strip cluster with detId:" << geomDetUnit->geographicalId().rawId()
                                            << "(module:" << module << ") " << std::endl;
  else
    std::cout << "no module?!" << std::endl;
  LogTrace("VectorHitsBuilderValidation")
      << "with pitch:" << topol.pitch().first << " , " << topol.pitch().second << std::endl;
  LogTrace("VectorHitsBuilderValidation") << " and width:" << theGeomDet->surface().bounds().width()
                                          << " , lenght:" << theGeomDet->surface().bounds().length() << std::endl;

  auto&& lparams = cpe_->localParameters(*cluster.cluster_phase2OT(), *theGeomDet);

  LogTrace("VectorHitsBuilderValidation")
      << "\t local  pos " << lparams.first << "with err " << lparams.second << std::endl;

  return;
}

void VectorHitsBuilderValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("src", "siPhase2Clusters");
  desc.add<edm::InputTag>("links", edm::InputTag("simSiPixelDigis", "Tracker"));
  desc.add<edm::InputTag>("VH_acc", edm::InputTag("siPhase2VectorHits", "accepted"));
  desc.add<edm::InputTag>("VH_rej", edm::InputTag("siPhase2VectorHits", "rejected"));
  desc.add<edm::ESInputTag>("CPE", edm::ESInputTag("phase2StripCPEESProducer", "Phase2StripCPE"));
  desc.add<edm::InputTag>("trackingParticleSrc", edm::InputTag("mix", "MergedTrackTruth"));
  descriptions.add("vectorHitsBuilderValidation", desc);
}

DEFINE_FWK_MODULE(VectorHitsBuilderValidation);
