// -*- C++ -*-
//
// Package:    CalibTracker/SiPixelLorentzAnglePCLWorker
// Class:      SiPixelLorentzAnglePCLWorker
//
/**\class SiPixelLorentzAnglePCLWorker SiPixelLorentzAnglePCLWorker.cc CalibTracker/SiPixelLorentzAnglePCLWorker/src/SiPixelLorentzAnglePCLWorker.cc
 Description: generates the intermediate ALCAPROMPT dataset for the measurement of the SiPixel Lorentz Angle in the Prompt Calibration Loop
 Implementation:
     Books and fills 2D histograms of the drift vs depth in bins of pixel module rings to be fed into the SiPixelLorentzAnglePCLHarvester
*/
//
// Original Author:  mmusich
//         Created:  Sat, 29 May 2021 14:46:19 GMT
//
//

// system includes
#include <string>
#include <fmt/printf.h>

// user include files
#include "CalibTracker/Records/interface/SiPixelTemplateDBObjectESProducerRcd.h"
#include "CalibTracker/SiPixelLorentzAngle/interface/SiPixelLorentzAngleCalibrationStruct.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplateDefs.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyMap.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

// ROOT includes
#include <TTree.h>
#include <TFile.h>
#include <fstream>

//
// class declaration
//

static const int maxpix = 1000;
struct Pixinfo {
  int npix;
  float row[maxpix];
  float col[maxpix];
  float adc[maxpix];
  float x[maxpix];
  float y[maxpix];
};

struct Hit {
  float x;
  float y;
  double alpha;
  double beta;
  double gamma;
};
struct Clust {
  float x;
  float y;
  float charge;
  int size_x;
  int size_y;
  int maxPixelCol;
  int maxPixelRow;
  int minPixelCol;
  int minPixelRow;
};
struct Rechit {
  float x;
  float y;
};

enum LorentzAngleAnalysisTypeEnum { eGrazingAngle, eMinimumClusterSize };

class SiPixelLorentzAnglePCLWorker : public DQMEDAnalyzer {
public:
  explicit SiPixelLorentzAnglePCLWorker(const edm::ParameterSet&);
  ~SiPixelLorentzAnglePCLWorker() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(edm::Event const&, edm::EventSetup const&) override;

  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;

  void dqmEndRun(edm::Run const&, edm::EventSetup const&);

  const Pixinfo fillPix(const SiPixelCluster& LocPix, const PixelTopology* topol) const;
  const std::pair<LocalPoint, LocalPoint> surface_deformation(const PixelTopology* topol,
                                                              TrajectoryStateOnSurface& tsos,
                                                              const SiPixelRecHit* recHitPix) const;
  LorentzAngleAnalysisTypeEnum convertStringToLorentzAngleAnalysisTypeEnum(std::string type);
  // ------------ member data ------------
  SiPixelLorentzAngleCalibrationHistograms iHists;

  // template stuff
  edm::ESWatcher<SiPixelTemplateDBObjectESProducerRcd> watchSiPixelTemplateRcd_;
  const SiPixelTemplateDBObject* templateDBobject_;
  const std::vector<SiPixelTemplateStore>* thePixelTemp_ = nullptr;

  LorentzAngleAnalysisTypeEnum analysisType_;
  std::string folder_;
  bool notInPCL_;
  std::string filename_;
  std::vector<std::string> newmodulelist_;

  // tree branches barrel
  int run_;
  long int event_;
  int lumiblock_;
  int bx_;
  int orbit_;
  int module_;
  int ladder_;
  int layer_;
  int isflipped_;
  float pt_;
  float eta_;
  float phi_;
  double chi2_;
  double ndof_;
  Pixinfo pixinfo_;
  Hit simhit_, trackhit_;
  Clust clust_;
  Rechit rechit_;
  Rechit rechitCorr_;
  float trackhitCorrX_;
  float trackhitCorrY_;
  float qScale_;
  float rQmQt_;

  // tree branches forward
  int sideF_;
  int diskF_;
  int bladeF_;
  int panelF_;
  int moduleF_;
  Pixinfo pixinfoF_;
  Hit simhitF_, trackhitF_;
  Clust clustF_;
  Rechit rechitF_;
  Rechit rechitCorrF_;
  float trackhitCorrXF_;
  float trackhitCorrYF_;
  float qScaleF_;
  float rQmQtF_;

  // parameters from config file
  double ptmin_;
  double normChi2Max_;
  std::vector<int> clustSizeYMin_;
  int clustSizeXMax_;
  double residualMax_;
  double clustChargeMaxPerLength_;
  int hist_depth_;
  int hist_drift_;

  std::unique_ptr<TFile> hFile_;
  std::unique_ptr<TTree> SiPixelLorentzAngleTreeBarrel_;
  std::unique_ptr<TTree> SiPixelLorentzAngleTreeForward_;

  // es consumes
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomEsToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoEsToken_;
  edm::ESGetToken<SiPixelTemplateDBObject, SiPixelTemplateDBObjectESProducerRcd> siPixelTemplateEsToken_;
  edm::ESGetToken<std::vector<SiPixelTemplateStore>, SiPixelTemplateDBObjectESProducerRcd> siPixelTemplateStoreEsToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoPerEventEsToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomPerEventEsToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;

  // event consumes
  edm::EDGetTokenT<TrajTrackAssociationCollection> t_trajTrack;
};

//
// constructors and destructor
//
SiPixelLorentzAnglePCLWorker::SiPixelLorentzAnglePCLWorker(const edm::ParameterSet& iConfig)
    : analysisType_(convertStringToLorentzAngleAnalysisTypeEnum(iConfig.getParameter<std::string>("analysisType"))),
      folder_(iConfig.getParameter<std::string>("folder")),
      notInPCL_(iConfig.getParameter<bool>("notInPCL")),
      filename_(iConfig.getParameter<std::string>("fileName")),
      newmodulelist_(iConfig.getParameter<std::vector<std::string>>("newmodulelist")),
      ptmin_(iConfig.getParameter<double>("ptMin")),
      normChi2Max_(iConfig.getParameter<double>("normChi2Max")),
      clustSizeYMin_(iConfig.getParameter<std::vector<int>>("clustSizeYMin")),
      clustSizeXMax_(iConfig.getParameter<int>("clustSizeXMax")),
      residualMax_(iConfig.getParameter<double>("residualMax")),
      clustChargeMaxPerLength_(iConfig.getParameter<double>("clustChargeMaxPerLength")),
      hist_depth_(iConfig.getParameter<int>("binsDepth")),
      hist_drift_(iConfig.getParameter<int>("binsDrift")),
      geomEsToken_(esConsumes<edm::Transition::BeginRun>()),
      topoEsToken_(esConsumes<edm::Transition::BeginRun>()),
      siPixelTemplateEsToken_(esConsumes<edm::Transition::BeginRun>()),
      siPixelTemplateStoreEsToken_(esConsumes<edm::Transition::BeginRun>()),
      topoPerEventEsToken_(esConsumes()),
      geomPerEventEsToken_(esConsumes()),
      magneticFieldToken_(esConsumes()) {
  t_trajTrack = consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("src"));

  // now do what ever initialization is needed
  int bufsize = 64000;
  //    create tree structure
  //    Barrel pixel
  if (notInPCL_) {
    hFile_ = std::make_unique<TFile>(filename_.c_str(), "RECREATE");
    SiPixelLorentzAngleTreeBarrel_ =
        std::make_unique<TTree>("SiPixelLorentzAngleTreeBarrel_", "SiPixel LorentzAngle tree barrel", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("run", &run_, "run/I", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("event", &event_, "event/l", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("lumiblock", &lumiblock_, "lumiblock/I", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("bx", &bx_, "bx/I", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("orbit", &orbit_, "orbit/I", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("module", &module_, "module/I", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("ladder", &ladder_, "ladder/I", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("layer", &layer_, "layer/I", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("isflipped", &isflipped_, "isflipped/I", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("pt", &pt_, "pt/F", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("eta", &eta_, "eta/F", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("phi", &phi_, "phi/F", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("chi2", &chi2_, "chi2/D", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("ndof", &ndof_, "ndof/D", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("trackhit", &trackhit_, "x/F:y/F:alpha/D:beta/D:gamma_/D", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("npix", &pixinfo_.npix, "npix/I", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("rowpix", pixinfo_.row, "row[npix]/F", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("colpix", pixinfo_.col, "col[npix]/F", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("adc", pixinfo_.adc, "adc[npix]/F", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("xpix", pixinfo_.x, "x[npix]/F", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("ypix", pixinfo_.y, "y[npix]/F", bufsize);

    SiPixelLorentzAngleTreeBarrel_->Branch(
        "clust",
        &clust_,
        "x/F:y/F:charge/F:size_x/I:size_y/I:maxPixelCol/I:maxPixelRow:minPixelCol/I:minPixelRow/I",
        bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("rechit", &rechit_, "x/F:y/F", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("rechit_corr", &rechitCorr_, "x/F:y/F", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("trackhitcorr_x", &trackhitCorrX_, "trackhitcorr_x/F", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("trackhitcorr_y", &trackhitCorrY_, "trackhitcorr_y/F", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("qScale", &qScale_, "qScale/F", bufsize);
    SiPixelLorentzAngleTreeBarrel_->Branch("rQmQt", &rQmQt_, "rQmQt/F", bufsize);
    //    Forward pixel

    SiPixelLorentzAngleTreeForward_ =
        std::make_unique<TTree>("SiPixelLorentzAngleTreeForward_", "SiPixel LorentzAngle tree forward", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("run", &run_, "run/I", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("event", &event_, "event/l", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("lumiblock", &lumiblock_, "lumiblock/I", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("bx", &bx_, "bx/I", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("orbit", &orbit_, "orbit/I", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("side", &sideF_, "side/I", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("disk", &diskF_, "disk/I", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("blade", &bladeF_, "blade/I", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("panel", &panelF_, "panel/I", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("module", &moduleF_, "module/I", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("pt", &pt_, "pt/F", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("eta", &eta_, "eta/F", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("phi", &phi_, "phi/F", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("chi2", &chi2_, "chi2/D", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("ndof", &ndof_, "ndof/D", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("trackhit", &trackhitF_, "x/F:y/F:alpha/D:beta/D:gamma_/D", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("npix", &pixinfoF_.npix, "npix/I", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("rowpix", pixinfoF_.row, "row[npix]/F", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("colpix", pixinfoF_.col, "col[npix]/F", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("adc", pixinfoF_.adc, "adc[npix]/F", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("xpix", pixinfoF_.x, "x[npix]/F", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("ypix", pixinfoF_.y, "y[npix]/F", bufsize);

    SiPixelLorentzAngleTreeForward_->Branch(
        "clust",
        &clustF_,
        "x/F:y/F:charge/F:size_x/I:size_y/I:maxPixelCol/I:maxPixelRow:minPixelCol/I:minPixelRow/I",
        bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("rechit", &rechitF_, "x/F:y/F", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("rechit_corr", &rechitCorrF_, "x/F:y/F", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("trackhitcorr_x", &trackhitCorrXF_, "trackhitcorr_x/F", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("trackhitcorr_y", &trackhitCorrYF_, "trackhitcorr_y/F", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("qScale", &qScaleF_, "qScale/F", bufsize);
    SiPixelLorentzAngleTreeForward_->Branch("rQmQt", &rQmQtF_, "rQmQt/F", bufsize);
  }
}

//
// member functions
//

// ------------ method called for each event  ------------

void SiPixelLorentzAnglePCLWorker::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  // Retrieve tracker topology from geometry
  const TrackerTopology* const tTopo = &iSetup.getData(topoPerEventEsToken_);

  // Retrieve track geometry
  const TrackerGeometry* tracker = &iSetup.getData(geomPerEventEsToken_);

  // Retrieve magnetic field
  const MagneticField* magField = &iSetup.getData(magneticFieldToken_);

  // get the association map between tracks and trajectories
  edm::Handle<TrajTrackAssociationCollection> trajTrackCollectionHandle;
  iEvent.getByToken(t_trajTrack, trajTrackCollectionHandle);

  module_ = -1;
  layer_ = -1;
  ladder_ = -1;
  isflipped_ = -1;
  pt_ = -999;
  eta_ = 999;
  phi_ = 999;
  pixinfo_.npix = 0;

  run_ = iEvent.id().run();
  event_ = iEvent.id().event();
  lumiblock_ = iEvent.luminosityBlock();
  bx_ = iEvent.bunchCrossing();
  orbit_ = iEvent.orbitNumber();

  if (!trajTrackCollectionHandle->empty()) {
    for (TrajTrackAssociationCollection::const_iterator it = trajTrackCollectionHandle->begin();
         it != trajTrackCollectionHandle->end();
         ++it) {
      const reco::Track& track = *it->val;
      const Trajectory& traj = *it->key;

      // get the trajectory measurements
      std::vector<TrajectoryMeasurement> tmColl = traj.measurements();
      pt_ = track.pt();
      eta_ = track.eta();
      phi_ = track.phi();
      chi2_ = traj.chiSquared();
      ndof_ = traj.ndof();

      if (pt_ < ptmin_)
        continue;

      iHists.h_trackEta_->Fill(eta_);
      iHists.h_trackPhi_->Fill(phi_);
      iHists.h_trackPt_->Fill(pt_);
      iHists.h_trackChi2_->Fill(chi2_ / ndof_);
      iHists.h_tracks_->Fill(0);
      bool pixeltrack = false;

      // iterate over trajectory measurements
      for (const auto& itTraj : tmColl) {
        if (!itTraj.updatedState().isValid())
          continue;
        const TransientTrackingRecHit::ConstRecHitPointer& recHit = itTraj.recHit();
        if (!recHit->isValid() || recHit->geographicalId().det() != DetId::Tracker)
          continue;
        unsigned int subDetID = (recHit->geographicalId().subdetId());
        if (subDetID == PixelSubdetector::PixelBarrel || subDetID == PixelSubdetector::PixelEndcap) {
          if (!pixeltrack) {
            iHists.h_tracks_->Fill(1);
          }
          pixeltrack = true;
        }

        if (subDetID == PixelSubdetector::PixelBarrel) {
          DetId detIdObj = recHit->geographicalId();
          const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(tracker->idToDet(detIdObj));
          if (!theGeomDet)
            continue;

          const PixelTopology* topol = &(theGeomDet->specificTopology());

          float ypitch_ = topol->pitch().second;
          float width_ = theGeomDet->surface().bounds().thickness();

          if (!topol)
            continue;

          layer_ = tTopo->pxbLayer(detIdObj);
          ladder_ = tTopo->pxbLadder(detIdObj);
          module_ = tTopo->pxbModule(detIdObj);

          float tmp1 = theGeomDet->surface().toGlobal(Local3DPoint(0., 0., 0.)).perp();
          float tmp2 = theGeomDet->surface().toGlobal(Local3DPoint(0., 0., 1.)).perp();

          isflipped_ = (tmp2 < tmp1) ? 1 : 0;

          const SiPixelRecHit* recHitPix = dynamic_cast<const SiPixelRecHit*>((*recHit).hit());
          if (!recHitPix)
            continue;
          rechit_.x = recHitPix->localPosition().x();
          rechit_.y = recHitPix->localPosition().y();
          SiPixelRecHit::ClusterRef const& cluster = recHitPix->cluster();

          pixinfo_ = fillPix(*cluster, topol);

          // fill entries in clust_

          clust_.x = (cluster)->x();
          clust_.y = (cluster)->y();
          clust_.charge = (cluster->charge()) / 1000.;  // clust_.charge: in the unit of 1000e
          clust_.size_x = cluster->sizeX();
          clust_.size_y = cluster->sizeY();
          clust_.maxPixelCol = cluster->maxPixelCol();
          clust_.maxPixelRow = cluster->maxPixelRow();
          clust_.minPixelCol = cluster->minPixelCol();
          clust_.minPixelRow = cluster->minPixelRow();

          // fill the trackhit info
          TrajectoryStateOnSurface tsos = itTraj.updatedState();
          if (!tsos.isValid()) {
            edm::LogWarning("SiPixelLorentzAnglePCLWorker") << "tsos not valid";
            continue;
          }
          LocalVector trackdirection = tsos.localDirection();
          LocalPoint trackposition = tsos.localPosition();

          if (trackdirection.z() == 0)
            continue;
          // the local position and direction
          trackhit_.alpha = atan2(trackdirection.z(), trackdirection.x());
          trackhit_.beta = atan2(trackdirection.z(), trackdirection.y());
          trackhit_.gamma = atan2(trackdirection.x(), trackdirection.y());
          trackhit_.x = trackposition.x();
          trackhit_.y = trackposition.y();

          // get qScale_ = templ.qscale() and  templ.r_qMeas_qTrue();
          float cotalpha = trackdirection.x() / trackdirection.z();
          float cotbeta = trackdirection.y() / trackdirection.z();
          float cotbeta_min = clustSizeYMin_[layer_ - 1] * ypitch_ / width_;
          if (std::abs(cotbeta) <= cotbeta_min)
            continue;
          double drdz = sqrt(1. + cotalpha * cotalpha + cotbeta * cotbeta);
          double clusterCharge_cut = clustChargeMaxPerLength_ * drdz;

          auto detId = detIdObj.rawId();
          int DetId_index = -1;

          const auto& newModIt = (std::find(iHists.BPixnewDetIds_.begin(), iHists.BPixnewDetIds_.end(), detId));
          bool isNewMod = (newModIt != iHists.BPixnewDetIds_.end());
          if (isNewMod) {
            DetId_index = std::distance(iHists.BPixnewDetIds_.begin(), newModIt);
          }

          if (notInPCL_) {
            // fill the template from the store (from dqmBeginRun)
            SiPixelTemplate theTemplate(*thePixelTemp_);

            float locBx = (cotbeta < 0.) ? -1 : 1.;
            float locBz = (cotalpha < 0.) ? -locBx : locBx;

            int TemplID = templateDBobject_->getTemplateID(detId);
            theTemplate.interpolate(TemplID, cotalpha, cotbeta, locBz, locBx);
            qScale_ = theTemplate.qscale();
            rQmQt_ = theTemplate.r_qMeas_qTrue();
          }

          // Surface deformation
          const auto& lp_pair = surface_deformation(topol, tsos, recHitPix);

          LocalPoint lp_track = lp_pair.first;
          LocalPoint lp_rechit = lp_pair.second;

          rechitCorr_.x = lp_rechit.x();
          rechitCorr_.y = lp_rechit.y();
          trackhitCorrX_ = lp_track.x();
          trackhitCorrY_ = lp_track.y();

          if (notInPCL_) {
            SiPixelLorentzAngleTreeBarrel_->Fill();
          }

          if (analysisType_ != eGrazingAngle)
            continue;
          // is one pixel in cluster a large pixel ? (hit will be excluded)
          bool large_pix = false;
          for (int j = 0; j < pixinfo_.npix; j++) {
            int colpos = static_cast<int>(pixinfo_.col[j]);
            if (pixinfo_.row[j] == 0 || pixinfo_.row[j] == 79 || pixinfo_.row[j] == 80 || pixinfo_.row[j] == 159 ||
                colpos % 52 == 0 || colpos % 52 == 51) {
              large_pix = true;
            }
          }

          double residualsq = (trackhitCorrX_ - rechitCorr_.x) * (trackhitCorrX_ - rechitCorr_.x) +
                              (trackhitCorrY_ - rechitCorr_.y) * (trackhitCorrY_ - rechitCorr_.y);

          double xlim1 = trackhitCorrX_ - width_ * cotalpha / 2.;
          double hypitch_ = ypitch_ / 2.;
          double ylim1 = trackhitCorrY_ - width_ * cotbeta / 2.;
          double ylim2 = trackhitCorrY_ + width_ * cotbeta / 2.;

          int clustSizeY_cut = clustSizeYMin_[layer_ - 1];

          if (!large_pix && (chi2_ / ndof_) < normChi2Max_ && cluster->sizeY() >= clustSizeY_cut &&
              residualsq < residualMax_ * residualMax_ && cluster->charge() < clusterCharge_cut &&
              cluster->sizeX() < clustSizeXMax_) {
            // iterate over pixels in hit
            for (int j = 0; j < pixinfo_.npix; j++) {
              // use trackhits and include bowing correction
              float ypixlow = pixinfo_.y[j] - hypitch_;
              float ypixhigh = pixinfo_.y[j] + hypitch_;
              if (cotbeta > 0.) {
                if (ylim1 > ypixlow)
                  ypixlow = ylim1;
                if (ylim2 < ypixhigh)
                  ypixhigh = ylim2;
              } else {
                if (ylim2 > ypixlow)
                  ypixlow = ylim2;
                if (ylim1 < ypixhigh)
                  ypixhigh = ylim1;
              }
              float ypixavg = 0.5f * (ypixlow + ypixhigh);

              float dx = (pixinfo_.x[j] - xlim1) * siPixelLACalibration::cmToum;  // dx: in the unit of micrometer
              float dy = (ypixavg - ylim1) * siPixelLACalibration::cmToum;        // dy: in the unit of micrometer
              float depth = dy * tan(trackhit_.beta);
              float drift = dx - dy * tan(trackhit_.gamma);

              if (isNewMod == false) {
                int i_index = module_ + (layer_ - 1) * iHists.nModules_[layer_ - 1];
                iHists.h_drift_depth_adc_[i_index]->Fill(drift, depth, pixinfo_.adc[j]);
                iHists.h_drift_depth_adc2_[i_index]->Fill(drift, depth, pixinfo_.adc[j] * pixinfo_.adc[j]);
                iHists.h_drift_depth_noadc_[i_index]->Fill(drift, depth, 1.);
                iHists.h_bySectOccupancy_->Fill(i_index - 1);  // histogram starts at 0

                if (tracker->getDetectorType(subDetID) == TrackerGeometry::ModuleType::Ph1PXB) {
                  if ((module_ == 3 || module_ == 5) && (layer_ == 3 || layer_ == 4)) {
                    int i_index_merge = i_index + 1;
                    iHists.h_drift_depth_adc_[i_index_merge]->Fill(drift, depth, pixinfo_.adc[j]);
                    iHists.h_drift_depth_adc2_[i_index_merge]->Fill(drift, depth, pixinfo_.adc[j] * pixinfo_.adc[j]);
                    iHists.h_drift_depth_noadc_[i_index_merge]->Fill(drift, depth, 1.);
                    iHists.h_bySectOccupancy_->Fill(i_index_merge - 1);
                  }
                  if ((module_ == 4 || module_ == 6) && (layer_ == 3 || layer_ == 4)) {
                    int i_index_merge = i_index - 1;
                    iHists.h_drift_depth_adc_[i_index_merge]->Fill(drift, depth, pixinfo_.adc[j]);
                    iHists.h_drift_depth_adc2_[i_index_merge]->Fill(drift, depth, pixinfo_.adc[j] * pixinfo_.adc[j]);
                    iHists.h_drift_depth_noadc_[i_index_merge]->Fill(drift, depth, 1.);
                    iHists.h_bySectOccupancy_->Fill(i_index_merge - 1);
                  }
                }

              } else {
                int new_index = iHists.nModules_[iHists.nlay - 1] +
                                (iHists.nlay - 1) * iHists.nModules_[iHists.nlay - 1] + 1 + DetId_index;

                iHists.h_drift_depth_adc_[new_index]->Fill(drift, depth, pixinfo_.adc[j]);
                iHists.h_drift_depth_adc2_[new_index]->Fill(drift, depth, pixinfo_.adc[j] * pixinfo_.adc[j]);
                iHists.h_drift_depth_noadc_[new_index]->Fill(drift, depth, 1.);
                iHists.h_bySectOccupancy_->Fill(new_index - 1);  // histogram starts at 0
              }
            }
          }
        } else if (subDetID == PixelSubdetector::PixelEndcap) {
          DetId detIdObj = recHit->geographicalId();
          const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(tracker->idToDet(detIdObj));
          if (!theGeomDet)
            continue;

          const PixelTopology* topol = &(theGeomDet->specificTopology());

          if (!topol)
            continue;

          sideF_ = tTopo->pxfSide(detIdObj);
          diskF_ = tTopo->pxfDisk(detIdObj);
          bladeF_ = tTopo->pxfBlade(detIdObj);
          panelF_ = tTopo->pxfPanel(detIdObj);
          moduleF_ = tTopo->pxfModule(detIdObj);

          const SiPixelRecHit* recHitPix = dynamic_cast<const SiPixelRecHit*>((*recHit).hit());
          if (!recHitPix)
            continue;
          rechitF_.x = recHitPix->localPosition().x();
          rechitF_.y = recHitPix->localPosition().y();
          SiPixelRecHit::ClusterRef const& cluster = recHitPix->cluster();

          pixinfoF_ = fillPix(*cluster, topol);

          // fill entries in clust_

          clustF_.x = (cluster)->x();
          clustF_.y = (cluster)->y();
          clustF_.charge = (cluster->charge()) / 1000.;  // clustF_.charge: in the unit of 1000e
          clustF_.size_x = cluster->sizeX();
          clustF_.size_y = cluster->sizeY();
          clustF_.maxPixelCol = cluster->maxPixelCol();
          clustF_.maxPixelRow = cluster->maxPixelRow();
          clustF_.minPixelCol = cluster->minPixelCol();
          clustF_.minPixelRow = cluster->minPixelRow();

          // fill the trackhit info
          TrajectoryStateOnSurface tsos = itTraj.updatedState();
          if (!tsos.isValid()) {
            edm::LogWarning("SiPixelLorentzAnglePCLWorker") << "tsos not valid";
            continue;
          }
          LocalVector trackdirection = tsos.localDirection();
          LocalPoint trackposition = tsos.localPosition();

          if (trackdirection.z() == 0)
            continue;
          // the local position and direction
          trackhitF_.alpha = atan2(trackdirection.z(), trackdirection.x());
          trackhitF_.beta = atan2(trackdirection.z(), trackdirection.y());
          trackhitF_.gamma = atan2(trackdirection.x(), trackdirection.y());
          trackhitF_.x = trackposition.x();
          trackhitF_.y = trackposition.y();

          float cotalpha = trackdirection.x() / trackdirection.z();
          float cotbeta = trackdirection.y() / trackdirection.z();

          auto detId = detIdObj.rawId();

          if (notInPCL_) {
            // fill the template from the store (from dqmBeginRun)
            SiPixelTemplate theTemplate(*thePixelTemp_);

            float locBx = (cotbeta < 0.) ? -1 : 1.;
            float locBz = (cotalpha < 0.) ? -locBx : locBx;

            int TemplID = templateDBobject_->getTemplateID(detId);
            theTemplate.interpolate(TemplID, cotalpha, cotbeta, locBz, locBx);
            qScaleF_ = theTemplate.qscale();
            rQmQtF_ = theTemplate.r_qMeas_qTrue();
          }

          // Surface deformation
          const auto& lp_pair = surface_deformation(topol, tsos, recHitPix);

          LocalPoint lp_track = lp_pair.first;
          LocalPoint lp_rechit = lp_pair.second;

          rechitCorrF_.x = lp_rechit.x();
          rechitCorrF_.y = lp_rechit.y();
          trackhitCorrXF_ = lp_track.x();
          trackhitCorrYF_ = lp_track.y();
          if (notInPCL_) {
            SiPixelLorentzAngleTreeForward_->Fill();
          }

          if (analysisType_ != eMinimumClusterSize)
            continue;

          int theMagField = magField->nominalValue();
          if (theMagField < 37 || theMagField > 39)
            continue;

          double chi2_ndof = chi2_ / ndof_;
          if (chi2_ndof >= normChi2Max_)
            continue;

          //--- large pixel cut
          bool large_pix = false;
          for (int j = 0; j < pixinfoF_.npix; j++) {
            int colpos = static_cast<int>(pixinfoF_.col[j]);
            if (pixinfoF_.row[j] == 0 || pixinfoF_.row[j] == 79 || pixinfoF_.row[j] == 80 || pixinfoF_.row[j] == 159 ||
                colpos % 52 == 0 || colpos % 52 == 51) {
              large_pix = true;
            }
          }

          if (large_pix)
            continue;

          //--- residual cut
          double residual = sqrt(pow(trackhitCorrXF_ - rechitCorrF_.x, 2) + pow(trackhitCorrYF_ - rechitCorrF_.y, 2));

          if (residual > residualMax_)
            continue;

          int ringIdx = bladeF_ <= 22 ? 0 : 1;
          int panelIdx = panelF_ - 1;
          int sideIdx = sideF_ - 1;
          int idx = iHists.nSides_ * iHists.nPanels_ * ringIdx + iHists.nSides_ * panelIdx + sideIdx;
          int idxBeta = iHists.betaStartIdx_ + idx;

          double cotanAlpha = std::tan(M_PI / 2. - trackhitF_.alpha);
          double cotanBeta = std::tan(M_PI / 2. - trackhitF_.beta);

          LocalVector Bfield = theGeomDet->surface().toLocal(magField->inTesla(theGeomDet->surface().position()));
          iHists.h_fpixMagField_[0][idx]->Fill(Bfield.x());
          iHists.h_fpixMagField_[1][idx]->Fill(Bfield.y());
          iHists.h_fpixMagField_[2][idx]->Fill(Bfield.z());

          if (clustF_.size_y >= 2) {
            iHists.h_fpixAngleSize_[idx]->Fill(cotanAlpha, clustF_.size_x);
          }

          if (clust_.size_x >= 0) {
            iHists.h_fpixAngleSize_[idxBeta]->Fill(cotanBeta, clustF_.size_y);
          }
        }
      }  //end iteration over trajectory measurements
    }    //end iteration over trajectories
  }
}

void SiPixelLorentzAnglePCLWorker::dqmBeginRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  // geometry
  const TrackerGeometry* geom = &iSetup.getData(geomEsToken_);
  const TrackerTopology* tTopo = &iSetup.getData(topoEsToken_);

  if (notInPCL_) {
    // Initialize 1D templates
    if (watchSiPixelTemplateRcd_.check(iSetup)) {
      templateDBobject_ = &iSetup.getData(siPixelTemplateEsToken_);
      thePixelTemp_ = &iSetup.getData(siPixelTemplateStoreEsToken_);
    }
  }

  PixelTopologyMap map = PixelTopologyMap(geom, tTopo);
  iHists.nlay = geom->numberOfLayers(PixelSubdetector::PixelBarrel);
  iHists.nModules_.resize(iHists.nlay);
  for (int i = 0; i < iHists.nlay; i++) {
    iHists.nModules_[i] = map.getPXBModules(i + 1);
  }

  // list of modules already filled, then return (we already entered here)
  if (!iHists.BPixnewDetIds_.empty() || !iHists.FPixnewDetIds_.empty())
    return;

  if (!newmodulelist_.empty()) {
    for (auto const& modulename : newmodulelist_) {
      if (modulename.find("BPix_") != std::string::npos) {
        PixelBarrelName bn(modulename, true);
        const auto& detId = bn.getDetId(tTopo);
        iHists.BPixnewmodulename_.push_back(modulename);
        iHists.BPixnewDetIds_.push_back(detId.rawId());
        iHists.BPixnewModule_.push_back(bn.moduleName());
        iHists.BPixnewLayer_.push_back(bn.layerName());
      } else if (modulename.find("FPix_") != std::string::npos) {
        PixelEndcapName en(modulename, true);
        const auto& detId = en.getDetId(tTopo);
        iHists.FPixnewmodulename_.push_back(modulename);
        iHists.FPixnewDetIds_.push_back(detId.rawId());
        iHists.FPixnewDisk_.push_back(en.diskName());
        iHists.FPixnewBlade_.push_back(en.bladeName());
      }
    }
  }
}

void SiPixelLorentzAnglePCLWorker::bookHistograms(DQMStore::IBooker& iBooker,
                                                  edm::Run const& run,
                                                  edm::EventSetup const& iSetup) {
  std::string name;
  std::string title;
  if (analysisType_ == eGrazingAngle) {
    // book the by partition monitoring
    const auto maxSect = iHists.nlay * iHists.nModules_[iHists.nlay - 1] + (int)iHists.BPixnewDetIds_.size();

    iBooker.setCurrentFolder(fmt::sprintf("%s/SectorMonitoring", folder_.data()));
    iHists.h_bySectOccupancy_ = iBooker.book1D(
        "h_bySectorOccupancy", "hit occupancy by sector;pixel sector;hits on track", maxSect, -0.5, maxSect - 0.5);

    iBooker.setCurrentFolder(folder_);
    static constexpr double min_depth_ = -100.;
    static constexpr double max_depth_ = 400.;
    static constexpr double min_drift_ = -500.;
    static constexpr double max_drift_ = 500.;

    // book the mean values projections and set the bin names of the by sector monitoring
    for (int i_layer = 1; i_layer <= iHists.nlay; i_layer++) {
      for (int i_module = 1; i_module <= iHists.nModules_[i_layer - 1]; i_module++) {
        unsigned int i_index = i_module + (i_layer - 1) * iHists.nModules_[i_layer - 1];
        std::string binName = fmt::sprintf("BPix Lay%i Mod%i", i_layer, i_module);
        LogDebug("SiPixelLorentzAnglePCLWorker") << " i_index: " << i_index << " bin name: " << binName
                                                 << " (i_layer: " << i_layer << " i_module:" << i_module << ")";

        iHists.h_bySectOccupancy_->setBinLabel(i_index, binName);

        name = fmt::sprintf("h_mean_layer%i_module%i", i_layer, i_module);
        title = fmt::sprintf(
            "average drift vs depth layer%i module%i; production depth [#mum]; #LTdrift#GT [#mum]", i_layer, i_module);
        iHists.h_mean_[i_index] = iBooker.book1D(name, title, hist_depth_, min_depth_, max_depth_);
      }
    }
    for (int i = 0; i < (int)iHists.BPixnewDetIds_.size(); i++) {
      name = fmt::sprintf("h_BPixnew_mean_%s", iHists.BPixnewmodulename_[i].c_str());
      title = fmt::sprintf("average drift vs depth %s; production depth [#mum]; #LTdrift#GT [#mum]",
                           iHists.BPixnewmodulename_[i].c_str());
      int new_index = iHists.nModules_[iHists.nlay - 1] + (iHists.nlay - 1) * iHists.nModules_[iHists.nlay - 1] + 1 + i;
      iHists.h_mean_[new_index] = iBooker.book1D(name, title, hist_depth_, min_depth_, max_depth_);

      LogDebug("SiPixelLorentzAnglePCLWorker")
          << "i_index" << new_index << " bin name: " << iHists.BPixnewmodulename_[i];

      iHists.h_bySectOccupancy_->setBinLabel(new_index, iHists.BPixnewmodulename_[i]);
    }

    //book the 2D histograms
    for (int i_layer = 1; i_layer <= iHists.nlay; i_layer++) {
      iBooker.setCurrentFolder(fmt::sprintf("%s/BPix/BPixLayer%i", folder_.data(), i_layer));
      for (int i_module = 1; i_module <= iHists.nModules_[i_layer - 1]; i_module++) {
        unsigned int i_index = i_module + (i_layer - 1) * iHists.nModules_[i_layer - 1];

        name = fmt::sprintf("h_drift_depth_adc_layer%i_module%i", i_layer, i_module);
        title = fmt::sprintf(
            "depth vs drift (ADC) layer%i module%i; drift [#mum]; production depth [#mum]", i_layer, i_module);
        iHists.h_drift_depth_adc_[i_index] =
            iBooker.book2D(name, title, hist_drift_, min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);

        name = fmt::sprintf("h_drift_depth_adc2_layer%i_module%i", i_layer, i_module);
        title = fmt::sprintf(
            "depth vs drift (ADC^{2}) layer%i module%i; drift [#mum]; production depth [#mum]", i_layer, i_module);
        iHists.h_drift_depth_adc2_[i_index] =
            iBooker.book2D(name, title, hist_drift_, min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);

        name = fmt::sprintf("h_drift_depth_noadc_layer%i_module%i", i_layer, i_module);
        title = fmt::sprintf(
            "depth vs drift (no ADC) layer%i module%i; drift [#mum]; production depth [#mum]", i_layer, i_module);
        iHists.h_drift_depth_noadc_[i_index] =
            iBooker.book2D(name, title, hist_drift_, min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);

        name = fmt::sprintf("h_drift_depth_layer%i_module%i", i_layer, i_module);
        title =
            fmt::sprintf("depth vs drift layer%i module%i; drift [#mum]; production depth [#mum]", i_layer, i_module);
        iHists.h_drift_depth_[i_index] =
            iBooker.book2D(name, title, hist_drift_, min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
      }
    }

    // book the "new" modules
    iBooker.setCurrentFolder(fmt::sprintf("%s/BPix/NewModules", folder_.data()));
    for (int i = 0; i < (int)iHists.BPixnewDetIds_.size(); i++) {
      int new_index = iHists.nModules_[iHists.nlay - 1] + (iHists.nlay - 1) * iHists.nModules_[iHists.nlay - 1] + 1 + i;

      name = fmt::sprintf("h_BPixnew_drift_depth_adc_%s", iHists.BPixnewmodulename_[i].c_str());
      title = fmt::sprintf("depth vs drift (ADC) %s; drift [#mum]; production depth [#mum]",
                           iHists.BPixnewmodulename_[i].c_str());
      iHists.h_drift_depth_adc_[new_index] =
          iBooker.book2D(name, title, hist_drift_, min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);

      name = fmt::sprintf("h_BPixnew_drift_depth_adc2_%s", iHists.BPixnewmodulename_[i].c_str());
      title = fmt::sprintf("depth vs drift (ADC^{2}) %s; drift [#mum]; production depth [#mum]",
                           iHists.BPixnewmodulename_[i].c_str());
      iHists.h_drift_depth_adc2_[new_index] =
          iBooker.book2D(name, title, hist_drift_, min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);

      name = fmt::sprintf("h_BPixnew_drift_depth_noadc_%s", iHists.BPixnewmodulename_[i].c_str());
      title = fmt::sprintf("depth vs drift (no ADC)%s; drift [#mum]; production depth [#mum]",
                           iHists.BPixnewmodulename_[i].c_str());
      iHists.h_drift_depth_noadc_[new_index] =
          iBooker.book2D(name, title, hist_drift_, min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);

      name = fmt::sprintf("h_BPixnew_drift_depth_%s", iHists.BPixnewmodulename_[i].c_str());
      title = fmt::sprintf("depth vs drift %s; drift [#mum]; production depth [#mum]",
                           iHists.BPixnewmodulename_[i].c_str());
      iHists.h_drift_depth_[new_index] =
          iBooker.book2D(name, title, hist_drift_, min_drift_, max_drift_, hist_depth_, min_depth_, max_depth_);
    }
  }  // end if GrazinAngleAnalysis
  else {
    iBooker.setCurrentFolder(folder_);
    std::string baseName;
    std::string baseTitle;

    for (int r = 0; r < iHists.nRings_; ++r) {
      for (int p = 0; p < iHists.nPanels_; ++p) {
        for (int s = 0; s < iHists.nSides_; ++s) {
          baseName = fmt::sprintf("R%d_P%d_z%d", r + 1, p + 1, s + 1);
          if (s == 0)
            baseTitle = fmt::sprintf("Ring%d_Panel%d_z-", r + 1, p + 1);
          else
            baseTitle = fmt::sprintf("Ring%d_Panel%d_z+", r + 1, p + 1);

          int idx = iHists.nSides_ * iHists.nPanels_ * r + iHists.nSides_ * p + s;
          int idxBeta = iHists.betaStartIdx_ + idx;

          name = fmt::sprintf("%s_alphaMean", baseName);
          title = fmt::sprintf("%s_alphaMean;cot(#alpha); Average cluster size x (pixel)", baseTitle);
          iHists.h_fpixMean_[idx] = iBooker.book1D(name, title, 60, -3., 3.);
          name = fmt::sprintf("%s_betaMean", baseName);
          title = fmt::sprintf("%s_betaMean;cot(#beta); Average cluster size y (pixel)", baseTitle);
          iHists.h_fpixMean_[idxBeta] = iBooker.book1D(name, title, 60, -3., 3.);

        }  // loop over sides
      }    // loop over panels
    }      // loop over rings
    iBooker.setCurrentFolder(fmt::sprintf("%s/FPix", folder_.data()));
    for (int r = 0; r < iHists.nRings_; ++r) {
      for (int p = 0; p < iHists.nPanels_; ++p) {
        for (int s = 0; s < iHists.nSides_; ++s) {
          baseName = fmt::sprintf("R%d_P%d_z%d", r + 1, p + 1, s + 1);
          if (s == 0)
            baseTitle = fmt::sprintf("Ring%d_Panel%d_z-", r + 1, p + 1);
          else
            baseTitle = fmt::sprintf("Ring%d_Panel%d_z+", r + 1, p + 1);

          int idx = iHists.nSides_ * iHists.nPanels_ * r + iHists.nSides_ * p + s;
          int idxBeta = iHists.betaStartIdx_ + idx;

          name = fmt::sprintf("%s_alpha", baseName);
          title = fmt::sprintf("%s_alpha;cot(#alpha); Cluster size x (pixel)", baseTitle);
          iHists.h_fpixAngleSize_[idx] = iBooker.book2D(name, title, 60, -3., 3., 10, 0.5, 10.5);
          name = fmt::sprintf("%s_beta", baseName);
          title = fmt::sprintf("%s_beta;cot(#beta); Cluster size y (pixel) ", baseTitle);
          iHists.h_fpixAngleSize_[idxBeta] = iBooker.book2D(name, title, 60, -3., 3., 10, 0.5, 10.5);
          for (int m = 0; m < 3; ++m) {
            name = fmt::sprintf("%s_B%d", baseName, m);
            char bComp = m == 0 ? 'x' : (m == 1 ? 'y' : 'z');
            title = fmt::sprintf("%s_magField%d;B_{%c} [T];Entries", baseTitle, m, bComp);
            iHists.h_fpixMagField_[m][idx] = iBooker.book1D(name, title, 10000, -5., 5.);
          }  // mag. field comps
        }    // loop over sides
      }      // loop over panels
    }        // loop over rings
  }          // if MinimalClusterSize

  // book the track monitoring plots
  iBooker.setCurrentFolder(fmt::sprintf("%s/TrackMonitoring", folder_.data()));
  iHists.h_tracks_ = iBooker.book1D("h_tracks", ";tracker volume;tracks", 2, -0.5, 1.5);
  iHists.h_tracks_->setBinLabel(1, "all tracks", 1);
  iHists.h_tracks_->setBinLabel(2, "has pixel hits", 1);
  iHists.h_trackEta_ = iBooker.book1D("h_trackEta", ";track #eta; #tracks", 30, -3., 3.);
  iHists.h_trackPhi_ = iBooker.book1D("h_trackPhi", ";track #phi; #tracks", 48, -M_PI, M_PI);
  iHists.h_trackPt_ = iBooker.book1D("h_trackPt", ";track p_{T} [GeV]; #tracks", 100, 0., 100.);
  iHists.h_trackChi2_ = iBooker.book1D("h_trackChi2ndof", ";track #chi^{2}/ndof; #tracks", 100, 0., 10.);
}

void SiPixelLorentzAnglePCLWorker::dqmEndRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  if (notInPCL_) {
    hFile_->cd();
    hFile_->Write();
    hFile_->Close();
  }
}

// method used to fill per pixel info
const Pixinfo SiPixelLorentzAnglePCLWorker::fillPix(const SiPixelCluster& LocPix, const PixelTopology* topol) const {
  Pixinfo pixinfo;
  const std::vector<SiPixelCluster::Pixel>& pixvector = LocPix.pixels();
  pixinfo.npix = 0;
  for (std::vector<SiPixelCluster::Pixel>::const_iterator itPix = pixvector.begin(); itPix != pixvector.end();
       itPix++) {
    pixinfo.row[pixinfo.npix] = itPix->x;
    pixinfo.col[pixinfo.npix] = itPix->y;
    pixinfo.adc[pixinfo.npix] = itPix->adc;
    LocalPoint lp = topol->localPosition(MeasurementPoint(itPix->x + 0.5, itPix->y + 0.5));
    pixinfo.x[pixinfo.npix] = lp.x();
    pixinfo.y[pixinfo.npix] = lp.y();
    pixinfo.npix++;
  }
  return pixinfo;
}

// method used to correct for the surface deformation
const std::pair<LocalPoint, LocalPoint> SiPixelLorentzAnglePCLWorker::surface_deformation(
    const PixelTopology* topol, TrajectoryStateOnSurface& tsos, const SiPixelRecHit* recHitPix) const {
  LocalPoint trackposition = tsos.localPosition();
  const LocalTrajectoryParameters& ltp = tsos.localParameters();
  const Topology::LocalTrackAngles localTrackAngles(ltp.dxdz(), ltp.dydz());

  std::pair<float, float> pixels_track = topol->pixel(trackposition, localTrackAngles);
  std::pair<float, float> pixels_rechit = topol->pixel(recHitPix->localPosition(), localTrackAngles);

  LocalPoint lp_track = topol->localPosition(MeasurementPoint(pixels_track.first, pixels_track.second));

  LocalPoint lp_rechit = topol->localPosition(MeasurementPoint(pixels_rechit.first, pixels_rechit.second));

  std::pair<LocalPoint, LocalPoint> lps = std::make_pair(lp_track, lp_rechit);
  return lps;
}

LorentzAngleAnalysisTypeEnum SiPixelLorentzAnglePCLWorker::convertStringToLorentzAngleAnalysisTypeEnum(
    std::string type) {
  return (type == "GrazingAngle") ? eGrazingAngle : eMinimumClusterSize;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiPixelLorentzAnglePCLWorker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Worker module of the SiPixel Lorentz Angle PCL monitoring workflow");
  desc.add<std::string>("analysisType", "GrazingAngle")
      ->setComment("analysis type - GrazingAngle (default) or MinimumClusterSize");
  desc.add<std::string>("folder", "AlCaReco/SiPixelLorentzAngle")->setComment("directory of PCL Worker output");
  desc.add<bool>("notInPCL", false)->setComment("create TTree (true) or not (false)");
  desc.add<std::string>("fileName", "testrun.root")->setComment("name of the TTree file if notInPCL = true");
  desc.add<std::vector<std::string>>("newmodulelist", {})->setComment("the list of DetIds for new sensors");
  desc.add<edm::InputTag>("src", edm::InputTag("TrackRefitter"))->setComment("input track collections");
  desc.add<double>("ptMin", 3.)->setComment("minimum pt on tracks");
  desc.add<double>("normChi2Max", 2.)->setComment("maximum reduced chi squared");
  desc.add<std::vector<int>>("clustSizeYMin", {4, 3, 3, 2})
      ->setComment("minimum cluster size on Y axis for all Barrel Layers");
  desc.add<int>("clustSizeXMax", 5)->setComment("maximum cluster size on X axis");
  desc.add<double>("residualMax", 0.005)->setComment("maximum residual");
  desc.add<double>("clustChargeMaxPerLength", 50000)
      ->setComment("maximum cluster charge per unit length of pixel depth (z)");
  desc.add<int>("binsDepth", 50)->setComment("# bins for electron production depth axis");
  desc.add<int>("binsDrift", 100)->setComment("# bins for electron drift axis");
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(SiPixelLorentzAnglePCLWorker);
