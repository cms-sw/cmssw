// -*- C++ -*-
//
// Package:    CalibTracker/SiStripLorentzAnglePCLMonitor
// Class:      SiStripLorentzAnglePCLMonitor
//
/**\class SiStripLorentzAnglePCLMonitor SiStripLorentzAnglePCLMonitor.cc CalibTracker/SiStripLorentzAnglePCLMonitor/plugins/SiStripLorentzAnglePCLMonitor.cc

 Description: class to book and fill histograms necessary for the online monitoring of the SiStripLorentzAngle

 Implementation:
     Largely taken from https://github.com/robervalwalsh/tracker-la/blob/master/SiStripLAMonitor.cc
*/
//
// Original Author:  musich
//         Created:  Sun, 07 May 2023 16:57:10 GMT
//
//

// system includes
#include <string>
#include <fmt/format.h>
#include <fmt/printf.h>

// user include files
#include "CalibFormats/SiStripObjects/interface/SiStripHashedDetId.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CalibTracker/SiStripCommon/interface/ShallowTools.h"
#include "CalibTracker/SiStripLorentzAngle/interface/SiStripLorentzAngleCalibrationHelpers.h"
#include "CalibTracker/SiStripLorentzAngle/interface/SiStripLorentzAngleCalibrationStruct.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

// ROOT includes
#include "TVector3.h"

//
// class declaration
//

class SiStripLorentzAnglePCLMonitor : public DQMEDAnalyzer {
public:
  explicit SiStripLorentzAnglePCLMonitor(const edm::ParameterSet&);
  ~SiStripLorentzAnglePCLMonitor() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;

  std::string moduleLocationType(const uint32_t& mod, const TrackerTopology* tTopo);

  // ------------ member data ------------
  SiStripClusterInfo m_clusterInfo;
  SiStripLorentzAngleCalibrationHistograms iHists_;
  SiStripHashedDetId m_hash;

  // for magnetic field conversion
  static constexpr float teslaToInverseGeV_ = 2.99792458e-3f;

  bool mismatchedBField_;
  bool mismatchedLatency_;
  const std::string folder_;
  const bool saveHistosMods_;
  const edm::EDGetTokenT<edm::View<reco::Track>> m_tracks_token;
  const edm::EDGetTokenT<TrajTrackAssociationCollection> m_association_token;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> m_tkGeomToken;

  const edm::ESGetToken<SiStripLatency, SiStripLatencyRcd> m_latencyTokenBR;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> m_topoEsTokenBR;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> m_tkGeomTokenBR;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> m_magFieldTokenBR;
  const edm::ESGetToken<SiStripLorentzAngle, SiStripLorentzAngleDepRcd> m_lorentzAngleTokenBR;

  struct OnTrackCluster {
    uint32_t det;
    const SiStripCluster* cluster;
    const Trajectory* traj;
    const reco::Track* track;
    const TrajectoryMeasurement& measurement;
    OnTrackCluster(uint32_t detId,
                   const SiStripCluster* stripCluster,
                   const Trajectory* trajectory,
                   const reco::Track* track_,
                   const TrajectoryMeasurement& measurement_)
        : det{detId}, cluster{stripCluster}, traj{trajectory}, track{track_}, measurement{measurement_} {}
  };
};

SiStripLorentzAnglePCLMonitor::SiStripLorentzAnglePCLMonitor(const edm::ParameterSet& iConfig)
    : m_clusterInfo(consumesCollector()),
      mismatchedBField_{false},
      mismatchedLatency_{false},
      folder_(iConfig.getParameter<std::string>("folder")),
      saveHistosMods_(iConfig.getParameter<bool>("saveHistoMods")),
      m_tracks_token(consumes<edm::View<reco::Track>>(iConfig.getParameter<edm::InputTag>("Tracks"))),
      m_association_token(consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("Tracks"))),
      m_tkGeomToken{esConsumes<>()},
      m_latencyTokenBR{esConsumes<edm::Transition::BeginRun>()},
      m_topoEsTokenBR{esConsumes<edm::Transition::BeginRun>()},
      m_tkGeomTokenBR{esConsumes<edm::Transition::BeginRun>()},
      m_magFieldTokenBR{esConsumes<edm::Transition::BeginRun>()},
      m_lorentzAngleTokenBR{esConsumes<edm::Transition::BeginRun>()} {}
//
// member functions
//

void SiStripLorentzAnglePCLMonitor::dqmBeginRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  const auto& tkGeom = iSetup.getData(m_tkGeomTokenBR);
  const auto& magField = iSetup.getData(m_magFieldTokenBR);
  const auto& lorentzAngle = iSetup.getData(m_lorentzAngleTokenBR);
  const TrackerTopology* tTopo = &iSetup.getData(m_topoEsTokenBR);

  // fast cachecd access
  const auto& theMagField = 1.f / (magField.inverseBzAtOriginInGeV() * teslaToInverseGeV_);

  if (iHists_.bfield_.empty()) {
    iHists_.bfield_ = siStripLACalibration::fieldAsString(theMagField);
  } else {
    if (iHists_.bfield_ != siStripLACalibration::fieldAsString(theMagField)) {
      mismatchedBField_ = true;
    }
  }

  const SiStripLatency* apvlat = &iSetup.getData(m_latencyTokenBR);
  if (iHists_.apvmode_.empty()) {
    iHists_.apvmode_ = siStripLACalibration::apvModeAsString(apvlat);
  } else {
    if (iHists_.apvmode_ != siStripLACalibration::apvModeAsString(apvlat)) {
      mismatchedLatency_ = true;
    }
  }

  std::vector<uint32_t> c_rawid;
  std::vector<float> c_globalZofunitlocalY, c_localB, c_BdotY, c_driftx, c_drifty, c_driftz, c_lorentzAngle;

  auto dets = tkGeom.detsTIB();
  //dets.insert(dets.end(), tkGeom.detsTID().begin(), tkGeom.detsTID().end()); // no LA in endcaps
  dets.insert(dets.end(), tkGeom.detsTOB().begin(), tkGeom.detsTOB().end());
  //dets.insert(dets.end(), tkGeom.detsTEC().begin(), tkGeom.detsTEC().end()); // no LA in endcaps

  for (auto det : dets) {
    auto detid = det->geographicalId().rawId();
    const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>(tkGeom.idToDet(det->geographicalId()));
    if (stripDet) {
      c_rawid.push_back(detid);
      c_globalZofunitlocalY.push_back(stripDet->toGlobal(LocalVector(0, 1, 0)).z());
      iHists_.orientation_[detid] = (stripDet->toGlobal(LocalVector(0, 1, 0)).z() < 0 ? -1 : 1);
      const auto locB = magField.inTesla(stripDet->surface().position());
      c_localB.push_back(locB.mag());
      c_BdotY.push_back(stripDet->surface().toLocal(locB).y());
      const auto drift = shallow::drift(stripDet, magField, lorentzAngle);
      c_driftx.push_back(drift.x());
      c_drifty.push_back(drift.y());
      c_driftz.push_back(drift.z());
      c_lorentzAngle.push_back(lorentzAngle.getLorentzAngle(detid));
      iHists_.la_db_[detid] = lorentzAngle.getLorentzAngle(detid);
      iHists_.moduleLocationType_[detid] = this->moduleLocationType(detid, tTopo);
    }
  }

  // Sorted DetId list gives max performance, anything else is worse
  std::sort(c_rawid.begin(), c_rawid.end());

  // initialized the hash map
  m_hash = SiStripHashedDetId(c_rawid);

  //reserve the size of the vector
  iHists_.h2_ct_w_m_.reserve(c_rawid.size());
  iHists_.h2_ct_var2_m_.reserve(c_rawid.size());
  iHists_.h2_ct_var3_m_.reserve(c_rawid.size());

  iHists_.h2_t_w_m_.reserve(c_rawid.size());
  iHists_.h2_t_var2_m_.reserve(c_rawid.size());
  iHists_.h2_t_var3_m_.reserve(c_rawid.size());
}

std::string SiStripLorentzAnglePCLMonitor::moduleLocationType(const uint32_t& mod, const TrackerTopology* tTopo) {
  const SiStripDetId detid(mod);
  std::string subdet = "";
  unsigned int layer = 0;
  if (detid.subDetector() == SiStripDetId::TIB) {
    subdet = "TIB";
    layer = tTopo->layer(mod);
  }

  if (detid.subDetector() == SiStripDetId::TOB) {
    subdet = "TOB";
    layer = tTopo->layer(mod);
  }

  std::string type = (detid.stereo() ? "s" : "a");
  std::string d_l_t = Form("%s_L%d%s", subdet.c_str(), layer, type.c_str());

  if (layer == 0)
    return subdet;
  return d_l_t;
}

// ------------ method called for each event  ------------
void SiStripLorentzAnglePCLMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // return immediately if the field is not consistent!
  if (mismatchedBField_)
    return;

  if (mismatchedLatency_)
    return;

  edm::Handle<edm::View<reco::Track>> tracks;
  iEvent.getByToken(m_tracks_token, tracks);
  edm::Handle<TrajTrackAssociationCollection> trajTrackAssociations;
  iEvent.getByToken(m_association_token, trajTrackAssociations);

  LogDebug(moduleDescription().moduleName()) << "I AM IN EVENT" << iEvent.id() << std::endl;

  std::vector<OnTrackCluster> clusters{};

  // first collect all the clusters
  for (const auto& assoc : *trajTrackAssociations) {
    const auto traj = assoc.key.get();
    const auto track = assoc.val.get();

    iHists_.h1_["track_pt"]->Fill(track->pt());
    iHists_.h1_["track_eta"]->Fill(track->eta());
    iHists_.h1_["track_phi"]->Fill(track->phi());
    iHists_.h1_["track_validhits"]->Fill(track->numberOfValidHits());
    iHists_.h1_["track_chi2ndof"]->Fill((track->chi2() / track->ndof()));
    iHists_.h2_["track_chi2xhits"]->Fill((track->chi2() / track->ndof()), track->numberOfValidHits());
    iHists_.h2_["track_ptxhits"]->Fill(track->pt(), track->numberOfValidHits());
    iHists_.h2_["track_etaxhits"]->Fill(track->eta(), track->numberOfValidHits());
    iHists_.h2_["track_ptxchi2"]->Fill(track->pt(), (track->chi2() / track->ndof()));
    iHists_.h2_["track_ptxeta"]->Fill(track->pt(), track->eta());
    iHists_.h2_["track_etaxchi2"]->Fill(track->eta(), (track->chi2() / track->ndof()));

    edm::LogInfo("SiStripLorentzAnglePCLMonitor")
        << " track pT()" << track->pt() << " track eta()" << track->eta() << std::endl;

    for (const auto& meas : traj->measurements()) {
      const auto& trajState = meas.updatedState();
      if (!trajState.isValid())
        continue;

      // there can be 2 (stereo module), 1 (no stereo module), or 0 (no strip hit) clusters per measurement
      const auto trechit = meas.recHit()->hit();
      const auto simple1d = dynamic_cast<const SiStripRecHit1D*>(trechit);
      const auto simple = dynamic_cast<const SiStripRecHit2D*>(trechit);
      const auto matched = dynamic_cast<const SiStripMatchedRecHit2D*>(trechit);
      if (matched) {
        clusters.emplace_back(matched->monoId(), &matched->monoCluster(), traj, track, meas);
        clusters.emplace_back(matched->stereoId(), &matched->stereoCluster(), traj, track, meas);
      } else if (simple) {
        clusters.emplace_back(simple->geographicalId().rawId(), simple->cluster().get(), traj, track, meas);
      } else if (simple1d) {
        clusters.emplace_back(simple1d->geographicalId().rawId(), simple1d->cluster().get(), traj, track, meas);
      }
    }
  }

  for (const auto clus : clusters) {
    uint32_t c_nstrips = clus.cluster->amplitudes().size();
    m_clusterInfo.setCluster(*clus.cluster, clus.det);
    float c_variance = m_clusterInfo.variance();
    const auto& trajState = clus.measurement.updatedState();
    const auto trackDir = trajState.localDirection();
    float c_localdirx = trackDir.x();
    float c_localdiry = trackDir.y();
    float c_localdirz = trackDir.z();
    const auto hit = clus.measurement.recHit()->hit();

    // not yet needed (might be used for Backplane correction later on
    /*
      const auto& tkGeom = iSetup.getData(m_tkGeomToken);
      const auto stripDet = dynamic_cast<const StripGeomDetUnit*>(tkGeom.idToDet(hit->geographicalId()));
      float c_barycenter = stripDet->specificTopology().localPosition(clus.cluster->barycenter()).x();
      float c_localx = stripDet->toLocal(trajState.globalPosition()).x();
      float c_rhlocalx = hit->localPosition().x();
      float c_rhlocalxerr = hit->localPositionError().xx();
    */

    const uint32_t mod = hit->geographicalId().rawId();

    std::string locationtype = iHists_.moduleLocationType_[mod];
    if (locationtype.empty())
      return;

    const auto& hashedIndex = m_hash.hashedIndex(mod);

    TVector3 localdir(c_localdirx, c_localdiry, c_localdirz);
    int sign = iHists_.orientation_[mod];
    float tantheta = TMath::Tan(localdir.Theta());
    float cosphi = TMath::Cos(localdir.Phi());
    float theta = localdir.Theta();

    iHists_.h1_[Form("%s_nstrips", locationtype.c_str())]->Fill(c_nstrips);
    iHists_.h1_[Form("%s_tanthetatrk", locationtype.c_str())]->Fill(sign * tantheta);
    iHists_.h1_[Form("%s_cosphitrk", locationtype.c_str())]->Fill(cosphi);

    // nstrips
    iHists_.h2_[Form("%s_tanthcosphtrk_nstrip", locationtype.c_str())]->Fill(sign * cosphi * tantheta, c_nstrips);
    iHists_.h2_[Form("%s_thetatrk_nstrip", locationtype.c_str())]->Fill(sign * theta * cosphi, c_nstrips);

    // variance for width == 2
    if (c_nstrips == 2) {
      iHists_.h1_[Form("%s_variance_w2", locationtype.c_str())]->Fill(c_variance);
      iHists_.h2_[Form("%s_tanthcosphtrk_var2", locationtype.c_str())]->Fill(sign * cosphi * tantheta, c_variance);
      iHists_.h2_[Form("%s_thcosphtrk_var2", locationtype.c_str())]->Fill(sign * cosphi * theta, c_variance);

      // not in PCL
      if (saveHistosMods_) {
        iHists_.h2_ct_var2_m_[hashedIndex]->Fill(sign * cosphi * tantheta, c_variance);
        iHists_.h2_t_var2_m_[hashedIndex]->Fill(sign * cosphi * theta, c_variance);
      }
    }
    // variance for width == 3
    if (c_nstrips == 3) {
      iHists_.h1_[Form("%s_variance_w3", locationtype.c_str())]->Fill(c_variance);
      iHists_.h2_[Form("%s_tanthcosphtrk_var3", locationtype.c_str())]->Fill(sign * cosphi * tantheta, c_variance);
      iHists_.h2_[Form("%s_thcosphtrk_var3", locationtype.c_str())]->Fill(sign * cosphi * theta, c_variance);

      // not in PCL
      if (saveHistosMods_) {
        iHists_.h2_ct_var3_m_[hashedIndex]->Fill(sign * cosphi * tantheta, c_variance);
        iHists_.h2_t_var3_m_[hashedIndex]->Fill(sign * cosphi * theta, c_variance);
      }
    }
    // not in PCL
    if (saveHistosMods_) {
      iHists_.h2_ct_w_m_[hashedIndex]->Fill(sign * cosphi * tantheta, c_nstrips);
      iHists_.h2_t_w_m_[hashedIndex]->Fill(sign * cosphi * theta, c_nstrips);
    }
  }
}

void SiStripLorentzAnglePCLMonitor::bookHistograms(DQMStore::IBooker& ibook,
                                                   edm::Run const& run,
                                                   edm::EventSetup const& iSetup) {
  std::string bvalue = (iHists_.bfield_ == "3.8") ? "B-ON" : "B-OFF";
  std::string folderToBook = fmt::format("{}/{}_{}", folder_, bvalue, iHists_.apvmode_);

  ibook.setCurrentFolder(folderToBook);
  edm::LogPrint(moduleDescription().moduleName()) << "booking in " << folderToBook;

  // prepare track histograms
  // clang-format off
  iHists_.h1_["track_pt"] = ibook.book1D("track_pt", "track p_{T};track p_{T} [GeV];n. tracks", 2000, 0, 1000);
  iHists_.h1_["track_eta"] = ibook.book1D("track_eta", "track #eta;track #eta;n. tracks", 100, -4, 4);
  iHists_.h1_["track_phi"] = ibook.book1D("track_phi", "track #phi;track #phi;n. tracks", 80, -3.2, 3.2);
  iHists_.h1_["track_validhits"] =
      ibook.book1D("track_validhits", "track n. valid hits;track n. valid hits;n. tracks", 50, 0, 50);
  iHists_.h1_["track_chi2ndof"] =
      ibook.book1D("track_chi2ndof", "track #chi^{2}/ndf;track #chi^{2}/ndf;n. tracks", 100, 0, 5);
  iHists_.h2_["track_chi2xhits"] =
      ibook.book2D("track_chi2xhits_2d",
                   "track track n. hits vs track #chi^{2}/ndf;track #chi^{2};track n. valid hits;tracks",
                   100, 0, 5, 50, 0, 50);
  iHists_.h2_["track_ptxhits"] = ibook.book2D(
      "track_ptxhits_2d", "track n. hits vs p_{T};track p_{T} [GeV];track n. valid hits;tracks", 200, 0, 100, 50, 0, 50);
  iHists_.h2_["track_etaxhits"] = ibook.book2D(
      "track_etaxhits_2d", "track n. hits vs track #eta;track #eta;track n. valid hits;tracks", 60, -3, 3, 50, 0, 50);
  iHists_.h2_["track_ptxchi2"] =
      ibook.book2D("track_ptxchi2_2d",
                   "track #chi^{2}/ndf vs track p_{T};track p_{T} [GeV]; track #chi^{2}/ndf;tracks",
                   200, 0, 100, 100, 0, 5);
  iHists_.h2_["track_ptxeta"] = ibook.book2D(
      "track_ptxeta_2d", "track #eta vs track p_{T};track p_{T} [GeV];track #eta;tracks", 200, 0, 100, 60, -3, 3);
  iHists_.h2_["track_etaxchi2"] = ibook.book2D(
      "track_etaxchi2_2d", "track #chi^{2}/ndf vs track #eta;track #eta;track #chi^{2};tracks", 60, -3, 3, 100, 0, 5);
  // clang-format on

  // fill in the module types
  iHists_.nlayers_["TIB"] = 4;
  iHists_.nlayers_["TOB"] = 6;
  iHists_.modtypes_.push_back("s");
  iHists_.modtypes_.push_back("a");

  // prepare type histograms
  for (auto& layers : iHists_.nlayers_) {
    std::string subdet = layers.first;
    for (int l = 1; l <= layers.second; ++l) {
      ibook.setCurrentFolder(folderToBook + Form("/%s/L%d", subdet.c_str(), l));
      for (auto& t : iHists_.modtypes_) {
        // do not fill stereo where there aren't
        if (l > 2 && t == "s")
          continue;
        std::string locType = Form("%s_L%d%s", subdet.c_str(), l, t.c_str());

        // clang-format off
	const char* titles = Form("n.strips in %s;n.strips;n. clusters", locType.c_str());
        iHists_.h1_[Form("%s_nstrips", locType.c_str())] = ibook.book1D(Form("%s_nstrips", locType.c_str()), titles, 20, 0, 20);

	titles =  Form("tan(#theta_{trk}) in %s;tan(#theta_{trk});n. clusters", locType.c_str());
        iHists_.h1_[Form("%s_tanthetatrk", locType.c_str())] = ibook.book1D(Form("%s_tanthetatrk", locType.c_str()), titles, 300, -1.5, 1.5);

	titles =  Form("cos(#phi_{trk}) in %s;cos(#phi_{trk});n. clusters", locType.c_str());
        iHists_.h1_[Form("%s_cosphitrk", locType.c_str())] = ibook.book1D(Form("%s_cosphitrk", locType.c_str()), titles, 40, -1, 1);

	titles = Form("Cluster variance (w=2) in %s;cluster variance (w=2);n. clusters", locType.c_str());
        iHists_.h1_[Form("%s_variance_w2", locType.c_str())] = ibook.book1D(Form("%s_variance_w2", locType.c_str()), titles, 100, 0, 1);

	titles = Form("Cluster variance (w=3) in %s;cluster variance (w=3);n. clusters", locType.c_str());
        iHists_.h1_[Form("%s_variance_w3", locType.c_str())] = ibook.book1D(Form("%s_variance_w3", locType.c_str()), titles, 100, 0, 1);

	titles = Form("tan(#theta_{trk})cos(#phi_{trk}) vs n. strips in %s;n. strips;tan(#theta_{trk})cos(#phi_{trk});n. clusters", locType.c_str());
        iHists_.h2_[Form("%s_tanthcosphtrk_nstrip", locType.c_str())] = ibook.book2D(Form("%s_tanthcosphtrk_nstrip", locType.c_str()), titles, 360, -0.9, 0.9, 20, 0, 20);

	titles = Form("#theta_{trk} vs n. strips in %s;n. strips;#theta_{trk} [rad];n. clusters", locType.c_str());
        iHists_.h2_[Form("%s_thetatrk_nstrip", locType.c_str())] = ibook.book2D(Form("%s_thetatrk_nstrip", locType.c_str()), titles, 360, -0.9, 0.9, 20, 0, 20);

	titles = Form("tan(#theta_{trk})cos(#phi_{trk}) vs cluster variance (w=2) in %s;cluster variance (w=2);tan(#theta_{trk})cos(#phi_{trk});n. clusters", locType.c_str());
        iHists_.h2_[Form("%s_tanthcosphtrk_var2", locType.c_str())] = ibook.book2D(Form("%s_tanthcosphtrk_var2", locType.c_str()), titles, 360, -0.9, 0.9, 50, 0, 1);

	titles =  Form("tan(#theta_{trk})cos(#phi_{trk}) vs cluster variance (w=3) in %s;cluster variance (w=3);tan(#theta_{trk})cos(#phi_{trk});n. clusters", locType.c_str());
        iHists_.h2_[Form("%s_tanthcosphtrk_var3", locType.c_str())] = ibook.book2D(Form("%s_tanthcosphtrk_var3", locType.c_str()), titles, 360, -0.9, 0.9, 50, 0, 1);

	titles =  Form("#theta_{trk}cos(#phi_{trk}) vs cluster variance (w=2) in %s;cluster variance (w=2);#theta_{trk}cos(#phi_{trk});n. clusters", locType.c_str());
        iHists_.h2_[Form("%s_thcosphtrk_var2", locType.c_str())] = ibook.book2D(Form("%s_thcosphtrk_var2", locType.c_str()), titles, 360, -0.9, 0.9, 50, 0, 1);

	titles = Form("#theta_{trk}cos(#phi_{trk}) vs cluster variance (w=3) in %s;cluster variance (w=3);#theta_{trk}cos(#phi_{trk});n. clusters", locType.c_str());
        iHists_.h2_[Form("%s_thcosphtrk_var3", locType.c_str())] = ibook.book2D(Form("%s_thcosphtrk_var3", locType.c_str()), titles, 360, -0.9, 0.9, 50, 0, 1);
        // clang-format on
      }
    }
  }

  // prepare module histograms
  if (saveHistosMods_) {
    ibook.setCurrentFolder(folderToBook + "/modules");
    for (const auto& [mod, locationType] : iHists_.moduleLocationType_) {
      // histograms for each module
      iHists_.h1_[Form("%s_%d_nstrips", locationType.c_str(), mod)] =
          ibook.book1D(Form("%s_%d_nstrips", locationType.c_str(), mod), "", 10, 0, 10);
      iHists_.h1_[Form("%s_%d_tanthetatrk", locationType.c_str(), mod)] =
          ibook.book1D(Form("%s_%d_tanthetatrk", locationType.c_str(), mod), "", 40, -1., 1.);
      iHists_.h1_[Form("%s_%d_cosphitrk", locationType.c_str(), mod)] =
          ibook.book1D(Form("%s_%d_cosphitrk", locationType.c_str(), mod), "", 40, -1, 1);
      iHists_.h1_[Form("%s_%d_variance_w2", locationType.c_str(), mod)] =
          ibook.book1D(Form("%s_%d_variance_w2", locationType.c_str(), mod), "", 20, 0, 1);
      iHists_.h1_[Form("%s_%d_variance_w3", locationType.c_str(), mod)] =
          ibook.book1D(Form("%s_%d_variance_w3", locationType.c_str(), mod), "", 20, 0, 1);
    }

    int counter{0};
    SiStripHashedDetId::const_iterator iter = m_hash.begin();
    for (; iter != m_hash.end(); ++iter) {
      const auto& locationType = iHists_.moduleLocationType_[(*iter)];
      iHists_.h2_ct_w_m_.push_back(
          ibook.book2D(Form("ct_w_m_%s_%d", locationType.c_str(), *iter), "", 90, -0.9, 0.9, 10, 0, 10));
      iHists_.h2_t_w_m_.push_back(
          ibook.book2D(Form("t_w_m_%s_%d", locationType.c_str(), *iter), "", 90, -0.9, 0.9, 10, 0, 10));
      iHists_.h2_ct_var2_m_.push_back(
          ibook.book2D(Form("ct_var2_m_%s_%d", locationType.c_str(), *iter), "", 90, -0.9, 0.9, 20, 0, 1));
      iHists_.h2_ct_var3_m_.push_back(
          ibook.book2D(Form("ct_var3_m_%s_%d", locationType.c_str(), *iter), "", 90, -0.9, 0.9, 20, 0, 1));
      iHists_.h2_t_var2_m_.push_back(
          ibook.book2D(Form("t_var2_m_%s_%d", locationType.c_str(), *iter), "", 90, -0.9, 0.9, 20, 0, 1));
      iHists_.h2_t_var3_m_.push_back(
          ibook.book2D(Form("t_var3_m_%s_%d", locationType.c_str(), *iter), "", 90, -0.9, 0.9, 20, 0, 1));
      counter++;
    }
    edm::LogPrint(moduleDescription().moduleName())
        << __PRETTY_FUNCTION__ << " Booked " << counter << " module level histograms!";
  }  // if saveHistoMods
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiStripLorentzAnglePCLMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "AlCaReco/SiStripLorentzAngle")->setComment("DQM folder to write into");
  desc.add<bool>("saveHistoMods", false)->setComment("save module level hisotgrams. Warning! takes a lot of space!");
  desc.add<edm::InputTag>("Tracks", edm::InputTag("SiStripCalCosmics"))->setComment("input track collection");
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(SiStripLorentzAnglePCLMonitor);
