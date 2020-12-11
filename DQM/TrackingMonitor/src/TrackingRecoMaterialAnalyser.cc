#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"

#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cassert>
#include <unordered_map>
#include <string>

// Values are not ordered randomly, but the order is taken from
// http://cmslxr.fnal.gov/dxr/CMSSW/source/Geometry/CommonDetUnit/interface/GeomDetEnumerators.h#15
static const std::vector<std::string> sDETS{"", "PXB", "PXF", "TIB", "TID", "TOB", "TEC"};

class TrackingRecoMaterialAnalyser : public DQMEDAnalyzer {
public:
  explicit TrackingRecoMaterialAnalyser(const edm::ParameterSet &);
  void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  ~TrackingRecoMaterialAnalyser() override;

private:
  inline bool isDoubleSided(const TrackerTopology *tTopo, DetId id) { return (tTopo->glued(id)); }
  TrackTransformer refitter_;
  const edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;
  const edm::EDGetTokenT<reco::VertexCollection> verticesToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryTokenRun_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  bool usePV_;
  std::string folder_;
  std::unordered_map<std::string, MonitorElement *> histosOriEta_;
  std::unordered_map<std::string, MonitorElement *> histosEta_;
  MonitorElement *histo_RZ_;
  MonitorElement *histo_RZ_Ori_;
  MonitorElement *deltaPt_in_out_2d_;
  MonitorElement *deltaP_in_out_vs_eta_;
  MonitorElement *deltaP_in_out_vs_z_;
  MonitorElement *deltaP_in_out_vs_eta_2d_;
  MonitorElement *deltaP_in_out_vs_eta_vs_phi_2d_;
  MonitorElement *deltaP_in_out_vs_z_2d_;
  MonitorElement *deltaPt_in_out_vs_eta_;
  MonitorElement *deltaPt_in_out_vs_z_;
  MonitorElement *deltaPl_in_out_vs_eta_;
  MonitorElement *deltaPl_in_out_vs_z_;
  MonitorElement *P_vs_eta_2d_;
};

//-------------------------------------------------------------------------
TrackingRecoMaterialAnalyser::TrackingRecoMaterialAnalyser(const edm::ParameterSet &iPSet)
    : refitter_(iPSet, consumesCollector()),
      tracksToken_(consumes<reco::TrackCollection>(iPSet.getParameter<edm::InputTag>("tracks"))),
      beamspotToken_(consumes<reco::BeamSpot>(iPSet.getParameter<edm::InputTag>("beamspot"))),
      verticesToken_(mayConsume<reco::VertexCollection>(iPSet.getParameter<edm::InputTag>("vertices"))),
      trackerGeometryTokenRun_(esConsumes<edm::Transition::BeginRun>()),
      tTopoToken_(esConsumes()),
      usePV_(iPSet.getParameter<bool>("usePV")),
      folder_(iPSet.getParameter<std::string>("folder")),
      histo_RZ_(nullptr),
      histo_RZ_Ori_(nullptr),
      deltaPt_in_out_2d_(nullptr),
      deltaP_in_out_vs_eta_(nullptr),
      deltaP_in_out_vs_z_(nullptr),
      deltaP_in_out_vs_eta_2d_(nullptr),
      deltaP_in_out_vs_eta_vs_phi_2d_(nullptr),
      deltaP_in_out_vs_z_2d_(nullptr),
      deltaPt_in_out_vs_eta_(nullptr),
      deltaPt_in_out_vs_z_(nullptr),
      deltaPl_in_out_vs_eta_(nullptr),
      deltaPl_in_out_vs_z_(nullptr),
      P_vs_eta_2d_(nullptr) {}

//-------------------------------------------------------------------------
TrackingRecoMaterialAnalyser::~TrackingRecoMaterialAnalyser(void) {}

//-------------------------------------------------------------------------
void TrackingRecoMaterialAnalyser::bookHistograms(DQMStore::IBooker &ibook,
                                                  edm::Run const &,
                                                  edm::EventSetup const &setup) {
  using namespace std;
  const TrackerGeometry &trackerGeometry = setup.getData(trackerGeometryTokenRun_);

  ibook.setCurrentFolder(folder_);

  // Histogram to store the radiation length map, in the R-Z plane,
  // gathering numbers directly from the trackerRecoMaterial.xml
  // file. The numbers are not corrected for the track angle.
  histo_RZ_Ori_ = ibook.bookProfile2D("OriRadLen", "Original_RadLen", 600, -300., 300, 120, 0., 120., 0., 1.);

  // Histogram to store the radiation length map, as before, but
  // correcting the numbers with the track angle. This represents the
  // real material seen by the track.
  histo_RZ_ = ibook.bookProfile2D("RadLen", "RadLen", 600, -300., 300, 120, 0., 120., 0., 1.);

  // Histogram to show the deltaP Out-In in the eta-phi plane.
  deltaP_in_out_vs_eta_vs_phi_2d_ = ibook.bookProfile2D(
      "DeltaP_in_out_vs_eta_vs_phi_2d", "DeltaP_in_out_vs_eta_vs_phi_2d", 100, -3.0, 3.0, 100, -3.15, 3.15, 0., 100.);

  // Histogram to show the deltaP Out-In vs eta.
  deltaP_in_out_vs_eta_2d_ =
      ibook.book2D("DeltaP_in_out_vs_eta_2d", "DeltaP_in_out_vs_eta_2d", 100, -3.0, 3.0, 100, 0., 1);

  // Histogram to show the deltaP Out-In vs Z. The Z coordinate is the
  // one computed at the outermost hit.
  deltaP_in_out_vs_z_2d_ = ibook.book2D("DeltaP_in_out_vs_z_2d", "DeltaP_in_out_vs_z_2d", 600, -300, 300, 200., -1, 1.);

  // Histogram to show the deltaP Out-In vs eta. The eta is the one
  // computed at the innermost hit.
  deltaP_in_out_vs_eta_ =
      ibook.bookProfile("DeltaP_in_out_vs_eta", "DeltaP_in_out_vs_eta", 100, -3.0, 3.0, -100., 100.);
  deltaP_in_out_vs_z_ = ibook.bookProfile("DeltaP_in_out_vs_z", "DeltaP_in_out_vs_z", 600, -300, 300, -100., 100.);

  // Histogram to show the delta_Pt Out-In vs eta. The eta is the one
  // computed at the innermost hit.
  deltaPt_in_out_vs_eta_ =
      ibook.bookProfile("DeltaPt_in_out_vs_eta", "DeltaPt_in_out_vs_eta", 100, -3.0, 3.0, -100., 100.);

  // Histogram to show the delta_Pt Out-In vs Z. The Z is the one
  // computed at the outermost hit.
  deltaPt_in_out_vs_z_ = ibook.bookProfile("DeltaPt_in_out_vs_z", "DeltaPt_in_out_vs_z", 600, -300, 300, -100., 100);

  // Histogram to show the delta_Pl Out-In vs eta. The eta is the one
  // computed at the innermost hit.
  deltaPl_in_out_vs_eta_ =
      ibook.bookProfile("DeltaPz_in_out_vs_eta", "DeltaPz_in_out_vs_eta", 100, -3.0, 3.0, -100., 100.);

  // Histogram to show the delta_Pl Out-In vs Z. The Z is the one
  // computed at the outermost hit.
  deltaPl_in_out_vs_z_ = ibook.bookProfile("DeltaPz_in_out_vs_z", "DeltaPz_in_out_vs_z", 600, -300, 300, -100., 100.);

  // Histogram to show the delta_Pt Out-In in the Z-R plane. Z and R
  // are related to the outermost hit.
  deltaPt_in_out_2d_ = ibook.bookProfile2D("DeltaPt 2D", "DeltaPt 2D", 600, -300., 300, 120, 0., 120., -100., 100.);

  // Histogram to show the distribution of p vs eta for all tracks.
  P_vs_eta_2d_ = ibook.book2D("P_vs_eta_2d", "P_vs_eta_2d", 100, -3.0, 3.0, 100., 0., 5.);
  char title[50];
  char key[20];
  for (unsigned int det = 1; det < sDETS.size(); ++det) {
    for (unsigned int sub_det = 1; sub_det <= trackerGeometry.numberOfLayers(det); ++sub_det) {
      memset(title, 0, sizeof(title));
      snprintf(title, sizeof(title), "Original_RadLen_vs_Eta_%s%d", sDETS[det].data(), sub_det);
      snprintf(key, sizeof(key), "%s%d", sDETS[det].data(), sub_det);
      histosOriEta_.insert(
          make_pair<string, MonitorElement *>(key, ibook.bookProfile(title, title, 250, -5.0, 5.0, 0., 1.)));
      snprintf(title, sizeof(title), "RadLen_vs_Eta_%s%d", sDETS[det].data(), sub_det);
      histosEta_.insert(
          make_pair<string, MonitorElement *>(key, ibook.bookProfile(title, title, 250, -5.0, 5.0, 0., 1.)));
    }
  }
}

//-------------------------------------------------------------------------
void TrackingRecoMaterialAnalyser::analyze(const edm::Event &event, const edm::EventSetup &setup) {
  using namespace edm;
  using namespace reco;
  using namespace std;

  refitter_.setServices(setup);

  Handle<TrackCollection> tracks;
  Handle<VertexCollection> vertices;

  // Get the TrackerTopology
  const TrackerTopology *const tTopo = &setup.getData(tTopoToken_);

  // Get Tracks
  event.getByToken(tracksToken_, tracks);
  if (!tracks.isValid() || tracks->empty()) {
    LogInfo("TrackingRecoMaterialAnalyser") << "Invalid or empty track collection" << endl;
    return;
  }

  // Track Selector
  auto selector = [](const Track &track, const reco::Vertex::Point &pv) -> bool {
    return (track.quality(track.qualityByName("highPurity")) && track.dxy(pv) < 0.01 &&
            track.hitPattern().numberOfLostTrackerHits(HitPattern::MISSING_OUTER_HITS) == 0);
  };

  // Get BeamSpot
  Handle<BeamSpot> beamSpot;
  event.getByToken(beamspotToken_, beamSpot);
  // Bail out if missing
  if (!beamSpot.isValid())
    return;

  reco::Vertex::Point pv(beamSpot->position());
  if (usePV_) {
    event.getByToken(verticesToken_, vertices);
    if (vertices.isValid() && !vertices->empty()) {
      // Since we need to use eta and Z information from the tracks, in
      // order not to have the reco material distribution washed out due
      // to geometrical effects, we need to confine the PV to some small
      // region.
      const Vertex &v = (*vertices)[0];
      if (!v.isFake() && v.ndof() > 4 && std::fabs(v.z()) < 24 && std::fabs(v.position().rho()) < 2)
        pv = v.position();
    }
  }

  // Main idea:
  // * select first good tracks in input, according to reasonable criteria
  // * refit the tracks so that we have access to the TrajectoryMeasurements
  //   that internally have all the information about the TSOS on all
  //   crossed layers. We need the refit track and not the original one so
  //   that we are able to correctly compute the path travelled by the track
  //   within the detector, using its updated TSOS. The material description
  //   can in principle be derived also directly from the rechits, via the
  //   det()[(->)GeomDet *]->mediumProperties chain, but that would simply give the
  //   face values, not the "real" ones used while propagating the track.
  // * Loop on all measurements, extract the information about the TSOS,
  //   its surface and its mediumProperties
  // * Make plots for the untouched material properties, but also for the
  //   ones corrected by the track direction, since the material properties,
  //   according to the documentation, should refer to normal incidence of
  //   the track, which is seldom the case, according to the current direction
  TrajectoryStateOnSurface current_tsos;
  DetId current_det;
  for (auto const &track : *tracks) {
    if (!selector(track, pv))
      continue;
    auto const &inner = track.innerMomentum();
    auto const &outer = track.outerMomentum();
    deltaP_in_out_vs_eta_->Fill(inner.eta(), inner.R() - outer.R());
    deltaP_in_out_vs_z_->Fill(track.outerZ(), inner.R() - outer.R());
    deltaP_in_out_vs_eta_2d_->Fill(inner.eta(), inner.R() - outer.R());
    deltaP_in_out_vs_eta_vs_phi_2d_->Fill(inner.eta(), inner.phi(), inner.R() - outer.R());
    deltaP_in_out_vs_z_2d_->Fill(track.outerZ(), inner.R() - outer.R());
    deltaPt_in_out_vs_eta_->Fill(inner.eta(), inner.rho() - outer.rho());
    deltaPt_in_out_vs_z_->Fill(track.outerZ(), inner.rho() - outer.rho());
    deltaPl_in_out_vs_eta_->Fill(inner.eta(), inner.z() - outer.z());
    deltaPl_in_out_vs_z_->Fill(track.outerZ(), inner.z() - outer.z());
    deltaPt_in_out_2d_->Fill(track.outerZ(), track.outerPosition().rho(), inner.rho() - outer.rho());
    P_vs_eta_2d_->Fill(track.eta(), track.p());
    vector<Trajectory> traj = refitter_.transform(track);
    if (traj.size() > 1 || traj.empty())
      continue;
    for (auto const &tm : traj.front().measurements()) {
      if (tm.recHit().get() &&
          (tm.recHitR().type() == TrackingRecHit::valid || tm.recHitR().type() == TrackingRecHit::missing)) {
        current_tsos = tm.updatedState().isValid() ? tm.updatedState() : tm.forwardPredictedState();
        auto const &localP = current_tsos.localMomentum();
        current_det = tm.recHit()->geographicalId();
        const Surface &surface = current_tsos.surface();
        assert(tm.recHit()->surface() == &surface);
        if (!surface.mediumProperties().isValid()) {
          LogError("TrackingRecoMaterialAnalyser")
              << "Medium properties for material linked to detector"
              << " are invalid at: " << current_tsos.globalPosition() << " " << (SiStripDetId)current_det << endl;
          assert(0);
          continue;
        }
        float p2 = localP.mag2();
        float xf = std::abs(std::sqrt(p2) / localP.z());
        float ori_xi = surface.mediumProperties().xi();
        float ori_radLen = surface.mediumProperties().radLen();
        float xi = ori_xi * xf;
        float radLen = ori_radLen * xf;

        // NOTA BENE: THIS ASSUMES THAT THE TRACKS HAVE BEEN PRODUCED
        // WITH SPLITTING, AS IS THE CASE FOR THE generalTracks
        // collection.

        // Since there are double-sided (glued) modules all over the
        // tracker, the material budget has been internally
        // partitioned in two equal components, so that each single
        // layer will receive half of the correct radLen. For this
        // reason, only for the double-sided components, we rescale
        // the obtained radLen by 2.

        // In particular see code here:
        // http://cmslxr.fnal.gov/dxr/CMSSW_8_0_5/source/Geometry/TrackerGeometryBuilder/src/TrackerGeomBuilderFromGeometricDet.cc#213
        // where, in the SiStrip Tracker, if the module has a partner
        // (i.e. it's a glued detector) the plane is built with a
        // scaling of 0.5. The actual plane is built few lines below:
        // http://cmslxr.fnal.gov/dxr/CMSSW_8_0_5/source/Geometry/TrackerGeometryBuilder/src/TrackerGeomBuilderFromGeometricDet.cc#287

        if (isDoubleSided(tTopo, current_det)) {
          LogTrace("TrackingRecoMaterialAnalyser")
              << "Eta: " << track.eta() << " " << sDETS[current_det.subdetId()] << tTopo->layer(current_det)
              << " has ori_radLen: " << ori_radLen << " and ori_xi: " << xi << " and has radLen: " << radLen
              << "  and xi: " << xi << endl;
          ori_radLen *= 2.;
          radLen *= 2.;
        }

        histosOriEta_[sDETS[current_det.subdetId()] + to_string(tTopo->layer(current_det))]->Fill(
            current_tsos.globalPosition().eta(), ori_radLen);
        histosEta_[sDETS[current_det.subdetId()] + to_string(tTopo->layer(current_det))]->Fill(
            current_tsos.globalPosition().eta(), radLen);
        histo_RZ_Ori_->Fill(current_tsos.globalPosition().z(), current_tsos.globalPosition().perp(), ori_radLen);
        histo_RZ_->Fill(current_tsos.globalPosition().z(), current_tsos.globalPosition().perp(), radLen);
        LogInfo("TrackingRecoMaterialAnalyser")
            << "Eta: " << track.eta() << " " << sDETS[current_det.subdetId()] << tTopo->layer(current_det)
            << " has ori_radLen: " << ori_radLen << " and ori_xi: " << xi << " and has radLen: " << radLen
            << "  and xi: " << xi << endl;
      }
    }
  }
}

//-------------------------------------------------------------------------
// define as a plugin
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackingRecoMaterialAnalyser);
