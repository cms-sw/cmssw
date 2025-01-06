/** \class AnalyticalTrackSelector
 *
 * selects a subset of a track collection, copying extra information on demand
 * 
 * \author Paolo Azzurri, Giovanni Petrucciani 
 *
 *
 *
 */

#include <utility>
#include <vector>
#include <memory>
#include <algorithm>
#include <map>

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "MultiTrackSelector.h"

using namespace reco;

class dso_hidden AnalyticalTrackSelector final : public MultiTrackSelector {
private:
public:
  /// constructor
  explicit AnalyticalTrackSelector(const edm::ParameterSet& cfg);
  /// destructor
  ~AnalyticalTrackSelector() override = default;
  /// fillDescriptions
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  typedef math::XYZPoint Point;
  /// process one event
  void run(edm::Event& evt, const edm::EventSetup& es) const override;

  /// copy only the tracks, not extras and rechits (for AOD)
  bool copyExtras_;
  /// copy also trajectories and trajectory->track associations
  bool copyTrajectories_;
  /// eta restrictions
  double minEta_;
  double maxEta_;

  edm::EDGetTokenT<std::vector<Trajectory>> srcTraj_;
  edm::EDGetTokenT<TrajTrackAssociationCollection> srcTass_;
};

#include "DataFormats/Common/interface/ValueMap.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <Math/DistFunc.h>
#include "TMath.h"

AnalyticalTrackSelector::AnalyticalTrackSelector(const edm::ParameterSet& cfg) : MultiTrackSelector() {
  //Spoof the pset for each track selector!
  //Size is always 1!!!
  qualityToSet_.reserve(1);
  vtxNumber_.reserve(1);
  vertexCut_.reserve(1);
  res_par_.reserve(1);
  chi2n_par_.reserve(1);
  chi2n_no1Dmod_par_.reserve(1);
  d0_par1_.reserve(1);
  dz_par1_.reserve(1);
  d0_par2_.reserve(1);
  dz_par2_.reserve(1);
  applyAdaptedPVCuts_.reserve(1);
  max_d0_.reserve(1);
  max_z0_.reserve(1);
  nSigmaZ_.reserve(1);
  min_layers_.reserve(1);
  min_3Dlayers_.reserve(1);
  max_lostLayers_.reserve(1);
  min_hits_bypass_.reserve(1);
  applyAbsCutsIfNoPV_.reserve(1);
  max_d0NoPV_.reserve(1);
  max_z0NoPV_.reserve(1);
  preFilter_.reserve(1);
  max_relpterr_.reserve(1);
  min_nhits_.reserve(1);
  max_minMissHitOutOrIn_.reserve(1);
  max_lostHitFraction_.reserve(1);
  min_eta_.reserve(1);
  max_eta_.reserve(1);
  forest_.reserve(1);
  mvaType_.reserve(1);
  useMVA_.reserve(1);

  produces<edm::ValueMap<float>>("MVAVals");
  //foward compatibility
  produces<MVACollection>("MVAValues");
  forest_[0] = nullptr;
  useAnyMVA_ = cfg.getParameter<bool>("useAnyMVA");

  src_ = consumes<reco::TrackCollection>(cfg.getParameter<edm::InputTag>("src"));
  hSrc_ = consumes<TrackingRecHitCollection>(cfg.getParameter<edm::InputTag>("src"));
  beamspot_ = consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamspot"));
  useVertices_ = cfg.getParameter<bool>("useVertices");
  useVtxError_ = cfg.getParameter<bool>("useVtxError");
  if (useVertices_)
    vertices_ = consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertices"));
  copyExtras_ = cfg.getUntrackedParameter<bool>("copyExtras", false);
  copyTrajectories_ = cfg.getUntrackedParameter<bool>("copyTrajectories", false);
  if (copyTrajectories_) {
    srcTraj_ = consumes<std::vector<Trajectory>>(cfg.getParameter<edm::InputTag>("src"));
    srcTass_ = consumes<TrajTrackAssociationCollection>(cfg.getParameter<edm::InputTag>("src"));
  }

  qualityToSet_.push_back(TrackBase::undefQuality);
  // parameters for vertex selection
  vtxNumber_.push_back(useVertices_ ? cfg.getParameter<int32_t>("vtxNumber") : 0);
  vertexCut_.push_back(useVertices_ ? cfg.getParameter<std::string>("vertexCut") : "");
  //  parameters for adapted optimal cuts on chi2 and primary vertex compatibility
  res_par_.push_back(cfg.getParameter<std::vector<double>>("res_par"));
  chi2n_par_.push_back(cfg.getParameter<double>("chi2n_par"));
  chi2n_no1Dmod_par_.push_back(cfg.getParameter<double>("chi2n_no1Dmod_par"));
  d0_par1_.push_back(cfg.getParameter<std::vector<double>>("d0_par1"));
  dz_par1_.push_back(cfg.getParameter<std::vector<double>>("dz_par1"));
  d0_par2_.push_back(cfg.getParameter<std::vector<double>>("d0_par2"));
  dz_par2_.push_back(cfg.getParameter<std::vector<double>>("dz_par2"));

  // Boolean indicating if adapted primary vertex compatibility cuts are to be applied.
  applyAdaptedPVCuts_.push_back(cfg.getParameter<bool>("applyAdaptedPVCuts"));
  // Impact parameter absolute cuts.
  max_d0_.push_back(cfg.getParameter<double>("max_d0"));
  max_z0_.push_back(cfg.getParameter<double>("max_z0"));
  nSigmaZ_.push_back(cfg.getParameter<double>("nSigmaZ"));
  // Cuts on numbers of layers with hits/3D hits/lost hits.
  min_layers_.push_back(cfg.getParameter<uint32_t>("minNumberLayers"));
  min_3Dlayers_.push_back(cfg.getParameter<uint32_t>("minNumber3DLayers"));
  max_lostLayers_.push_back(cfg.getParameter<uint32_t>("maxNumberLostLayers"));
  min_hits_bypass_.push_back(cfg.getParameter<uint32_t>("minHitsToBypassChecks"));
  max_relpterr_.push_back(cfg.getParameter<double>("max_relpterr"));
  min_nhits_.push_back(cfg.getParameter<uint32_t>("min_nhits"));
  max_minMissHitOutOrIn_.push_back(cfg.getParameter<int32_t>("max_minMissHitOutOrIn"));
  max_lostHitFraction_.push_back(cfg.getParameter<double>("max_lostHitFraction"));
  min_eta_.push_back(cfg.getParameter<double>("min_eta"));
  max_eta_.push_back(cfg.getParameter<double>("max_eta"));

  // Flag to apply absolute cuts if no PV passes the selection
  applyAbsCutsIfNoPV_.push_back(cfg.getParameter<bool>("applyAbsCutsIfNoPV"));
  keepAllTracks_.push_back(cfg.getParameter<bool>("keepAllTracks"));

  setQualityBit_.push_back(false);
  std::string qualityStr = cfg.getParameter<std::string>("qualityBit");

  if (d0_par1_[0].size() != 2 || dz_par1_[0].size() != 2 || d0_par2_[0].size() != 2 || dz_par2_[0].size() != 2) {
    edm::LogError("MisConfiguration") << "vector of size less then 2";
    throw;
  }

  if (!qualityStr.empty()) {
    setQualityBit_[0] = true;
    qualityToSet_[0] = TrackBase::qualityByName(cfg.getParameter<std::string>("qualityBit"));
  }

  if (keepAllTracks_[0] && !setQualityBit_[0])
    throw cms::Exception("Configuration")
        << "If you set 'keepAllTracks' to true, you must specify which qualityBit to set.\n";
  if (setQualityBit_[0] && (qualityToSet_[0] == TrackBase::undefQuality))
    throw cms::Exception("Configuration")
        << "You can't set the quality bit " << cfg.getParameter<std::string>("qualityBit")
        << " as it is 'undefQuality' or unknown.\n";
  if (applyAbsCutsIfNoPV_[0]) {
    max_d0NoPV_.push_back(cfg.getParameter<double>("max_d0NoPV"));
    max_z0NoPV_.push_back(cfg.getParameter<double>("max_z0NoPV"));
  } else {  //dummy values
    max_d0NoPV_.push_back(0.);
    max_z0NoPV_.push_back(0.);
  }

  if (useAnyMVA_) {
    bool thisMVA = cfg.getParameter<bool>("useMVA");
    useMVA_.push_back(thisMVA);
    if (thisMVA) {
      double minVal = cfg.getParameter<double>("minMVA");
      min_MVA_.push_back(minVal);
      mvaType_.push_back(cfg.getParameter<std::string>("mvaType"));
      forestLabel_.push_back(cfg.getParameter<std::string>("GBRForestLabel"));
      useMVAonly_.push_back(cfg.getParameter<bool>("useMVAonly"));
    } else {
      min_MVA_.push_back(-9999.0);
      useMVAonly_.push_back(false);
      mvaType_.push_back("Detached");
      forestLabel_.push_back("MVASelectorIter0");
    }
  } else {
    useMVA_.push_back(false);
    useMVAonly_.push_back(false);
    min_MVA_.push_back(-9999.0);
    mvaType_.push_back("Detached");
    forestLabel_.push_back("MVASelectorIter0");
  }

  std::string alias(cfg.getParameter<std::string>("@module_label"));
  if (copyExtras_) {
    produces<reco::TrackExtraCollection>().setBranchAlias(alias + "TrackExtras");
    produces<TrackingRecHitCollection>().setBranchAlias(alias + "RecHits");
  }
  if (copyTrajectories_) {
    produces<std::vector<Trajectory>>().setBranchAlias(alias + "Trajectories");
    produces<TrajTrackAssociationCollection>().setBranchAlias(alias + "TrajectoryTrackAssociations");
  }
  // TrackCollection refers to TrackingRechit and TrackExtra
  // collections, need to declare its production after them to work
  // around a rare race condition in framework scheduling
  produces<reco::TrackCollection>().setBranchAlias(alias + "Tracks");
}

void AnalyticalTrackSelector::run(edm::Event& evt, const edm::EventSetup& es) const {
  // storage....
  std::unique_ptr<reco::TrackCollection> selTracks_;
  std::unique_ptr<reco::TrackExtraCollection> selTrackExtras_;
  std::unique_ptr<TrackingRecHitCollection> selHits_;
  std::unique_ptr<std::vector<Trajectory>> selTrajs_;
  std::unique_ptr<std::vector<const Trajectory*>> selTrajPtrs_;
  std::unique_ptr<TrajTrackAssociationCollection> selTTAss_;
  reco::TrackRefProd rTracks_;
  reco::TrackExtraRefProd rTrackExtras_;
  TrackingRecHitRefProd rHits_;
  edm::RefProd<std::vector<Trajectory>> rTrajectories_;
  std::vector<reco::TrackRef> trackRefs_;

  using namespace std;
  using namespace edm;
  using namespace reco;

  Handle<TrackCollection> hSrcTrack;
  Handle<vector<Trajectory>> hTraj;
  Handle<vector<Trajectory>> hTrajP;
  Handle<TrajTrackAssociationCollection> hTTAss;

  // looking for the beam spot
  edm::Handle<reco::BeamSpot> hBsp;
  evt.getByToken(beamspot_, hBsp);
  reco::BeamSpot vertexBeamSpot;
  vertexBeamSpot = *hBsp;

  // Select good primary vertices for use in subsequent track selection
  const reco::VertexCollection dummyVtx;
  const reco::VertexCollection* vtxPtr = &dummyVtx;
  std::vector<Point> points;
  std::vector<float> vterr, vzerr;
  if (useVertices_) {
    vtxPtr = &evt.get(vertices_);
    selectVertices(0, *vtxPtr, points, vterr, vzerr);
    // Debug
    LogDebug("SelectVertex") << points.size() << " good pixel vertices";
  }

  // Get tracks
  evt.getByToken(src_, hSrcTrack);
  // get hits in track..
  Handle<TrackingRecHitCollection> hSrcHits;
  evt.getByToken(hSrc_, hSrcHits);
  const TrackingRecHitCollection& srcHits(*hSrcHits);

  selTracks_ = std::make_unique<TrackCollection>();
  rTracks_ = evt.getRefBeforePut<TrackCollection>();
  if (copyExtras_) {
    selTrackExtras_ = std::make_unique<TrackExtraCollection>();
    selHits_ = std::make_unique<TrackingRecHitCollection>();
    rHits_ = evt.getRefBeforePut<TrackingRecHitCollection>();
    rTrackExtras_ = evt.getRefBeforePut<TrackExtraCollection>();
  }

  if (copyTrajectories_)
    trackRefs_.resize(hSrcTrack->size());

  std::vector<float> mvaVals_(hSrcTrack->size(), -99.f);
  processMVA(evt, es, vertexBeamSpot, *vtxPtr, 0, mvaVals_, true);

  // Loop over tracks
  size_t current = 0;
  for (TrackCollection::const_iterator it = hSrcTrack->begin(), ed = hSrcTrack->end(); it != ed; ++it, ++current) {
    const Track& trk = *it;
    // Check if this track passes cuts

    LogTrace("TrackSelection") << "ready to check track with pt=" << trk.pt();

    float mvaVal = 0;
    if (useAnyMVA_)
      mvaVal = mvaVals_[current];
    bool ok = select(0, vertexBeamSpot, srcHits, trk, points, vterr, vzerr, mvaVal);
    if (!ok) {
      LogTrace("TrackSelection") << "track with pt=" << trk.pt() << " NOT selected";

      if (copyTrajectories_)
        trackRefs_[current] = reco::TrackRef();
      if (!keepAllTracks_[0])
        continue;
    }
    LogTrace("TrackSelection") << "track with pt=" << trk.pt() << " selected";
    selTracks_->push_back(Track(trk));  // clone and store
    if (ok && setQualityBit_[0]) {
      selTracks_->back().setQuality(qualityToSet_[0]);
      if (qualityToSet_[0] == TrackBase::tight) {
        selTracks_->back().setQuality(TrackBase::loose);
      } else if (qualityToSet_[0] == TrackBase::highPurity) {
        selTracks_->back().setQuality(TrackBase::loose);
        selTracks_->back().setQuality(TrackBase::tight);
      }
      if (!points.empty()) {
        if (qualityToSet_[0] == TrackBase::loose) {
          selTracks_->back().setQuality(TrackBase::looseSetWithPV);
        } else if (qualityToSet_[0] == TrackBase::highPurity) {
          selTracks_->back().setQuality(TrackBase::looseSetWithPV);
          selTracks_->back().setQuality(TrackBase::highPuritySetWithPV);
        }
      }
    }
    if (copyExtras_) {
      // TrackExtras
      selTrackExtras_->push_back(TrackExtra(trk.outerPosition(),
                                            trk.outerMomentum(),
                                            trk.outerOk(),
                                            trk.innerPosition(),
                                            trk.innerMomentum(),
                                            trk.innerOk(),
                                            trk.outerStateCovariance(),
                                            trk.outerDetId(),
                                            trk.innerStateCovariance(),
                                            trk.innerDetId(),
                                            trk.seedDirection(),
                                            trk.seedRef()));
      selTracks_->back().setExtra(TrackExtraRef(rTrackExtras_, selTrackExtras_->size() - 1));
      TrackExtra& tx = selTrackExtras_->back();
      tx.setResiduals(trk.residuals());
      // TrackingRecHits
      auto const firstHitIndex = selHits_->size();
      for (trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++hit) {
        selHits_->push_back((*hit)->clone());
      }
      tx.setHits(rHits_, firstHitIndex, selHits_->size() - firstHitIndex);
      tx.setTrajParams(trk.extra()->trajParams(), trk.extra()->chi2sX5());
    }
    if (copyTrajectories_) {
      trackRefs_[current] = TrackRef(rTracks_, selTracks_->size() - 1);
    }
  }
  if (copyTrajectories_) {
    Handle<vector<Trajectory>> hTraj;
    Handle<TrajTrackAssociationCollection> hTTAss;
    evt.getByToken(srcTass_, hTTAss);
    evt.getByToken(srcTraj_, hTraj);
    selTrajs_ = std::make_unique<std::vector<Trajectory>>();
    rTrajectories_ = evt.getRefBeforePut<vector<Trajectory>>();
    selTTAss_ = std::make_unique<TrajTrackAssociationCollection>(rTrajectories_, rTracks_);
    for (size_t i = 0, n = hTraj->size(); i < n; ++i) {
      Ref<vector<Trajectory>> trajRef(hTraj, i);
      TrajTrackAssociationCollection::const_iterator match = hTTAss->find(trajRef);
      if (match != hTTAss->end()) {
        const Ref<TrackCollection>& trkRef = match->val;
        short oldKey = static_cast<short>(trkRef.key());
        if (trackRefs_[oldKey].isNonnull()) {
          selTrajs_->push_back(Trajectory(*trajRef));
          selTTAss_->insert(Ref<vector<Trajectory>>(rTrajectories_, selTrajs_->size() - 1), trackRefs_[oldKey]);
        }
      }
    }
  }

  evt.put(std::move(selTracks_));
  if (copyExtras_) {
    evt.put(std::move(selTrackExtras_));
    evt.put(std::move(selHits_));
  }
  if (copyTrajectories_) {
    evt.put(std::move(selTrajs_));
    evt.put(std::move(selTTAss_));
  }
}

void AnalyticalTrackSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("generalTracks"));
  desc.add<bool>("keepAllTracks", false)
      ->setComment("if set to true tracks failing this filter are kept in the output");
  desc.add<edm::InputTag>("beamspot", edm::InputTag("offlineBeamSpot"));

  // vertex selection
  desc.add<bool>("useVertices", true);
  desc.add<bool>("useVtxError", false);
  desc.add<edm::InputTag>("vertices", edm::InputTag("firstStepPrimaryVertices"));
  desc.add<int32_t>("vtxNumber", -1);
  desc.add<std::string>("vertexCut", "ndof>=2&!isFake");

  desc.addUntracked<bool>("copyExtras", false);
  desc.addUntracked<bool>("copyTrajectories", false);
  desc.add<std::string>("qualityBit", std::string(""))->setComment("set to ''if you don't want to set the bit");

  // parameters for adapted optimal cuts on chi2 and primary vertex compatibility
  desc.add<double>("chi2n_no1Dmod_par", 9999.)
      ->setComment("parameter for adapted optimal cuts on chi2 and primary vertex compatibility");
  desc.add<double>("chi2n_par", 1.6)
      ->setComment("parameter for adapted optimal cuts on chi2 and primary vertex compatibility");
  desc.add<std::vector<double>>("res_par", {0.003, 0.01})->setComment("default: Loose");
  desc.add<std::vector<double>>("d0_par1", {0.55, 4.0})->setComment("default: Loose");
  desc.add<std::vector<double>>("d0_par2", {0.65, 4.0})->setComment("default: Loose");
  desc.add<std::vector<double>>("dz_par1", {0.55, 4.0})->setComment("default: Loose");
  desc.add<std::vector<double>>("dz_par2", {0.45, 4.0})->setComment("default: Loose");
  desc.add<bool>("applyAdaptedPVCuts", true)
      ->setComment("Boolean indicating if adapted primary vertex compatibility cuts are to be applied.");

  // Impact parameter absolute cuts.
  desc.add<double>("max_d0", 100.)->setComment("transverse impact parameter absolute cut");
  desc.add<double>("max_z0", 100.)->setComment("longitudinal impact parameter absolute cut");
  desc.add<double>("nSigmaZ", 4.);

  // Cuts on numbers of layers with hits/3D hits/lost hits.
  desc.add<uint32_t>("minNumberLayers", 0);
  desc.add<uint32_t>("minNumber3DLayers", 0);
  desc.add<uint32_t>("minHitsToBypassChecks", 20);
  desc.add<uint32_t>("maxNumberLostLayers", 999);

  // Absolute cuts in case of no PV. If yes, please define also max_d0NoPV and max_z0NoPV
  desc.add<bool>("applyAbsCutsIfNoPV", false);
  desc.add<double>("max_d0NoPV", 100.);
  desc.add<double>("max_z0NoPV", 100.);

  // parameters for cutting on pterror/pt and number of valid hits
  desc.add<double>("max_relpterr", 9999)->setComment("parameter for cutting on pterror/pt");
  desc.add<uint32_t>("min_nhits", 0)->setComment("parameter for cutting on number of valid hits");

  desc.add<double>("max_lostHitFraction", 1.0);
  desc.add<int32_t>("max_minMissHitOutOrIn", 99);

  // parameters for cutting on eta
  desc.add<double>("max_eta", 9999.);
  desc.add<double>("min_eta", -9999.);

  // optional parameters for MVA selection
  desc.add<bool>("useMVA", false);
  desc.add<bool>("useAnyMVA", false);
  desc.add<bool>("useMVAonly", false);
  desc.add<double>("minMVA", -1.);
  desc.add<std::string>("GBRForestLabel", "MVASelectorIter0");
  desc.add<std::string>("mvaType", "Detached");

  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(AnalyticalTrackSelector);
