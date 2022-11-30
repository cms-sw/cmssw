#include "HIMultiTrackSelector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <Math/DistFunc.h>
#include <TMath.h>
#include <TFile.h>

namespace {
  // not a generic solution (wrong for N negative for instance)
  template <int N>
  struct PowN {
    template <typename T>
    static T op(T t) {
      return PowN<N / 2>::op(t) * PowN<(N + 1) / 2>::op(t);
    }
  };
  template <>
  struct PowN<0> {
    template <typename T>
    static T op(T t) {
      return T(1);
    }
  };
  template <>
  struct PowN<1> {
    template <typename T>
    static T op(T t) {
      return t;
    }
  };
  template <>
  struct PowN<2> {
    template <typename T>
    static T op(T t) {
      return t * t;
    }
  };

  template <typename T>
  T powN(T t, int n) {
    switch (n) {
      case 4:
        return PowN<4>::op(t);  // the only one that matters
      case 3:
        return PowN<3>::op(t);  // and this
      case 8:
        return PowN<8>::op(t);  // used in conversion????
      case 2:
        return PowN<2>::op(t);
      case 5:
        return PowN<5>::op(t);
      case 6:
        return PowN<6>::op(t);
      case 7:
        return PowN<7>::op(t);
      case 0:
        return PowN<0>::op(t);
      case 1:
        return PowN<1>::op(t);
      default:
        return std::pow(t, T(n));
    }
  }

}  // namespace

using namespace reco;

HIMultiTrackSelector::HIMultiTrackSelector() {
  useForestFromDB_ = true;
  forest_ = nullptr;
}

void HIMultiTrackSelector::ParseForestVars() {
  mvavars_indices.clear();
  for (unsigned i = 0; i < forestVars_.size(); i++) {
    std::string v = forestVars_[i];
    int ind = -1;
    if (v == "chi2perdofperlayer")
      ind = chi2perdofperlayer;
    if (v == "dxyperdxyerror")
      ind = dxyperdxyerror;
    if (v == "dzperdzerror")
      ind = dzperdzerror;
    if (v == "relpterr")
      ind = relpterr;
    if (v == "lostmidfrac")
      ind = lostmidfrac;
    if (v == "minlost")
      ind = minlost;
    if (v == "nhits")
      ind = nhits;
    if (v == "eta")
      ind = eta;
    if (v == "chi2n_no1dmod")
      ind = chi2n_no1dmod;
    if (v == "chi2n")
      ind = chi2n;
    if (v == "nlayerslost")
      ind = nlayerslost;
    if (v == "nlayers3d")
      ind = nlayers3d;
    if (v == "nlayers")
      ind = nlayers;
    if (v == "ndof")
      ind = ndof;
    if (v == "etaerror")
      ind = etaerror;

    if (ind == -1)
      edm::LogWarning("HIMultiTrackSelector")
          << "Unknown forest variable " << v << ". Please make sure it's in the list of supported variables\n";

    mvavars_indices.push_back(ind);
  }
}

HIMultiTrackSelector::HIMultiTrackSelector(const edm::ParameterSet &cfg)
    : src_(consumes<reco::TrackCollection>(cfg.getParameter<edm::InputTag>("src"))),
      hSrc_(consumes<TrackingRecHitCollection>(cfg.getParameter<edm::InputTag>("src"))),
      beamspot_(consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamspot"))),
      useVertices_(cfg.getParameter<bool>("useVertices")),
      useVtxError_(cfg.getParameter<bool>("useVtxError"))
// now get the pset for each selector
{
  if (useVertices_)
    vertices_ = consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertices"));

  applyPixelMergingCuts_ = false;
  if (cfg.exists("applyPixelMergingCuts"))
    applyPixelMergingCuts_ = cfg.getParameter<bool>("applyPixelMergingCuts");

  useAnyMVA_ = false;
  forestLabel_ = "MVASelectorIter0";
  std::string type = "BDTG";
  useForestFromDB_ = true;
  dbFileName_ = "";

  forest_ = nullptr;

  if (cfg.exists("useAnyMVA"))
    useAnyMVA_ = cfg.getParameter<bool>("useAnyMVA");
  if (useAnyMVA_) {
    if (cfg.exists("mvaType"))
      type = cfg.getParameter<std::string>("mvaType");
    if (cfg.exists("GBRForestLabel"))
      forestLabel_ = cfg.getParameter<std::string>("GBRForestLabel");
    if (cfg.exists("GBRForestVars")) {
      forestVars_ = cfg.getParameter<std::vector<std::string>>("GBRForestVars");
      ParseForestVars();
    }
    if (cfg.exists("GBRForestFileName")) {
      dbFileName_ = cfg.getParameter<std::string>("GBRForestFileName");
      useForestFromDB_ = false;
    }
    mvaType_ = type;
  }
  if (useForestFromDB_) {
    forestToken_ = esConsumes(edm::ESInputTag("", forestLabel_));
  }
  std::vector<edm::ParameterSet> trkSelectors(cfg.getParameter<std::vector<edm::ParameterSet>>("trackSelectors"));
  qualityToSet_.reserve(trkSelectors.size());
  vtxNumber_.reserve(trkSelectors.size());
  vertexCut_.reserve(trkSelectors.size());
  res_par_.reserve(trkSelectors.size());
  chi2n_par_.reserve(trkSelectors.size());
  chi2n_no1Dmod_par_.reserve(trkSelectors.size());
  d0_par1_.reserve(trkSelectors.size());
  dz_par1_.reserve(trkSelectors.size());
  d0_par2_.reserve(trkSelectors.size());
  dz_par2_.reserve(trkSelectors.size());
  applyAdaptedPVCuts_.reserve(trkSelectors.size());
  max_d0_.reserve(trkSelectors.size());
  max_z0_.reserve(trkSelectors.size());
  nSigmaZ_.reserve(trkSelectors.size());
  pixel_pTMinCut_.reserve(trkSelectors.size());
  pixel_pTMaxCut_.reserve(trkSelectors.size());
  min_layers_.reserve(trkSelectors.size());
  min_3Dlayers_.reserve(trkSelectors.size());
  max_lostLayers_.reserve(trkSelectors.size());
  min_hits_bypass_.reserve(trkSelectors.size());
  applyAbsCutsIfNoPV_.reserve(trkSelectors.size());
  max_d0NoPV_.reserve(trkSelectors.size());
  max_z0NoPV_.reserve(trkSelectors.size());
  preFilter_.reserve(trkSelectors.size());
  max_relpterr_.reserve(trkSelectors.size());
  min_nhits_.reserve(trkSelectors.size());
  max_minMissHitOutOrIn_.reserve(trkSelectors.size());
  max_lostHitFraction_.reserve(trkSelectors.size());
  min_eta_.reserve(trkSelectors.size());
  max_eta_.reserve(trkSelectors.size());
  useMVA_.reserve(trkSelectors.size());
  //mvaReaders_.reserve(trkSelectors.size());
  min_MVA_.reserve(trkSelectors.size());
  //mvaType_.reserve(trkSelectors.size());

  produces<edm::ValueMap<float>>("MVAVals");

  for (unsigned int i = 0; i < trkSelectors.size(); i++) {
    qualityToSet_.push_back(TrackBase::undefQuality);
    // parameters for vertex selection
    vtxNumber_.push_back(useVertices_ ? trkSelectors[i].getParameter<int32_t>("vtxNumber") : 0);
    vertexCut_.push_back(useVertices_ ? trkSelectors[i].getParameter<std::string>("vertexCut") : nullptr);
    //  parameters for adapted optimal cuts on chi2 and primary vertex compatibility
    res_par_.push_back(trkSelectors[i].getParameter<std::vector<double>>("res_par"));
    chi2n_par_.push_back(trkSelectors[i].getParameter<double>("chi2n_par"));
    chi2n_no1Dmod_par_.push_back(trkSelectors[i].getParameter<double>("chi2n_no1Dmod_par"));
    d0_par1_.push_back(trkSelectors[i].getParameter<std::vector<double>>("d0_par1"));
    dz_par1_.push_back(trkSelectors[i].getParameter<std::vector<double>>("dz_par1"));
    d0_par2_.push_back(trkSelectors[i].getParameter<std::vector<double>>("d0_par2"));
    dz_par2_.push_back(trkSelectors[i].getParameter<std::vector<double>>("dz_par2"));
    // Boolean indicating if adapted primary vertex compatibility cuts are to be applied.
    applyAdaptedPVCuts_.push_back(trkSelectors[i].getParameter<bool>("applyAdaptedPVCuts"));
    // Impact parameter absolute cuts.
    max_d0_.push_back(trkSelectors[i].getParameter<double>("max_d0"));
    max_z0_.push_back(trkSelectors[i].getParameter<double>("max_z0"));
    nSigmaZ_.push_back(trkSelectors[i].getParameter<double>("nSigmaZ"));
    // Cuts on numbers of layers with hits/3D hits/lost hits.
    min_layers_.push_back(trkSelectors[i].getParameter<uint32_t>("minNumberLayers"));
    min_3Dlayers_.push_back(trkSelectors[i].getParameter<uint32_t>("minNumber3DLayers"));
    max_lostLayers_.push_back(trkSelectors[i].getParameter<uint32_t>("maxNumberLostLayers"));
    min_hits_bypass_.push_back(trkSelectors[i].getParameter<uint32_t>("minHitsToBypassChecks"));
    // Flag to apply absolute cuts if no PV passes the selection
    applyAbsCutsIfNoPV_.push_back(trkSelectors[i].getParameter<bool>("applyAbsCutsIfNoPV"));
    keepAllTracks_.push_back(trkSelectors[i].getParameter<bool>("keepAllTracks"));
    max_relpterr_.push_back(trkSelectors[i].getParameter<double>("max_relpterr"));
    min_nhits_.push_back(trkSelectors[i].getParameter<uint32_t>("min_nhits"));
    max_minMissHitOutOrIn_.push_back(trkSelectors[i].existsAs<int32_t>("max_minMissHitOutOrIn")
                                         ? trkSelectors[i].getParameter<int32_t>("max_minMissHitOutOrIn")
                                         : 99);
    max_lostHitFraction_.push_back(trkSelectors[i].existsAs<double>("max_lostHitFraction")
                                       ? trkSelectors[i].getParameter<double>("max_lostHitFraction")
                                       : 1.0);
    min_eta_.push_back(trkSelectors[i].existsAs<double>("min_eta") ? trkSelectors[i].getParameter<double>("min_eta")
                                                                   : -9999);
    max_eta_.push_back(trkSelectors[i].existsAs<double>("max_eta") ? trkSelectors[i].getParameter<double>("max_eta")
                                                                   : 9999);

    setQualityBit_.push_back(false);
    std::string qualityStr = trkSelectors[i].getParameter<std::string>("qualityBit");
    if (!qualityStr.empty()) {
      setQualityBit_[i] = true;
      qualityToSet_[i] = TrackBase::qualityByName(trkSelectors[i].getParameter<std::string>("qualityBit"));
    }

    if (setQualityBit_[i] && (qualityToSet_[i] == TrackBase::undefQuality))
      throw cms::Exception("Configuration")
          << "You can't set the quality bit " << trkSelectors[i].getParameter<std::string>("qualityBit")
          << " as it is 'undefQuality' or unknown.\n";

    if (applyAbsCutsIfNoPV_[i]) {
      max_d0NoPV_.push_back(trkSelectors[i].getParameter<double>("max_d0NoPV"));
      max_z0NoPV_.push_back(trkSelectors[i].getParameter<double>("max_z0NoPV"));
    } else {  //dummy values
      max_d0NoPV_.push_back(0.);
      max_z0NoPV_.push_back(0.);
    }

    name_.push_back(trkSelectors[i].getParameter<std::string>("name"));

    preFilter_[i] = trkSelectors.size();  // no prefilter

    std::string pfName = trkSelectors[i].getParameter<std::string>("preFilterName");
    if (!pfName.empty()) {
      bool foundPF = false;
      for (unsigned int j = 0; j < i; j++)
        if (name_[j] == pfName) {
          foundPF = true;
          preFilter_[i] = j;
        }
      if (!foundPF)
        throw cms::Exception("Configuration") << "Invalid prefilter name in HIMultiTrackSelector "
                                              << trkSelectors[i].getParameter<std::string>("preFilterName");
    }

    if (applyPixelMergingCuts_) {
      pixel_pTMinCut_.push_back(trkSelectors[i].getParameter<std::vector<double>>("pixel_pTMinCut"));
      pixel_pTMaxCut_.push_back(trkSelectors[i].getParameter<std::vector<double>>("pixel_pTMaxCut"));
    }

    //    produces<std::vector<int> >(name_[i]).setBranchAlias( name_[i] + "TrackQuals");
    produces<edm::ValueMap<int>>(name_[i]).setBranchAlias(name_[i] + "TrackQuals");
    if (useAnyMVA_) {
      bool thisMVA = false;
      if (trkSelectors[i].exists("useMVA"))
        thisMVA = trkSelectors[i].getParameter<bool>("useMVA");
      useMVA_.push_back(thisMVA);
      if (thisMVA) {
        double minVal = -1;
        if (trkSelectors[i].exists("minMVA"))
          minVal = trkSelectors[i].getParameter<double>("minMVA");
        min_MVA_.push_back(minVal);

      } else {
        min_MVA_.push_back(-9999.0);
      }
    } else {
      min_MVA_.push_back(-9999.0);
    }
  }
}

HIMultiTrackSelector::~HIMultiTrackSelector() { delete forest_; }

void HIMultiTrackSelector::beginStream(edm::StreamID) {
  if (!useForestFromDB_) {
    TFile gbrfile(dbFileName_.c_str());
    forest_ = (GBRForest *)gbrfile.Get(forestLabel_.c_str());
  }
}

void HIMultiTrackSelector::run(edm::Event &evt, const edm::EventSetup &es) const {
  using namespace std;
  using namespace edm;
  using namespace reco;

  // Get tracks
  Handle<TrackCollection> hSrcTrack;
  evt.getByToken(src_, hSrcTrack);
  const TrackCollection &srcTracks(*hSrcTrack);

  // get hits in track..
  Handle<TrackingRecHitCollection> hSrcHits;
  evt.getByToken(hSrc_, hSrcHits);
  const TrackingRecHitCollection &srcHits(*hSrcHits);

  // looking for the beam spot
  edm::Handle<reco::BeamSpot> hBsp;
  evt.getByToken(beamspot_, hBsp);
  const reco::BeamSpot &vertexBeamSpot(*hBsp);

  // Select good primary vertices for use in subsequent track selection
  edm::Handle<reco::VertexCollection> hVtx;
  if (useVertices_)
    evt.getByToken(vertices_, hVtx);

  unsigned int trkSize = srcTracks.size();
  std::vector<int> selTracksSave(qualityToSet_.size() * trkSize, 0);

  std::vector<float> mvaVals_(srcTracks.size(), -99.f);
  processMVA(evt, es, mvaVals_, *hVtx);

  for (unsigned int i = 0; i < qualityToSet_.size(); i++) {
    std::vector<int> selTracks(trkSize, 0);
    auto selTracksValueMap = std::make_unique<edm::ValueMap<int>>();
    edm::ValueMap<int>::Filler filler(*selTracksValueMap);

    std::vector<Point> points;
    std::vector<float> vterr, vzerr;
    if (useVertices_)
      selectVertices(i, *hVtx, points, vterr, vzerr);

    // Loop over tracks
    size_t current = 0;
    for (TrackCollection::const_iterator it = srcTracks.begin(), ed = srcTracks.end(); it != ed; ++it, ++current) {
      const Track &trk = *it;
      // Check if this track passes cuts

      LogTrace("TrackSelection") << "ready to check track with pt=" << trk.pt();

      //already removed
      bool ok = true;
      float mvaVal = 0;
      if (preFilter_[i] < i && selTracksSave[preFilter_[i] * trkSize + current] < 0) {
        selTracks[current] = -1;
        ok = false;
        if (!keepAllTracks_[i])
          continue;
      } else {
        if (useAnyMVA_)
          mvaVal = mvaVals_[current];
        ok = select(i, vertexBeamSpot, srcHits, trk, points, vterr, vzerr, mvaVal);
        if (!ok) {
          LogTrace("TrackSelection") << "track with pt=" << trk.pt() << " NOT selected";
          if (!keepAllTracks_[i]) {
            selTracks[current] = -1;
            continue;
          }
        } else
          LogTrace("TrackSelection") << "track with pt=" << trk.pt() << " selected";
      }

      if (preFilter_[i] < i) {
        selTracks[current] = selTracksSave[preFilter_[i] * trkSize + current];
      } else {
        selTracks[current] = trk.qualityMask();
      }
      if (ok && setQualityBit_[i]) {
        selTracks[current] = (selTracks[current] | (1 << qualityToSet_[i]));
        if (qualityToSet_[i] == TrackBase::tight) {
          selTracks[current] = (selTracks[current] | (1 << TrackBase::loose));
        } else if (qualityToSet_[i] == TrackBase::highPurity) {
          selTracks[current] = (selTracks[current] | (1 << TrackBase::loose));
          selTracks[current] = (selTracks[current] | (1 << TrackBase::tight));
        }

        if (!points.empty()) {
          if (qualityToSet_[i] == TrackBase::loose) {
            selTracks[current] = (selTracks[current] | (1 << TrackBase::looseSetWithPV));
          } else if (qualityToSet_[i] == TrackBase::highPurity) {
            selTracks[current] = (selTracks[current] | (1 << TrackBase::looseSetWithPV));
            selTracks[current] = (selTracks[current] | (1 << TrackBase::highPuritySetWithPV));
          }
        }
      }
    }
    for (unsigned int j = 0; j < trkSize; j++)
      selTracksSave[j + i * trkSize] = selTracks[j];
    filler.insert(hSrcTrack, selTracks.begin(), selTracks.end());
    filler.fill();

    //    evt.put(std::move(selTracks),name_[i]);
    evt.put(std::move(selTracksValueMap), name_[i]);
  }
}

bool HIMultiTrackSelector::select(unsigned int tsNum,
                                  const reco::BeamSpot &vertexBeamSpot,
                                  const TrackingRecHitCollection &recHits,
                                  const reco::Track &tk,
                                  const std::vector<Point> &points,
                                  std::vector<float> &vterr,
                                  std::vector<float> &vzerr,
                                  double mvaVal) const {
  // Decide if the given track passes selection cuts.

  using namespace std;

  //cuts on number of valid hits
  auto nhits = tk.numberOfValidHits();
  if (nhits >= min_hits_bypass_[tsNum])
    return true;
  if (nhits < min_nhits_[tsNum])
    return false;

  if (tk.ndof() < 1E-5)
    return false;

  //////////////////////////////////////////////////
  //Adding the MVA selection before any other cut//
  ////////////////////////////////////////////////
  if (useAnyMVA_ && useMVA_[tsNum]) {
    if (mvaVal < min_MVA_[tsNum])
      return false;
    else
      return true;
  }
  /////////////////////////////////
  //End of MVA selection section//
  ///////////////////////////////

  // Cuts on numbers of layers with hits/3D hits/lost hits.
  uint32_t nlayers = tk.hitPattern().trackerLayersWithMeasurement();
  uint32_t nlayers3D =
      tk.hitPattern().pixelLayersWithMeasurement() + tk.hitPattern().numberOfValidStripLayersWithMonoAndStereo();
  uint32_t nlayersLost = tk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
  LogDebug("TrackSelection") << "cuts on nlayers: " << nlayers << " " << nlayers3D << " " << nlayersLost << " vs "
                             << min_layers_[tsNum] << " " << min_3Dlayers_[tsNum] << " " << max_lostLayers_[tsNum];
  if (nlayers < min_layers_[tsNum])
    return false;
  if (nlayers3D < min_3Dlayers_[tsNum])
    return false;
  if (nlayersLost > max_lostLayers_[tsNum])
    return false;
  LogTrace("TrackSelection") << "cuts on nlayers passed";

  float chi2n = tk.normalizedChi2();
  float chi2n_no1Dmod = chi2n;

  int count1dhits = 0;
  auto ith = tk.extra()->firstRecHit();
  auto edh = ith + tk.recHitsSize();
  for (; ith < edh; ++ith) {
    const TrackingRecHit &hit = recHits[ith];
    if (hit.dimension() == 1)
      ++count1dhits;
  }
  if (count1dhits > 0) {
    float chi2 = tk.chi2();
    float ndof = tk.ndof();
    chi2n = (chi2 + count1dhits) / float(ndof + count1dhits);
  }
  // For each 1D rechit, the chi^2 and ndof is increased by one.  This is a way of retaining approximately
  // the same normalized chi^2 distribution as with 2D rechits.
  if (chi2n > chi2n_par_[tsNum] * nlayers)
    return false;

  if (chi2n_no1Dmod > chi2n_no1Dmod_par_[tsNum] * nlayers)
    return false;

  // Get track parameters
  float pt = std::max(float(tk.pt()), 0.000001f);
  float eta = tk.eta();
  if (eta < min_eta_[tsNum] || eta > max_eta_[tsNum])
    return false;

  //cuts on relative error on pt
  float relpterr = float(tk.ptError()) / pt;
  if (relpterr > max_relpterr_[tsNum])
    return false;

  int lostIn = tk.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_INNER_HITS);
  int lostOut = tk.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_OUTER_HITS);
  int minLost = std::min(lostIn, lostOut);
  if (minLost > max_minMissHitOutOrIn_[tsNum])
    return false;
  float lostMidFrac =
      tk.numberOfLostHits() == 0 ? 0. : tk.numberOfLostHits() / (tk.numberOfValidHits() + tk.numberOfLostHits());
  if (lostMidFrac > max_lostHitFraction_[tsNum])
    return false;

  // Pixel Track Merging pT dependent cuts
  if (applyPixelMergingCuts_) {
    // hard cut at absolute min/max pt
    if (pt < pixel_pTMinCut_[tsNum][0])
      return false;
    if (pt > pixel_pTMaxCut_[tsNum][0])
      return false;
    // tapering cuts with chi2n_no1Dmod
    double pTMaxCutPos = (pixel_pTMaxCut_[tsNum][0] - pt) / (pixel_pTMaxCut_[tsNum][0] - pixel_pTMaxCut_[tsNum][1]);
    double pTMinCutPos = (pt - pixel_pTMinCut_[tsNum][0]) / (pixel_pTMinCut_[tsNum][1] - pixel_pTMinCut_[tsNum][0]);
    if (pt > pixel_pTMaxCut_[tsNum][1] &&
        chi2n_no1Dmod > pixel_pTMaxCut_[tsNum][2] * nlayers * pow(pTMaxCutPos, pixel_pTMaxCut_[tsNum][3]))
      return false;
    if (pt < pixel_pTMinCut_[tsNum][1] &&
        chi2n_no1Dmod > pixel_pTMinCut_[tsNum][2] * nlayers * pow(pTMinCutPos, pixel_pTMinCut_[tsNum][3]))
      return false;
  }

  //other track parameters
  float d0 = -tk.dxy(vertexBeamSpot.position()), d0E = tk.d0Error(), dz = tk.dz(vertexBeamSpot.position()),
        dzE = tk.dzError();

  // parametrized d0 resolution for the track pt
  float nomd0E = sqrt(res_par_[tsNum][0] * res_par_[tsNum][0] + (res_par_[tsNum][1] / pt) * (res_par_[tsNum][1] / pt));
  // parametrized z0 resolution for the track pt and eta
  float nomdzE = nomd0E * (std::cosh(eta));

  float dzCut = std::min(powN(dz_par1_[tsNum][0] * nlayers, int(dz_par1_[tsNum][1] + 0.5)) * nomdzE,
                         powN(dz_par2_[tsNum][0] * nlayers, int(dz_par2_[tsNum][1] + 0.5)) * dzE);
  float d0Cut = std::min(powN(d0_par1_[tsNum][0] * nlayers, int(d0_par1_[tsNum][1] + 0.5)) * nomd0E,
                         powN(d0_par2_[tsNum][0] * nlayers, int(d0_par2_[tsNum][1] + 0.5)) * d0E);

  // ---- PrimaryVertex compatibility cut
  bool primaryVertexZCompatibility(false);
  bool primaryVertexD0Compatibility(false);

  if (points.empty()) {  //If not primaryVertices are reconstructed, check just the compatibility with the BS
    //z0 within (n sigma + dzCut) of the beam spot z, if no good vertex is found
    if (abs(dz) < hypot(vertexBeamSpot.sigmaZ() * nSigmaZ_[tsNum], dzCut))
      primaryVertexZCompatibility = true;
    // d0 compatibility with beam line
    if (abs(d0) < d0Cut)
      primaryVertexD0Compatibility = true;
  }

  int iv = 0;
  for (std::vector<Point>::const_iterator point = points.begin(), end = points.end(); point != end; ++point) {
    LogTrace("TrackSelection") << "Test track w.r.t. vertex with z position " << point->z();
    if (primaryVertexZCompatibility && primaryVertexD0Compatibility)
      break;
    float dzPV = tk.dz(*point);   //re-evaluate the dz with respect to the vertex position
    float d0PV = tk.dxy(*point);  //re-evaluate the dxy with respect to the vertex position
    if (useVtxError_) {
      float dzErrPV = std::sqrt(dzE * dzE + vzerr[iv] * vzerr[iv]);  // include vertex error in z
      float d0ErrPV = std::sqrt(d0E * d0E + vterr[iv] * vterr[iv]);  // include vertex error in xy
      iv++;
      if (abs(dzPV) < dz_par1_[tsNum][0] * pow(nlayers, dz_par1_[tsNum][1]) * nomdzE &&
          abs(dzPV) < dz_par2_[tsNum][0] * pow(nlayers, dz_par2_[tsNum][1]) * dzErrPV && abs(dzPV) < max_z0_[tsNum])
        primaryVertexZCompatibility = true;
      if (abs(d0PV) < d0_par1_[tsNum][0] * pow(nlayers, d0_par1_[tsNum][1]) * nomd0E &&
          abs(d0PV) < d0_par2_[tsNum][0] * pow(nlayers, d0_par2_[tsNum][1]) * d0ErrPV && abs(d0PV) < max_d0_[tsNum])
        primaryVertexD0Compatibility = true;
    } else {
      if (abs(dzPV) < dzCut)
        primaryVertexZCompatibility = true;
      if (abs(d0PV) < d0Cut)
        primaryVertexD0Compatibility = true;
    }
    LogTrace("TrackSelection") << "distances " << dzPV << " " << d0PV << " vs " << dzCut << " " << d0Cut;
  }

  if (points.empty() && applyAbsCutsIfNoPV_[tsNum]) {
    if (abs(dz) > max_z0NoPV_[tsNum] || abs(d0) > max_d0NoPV_[tsNum])
      return false;
  } else {
    // Absolute cuts on all tracks impact parameters with respect to beam-spot.
    // If BS is not compatible, verify if at least the reco-vertex is compatible (useful for incorrect BS settings)
    if (abs(d0) > max_d0_[tsNum] && !primaryVertexD0Compatibility)
      return false;
    LogTrace("TrackSelection") << "absolute cuts on d0 passed";
    if (abs(dz) > max_z0_[tsNum] && !primaryVertexZCompatibility)
      return false;
    LogTrace("TrackSelection") << "absolute cuts on dz passed";
  }

  LogTrace("TrackSelection") << "cuts on PV: apply adapted PV cuts? " << applyAdaptedPVCuts_[tsNum]
                             << " d0 compatibility? " << primaryVertexD0Compatibility << " z compatibility? "
                             << primaryVertexZCompatibility;

  if (applyAdaptedPVCuts_[tsNum]) {
    return (primaryVertexD0Compatibility && primaryVertexZCompatibility);
  } else {
    return true;
  }
}

void HIMultiTrackSelector::selectVertices(unsigned int tsNum,
                                          const reco::VertexCollection &vtxs,
                                          std::vector<Point> &points,
                                          std::vector<float> &vterr,
                                          std::vector<float> &vzerr) const {
  // Select good primary vertices
  using namespace reco;
  int32_t toTake = vtxNumber_[tsNum];
  for (VertexCollection::const_iterator it = vtxs.begin(), ed = vtxs.end(); it != ed; ++it) {
    LogDebug("SelectVertex") << " select vertex with z position " << it->z() << " " << it->chi2() << " " << it->ndof()
                             << " " << TMath::Prob(it->chi2(), static_cast<int32_t>(it->ndof()));
    const Vertex &vtx = *it;
    bool pass = vertexCut_[tsNum](vtx);
    if (pass) {
      points.push_back(it->position());
      vterr.push_back(sqrt(it->yError() * it->xError()));
      vzerr.push_back(it->zError());
      LogTrace("SelectVertex") << " SELECTED vertex with z position " << it->z();
      toTake--;
      if (toTake == 0)
        break;
    }
  }
}

void HIMultiTrackSelector::processMVA(edm::Event &evt,
                                      const edm::EventSetup &es,
                                      std::vector<float> &mvaVals_,
                                      const reco::VertexCollection &vertices) const {
  using namespace std;
  using namespace edm;
  using namespace reco;

  // Get tracks
  Handle<TrackCollection> hSrcTrack;
  evt.getByToken(src_, hSrcTrack);
  const TrackCollection &srcTracks(*hSrcTrack);
  assert(mvaVals_.size() == srcTracks.size());

  // get hits in track..
  Handle<TrackingRecHitCollection> hSrcHits;
  evt.getByToken(hSrc_, hSrcHits);
  const TrackingRecHitCollection &srcHits(*hSrcHits);

  auto mvaValValueMap = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler mvaFiller(*mvaValValueMap);

  if (!useAnyMVA_) {
    // mvaVals_ already initalized...
    mvaFiller.insert(hSrcTrack, mvaVals_.begin(), mvaVals_.end());
    mvaFiller.fill();
    evt.put(std::move(mvaValValueMap), "MVAVals");
    return;
  }

  bool checkvertex =
      std::find(mvavars_indices.begin(), mvavars_indices.end(), dxyperdxyerror) != mvavars_indices.end() ||
      std::find(mvavars_indices.begin(), mvavars_indices.end(), dzperdzerror) != mvavars_indices.end();

  size_t current = 0;
  for (TrackCollection::const_iterator it = srcTracks.begin(), ed = srcTracks.end(); it != ed; ++it, ++current) {
    const Track &trk = *it;

    float mvavalues[15];
    mvavalues[ndof] = trk.ndof();
    mvavalues[nlayers] = trk.hitPattern().trackerLayersWithMeasurement();
    mvavalues[nlayers3d] =
        trk.hitPattern().pixelLayersWithMeasurement() + trk.hitPattern().numberOfValidStripLayersWithMonoAndStereo();
    mvavalues[nlayerslost] = trk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
    mvavalues[chi2n_no1dmod] = trk.normalizedChi2();
    mvavalues[chi2perdofperlayer] = mvavalues[chi2n_no1dmod] / mvavalues[nlayers];

    float chi2n1d = trk.normalizedChi2();
    int count1dhits = 0;
    auto ith = trk.extra()->firstRecHit();
    auto edh = ith + trk.recHitsSize();
    for (; ith < edh; ++ith) {
      const TrackingRecHit &hit = srcHits[ith];
      if (hit.dimension() == 1)
        ++count1dhits;
    }
    if (count1dhits > 0) {
      float chi2 = trk.chi2();
      float ndof = trk.ndof();
      chi2n1d = (chi2 + count1dhits) / float(ndof + count1dhits);
    }

    mvavalues[chi2n] = chi2n1d;  //chi2 and 1d modes

    mvavalues[eta] = trk.eta();
    mvavalues[relpterr] = float(trk.ptError()) / std::max(float(trk.pt()), 0.000001f);
    mvavalues[nhits] = trk.numberOfValidHits();

    int lostIn = trk.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_INNER_HITS);
    int lostOut = trk.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_OUTER_HITS);
    mvavalues[minlost] = std::min(lostIn, lostOut);
    mvavalues[lostmidfrac] = trk.numberOfLostHits() / (trk.numberOfValidHits() + trk.numberOfLostHits());

    mvavalues[etaerror] = trk.etaError();

    float reldz = 0;
    float reldxy = 0;
    if (checkvertex) {
      int vtxind = 0;  // only first vertex is taken into account for the speed purposes
      float dxy = trk.dxy(vertices[vtxind].position()),
            dxyE = sqrt(trk.dxyError() * trk.dxyError() + vertices[vtxind].xError() * vertices[vtxind].yError());
      float dz = trk.dz(vertices[vtxind].position()),
            dzE = sqrt(trk.dzError() * trk.dzError() + vertices[vtxind].zError() * vertices[vtxind].zError());
      reldz = dz / dzE;
      reldxy = dxy / dxyE;
    }
    mvavalues[dxyperdxyerror] = reldxy;
    mvavalues[dzperdzerror] = reldz;

    std::vector<float> gbrValues;

    //fill in the gbrValues vector with the necessary variables
    for (unsigned i = 0; i < mvavars_indices.size(); i++) {
      gbrValues.push_back(mvavalues[mvavars_indices[i]]);
    }

    GBRForest const *forest = forest_;
    if (useForestFromDB_) {
      forest = &es.getData(forestToken_);
    }

    auto gbrVal = forest->GetClassifier(&gbrValues[0]);
    mvaVals_[current] = gbrVal;
  }
  mvaFiller.insert(hSrcTrack, mvaVals_.begin(), mvaVals_.end());
  mvaFiller.fill();
  evt.put(std::move(mvaValValueMap), "MVAVals");
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HIMultiTrackSelector);
