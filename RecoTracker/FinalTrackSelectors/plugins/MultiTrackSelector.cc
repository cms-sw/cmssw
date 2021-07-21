#include "MultiTrackSelector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/transform.h"

#include <Math/DistFunc.h>
#include <TMath.h>
#include <TFile.h>

#include "powN.h"

using namespace reco;

MultiTrackSelector::MultiTrackSelector() { useForestFromDB_ = true; }

MultiTrackSelector::MultiTrackSelector(const edm::ParameterSet& cfg)
    : src_(consumes<reco::TrackCollection>(cfg.getParameter<edm::InputTag>("src"))),
      hSrc_(consumes<TrackingRecHitCollection>(cfg.getParameter<edm::InputTag>("src"))),
      beamspot_(consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamspot"))),
      useVertices_(cfg.getParameter<bool>("useVertices")),
      useVtxError_(cfg.getParameter<bool>("useVtxError"))
// now get the pset for each selector
{
  if (useVertices_)
    vertices_ = consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertices"));
  if (useVtxError_) {
    edm::LogWarning("MultiTRackSelector") << "you are executing buggy code, if intentional please help to fix it";
  }
  useAnyMVA_ = false;
  useForestFromDB_ = true;
  dbFileName_ = "";

  if (cfg.exists("useAnyMVA"))
    useAnyMVA_ = cfg.getParameter<bool>("useAnyMVA");

  if (useAnyMVA_) {
    if (cfg.exists("GBRForestFileName")) {
      dbFileName_ = cfg.getParameter<std::string>("GBRForestFileName");
      useForestFromDB_ = false;
    }
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
  useMVAonly_.reserve(trkSelectors.size());
  //mvaReaders_.reserve(trkSelectors.size());
  min_MVA_.reserve(trkSelectors.size());
  mvaType_.reserve(trkSelectors.size());
  forestLabel_.reserve(trkSelectors.size());
  forest_.reserve(trkSelectors.size());

  produces<edm::ValueMap<float>>("MVAVals");

  //foward compatibility
  produces<MVACollection>("MVAValues");

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
        throw cms::Exception("Configuration") << "Invalid prefilter name in MultiTrackSelector "
                                              << trkSelectors[i].getParameter<std::string>("preFilterName");
    }

    //    produces<std::vector<int> >(name_[i]).setBranchAlias( name_[i] + "TrackQuals");
    produces<edm::ValueMap<int>>(name_[i]).setBranchAlias(name_[i] + "TrackQuals");
    produces<QualityMaskCollection>(name_[i]).setBranchAlias(name_[i] + "QualityMasks");
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
        mvaType_.push_back(trkSelectors[i].exists("mvaType") ? trkSelectors[i].getParameter<std::string>("mvaType")
                                                             : "Detached");
        forestLabel_.push_back(trkSelectors[i].exists("GBRForestLabel")
                                   ? trkSelectors[i].getParameter<std::string>("GBRForestLabel")
                                   : "MVASelectorIter0");
        useMVAonly_.push_back(trkSelectors[i].exists("useMVAonly") ? trkSelectors[i].getParameter<bool>("useMVAonly")
                                                                   : false);
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
  }

  if (useForestFromDB_) {
    forestToken_ = edm::vector_transform(forestLabel_, [this](auto const& label) {
      return esConsumes<GBRForest, GBRWrapperRcd>(edm::ESInputTag("", label));
    });
  }
}

MultiTrackSelector::~MultiTrackSelector() {
  for (auto forest : forest_)
    delete forest;
}

void MultiTrackSelector::beginStream(edm::StreamID) {
  if (!useForestFromDB_) {
    TFile gbrfile(dbFileName_.c_str());
    for (int i = 0; i < (int)forestLabel_.size(); i++) {
      forest_[i] = (GBRForest*)gbrfile.Get(forestLabel_[i].c_str());
    }
  }
}

void MultiTrackSelector::run(edm::Event& evt, const edm::EventSetup& es) const {
  using namespace std;
  using namespace edm;
  using namespace reco;

  // Get tracks
  Handle<TrackCollection> hSrcTrack;
  evt.getByToken(src_, hSrcTrack);

  const TrackCollection& srcTracks(*hSrcTrack);
  if (hSrcTrack.failedToGet())
    edm::LogWarning("MultiTrackSelector") << "could not get Track collection";

  // get hits in track..
  Handle<TrackingRecHitCollection> hSrcHits;
  evt.getByToken(hSrc_, hSrcHits);
  const TrackingRecHitCollection& srcHits(*hSrcHits);

  // looking for the beam spot
  edm::Handle<reco::BeamSpot> hBsp;
  evt.getByToken(beamspot_, hBsp);
  const reco::BeamSpot& vertexBeamSpot(*hBsp);

  // Select good primary vertices for use in subsequent track selection
  edm::Handle<reco::VertexCollection> hVtx;
  if (useVertices_) {
    evt.getByToken(vertices_, hVtx);
    if (hVtx.failedToGet())
      edm::LogWarning("MultiTrackSelector") << "could not get Vertex collection";
  }

  unsigned int trkSize = srcTracks.size();
  std::vector<int> selTracksSave(qualityToSet_.size() * trkSize, 0);

  std::vector<Point> points;
  std::vector<float> vterr, vzerr;
  if (useVertices_)
    selectVertices(0, *hVtx, points, vterr, vzerr);
  //auto vtxP = points.empty() ? vertexBeamSpot.position() : points[0]; // rare, very rare, still happens!
  for (unsigned int i = 0; i < qualityToSet_.size(); i++) {
    std::vector<float> mvaVals_(srcTracks.size(), -99.f);
    processMVA(evt, es, vertexBeamSpot, *(hVtx.product()), i, mvaVals_, i == 0 ? true : false);
    std::vector<int> selTracks(trkSize, 0);
    auto selTracksValueMap = std::make_unique<edm::ValueMap<int>>();
    edm::ValueMap<int>::Filler filler(*selTracksValueMap);

    if (useVertices_)
      selectVertices(i, *hVtx, points, vterr, vzerr);

    // Loop over tracks
    size_t current = 0;
    for (TrackCollection::const_iterator it = srcTracks.begin(), ed = srcTracks.end(); it != ed; ++it, ++current) {
      const Track& trk = *it;
      // Check if this track passes cuts

      LogTrace("TrackSelection") << "ready to check track with pt=" << trk.pt();

      //already removed
      bool ok = true;
      if (preFilter_[i] < i && selTracksSave[preFilter_[i] * trkSize + current] < 0) {
        selTracks[current] = -1;
        ok = false;
        if (!keepAllTracks_[i])
          continue;
      } else {
        float mvaVal = 0;
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
    for (auto& q : selTracks)
      q = std::max(q, 0);
    auto quals = std::make_unique<QualityMaskCollection>(selTracks.begin(), selTracks.end());
    evt.put(std::move(quals), name_[i]);
  }
}

bool MultiTrackSelector::select(unsigned int tsNum,
                                const reco::BeamSpot& vertexBeamSpot,
                                const TrackingRecHitCollection& recHits,
                                const reco::Track& tk,
                                const std::vector<Point>& points,
                                std::vector<float>& vterr,
                                std::vector<float>& vzerr,
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
    if (useMVAonly_[tsNum])
      return mvaVal > min_MVA_[tsNum];
    if (mvaVal < min_MVA_[tsNum])
      return false;
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
    const TrackingRecHit& hit = recHits[ith];
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
  float lostMidFrac = tk.numberOfLostHits() / (tk.numberOfValidHits() + tk.numberOfLostHits());
  if (lostMidFrac > max_lostHitFraction_[tsNum])
    return false;

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

void MultiTrackSelector::selectVertices(unsigned int tsNum,
                                        const reco::VertexCollection& vtxs,
                                        std::vector<Point>& points,
                                        std::vector<float>& vterr,
                                        std::vector<float>& vzerr) const {
  // Select good primary vertices
  using namespace reco;
  int32_t toTake = vtxNumber_[tsNum];
  for (VertexCollection::const_iterator it = vtxs.begin(), ed = vtxs.end(); it != ed; ++it) {
    LogDebug("SelectVertex") << " select vertex with z position " << it->z() << " " << it->chi2() << " " << it->ndof()
                             << " " << TMath::Prob(it->chi2(), static_cast<int32_t>(it->ndof()));
    Vertex vtx = *it;
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

void MultiTrackSelector::processMVA(edm::Event& evt,
                                    const edm::EventSetup& es,
                                    const reco::BeamSpot& beamspot,
                                    const reco::VertexCollection& vertices,
                                    int selIndex,
                                    std::vector<float>& mvaVals_,
                                    bool writeIt) const {
  using namespace std;
  using namespace edm;
  using namespace reco;

  // Get tracks
  Handle<TrackCollection> hSrcTrack;
  evt.getByToken(src_, hSrcTrack);
  const TrackCollection& srcTracks(*hSrcTrack);
  RefToBaseProd<Track> rtbpTrackCollection(hSrcTrack);
  assert(mvaVals_.size() == srcTracks.size());

  // get hits in track..
  Handle<TrackingRecHitCollection> hSrcHits;
  evt.getByToken(hSrc_, hSrcHits);
  const TrackingRecHitCollection& srcHits(*hSrcHits);

  auto mvaValValueMap = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler mvaFiller(*mvaValValueMap);

  if (!useAnyMVA_ && writeIt) {
    // mvaVals_ already initalized...
    mvaFiller.insert(hSrcTrack, mvaVals_.begin(), mvaVals_.end());
    mvaFiller.fill();
    evt.put(std::move(mvaValValueMap), "MVAVals");
    auto mvas = std::make_unique<MVACollection>(mvaVals_.begin(), mvaVals_.end());
    evt.put(std::move(mvas), "MVAValues");
    return;
  }

  if (!useMVA_[selIndex] && !writeIt)
    return;

  size_t current = 0;
  for (TrackCollection::const_iterator it = srcTracks.begin(), ed = srcTracks.end(); it != ed; ++it, ++current) {
    const Track& trk = *it;
    RefToBase<Track> trackRef(rtbpTrackCollection, current);
    auto tmva_ndof_ = trk.ndof();
    auto tmva_nlayers_ = trk.hitPattern().trackerLayersWithMeasurement();
    auto tmva_nlayers3D_ =
        trk.hitPattern().pixelLayersWithMeasurement() + trk.hitPattern().numberOfValidStripLayersWithMonoAndStereo();
    auto tmva_nlayerslost_ = trk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
    float chi2n = trk.normalizedChi2();
    float chi2n_no1Dmod = chi2n;

    int count1dhits = 0;
    auto ith = trk.extra()->firstRecHit();
    auto edh = ith + trk.recHitsSize();
    for (; ith < edh; ++ith) {
      const TrackingRecHit& hit = srcHits[ith];
      if (hit.dimension() == 1)
        ++count1dhits;
    }
    if (count1dhits > 0) {
      float chi2 = trk.chi2();
      float ndof = trk.ndof();
      chi2n = (chi2 + count1dhits) / float(ndof + count1dhits);
    }
    auto tmva_chi2n_ = chi2n;
    auto tmva_chi2n_no1dmod_ = chi2n_no1Dmod;
    auto tmva_eta_ = trk.eta();
    auto tmva_relpterr_ = float(trk.ptError()) / std::max(float(trk.pt()), 0.000001f);
    auto tmva_nhits_ = trk.numberOfValidHits();
    int lostIn = trk.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_INNER_HITS);
    int lostOut = trk.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_OUTER_HITS);
    auto tmva_minlost_ = std::min(lostIn, lostOut);
    auto tmva_lostmidfrac_ = trk.numberOfLostHits() / (trk.numberOfValidHits() + trk.numberOfLostHits());
    auto tmva_absd0_ = fabs(-trk.dxy(beamspot.position()));
    auto tmva_absdz_ = fabs(trk.dz(beamspot.position()));
    Point bestVertex = getBestVertex(trackRef, vertices);
    auto tmva_absd0PV_ = fabs(trk.dxy(bestVertex));
    auto tmva_absdzPV_ = fabs(trk.dz(bestVertex));
    auto tmva_pt_ = trk.pt();

    GBRForest const* forest = forest_[selIndex];
    if (useForestFromDB_) {
      forest = &es.getData(forestToken_[selIndex]);
    }

    float gbrVals_[16];
    gbrVals_[0] = tmva_pt_;
    gbrVals_[1] = tmva_lostmidfrac_;
    gbrVals_[2] = tmva_minlost_;
    gbrVals_[3] = tmva_nhits_;
    gbrVals_[4] = tmva_relpterr_;
    gbrVals_[5] = tmva_eta_;
    gbrVals_[6] = tmva_chi2n_no1dmod_;
    gbrVals_[7] = tmva_chi2n_;
    gbrVals_[8] = tmva_nlayerslost_;
    gbrVals_[9] = tmva_nlayers3D_;
    gbrVals_[10] = tmva_nlayers_;
    gbrVals_[11] = tmva_ndof_;
    gbrVals_[12] = tmva_absd0PV_;
    gbrVals_[13] = tmva_absdzPV_;
    gbrVals_[14] = tmva_absdz_;
    gbrVals_[15] = tmva_absd0_;

    if (mvaType_[selIndex] == "Prompt") {
      auto gbrVal = forest->GetClassifier(gbrVals_);
      mvaVals_[current] = gbrVal;
    } else {
      float detachedGbrVals_[12];
      for (int jjj = 0; jjj < 12; jjj++)
        detachedGbrVals_[jjj] = gbrVals_[jjj];
      auto gbrVal = forest->GetClassifier(detachedGbrVals_);
      mvaVals_[current] = gbrVal;
    }
  }

  if (writeIt) {
    mvaFiller.insert(hSrcTrack, mvaVals_.begin(), mvaVals_.end());
    mvaFiller.fill();
    evt.put(std::move(mvaValValueMap), "MVAVals");
    auto mvas = std::make_unique<MVACollection>(mvaVals_.begin(), mvaVals_.end());
    evt.put(std::move(mvas), "MVAValues");
  }
}

MultiTrackSelector::Point MultiTrackSelector::getBestVertex(TrackBaseRef track, VertexCollection vertices) const {
  Point p(0, 0, -99999);
  Point p_dz(0, 0, -99999);
  float bestWeight = 0;
  float dzmin = 10000;
  bool weightMatch = false;
  for (auto const& vertex : vertices) {
    float w = vertex.trackWeight(track);
    const Point& v_pos = vertex.position();
    if (w > bestWeight) {
      p = v_pos;
      bestWeight = w;
      weightMatch = true;
    }
    float dz = fabs(track.get()->dz(v_pos));
    if (dz < dzmin) {
      p_dz = v_pos;
      dzmin = dz;
    }
  }
  if (weightMatch)
    return p;
  else
    return p_dz;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MultiTrackSelector);
