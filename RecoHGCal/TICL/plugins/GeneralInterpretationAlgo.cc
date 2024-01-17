#include "RecoHGCal/TICL/interface/TICLInterpretationAlgoBase.h"
#include "RecoHGCal/TICL/plugins/GeneralInterpretationAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

using namespace ticl;

using Vector = ticl::Trackster::Vector;

GeneralInterpretationAlgo::~GeneralInterpretationAlgo(){};

GeneralInterpretationAlgo::GeneralInterpretationAlgo(const edm::ParameterSet &conf, edm::ConsumesCollector cc)
    : TICLInterpretationAlgoBase(conf, cc),
      del_tk_ts_layer1_(conf.getParameter<double>("delta_tk_ts_layer1")),
      del_tk_ts_int_(conf.getParameter<double>("delta_tk_ts_interface")),
      del_ts_em_had_(conf.getParameter<double>("delta_ts_em_had")),
      del_ts_had_had_(conf.getParameter<double>("delta_ts_had_had")) {}

void GeneralInterpretationAlgo::initialize(const HGCalDDDConstants *hgcons,
                                           const hgcal::RecHitTools rhtools,
                                           const edm::ESHandle<MagneticField> bfieldH,
                                           const edm::ESHandle<Propagator> propH) {
  hgcons_ = hgcons;
  rhtools_ = rhtools;
  buildLayers();

  bfield_ = bfieldH;
  propagator_ = propH;
}

void GeneralInterpretationAlgo::buildLayers() {
  // build disks at HGCal front & EM-Had interface for track propagation

  float zVal = hgcons_->waferZ(1, true);
  std::pair<float, float> rMinMax = hgcons_->rangeR(zVal, true);

  float zVal_interface = rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z();
  std::pair<float, float> rMinMax_interface = hgcons_->rangeR(zVal_interface, true);

  for (int iSide = 0; iSide < 2; ++iSide) {
    float zSide = (iSide == 0) ? (-1. * zVal) : zVal;
    firstDisk_[iSide] =
        std::make_unique<GeomDet>(Disk::build(Disk::PositionType(0, 0, zSide),
                                              Disk::RotationType(),
                                              SimpleDiskBounds(rMinMax.first, rMinMax.second, zSide - 0.5, zSide + 0.5))
                                      .get());

    zSide = (iSide == 0) ? (-1. * zVal_interface) : zVal_interface;
    interfaceDisk_[iSide] = std::make_unique<GeomDet>(
        Disk::build(Disk::PositionType(0, 0, zSide),
                    Disk::RotationType(),
                    SimpleDiskBounds(rMinMax_interface.first, rMinMax_interface.second, zSide - 0.5, zSide + 0.5))
            .get());
  }
}
Vector GeneralInterpretationAlgo::propagateTrackster(const Trackster &t,
                                                     const unsigned idx,
                                                     float zVal,
                                                     std::array<TICLLayerTile, 2> &tracksterTiles) {
  // needs only the positive Z co-ordinate of the surface to propagate to
  // the correct sign is calculated inside according to the barycenter of trackster
  Vector const &baryc = t.barycenter();
  Vector directnv = t.eigenvectors(0);

  // barycenter as direction for tracksters w/ poor PCA
  // propagation still done to get the cartesian coords
  // which are anyway converted to eta, phi in linking
  // -> can be simplified later

  //FP: disable PCA propagation for the moment and fallback to barycenter position
  // if (t.eigenvalues()[0] / t.eigenvalues()[1] < 20)
  directnv = baryc.unit();
  zVal *= (baryc.Z() > 0) ? 1 : -1;
  float par = (zVal - baryc.Z()) / directnv.Z();
  float xOnSurface = par * directnv.X() + baryc.X();
  float yOnSurface = par * directnv.Y() + baryc.Y();
  Vector tPoint(xOnSurface, yOnSurface, zVal);
  if (tPoint.Eta() > 0) {
    tracksterTiles[1].fill(tPoint.Eta(), tPoint.Phi(), idx);
  } else if (tPoint.Eta() < 0) {
    tracksterTiles[0].fill(tPoint.Eta(), tPoint.Phi(), idx);
  }

  return tPoint;
}

void GeneralInterpretationAlgo::findTrackstersInWindow(const MultiVectorManager<Trackster> &tracksters,
                                                       const std::vector<std::pair<Vector, unsigned>> &seedingCollection,
                                                       const std::array<TICLLayerTile, 2> &tracksterTiles,
                                                       const std::vector<Vector> &tracksterPropPoints,
                                                       const float delta,
                                                       unsigned trackstersSize,
                                                       std::vector<std::vector<unsigned>> &resultCollection,
                                                       bool useMask = false) {
  // Finds tracksters in tracksterTiles within an eta-phi window
  // (given by delta) of the objects (track/trackster) in the seedingCollection.
  // Element i in resultCollection is the vector of trackster
  // indices found close to the i-th object in the seedingCollection.
  // If specified, Tracksters are masked once found as close to an object.
  std::vector<int> mask(trackstersSize, 0);
  const float delta2 = delta * delta;

  for (auto &i : seedingCollection) {
    float seed_eta = i.first.Eta();
    float seed_phi = i.first.Phi();
    unsigned seedId = i.second;
    auto sideZ = seed_eta > 0;  //forward or backward region
    const TICLLayerTile &tile = tracksterTiles[sideZ];
    float eta_min = std::max(std::fabs(seed_eta) - delta, (float)TileConstants::minEta);
    float eta_max = std::min(std::fabs(seed_eta) + delta, (float)TileConstants::maxEta);

    // get range of bins touched by delta
    std::array<int, 4> search_box = tile.searchBoxEtaPhi(eta_min, eta_max, seed_phi - delta, seed_phi + delta);

    std::vector<unsigned> in_delta;
    // std::vector<float> distances2;
    std::vector<float> energies;
    for (int eta_i = search_box[0]; eta_i <= search_box[1]; ++eta_i) {
      for (int phi_i = search_box[2]; phi_i <= search_box[3]; ++phi_i) {
        const auto &in_tile = tile[tile.globalBin(eta_i, (phi_i % TileConstants::nPhiBins))];
        for (const unsigned &t_i : in_tile) {
          // calculate actual distances of tracksters to the seed for a more accurate cut
          auto sep2 = (tracksterPropPoints[t_i].Eta() - seed_eta) * (tracksterPropPoints[t_i].Eta() - seed_eta) +
                      (tracksterPropPoints[t_i].Phi() - seed_phi) * (tracksterPropPoints[t_i].Phi() - seed_phi);
          if (sep2 < delta2) {
            in_delta.push_back(t_i);
            // distances2.push_back(sep2);
            energies.push_back(tracksters[t_i].raw_energy());
          }
        }
      }
    }

    // sort tracksters found in ascending order of their distances from the seed
    std::vector<unsigned> indices(in_delta.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](unsigned i, unsigned j) { return energies[i] > energies[j]; });

    // push back sorted tracksters in the result collection
    for (const unsigned &index : indices) {
      const auto &t_i = in_delta[index];
      if (!mask[t_i]) {
        resultCollection[seedId].push_back(t_i);
        if (useMask)
          mask[t_i] = 1;
      }
    }

  }  // seeding collection loop
}

bool GeneralInterpretationAlgo::timeAndEnergyCompatible(float &total_raw_energy,
                                                        const reco::Track &track,
                                                        const Trackster &trackster,
                                                        const float &tkT,
                                                        const float &tkTErr,
                                                        const float &tkBeta,
                                                        const GlobalPoint &tkMtdPos,
                                                        bool useMTDTiming) {
  float threshold = std::min(0.2 * trackster.raw_energy(), 10.0);
  bool energyCompatible = (total_raw_energy + trackster.raw_energy() < track.p() + threshold);

  if (!useMTDTiming)
    return energyCompatible;

  // compatible if trackster time is within 3sigma of
  // track time; compatible if either: no time assigned
  // to trackster or track

  float tsT = trackster.time();
  float tsTErr = trackster.timeError();

  bool timeCompatible = false;
  if (tsT == -99. or tkTErr == -1)
    timeCompatible = true;
  else {
    const auto &barycenter = trackster.barycenter();

    const auto deltaSoverV = std::sqrt((barycenter.x() - tkMtdPos.x()) * (barycenter.x() - tkMtdPos.x()) +
                                       (barycenter.y() - tkMtdPos.y()) * (barycenter.y() - tkMtdPos.y()) +
                                       (barycenter.z() - tkMtdPos.z()) * (barycenter.z() - tkMtdPos.z())) /
                             (tkBeta * 29.9792458);

    const auto deltaT = tsT - tkT;

    //  timeCompatible = (std::abs(deltaSoverV - deltaT) < maxDeltaT_ * sqrt(tsTErr * tsTErr + tkTErr * tkTErr));
    // use sqrt(2) * error on the track for the total error, because the time of the trackster is too small
    timeCompatible = std::abs(deltaSoverV - deltaT) < maxDeltaT_ * std::sqrt(tsTErr * tsTErr + tkTErr * tkTErr);
  }

  if (TICLInterpretationAlgoBase::algo_verbosity_ > VerbosityLevel::Advanced) {
    if (!(energyCompatible))
      LogDebug("GeneralInterpretationAlgo")
          << "energy incompatible : track p " << track.p() << " trackster energy " << trackster.raw_energy() << "\n"
          << "                      total_raw_energy " << total_raw_energy << " greater than track p + threshold "
          << track.p() + threshold << "\n";
    if (!(timeCompatible))
      LogDebug("GeneralInterpretationAlgo") << "time incompatible : track time " << tkT << " +/- " << tkTErr
                                            << " trackster time " << tsT << " +/- " << tsTErr << "\n";
  }

  return energyCompatible && timeCompatible;
}

void GeneralInterpretationAlgo::makeCandidates(const Inputs &input,
                                               const TrackTimingInformation &inputTiming,
                                               std::vector<Trackster> &resultTracksters,
                                               std::vector<int> &resultCandidate) {
  bool useMTDTiming = inputTiming.tkTime_h.isValid();
  std::cout << "GeneralInterpretationAlgo " << std::endl;
  const auto tkH = input.tracksHandle;
  const auto maskTracks = input.maskedTracks;
  const auto &tracks = *tkH;
  const auto &tracksters = input.tracksters;

  auto bFieldProd = bfield_.product();
  const Propagator &prop = (*propagator_);

  // propagated point collections
  // elements in the propagated points collecions are used
  // to look for potential linkages in the appropriate tiles
  std::vector<std::pair<Vector, unsigned>> trackPColl;     // propagated track points and index of track in collection
  std::vector<std::pair<Vector, unsigned>> tkPropIntColl;  // tracks propagated to lastLayerEE

  trackPColl.reserve(tracks.size());
  tkPropIntColl.reserve(tracks.size());

  std::array<TICLLayerTile, 2> tracksterPropTiles = {};  // all Tracksters, propagated to layer 1
  std::array<TICLLayerTile, 2> tsPropIntTiles = {};      // all Tracksters, propagated to lastLayerEE

  if (TICLInterpretationAlgoBase::algo_verbosity_ > VerbosityLevel::Advanced)
    LogDebug("GeneralInterpretationAlgo") << "------- Geometric Linking ------- \n";

  // Propagate tracks
  std::vector<unsigned> candidateTrackIds;
  candidateTrackIds.reserve(tracks.size());
  for (unsigned i = 0; i < tracks.size(); ++i) {
    if (!maskTracks.at(i))
      continue;
    candidateTrackIds.push_back(i);
  }

  std::sort(candidateTrackIds.begin(), candidateTrackIds.end(), [&](unsigned i, unsigned j) {
    return tracks[i].p() > tracks[j].p();
  });

  for (auto const i : candidateTrackIds) {
    const auto &tk = tracks[i];
    int iSide = int(tk.eta() > 0);
    const auto &fts = trajectoryStateTransform::outerFreeState((tk), bFieldProd);
    // to the HGCal front
    const auto &tsos = prop.propagate(fts, firstDisk_[iSide]->surface());
    if (tsos.isValid()) {
      Vector trackP(tsos.globalPosition().x(), tsos.globalPosition().y(), tsos.globalPosition().z());
      trackPColl.emplace_back(trackP, i);
    }
    // to lastLayerEE
    const auto &tsos_int = prop.propagate(fts, interfaceDisk_[iSide]->surface());
    if (tsos_int.isValid()) {
      Vector trackP(tsos_int.globalPosition().x(), tsos_int.globalPosition().y(), tsos_int.globalPosition().z());
      tkPropIntColl.emplace_back(trackP, i);
    }
  }  // Tracks
  tkPropIntColl.shrink_to_fit();
  trackPColl.shrink_to_fit();
  candidateTrackIds.shrink_to_fit();

  // Propagate tracksters

  // Record postions of all tracksters propagated to layer 1 and lastLayerEE,
  // to be used later for distance calculation in the link finding stage
  // indexed by trackster index in event collection
  std::vector<Vector> tsAllProp;
  std::vector<Vector> tsAllPropInt;
  tsAllProp.reserve(tracksters.size());
  tsAllPropInt.reserve(tracksters.size());
  // Propagate tracksters

  for (unsigned i = 0; i < tracksters.size(); ++i) {
    const auto &t = tracksters[i];
    if (TICLInterpretationAlgoBase::algo_verbosity_ > VerbosityLevel::Advanced)
      LogDebug("GeneralInterpretationAlgo")
          << "trackster " << i << " - eta " << t.barycenter().eta() << " phi " << t.barycenter().phi() << " time "
          << t.time() << " energy " << t.raw_energy() << "\n";

    // to HGCal front
    float zVal = hgcons_->waferZ(1, true);
    auto tsP = propagateTrackster(t, i, zVal, tracksterPropTiles);
    tsAllProp.emplace_back(tsP);

    // to lastLayerEE
    zVal = rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z();
    tsP = propagateTrackster(t, i, zVal, tsPropIntTiles);
    tsAllPropInt.emplace_back(tsP);

  }  // TS

  std::vector<std::vector<unsigned>> tsNearTk(tracks.size());
  findTrackstersInWindow(
      tracksters, trackPColl, tracksterPropTiles, tsAllProp, del_tk_ts_layer1_, tracksters.size(), tsNearTk);
  // step 4: tracks -> all tracksters, at lastLayerEE

  std::vector<std::vector<unsigned>> tsNearTkAtInt(tracks.size());
  findTrackstersInWindow(
      tracksters, tkPropIntColl, tsPropIntTiles, tsAllPropInt, del_tk_ts_int_, tracksters.size(), tsNearTkAtInt);

  std::vector<unsigned int> chargedHadronsFromTk;
  std::vector<std::vector<unsigned int>> trackstersInTrackIndices;
  trackstersInTrackIndices.resize(tracks.size());

  std::vector<bool> chargedMask(tracksters.size(), true);
  for (unsigned &i : candidateTrackIds) {
    if (tsNearTk[i].empty() && tsNearTkAtInt[i].empty()) {  // nothing linked to track, make charged hadrons
      continue;
    }

    std::vector<unsigned int> chargedCandidate;
    float total_raw_energy = 0.f;

    auto tkRef = reco::TrackRef(tkH, i);
    float track_time = 0.f;
    float track_timeErr = 0.f;
    float track_beta = 0.f;
    GlobalPoint track_MtdPos{0.f, 0.f, 0.f};
    if (useMTDTiming) {
      track_time = (*inputTiming.tkTime_h)[tkRef];
      track_timeErr = (*inputTiming.tkTimeErr_h)[tkRef];
      track_beta = (*inputTiming.tkBeta_h)[tkRef];
      track_MtdPos = (*inputTiming.tkMtdPos_h)[tkRef];
    }

    for (auto const tsIdx : tsNearTk[i]) {
      if (chargedMask[tsIdx] && timeAndEnergyCompatible(total_raw_energy,
                                                        tracks[i],
                                                        tracksters[tsIdx],
                                                        track_time,
                                                        track_timeErr,
                                                        track_beta,
                                                        track_MtdPos,
                                                        useMTDTiming)) {
        chargedCandidate.push_back(tsIdx);
        chargedMask[tsIdx] = false;
      }
    }
    for (const unsigned tsIdx : tsNearTkAtInt[i]) {  // do the same for tk -> ts links at the interface
      if (chargedMask[tsIdx] && timeAndEnergyCompatible(total_raw_energy,
                                                        tracks[i],
                                                        tracksters[tsIdx],
                                                        track_time,
                                                        track_timeErr,
                                                        track_beta,
                                                        track_MtdPos,
                                                        useMTDTiming)) {
        chargedCandidate.push_back(tsIdx);
        chargedMask[tsIdx] = false;
      }
    }
    trackstersInTrackIndices[i] = chargedCandidate;
  }

  for (size_t iTrack = 0; iTrack < trackstersInTrackIndices.size(); iTrack++) {
    if (!trackstersInTrackIndices[iTrack].empty()) {
      Trackster outTrackster;
      for (auto const tracksterId : trackstersInTrackIndices[iTrack]) {
        //maskTracksters[tracksterId] = 0;
        outTrackster.mergeTracksters(input.tracksters[tracksterId]);
      }
      resultCandidate[iTrack] = resultTracksters.size();
      resultTracksters.push_back(outTrackster);
    }
  }

  for (size_t iTrackster = 0; iTrackster < input.tracksters.size(); iTrackster++) {
    if (chargedMask[iTrackster]) {
      resultTracksters.push_back(input.tracksters[iTrackster]);
    }
  }
};

void GeneralInterpretationAlgo::fillPSetDescription(edm::ParameterSetDescription &desc) {
  desc.add<std::string>("cutTk",
                        "1.48 < abs(eta) < 3.0 && pt > 1. && quality(\"highPurity\") && "
                        "hitPattern().numberOfLostHits(\"MISSING_OUTER_HITS\") < 5");
  desc.add<double>("delta_tk_ts_layer1", 0.02);
  desc.add<double>("delta_tk_ts_interface", 0.03);
  desc.add<double>("delta_ts_em_had", 0.03);
  desc.add<double>("delta_ts_had_had", 0.03);
  TICLInterpretationAlgoBase::fillPSetDescription(desc);
}
