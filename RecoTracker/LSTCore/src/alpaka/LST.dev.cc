#include "RecoTracker/LSTCore/interface/alpaka/LST.h"

#include "Event.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE::lst;

#include "Math/Vector3D.h"
#include "Math/VectorUtil.h"
using XYZVector = ROOT::Math::XYZVector;

namespace {
  XYZVector calculateR3FromPCA(const XYZVector& p3, float dxy, float dz) {
    const float pt = p3.rho();
    const float p = p3.r();
    const float vz = dz * pt * pt / p / p;

    const float vx = -dxy * p3.y() / pt - p3.x() / p * p3.z() / p * dz;
    const float vy = dxy * p3.x() / pt - p3.y() / p * p3.z() / p * dz;
    return {vx, vy, vz};
  }
}  // namespace

void LST::prepareInput(std::vector<float> const& see_px,
                       std::vector<float> const& see_py,
                       std::vector<float> const& see_pz,
                       std::vector<float> const& see_dxy,
                       std::vector<float> const& see_dz,
                       std::vector<float> const& see_ptErr,
                       std::vector<float> const& see_etaErr,
                       std::vector<float> const& see_stateTrajGlbX,
                       std::vector<float> const& see_stateTrajGlbY,
                       std::vector<float> const& see_stateTrajGlbZ,
                       std::vector<float> const& see_stateTrajGlbPx,
                       std::vector<float> const& see_stateTrajGlbPy,
                       std::vector<float> const& see_stateTrajGlbPz,
                       std::vector<int> const& see_q,
                       std::vector<std::vector<int>> const& see_hitIdx,
                       std::vector<unsigned int> const& ph2_detId,
                       std::vector<float> const& ph2_x,
                       std::vector<float> const& ph2_y,
                       std::vector<float> const& ph2_z) {
  unsigned int count = 0;
  auto n_see = see_stateTrajGlbPx.size();
  std::vector<float> px_vec;
  px_vec.reserve(n_see);
  std::vector<float> py_vec;
  py_vec.reserve(n_see);
  std::vector<float> pz_vec;
  pz_vec.reserve(n_see);
  std::vector<unsigned int> hitIndices_vec0;
  hitIndices_vec0.reserve(n_see);
  std::vector<unsigned int> hitIndices_vec1;
  hitIndices_vec1.reserve(n_see);
  std::vector<unsigned int> hitIndices_vec2;
  hitIndices_vec2.reserve(n_see);
  std::vector<unsigned int> hitIndices_vec3;
  hitIndices_vec3.reserve(n_see);
  std::vector<float> ptIn_vec;
  ptIn_vec.reserve(n_see);
  std::vector<float> ptErr_vec;
  ptErr_vec.reserve(n_see);
  std::vector<float> etaErr_vec;
  etaErr_vec.reserve(n_see);
  std::vector<float> eta_vec;
  eta_vec.reserve(n_see);
  std::vector<float> phi_vec;
  phi_vec.reserve(n_see);
  std::vector<int> charge_vec;
  charge_vec.reserve(n_see);
  std::vector<unsigned int> seedIdx_vec;
  seedIdx_vec.reserve(n_see);
  std::vector<float> deltaPhi_vec;
  deltaPhi_vec.reserve(n_see);
  std::vector<float> trkX = ph2_x;
  std::vector<float> trkY = ph2_y;
  std::vector<float> trkZ = ph2_z;
  std::vector<unsigned int> hitId = ph2_detId;
  std::vector<unsigned int> hitIdxs(ph2_detId.size());

  std::vector<int> superbin_vec;
  std::vector<PixelType> pixelType_vec;
  std::vector<char> isQuad_vec;
  std::iota(hitIdxs.begin(), hitIdxs.end(), 0);
  const int hit_size = trkX.size();

  for (size_t iSeed = 0; iSeed < n_see; iSeed++) {
    XYZVector p3LH(see_stateTrajGlbPx[iSeed], see_stateTrajGlbPy[iSeed], see_stateTrajGlbPz[iSeed]);
    float ptIn = p3LH.rho();
    float eta = p3LH.eta();
    float ptErr = see_ptErr[iSeed];

    if ((ptIn > 0.8 - 2 * ptErr)) {
      XYZVector r3LH(see_stateTrajGlbX[iSeed], see_stateTrajGlbY[iSeed], see_stateTrajGlbZ[iSeed]);
      XYZVector p3PCA(see_px[iSeed], see_py[iSeed], see_pz[iSeed]);
      XYZVector r3PCA(calculateR3FromPCA(p3PCA, see_dxy[iSeed], see_dz[iSeed]));

      // The charge could be used directly in the line below
      float pixelSegmentDeltaPhiChange = ROOT::Math::VectorUtil::DeltaPhi(p3LH, r3LH);
      float etaErr = see_etaErr[iSeed];
      float px = p3LH.x();
      float py = p3LH.y();
      float pz = p3LH.z();

      int charge = see_q[iSeed];
      PixelType pixtype = PixelType::kInvalid;

      if (ptIn >= 2.0)
        pixtype = PixelType::kHighPt;
      else if (ptIn >= (0.8 - 2 * ptErr) and ptIn < 2.0) {
        if (pixelSegmentDeltaPhiChange >= 0)
          pixtype = PixelType::kLowPtPosCurv;
        else
          pixtype = PixelType::kLowPtNegCurv;
      } else
        continue;

      unsigned int hitIdx0 = hit_size + count;
      count++;
      unsigned int hitIdx1 = hit_size + count;
      count++;
      unsigned int hitIdx2 = hit_size + count;
      count++;
      unsigned int hitIdx3;
      if (see_hitIdx[iSeed].size() <= 3)
        hitIdx3 = hitIdx2;
      else {
        hitIdx3 = hit_size + count;
        count++;
      }

      trkX.push_back(r3PCA.x());
      trkY.push_back(r3PCA.y());
      trkZ.push_back(r3PCA.z());
      trkX.push_back(p3PCA.rho());
      float p3PCA_Eta = p3PCA.eta();
      trkY.push_back(p3PCA_Eta);
      float p3PCA_Phi = p3PCA.phi();
      trkZ.push_back(p3PCA_Phi);
      trkX.push_back(r3LH.x());
      trkY.push_back(r3LH.y());
      trkZ.push_back(r3LH.z());
      hitId.push_back(1);
      hitId.push_back(1);
      hitId.push_back(1);
      if (see_hitIdx[iSeed].size() > 3) {
        trkX.push_back(r3LH.x());
        trkY.push_back(see_dxy[iSeed]);
        trkZ.push_back(see_dz[iSeed]);
        hitId.push_back(1);
      }
      px_vec.push_back(px);
      py_vec.push_back(py);
      pz_vec.push_back(pz);

      hitIndices_vec0.push_back(hitIdx0);
      hitIndices_vec1.push_back(hitIdx1);
      hitIndices_vec2.push_back(hitIdx2);
      hitIndices_vec3.push_back(hitIdx3);
      ptIn_vec.push_back(ptIn);
      ptErr_vec.push_back(ptErr);
      etaErr_vec.push_back(etaErr);
      eta_vec.push_back(eta);
      float phi = p3LH.phi();
      phi_vec.push_back(phi);
      charge_vec.push_back(charge);
      seedIdx_vec.push_back(iSeed);
      deltaPhi_vec.push_back(pixelSegmentDeltaPhiChange);

      hitIdxs.push_back(see_hitIdx[iSeed][0]);
      hitIdxs.push_back(see_hitIdx[iSeed][1]);
      hitIdxs.push_back(see_hitIdx[iSeed][2]);
      char isQuad = false;
      if (see_hitIdx[iSeed].size() > 3) {
        isQuad = true;
        hitIdxs.push_back(see_hitIdx[iSeed][3]);
      }
      float neta = 25.;
      float nphi = 72.;
      float nz = 25.;
      int etabin = (p3PCA_Eta + 2.6) / ((2 * 2.6) / neta);
      int phibin = (p3PCA_Phi + 3.14159265358979323846) / ((2. * 3.14159265358979323846) / nphi);
      int dzbin = (see_dz[iSeed] + 30) / (2 * 30 / nz);
      int isuperbin = (nz * nphi) * etabin + (nz)*phibin + dzbin;
      superbin_vec.push_back(isuperbin);
      pixelType_vec.push_back(pixtype);
      isQuad_vec.push_back(isQuad);
    }
  }

  in_trkX_ = trkX;
  in_trkY_ = trkY;
  in_trkZ_ = trkZ;
  in_hitId_ = hitId;
  in_hitIdxs_ = hitIdxs;
  in_hitIndices_vec0_ = hitIndices_vec0;
  in_hitIndices_vec1_ = hitIndices_vec1;
  in_hitIndices_vec2_ = hitIndices_vec2;
  in_hitIndices_vec3_ = hitIndices_vec3;
  in_deltaPhi_vec_ = deltaPhi_vec;
  in_ptIn_vec_ = ptIn_vec;
  in_ptErr_vec_ = ptErr_vec;
  in_px_vec_ = px_vec;
  in_py_vec_ = py_vec;
  in_pz_vec_ = pz_vec;
  in_eta_vec_ = eta_vec;
  in_etaErr_vec_ = etaErr_vec;
  in_phi_vec_ = phi_vec;
  in_charge_vec_ = charge_vec;
  in_seedIdx_vec_ = seedIdx_vec;
  in_superbin_vec_ = superbin_vec;
  in_pixelType_vec_ = pixelType_vec;
  in_isQuad_vec_ = isQuad_vec;
}

std::vector<unsigned int> LST::getHitIdxs(short trackCandidateType,
                                          unsigned int TCIdx,
                                          unsigned int const* TCHitIndices,
                                          unsigned int const* hitIndices) {
  std::vector<unsigned int> hits;

  unsigned int maxNHits = 0;
  if (trackCandidateType == 7)
    maxNHits = Params_pT5::kHits;  // pT5
  else if (trackCandidateType == 5)
    maxNHits = Params_pT3::kHits;  // pT3
  else if (trackCandidateType == 4)
    maxNHits = Params_T5::kHits;  // T5
  else if (trackCandidateType == 8)
    maxNHits = Params_pLS::kHits;  // pLS

  for (unsigned int i = 0; i < maxNHits; i++) {
    unsigned int hitIdxInGPU = TCHitIndices[Params_pT5::kHits * TCIdx + i];
    unsigned int hitIdx =
        (trackCandidateType == 8)
            ? hitIdxInGPU
            : hitIndices[hitIdxInGPU];  // Hit indices are stored differently in the standalone for pLS.

    // For p objects, the 3rd and 4th hit maybe the same,
    // due to the way pLS hits are stored in the standalone.
    // This is because pixel seeds can be either triplets or quadruplets.
    if (trackCandidateType != 4 && hits.size() == 3 && hits.back() == hitIdx)  // Remove duplicate 4th hits.
      continue;

    hits.push_back(hitIdx);
  }

  return hits;
}

void LST::getOutput(Event& event) {
  std::vector<std::vector<unsigned int>> tc_hitIdxs;
  std::vector<unsigned int> tc_len;
  std::vector<int> tc_seedIdx;
  std::vector<short> tc_trackCandidateType;

  HitsBuffer<alpaka::DevCpu>& hitsInGPU = event.getHitsInCMSSW(false);  // sync on next line
  TrackCandidates const* trackCandidates = event.getTrackCandidatesInCMSSW().data();

  unsigned int nTrackCandidates = *trackCandidates->nTrackCandidates;

  for (unsigned int idx = 0; idx < nTrackCandidates; idx++) {
    short trackCandidateType = trackCandidates->trackCandidateType[idx];
    std::vector<unsigned int> hit_idx =
        getHitIdxs(trackCandidateType, idx, trackCandidates->hitIndices, hitsInGPU.data()->idxs);

    tc_hitIdxs.push_back(hit_idx);
    tc_len.push_back(hit_idx.size());
    tc_seedIdx.push_back(trackCandidates->pixelSeedIndex[idx]);
    tc_trackCandidateType.push_back(trackCandidateType);
  }

  out_tc_hitIdxs_ = tc_hitIdxs;
  out_tc_len_ = tc_len;
  out_tc_seedIdx_ = tc_seedIdx;
  out_tc_trackCandidateType_ = tc_trackCandidateType;
}

void LST::run(Queue& queue,
              bool verbose,
              LSTESData<Device> const* deviceESData,
              std::vector<float> const& see_px,
              std::vector<float> const& see_py,
              std::vector<float> const& see_pz,
              std::vector<float> const& see_dxy,
              std::vector<float> const& see_dz,
              std::vector<float> const& see_ptErr,
              std::vector<float> const& see_etaErr,
              std::vector<float> const& see_stateTrajGlbX,
              std::vector<float> const& see_stateTrajGlbY,
              std::vector<float> const& see_stateTrajGlbZ,
              std::vector<float> const& see_stateTrajGlbPx,
              std::vector<float> const& see_stateTrajGlbPy,
              std::vector<float> const& see_stateTrajGlbPz,
              std::vector<int> const& see_q,
              std::vector<std::vector<int>> const& see_hitIdx,
              std::vector<unsigned int> const& ph2_detId,
              std::vector<float> const& ph2_x,
              std::vector<float> const& ph2_y,
              std::vector<float> const& ph2_z,
              bool no_pls_dupclean,
              bool tc_pls_triplets) {
  auto event = Event(verbose, queue, deviceESData);
  prepareInput(see_px,
               see_py,
               see_pz,
               see_dxy,
               see_dz,
               see_ptErr,
               see_etaErr,
               see_stateTrajGlbX,
               see_stateTrajGlbY,
               see_stateTrajGlbZ,
               see_stateTrajGlbPx,
               see_stateTrajGlbPy,
               see_stateTrajGlbPz,
               see_q,
               see_hitIdx,
               ph2_detId,
               ph2_x,
               ph2_y,
               ph2_z);

  event.addHitToEvent(in_trkX_, in_trkY_, in_trkZ_, in_hitId_, in_hitIdxs_);
  event.addPixelSegmentToEvent(in_hitIndices_vec0_,
                               in_hitIndices_vec1_,
                               in_hitIndices_vec2_,
                               in_hitIndices_vec3_,
                               in_deltaPhi_vec_,
                               in_ptIn_vec_,
                               in_ptErr_vec_,
                               in_px_vec_,
                               in_py_vec_,
                               in_pz_vec_,
                               in_eta_vec_,
                               in_etaErr_vec_,
                               in_phi_vec_,
                               in_charge_vec_,
                               in_seedIdx_vec_,
                               in_superbin_vec_,
                               in_pixelType_vec_,
                               in_isQuad_vec_);
  event.createMiniDoublets();
  if (verbose) {
    alpaka::wait(queue);  // event calls are asynchronous: wait before printing
    printf("# of Mini-doublets produced: %d\n", event.getNumberOfMiniDoublets());
    printf("# of Mini-doublets produced barrel layer 1: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(0));
    printf("# of Mini-doublets produced barrel layer 2: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(1));
    printf("# of Mini-doublets produced barrel layer 3: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(2));
    printf("# of Mini-doublets produced barrel layer 4: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(3));
    printf("# of Mini-doublets produced barrel layer 5: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(4));
    printf("# of Mini-doublets produced barrel layer 6: %d\n", event.getNumberOfMiniDoubletsByLayerBarrel(5));
    printf("# of Mini-doublets produced endcap layer 1: %d\n", event.getNumberOfMiniDoubletsByLayerEndcap(0));
    printf("# of Mini-doublets produced endcap layer 2: %d\n", event.getNumberOfMiniDoubletsByLayerEndcap(1));
    printf("# of Mini-doublets produced endcap layer 3: %d\n", event.getNumberOfMiniDoubletsByLayerEndcap(2));
    printf("# of Mini-doublets produced endcap layer 4: %d\n", event.getNumberOfMiniDoubletsByLayerEndcap(3));
    printf("# of Mini-doublets produced endcap layer 5: %d\n", event.getNumberOfMiniDoubletsByLayerEndcap(4));
  }

  event.createSegmentsWithModuleMap();
  if (verbose) {
    alpaka::wait(queue);  // event calls are asynchronous: wait before printing
    printf("# of Segments produced: %d\n", event.getNumberOfSegments());
    printf("# of Segments produced layer 1-2:  %d\n", event.getNumberOfSegmentsByLayerBarrel(0));
    printf("# of Segments produced layer 2-3:  %d\n", event.getNumberOfSegmentsByLayerBarrel(1));
    printf("# of Segments produced layer 3-4:  %d\n", event.getNumberOfSegmentsByLayerBarrel(2));
    printf("# of Segments produced layer 4-5:  %d\n", event.getNumberOfSegmentsByLayerBarrel(3));
    printf("# of Segments produced layer 5-6:  %d\n", event.getNumberOfSegmentsByLayerBarrel(4));
    printf("# of Segments produced endcap layer 1:  %d\n", event.getNumberOfSegmentsByLayerEndcap(0));
    printf("# of Segments produced endcap layer 2:  %d\n", event.getNumberOfSegmentsByLayerEndcap(1));
    printf("# of Segments produced endcap layer 3:  %d\n", event.getNumberOfSegmentsByLayerEndcap(2));
    printf("# of Segments produced endcap layer 4:  %d\n", event.getNumberOfSegmentsByLayerEndcap(3));
    printf("# of Segments produced endcap layer 5:  %d\n", event.getNumberOfSegmentsByLayerEndcap(4));
  }

  event.createTriplets();
  if (verbose) {
    alpaka::wait(queue);  // event calls are asynchronous: wait before printing
    printf("# of T3s produced: %d\n", event.getNumberOfTriplets());
    printf("# of T3s produced layer 1-2-3: %d\n", event.getNumberOfTripletsByLayerBarrel(0));
    printf("# of T3s produced layer 2-3-4: %d\n", event.getNumberOfTripletsByLayerBarrel(1));
    printf("# of T3s produced layer 3-4-5: %d\n", event.getNumberOfTripletsByLayerBarrel(2));
    printf("# of T3s produced layer 4-5-6: %d\n", event.getNumberOfTripletsByLayerBarrel(3));
    printf("# of T3s produced endcap layer 1-2-3: %d\n", event.getNumberOfTripletsByLayerEndcap(0));
    printf("# of T3s produced endcap layer 2-3-4: %d\n", event.getNumberOfTripletsByLayerEndcap(1));
    printf("# of T3s produced endcap layer 3-4-5: %d\n", event.getNumberOfTripletsByLayerEndcap(2));
    printf("# of T3s produced endcap layer 1: %d\n", event.getNumberOfTripletsByLayerEndcap(0));
    printf("# of T3s produced endcap layer 2: %d\n", event.getNumberOfTripletsByLayerEndcap(1));
    printf("# of T3s produced endcap layer 3: %d\n", event.getNumberOfTripletsByLayerEndcap(2));
    printf("# of T3s produced endcap layer 4: %d\n", event.getNumberOfTripletsByLayerEndcap(3));
    printf("# of T3s produced endcap layer 5: %d\n", event.getNumberOfTripletsByLayerEndcap(4));
  }

  event.createQuintuplets();
  if (verbose) {
    alpaka::wait(queue);  // event calls are asynchronous: wait before printing
    printf("# of Quintuplets produced: %d\n", event.getNumberOfQuintuplets());
    printf("# of Quintuplets produced layer 1-2-3-4-5-6: %d\n", event.getNumberOfQuintupletsByLayerBarrel(0));
    printf("# of Quintuplets produced layer 2: %d\n", event.getNumberOfQuintupletsByLayerBarrel(1));
    printf("# of Quintuplets produced layer 3: %d\n", event.getNumberOfQuintupletsByLayerBarrel(2));
    printf("# of Quintuplets produced layer 4: %d\n", event.getNumberOfQuintupletsByLayerBarrel(3));
    printf("# of Quintuplets produced layer 5: %d\n", event.getNumberOfQuintupletsByLayerBarrel(4));
    printf("# of Quintuplets produced layer 6: %d\n", event.getNumberOfQuintupletsByLayerBarrel(5));
    printf("# of Quintuplets produced endcap layer 1: %d\n", event.getNumberOfQuintupletsByLayerEndcap(0));
    printf("# of Quintuplets produced endcap layer 2: %d\n", event.getNumberOfQuintupletsByLayerEndcap(1));
    printf("# of Quintuplets produced endcap layer 3: %d\n", event.getNumberOfQuintupletsByLayerEndcap(2));
    printf("# of Quintuplets produced endcap layer 4: %d\n", event.getNumberOfQuintupletsByLayerEndcap(3));
    printf("# of Quintuplets produced endcap layer 5: %d\n", event.getNumberOfQuintupletsByLayerEndcap(4));
  }

  event.pixelLineSegmentCleaning(no_pls_dupclean);

  event.createPixelQuintuplets();
  if (verbose) {
    alpaka::wait(queue);  // event calls are asynchronous: wait before printing
    printf("# of Pixel Quintuplets produced: %d\n", event.getNumberOfPixelQuintuplets());
  }

  event.createPixelTriplets();
  if (verbose) {
    alpaka::wait(queue);  // event calls are asynchronous: wait before printing
    printf("# of Pixel T3s produced: %d\n", event.getNumberOfPixelTriplets());
  }

  event.createTrackCandidates(no_pls_dupclean, tc_pls_triplets);
  if (verbose) {
    alpaka::wait(queue);  // event calls are asynchronous: wait before printing
    printf("# of TrackCandidates produced: %d\n", event.getNumberOfTrackCandidates());
    printf("        # of Pixel TrackCandidates produced: %d\n", event.getNumberOfPixelTrackCandidates());
    printf("        # of pT5 TrackCandidates produced: %d\n", event.getNumberOfPT5TrackCandidates());
    printf("        # of pT3 TrackCandidates produced: %d\n", event.getNumberOfPT3TrackCandidates());
    printf("        # of pLS TrackCandidates produced: %d\n", event.getNumberOfPLSTrackCandidates());
    printf("        # of T5 TrackCandidates produced: %d\n", event.getNumberOfT5TrackCandidates());
  }

  getOutput(event);

  event.resetEventSync();
}
