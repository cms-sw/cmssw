#include "RecoTracker/LSTCore/interface/alpaka/LST.h"

#include "LSTEvent.h"

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

  using namespace ALPAKA_ACCELERATOR_NAMESPACE::lst;
  std::vector<unsigned int> getHitIdxs(short trackCandidateType,
                                       Params_pT5::ArrayUxHits const& tcHitIndices,
                                       unsigned int const* hitIndices) {
    std::vector<unsigned int> hits;

    unsigned int maxNHits = 0;
    if (trackCandidateType == LSTObjType::pT5)
      maxNHits = Params_pT5::kHits;
    else if (trackCandidateType == LSTObjType::pT3)
      maxNHits = Params_pT3::kHits;
    else if (trackCandidateType == LSTObjType::T5)
      maxNHits = Params_T5::kHits;
    else if (trackCandidateType == LSTObjType::pLS)
      maxNHits = Params_pLS::kHits;

    for (unsigned int i = 0; i < maxNHits; i++) {
      unsigned int hitIdxDev = tcHitIndices[i];
      unsigned int hitIdx =
          (trackCandidateType == LSTObjType::pLS)
              ? hitIdxDev
              : hitIndices[hitIdxDev];  // Hit indices are stored differently in the standalone for pLS.

      // For p objects, the 3rd and 4th hit maybe the same,
      // due to the way pLS hits are stored in the standalone.
      // This is because pixel seeds can be either triplets or quadruplets.
      if (trackCandidateType != LSTObjType::T5 && hits.size() == 3 &&
          hits.back() == hitIdx)  // Remove duplicate 4th hits.
        continue;

      hits.push_back(hitIdx);
    }

    return hits;
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
                       std::vector<float> const& ph2_z,
                       float const ptCut) {
  in_trkX_.clear();
  in_trkY_.clear();
  in_trkZ_.clear();
  in_hitId_.clear();
  in_hitIdxs_.clear();
  in_hitIndices_vec0_.clear();
  in_hitIndices_vec1_.clear();
  in_hitIndices_vec2_.clear();
  in_hitIndices_vec3_.clear();
  in_deltaPhi_vec_.clear();
  in_ptIn_vec_.clear();
  in_ptErr_vec_.clear();
  in_px_vec_.clear();
  in_py_vec_.clear();
  in_pz_vec_.clear();
  in_eta_vec_.clear();
  in_etaErr_vec_.clear();
  in_phi_vec_.clear();
  in_charge_vec_.clear();
  in_seedIdx_vec_.clear();
  in_superbin_vec_.clear();
  in_pixelType_vec_.clear();
  in_isQuad_vec_.clear();

  unsigned int count = 0;
  auto n_see = see_stateTrajGlbPx.size();
  in_px_vec_.reserve(n_see);
  in_py_vec_.reserve(n_see);
  in_pz_vec_.reserve(n_see);
  in_hitIndices_vec0_.reserve(n_see);
  in_hitIndices_vec1_.reserve(n_see);
  in_hitIndices_vec2_.reserve(n_see);
  in_hitIndices_vec3_.reserve(n_see);
  in_ptIn_vec_.reserve(n_see);
  in_ptErr_vec_.reserve(n_see);
  in_etaErr_vec_.reserve(n_see);
  in_eta_vec_.reserve(n_see);
  in_phi_vec_.reserve(n_see);
  in_charge_vec_.reserve(n_see);
  in_seedIdx_vec_.reserve(n_see);
  in_deltaPhi_vec_.reserve(n_see);
  in_trkX_ = ph2_x;
  in_trkY_ = ph2_y;
  in_trkZ_ = ph2_z;
  in_hitId_ = ph2_detId;
  in_hitIdxs_.resize(ph2_detId.size());

  std::iota(in_hitIdxs_.begin(), in_hitIdxs_.end(), 0);
  const int hit_size = in_trkX_.size();

  for (size_t iSeed = 0; iSeed < n_see; iSeed++) {
    XYZVector p3LH(see_stateTrajGlbPx[iSeed], see_stateTrajGlbPy[iSeed], see_stateTrajGlbPz[iSeed]);
    float ptIn = p3LH.rho();
    float eta = p3LH.eta();
    float ptErr = see_ptErr[iSeed];

    if ((ptIn > ptCut - 2 * ptErr)) {
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
      else if (ptIn >= (ptCut - 2 * ptErr) and ptIn < 2.0) {
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

      in_trkX_.push_back(r3PCA.x());
      in_trkY_.push_back(r3PCA.y());
      in_trkZ_.push_back(r3PCA.z());
      in_trkX_.push_back(p3PCA.rho());
      float p3PCA_Eta = p3PCA.eta();
      in_trkY_.push_back(p3PCA_Eta);
      float p3PCA_Phi = p3PCA.phi();
      in_trkZ_.push_back(p3PCA_Phi);
      in_trkX_.push_back(r3LH.x());
      in_trkY_.push_back(r3LH.y());
      in_trkZ_.push_back(r3LH.z());
      in_hitId_.push_back(1);
      in_hitId_.push_back(1);
      in_hitId_.push_back(1);
      if (see_hitIdx[iSeed].size() > 3) {
        in_trkX_.push_back(r3LH.x());
        in_trkY_.push_back(see_dxy[iSeed]);
        in_trkZ_.push_back(see_dz[iSeed]);
        in_hitId_.push_back(1);
      }
      in_px_vec_.push_back(px);
      in_py_vec_.push_back(py);
      in_pz_vec_.push_back(pz);

      in_hitIndices_vec0_.push_back(hitIdx0);
      in_hitIndices_vec1_.push_back(hitIdx1);
      in_hitIndices_vec2_.push_back(hitIdx2);
      in_hitIndices_vec3_.push_back(hitIdx3);
      in_ptIn_vec_.push_back(ptIn);
      in_ptErr_vec_.push_back(ptErr);
      in_etaErr_vec_.push_back(etaErr);
      in_eta_vec_.push_back(eta);
      float phi = p3LH.phi();
      in_phi_vec_.push_back(phi);
      in_charge_vec_.push_back(charge);
      in_seedIdx_vec_.push_back(iSeed);
      in_deltaPhi_vec_.push_back(pixelSegmentDeltaPhiChange);

      in_hitIdxs_.push_back(see_hitIdx[iSeed][0]);
      in_hitIdxs_.push_back(see_hitIdx[iSeed][1]);
      in_hitIdxs_.push_back(see_hitIdx[iSeed][2]);
      char isQuad = false;
      if (see_hitIdx[iSeed].size() > 3) {
        isQuad = true;
        in_hitIdxs_.push_back(see_hitIdx[iSeed][3]);
      }
      float neta = 25.;
      float nphi = 72.;
      float nz = 25.;
      int etabin = (p3PCA_Eta + 2.6) / ((2 * 2.6) / neta);
      int phibin = (p3PCA_Phi + kPi) / ((2. * kPi) / nphi);
      int dzbin = (see_dz[iSeed] + 30) / (2 * 30 / nz);
      int isuperbin = (nz * nphi) * etabin + (nz)*phibin + dzbin;
      in_superbin_vec_.push_back(isuperbin);
      in_pixelType_vec_.push_back(pixtype);
      in_isQuad_vec_.push_back(isQuad);
    }
  }
}

void LST::getOutput(LSTEvent& event) {
  out_tc_hitIdxs_.clear();
  out_tc_len_.clear();
  out_tc_seedIdx_.clear();
  out_tc_trackCandidateType_.clear();

  auto const hits = event.getHits<HitsSoA>(/*inCMSSW*/ true, /*sync*/ false);  // sync on next line
  auto const& trackCandidates = event.getTrackCandidates(/*inCMSSW*/ true, /*sync*/ true);

  unsigned int nTrackCandidates = trackCandidates.nTrackCandidates();

  for (unsigned int idx = 0; idx < nTrackCandidates; idx++) {
    short trackCandidateType = trackCandidates.trackCandidateType()[idx];
    std::vector<unsigned int> hit_idx = getHitIdxs(trackCandidateType, trackCandidates.hitIndices()[idx], hits.idxs());

    out_tc_hitIdxs_.push_back(hit_idx);
    out_tc_len_.push_back(hit_idx.size());
    out_tc_seedIdx_.push_back(trackCandidates.pixelSeedIndex()[idx]);
    out_tc_trackCandidateType_.push_back(trackCandidateType);
  }
}

void LST::run(Queue& queue,
              bool verbose,
              float const ptCut,
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
  auto event = LSTEvent(verbose, ptCut, queue, deviceESData);
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
               ph2_z,
               ptCut);

  event.addHitToEvent(in_trkX_, in_trkY_, in_trkZ_, in_hitId_, in_hitIdxs_);
  event.addPixelSegmentToEventStart(in_ptIn_vec_,
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

  event.addPixelSegmentToEventFinalize(
      in_hitIndices_vec0_, in_hitIndices_vec1_, in_hitIndices_vec2_, in_hitIndices_vec3_, in_deltaPhi_vec_);

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
}
