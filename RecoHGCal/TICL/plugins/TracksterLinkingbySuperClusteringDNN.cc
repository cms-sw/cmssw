/*
TICL plugin for electron superclustering in HGCAL using a DNN. 
DNN designed by Alessandro Tarabini, Florian Beaudette, Gamze Sokmen, Shamik Ghosh, Theo Cuisset.

Inputs are CLUE3D EM tracksters. Outputs are superclusters (as vectors of IDs of trackster)
"Seed trackster" : seed of supercluster, always highest pT trackster of supercluster, normally should be an electron
"Candidate trackster" : trackster that is considered for superclustering with a seed

Algorithm description :
1) Tracksters are ordered by decreasing pT.
2) We iterate over candidate tracksters, then over seed tracksters with higher pT than the candidate.
   If the pair seed-candidate is in a compatible eta-phi window and passes some selections (seed pT, energy, etc), then we add the DNN features of the pair to a tensor for later inference.
3) We run the inference with the DNN on the pairs (in minibatches to reduce memory usage)
4) We iterate over candidate and seed pairs inference results. For each candidate, we take the seed for which the DNN score for the seed-candidate score is best.
   If the score is also above a working point, then we add the candidate to the supercluster of the seed, and mask the candidate so it cannot be considered as a seed further

The loop is first on candidate, then on seeds as it is more efficient for step 4 to find the best seed for each candidate.

Authors : Theo Cuisset <theo.cuisset@cern.ch>, Shamik Ghosh <shamik.ghosh@cern.ch>
Date : 11/2023

Updates : Logic works as it should and switching to v3 (Shamik)
Date: 07/2025

Modified by Felice Pantaleo <felice.pantaleo@cern.ch>
Improved memory usage and inference performance. 
Date: 02/2026
*/

#include <string>
#include <memory>
#include <algorithm>
#include <vector>
#include <numeric>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "RecoHGCal/TICL/plugins/TracksterLinkingbySuperClusteringDNN.h"

using namespace ticl;

TracksterLinkingbySuperClusteringDNN::TracksterLinkingbySuperClusteringDNN(const edm::ParameterSet& ps,
                                                                           edm::ConsumesCollector iC,
                                                                           cms::Ort::ONNXRuntime const* onnxRuntime)
    : TracksterLinkingAlgoBase(ps, iC, onnxRuntime),
      dnnInputs_(makeSuperclusteringDNNInputFromString(ps.getParameter<std::string>("dnnInputsVersion"))),
      inferenceBatchSize_(ps.getParameter<unsigned int>("inferenceBatchSize")),
      nnWorkingPoint_(ps.getParameter<double>("nnWorkingPoint")),
      deltaEtaWindow_(ps.getParameter<double>("deltaEtaWindow")),
      deltaPhiWindow_(ps.getParameter<double>("deltaPhiWindow")),
      seedPtThreshold_(ps.getParameter<double>("seedPtThreshold")),
      candidateEnergyThreshold_(ps.getParameter<double>("candidateEnergyThreshold")),
      explVarRatioCut_energyBoundary_(ps.getParameter<double>("candidateEnergyThreshold")),
      explVarRatioMinimum_lowEnergy_(ps.getParameter<double>("explVarRatioMinimum_lowEnergy")),
      explVarRatioMinimum_highEnergy_(ps.getParameter<double>("explVarRatioMinimum_highEnergy")),
      filterByTracksterPID_(ps.getParameter<bool>("filterByTracksterPID")),
      tracksterPIDCategoriesToFilter_(ps.getParameter<std::vector<int>>("tracksterPIDCategoriesToFilter")),
      PIDThreshold_(ps.getParameter<double>("PIDThreshold")) {
  const auto model = ps.getParameter<std::string>("onnxModelPath");
  if (model.empty()) {
    throw cms::Exception("Configuration")
        << "TracksterLinkingbySuperClusteringDNN requires a non-empty 'onnxModelPath'.";
  }
  if (!onnxRuntime_) {
    throw cms::Exception("Configuration")
        << "TracksterLinkingbySuperClusteringDNN could not retrieve an ONNX session for 'onnxModelPath' = " << model;
  }
}

void TracksterLinkingbySuperClusteringDNN::initialize(const HGCalDDDConstants* hgcons,
                                                      const hgcal::RecHitTools rhtools,
                                                      const edm::ESHandle<MagneticField> bfieldH,
                                                      const edm::ESHandle<Propagator> propH) {}

/** 
 * Check if trackster passes cut on explained variance ratio. The DNN is trained only on pairs where both seed and candidate pass this cut
 * Explained variance ratio is (largest PCA eigenvalue) / (sum of PCA eigenvalues)
*/
bool TracksterLinkingbySuperClusteringDNN::checkExplainedVarianceRatioCut(ticl::Trackster const& ts) const {
  float explVar_denominator =
      std::accumulate(std::begin(ts.eigenvalues()), std::end(ts.eigenvalues()), 0.f, std::plus<float>());
  if (explVar_denominator != 0.) {
    float explVarRatio = ts.eigenvalues()[0] / explVar_denominator;
    if (ts.raw_energy() > explVarRatioCut_energyBoundary_)
      return explVarRatio >= explVarRatioMinimum_highEnergy_;
    else
      return explVarRatio >= explVarRatioMinimum_lowEnergy_;
  } else
    return false;
}

bool TracksterLinkingbySuperClusteringDNN::checkExplainedVarianceRatioCut(
    edm::MultiSpan<ticl::Trackster> const& tracksters,
    unsigned int index,
    std::unordered_map<unsigned int, bool>& cache) const {
  auto cache_it = cache.find(index);
  if (cache_it != cache.end()) {
    return cache_it->second;
  } else {
    bool result = checkExplainedVarianceRatioCut(tracksters[index]);
    cache[index] = result;
    return result;
  }
}

bool TracksterLinkingbySuperClusteringDNN::trackstersPassesPIDCut(const Trackster& tst) const {
  if (filterByTracksterPID_) {
    float probTotal = 0.0f;
    for (int cat : tracksterPIDCategoriesToFilter_) {
      probTotal += tst.id_probabilities(cat);
    }
    return probTotal >= PIDThreshold_;
  } else
    return true;
}

/**
 * resultTracksters : superclusters as tracksters (ie merging of tracksters that have been superclustered together)
 * outputSuperclusters : same as linkedTracksterIdToInputTracksterId. Probably should use only one of the two.
 * linkedTracksterIdToInputTracksterId : maps indices from resultTracksters back into input tracksters.
 *    resultTracksters[i] has seed input.tracksters[linkedTracksterIdToInputTracksterId[i][0]], linked with tracksters input.tracksters[linkedTracksterIdToInputTracksterId[i][1..N]]
*/
void TracksterLinkingbySuperClusteringDNN::linkTracksters(
    const Inputs& input,
    std::vector<Trackster>& resultTracksters,
    std::vector<std::vector<unsigned int>>& outputSuperclusters,
    std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) {
  auto const& inputTracksters = input.tracksters;
  const auto tracksterCount = static_cast<unsigned int>(inputTracksters.size());
  if (tracksterCount == 0) {
    return;
  }

  // ---- sort by decreasing pT (indices only)
  std::vector<unsigned int> trackstersIndicesPt(tracksterCount);
  std::iota(trackstersIndicesPt.begin(), trackstersIndicesPt.end(), 0u);
  std::stable_sort(
      trackstersIndicesPt.begin(), trackstersIndicesPt.end(), [&inputTracksters](unsigned int a, unsigned int b) {
        return inputTracksters[a].raw_pt() > inputTracksters[b].raw_pt();
      });

  // ---- batch sizing (rows = pairs, cols = featureCount)
  const auto featuresPerPair = static_cast<unsigned int>(dnnInputs_->featureCount());
  const auto maxPairsPerBatch = static_cast<unsigned int>(inferenceBatchSize_) / featuresPerPair;  // number of rows

  if (maxPairsPerBatch == 0u) {
    throw cms::Exception("Configuration") << "inferenceBatchSize (" << inferenceBatchSize_
                                          << ") is smaller than featureCount (" << featuresPerPair << ").";
  }

  // ---- reusable batch buffers
  std::vector<float> inputBatch;
  inputBatch.reserve(static_cast<size_t>(maxPairsPerBatch) * featuresPerPair);

  std::vector<std::pair<unsigned int, unsigned int>> batchPairs;
  batchPairs.reserve(maxPairsPerBatch);

  // ONNX I/O buffers reused across flushes
  cms::Ort::FloatArrays inputs_for_onnx;
  cms::Ort::FloatArrays outputs_for_onnx;
  inputs_for_onnx.resize(1);

  // Reuse shapes buffer: one input tensor of rank 2: [batch, features]
  std::vector<std::vector<int64_t>> input_shapes(1, std::vector<int64_t>(2, 0));

  static const std::vector<std::string> kInputNames = {"input"};

  // ---- tiles (one per endcap)
  std::array<TICLLayerTile, 2> tracksterTilesBothEndcaps_pt;
  for (unsigned int i_pt = 0; i_pt < tracksterCount; ++i_pt) {
    auto const& ts = inputTracksters[trackstersIndicesPt[i_pt]];
    tracksterTilesBothEndcaps_pt[ts.barycenter().eta() > 0.f].fill(ts.barycenter().eta(), ts.barycenter().phi(), i_pt);
  }

  // ---- bookkeeping masks
  std::vector<bool> tracksterMask(tracksterCount, false);
  std::vector<bool> usedAsCandidate(tracksterCount, false);

  constexpr auto kInvalid = std::numeric_limits<unsigned int>::max();
  unsigned int previousCand = kInvalid;
  unsigned int bestSeed = kInvalid;
  float bestScore = nnWorkingPoint_;

  auto onCandidateTransition = [&](unsigned int candIdx) {
    if (bestSeed == kInvalid) {
      return;
    }

    tracksterMask[candIdx] = true;
    usedAsCandidate[candIdx] = true;

    auto seed_it = std::find_if(outputSuperclusters.begin(), outputSuperclusters.end(), [bestSeed](auto const& sc) {
      return sc[0] == bestSeed;
    });

    if (seed_it == outputSuperclusters.end()) {
      outputSuperclusters.emplace_back(std::initializer_list<unsigned int>{bestSeed});
      resultTracksters.emplace_back(inputTracksters[bestSeed]);
      linkedTracksterIdToInputTracksterId.emplace_back(std::initializer_list<unsigned int>{bestSeed});
      seed_it = std::prev(outputSuperclusters.end());
      tracksterMask[bestSeed] = true;
    }

    const auto outIdx = static_cast<unsigned int>(seed_it - outputSuperclusters.begin());
    seed_it->push_back(candIdx);
    resultTracksters[outIdx].mergeTracksters(inputTracksters[candIdx]);
    linkedTracksterIdToInputTracksterId[outIdx].push_back(candIdx);

    bestSeed = kInvalid;
    bestScore = nnWorkingPoint_;
  };

  auto flushBatch = [&]() {
    const auto pairsInBatch = static_cast<unsigned int>(batchPairs.size());
    if (pairsInBatch == 0u) {
      return;
    }

    // shape: [pairs, features]
    input_shapes[0][0] = static_cast<int64_t>(pairsInBatch);
    input_shapes[0][1] = static_cast<int64_t>(featuresPerPair);

    // Provide ONNX with the batch buffer without realloc/copy
    inputs_for_onnx[0].swap(inputBatch);

    outputs_for_onnx.clear();
    onnxRuntime_->runInto(kInputNames,
                          inputs_for_onnx,
                          input_shapes,
                          {},                // all outputs
                          outputs_for_onnx,  // resized as needed
                          {},                // optional output_shapes
                          static_cast<int64_t>(pairsInBatch));

    if (outputs_for_onnx.empty()) {
      throw cms::Exception("RuntimeError") << "ONNX model returned no outputs.";
    }

    auto const& out = outputs_for_onnx[0];
    if (out.size() < pairsInBatch) {
      throw cms::Exception("RuntimeError")
          << "ONNX output has size " << out.size() << " but expected at least " << pairsInBatch;
    }

    // Consume in-order
    for (unsigned int i = 0; i < pairsInBatch; ++i) {
      const auto [seedIdx, candIdx] = batchPairs[i];

      if (previousCand != kInvalid && candIdx != previousCand) {
        onCandidateTransition(previousCand);
      }

      const float score = out[i];

      // Ignore seed if it was previously used as a candidate
      if (score > bestScore && !usedAsCandidate[seedIdx]) {
        bestSeed = seedIdx;
        bestScore = score;
      }

      previousCand = candIdx;
    }

    // Restore storage for reuse
    inputBatch.swap(inputs_for_onnx[0]);
    inputBatch.clear();
    batchPairs.clear();
  };

  // Cache for explained variance ratio cut results to avoid expensive redundant calculations across seeds and candidates
  std::unordered_map<unsigned int, bool> checkExplainedVarianceRatioCut_cache;

  // ---- main loops: candidate then seed
  for (unsigned int cand_pt = 1; cand_pt < tracksterCount; ++cand_pt) {
    auto const& ts_cand = inputTracksters[trackstersIndicesPt[cand_pt]];

    if (ts_cand.raw_energy() < candidateEnergyThreshold_) {
      continue;
    }

    bool passes_cut = checkExplainedVarianceRatioCut(
        inputTracksters, trackstersIndicesPt[cand_pt], checkExplainedVarianceRatioCut_cache);

    if (!passes_cut) {
      continue;
    }

    auto& tiles = tracksterTilesBothEndcaps_pt[ts_cand.barycenter().eta() > 0.f];
    const auto search_box = tiles.searchBoxEtaPhi(ts_cand.barycenter().Eta() - deltaEtaWindow_,
                                                  ts_cand.barycenter().Eta() + deltaEtaWindow_,
                                                  ts_cand.barycenter().Phi() - deltaPhiWindow_,
                                                  ts_cand.barycenter().Phi() + deltaPhiWindow_);

    for (int eta_i = search_box[0]; eta_i <= search_box[1]; ++eta_i) {
      for (int phi_i = search_box[2]; phi_i <= search_box[3]; ++phi_i) {
        const auto bin = tiles.globalBin(eta_i, (phi_i % TileConstants::nPhiBins));
        for (unsigned int seed_pt : tiles[bin]) {
          if (seed_pt >= cand_pt) {
            continue;  // only higher-pT seeds
          }

          auto const& ts_seed = inputTracksters[trackstersIndicesPt[seed_pt]];

          if (ts_seed.raw_pt() < seedPtThreshold_) {
            break;  // due to pT ordering
          }

          bool passes_cut = checkExplainedVarianceRatioCut(
              inputTracksters, trackstersIndicesPt[seed_pt], checkExplainedVarianceRatioCut_cache);

          if (!passes_cut || !trackstersPassesPIDCut(ts_seed)) {
            continue;
          }

          if (std::abs(ts_seed.barycenter().Eta() - ts_cand.barycenter().Eta()) >= deltaEtaWindow_) {
            continue;
          }
          if (std::abs(deltaPhi(ts_seed.barycenter().Phi(), ts_cand.barycenter().Phi())) >= deltaPhiWindow_) {
            continue;
          }

          if (batchPairs.size() == maxPairsPerBatch) {
            flushBatch();
          }

          // Append one feature row directly into the flat buffer
          const size_t base = inputBatch.size();
          inputBatch.resize(base + featuresPerPair);

          dnnInputs_->computeInto(ts_seed, ts_cand, std::span<float>(inputBatch.data() + base, featuresPerPair));
          batchPairs.emplace_back(trackstersIndicesPt[seed_pt], trackstersIndicesPt[cand_pt]);
        }
      }
    }
  }

  // Flush tail and finalize last candidate
  flushBatch();
  onCandidateTransition(previousCand);

  // Singleton superclusters for unused tracksters with enough pt
  for (unsigned int ts_id = 0; ts_id < tracksterCount; ++ts_id) {
    if (!tracksterMask[ts_id] && inputTracksters[ts_id].raw_pt() >= seedPtThreshold_) {
      outputSuperclusters.emplace_back(std::initializer_list<unsigned int>{ts_id});
      resultTracksters.emplace_back(inputTracksters[ts_id]);
      linkedTracksterIdToInputTracksterId.emplace_back(std::initializer_list<unsigned int>{ts_id});
    }
  }

#ifdef EDM_ML_DEBUG
  for (std::vector<unsigned int> const& sc : outputSuperclusters) {
    std::ostringstream s;
    for (unsigned int trackster_id : sc)
      s << trackster_id << " ";
    LogDebug("HGCalTICLSuperclustering") << "Created supercluster of size " << sc.size()
                                         << " holding tracksters (first one is seed) " << s.str();
  }
#endif
}

void TracksterLinkingbySuperClusteringDNN::fillPSetDescription(edm::ParameterSetDescription& desc) {
  TracksterLinkingAlgoBase::fillPSetDescription(desc);  // adds algo_verbosity
  desc.add<std::string>("onnxModelPath")->setComment("Path to DNN (as ONNX model), empty disables loading");
  desc.ifValue(edm::ParameterDescription<std::string>("dnnInputsVersion", "v3", true),
               edm::allowedValues<std::string>("v1", "v2", "v3"))
      ->setComment(
          "DNN inputs version tag. Defines which set of features is fed to the DNN. Must match with the actual DNN.");
  desc.add<unsigned int>("inferenceBatchSize", 1e5)
      ->setComment(
          "Size of inference batches fed to DNN. Increasing it should produce faster inference but higher memory "
          "usage. "
          "Has no physics impact.");
  desc.add<double>("nnWorkingPoint")
      ->setComment("Working point of DNN (in [0, 1]). DNN score above WP will attempt to supercluster.");
  desc.add<double>("deltaEtaWindow", 0.1)
      ->setComment(
          "Size of delta eta window to consider for superclustering. Seed-candidate pairs outside this window "
          "are not considered for DNN inference.");
  desc.add<double>("deltaPhiWindow", 0.5)
      ->setComment(
          "Size of delta phi window to consider for superclustering. Seed-candidate pairs outside this window "
          "are not considered for DNN inference.");
  desc.add<double>("seedPtThreshold", 4.)
      ->setComment("Minimum transverse energy of trackster to be considered as seed of a supercluster");
  desc.add<double>("candidateEnergyThreshold", 2.)
      ->setComment("Minimum energy of trackster to be considered as candidate for superclustering");
  desc.add<double>("explVarRatioCut_energyBoundary", 50.)
      ->setComment("Boundary energy between low and high energy explVarRatio cut threshold");
  desc.add<double>("explVarRatioMinimum_lowEnergy", 0.92)
      ->setComment(
          "Cut on explained variance ratio of tracksters to be considered as candidate, "
          "for trackster raw_energy < explVarRatioCut_energyBoundary");
  desc.add<double>("explVarRatioMinimum_highEnergy", 0.95)
      ->setComment(
          "Cut on explained variance ratio of tracksters to be considered as candidate, "
          "for trackster raw_energy > explVarRatioCut_energyBoundary");
  desc.add<bool>("filterByTracksterPID", true)->setComment("Filter tracksters before superclustering by PID score");
  desc.add<std::vector<int>>(
          "tracksterPIDCategoriesToFilter",
          {static_cast<int>(Trackster::ParticleType::photon), static_cast<int>(Trackster::ParticleType::electron)})
      ->setComment("List of PID particle types (ticl::Trackster::ParticleType enum) to consider for PID filtering");
  desc.add<double>("PIDThreshold", 0.8)->setComment("PID score threshold");
}
