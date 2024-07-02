#include <string>
#include <memory>
#include <algorithm>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include "RecoHGCal/TICL/plugins/TracksterLinkingbySuperClusteringMustache.h"

using namespace ticl;

TracksterLinkingbySuperClusteringMustache::TracksterLinkingbySuperClusteringMustache(
    const edm::ParameterSet& ps, edm::ConsumesCollector iC, cms::Ort::ONNXRuntime const* onnxRuntime)
    : TracksterLinkingAlgoBase(ps, iC, onnxRuntime),
      ecalMustacheSCParametersToken_(iC.esConsumes<EcalMustacheSCParameters, EcalMustacheSCParametersRcd>()),
      ecalSCDynamicDPhiParametersToken_(iC.esConsumes<EcalSCDynamicDPhiParameters, EcalSCDynamicDPhiParametersRcd>()),
      seedThresholdPt_(ps.getParameter<double>("seedThresholdPt")),
      candidateEnergyThreshold_(ps.getParameter<double>("candidateEnergyThreshold")),
      filterByTracksterPID_(ps.getParameter<bool>("filterByTracksterPID")),
      tracksterPIDCategoriesToFilter_(ps.getParameter<std::vector<int>>("tracksterPIDCategoriesToFilter")),
      PIDThreshold_(ps.getParameter<double>("PIDThreshold")) {}

void TracksterLinkingbySuperClusteringMustache::initialize(const HGCalDDDConstants* hgcons,
                                                           const hgcal::RecHitTools rhtools,
                                                           const edm::ESHandle<MagneticField> bfieldH,
                                                           const edm::ESHandle<Propagator> propH) {}

void TracksterLinkingbySuperClusteringMustache::setEvent(edm::Event& iEvent, edm::EventSetup const& iEventSetup) {
  mustacheSCParams_ = &iEventSetup.getData(ecalMustacheSCParametersToken_);
  scDynamicDPhiParams_ = &iEventSetup.getData(ecalSCDynamicDPhiParametersToken_);
}

bool TracksterLinkingbySuperClusteringMustache::trackstersPassesPIDCut(const Trackster& tst) const {
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
void TracksterLinkingbySuperClusteringMustache::linkTracksters(
    const Inputs& input,
    std::vector<Trackster>& resultTracksters,
    std::vector<std::vector<unsigned int>>& outputSuperclusters,
    std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) {
  // For now we use all input tracksters for superclustering. At some point there might be a filter here for EM tracksters (electromagnetic identification with DNN ?)
  auto const& inputTracksters = input.tracksters;
  const unsigned int tracksterCount = inputTracksters.size();

  /* Sorting tracksters by decreasing order of pT (out-of-place sort). 
  inputTracksters[trackstersIndicesPt[0]], ..., inputTracksters[trackstersIndicesPt[N]] makes a list of tracksters sorted by decreasing pT
  Indices into this pT sorted collection will have the suffix _pt. Thus inputTracksters[index] and inputTracksters[trackstersIndicesPt[index_pt]] are correct
  */
  std::vector<unsigned int> trackstersIndicesPt(inputTracksters.size());
  std::iota(trackstersIndicesPt.begin(), trackstersIndicesPt.end(), 0);
  std::stable_sort(
      trackstersIndicesPt.begin(), trackstersIndicesPt.end(), [&inputTracksters](unsigned int i1, unsigned int i2) {
        return inputTracksters[i1].raw_pt() > inputTracksters[i2].raw_pt();
      });

  std::vector<bool> tracksterMask_pt(tracksterCount, false);  // Mask for already superclustered tracksters
  // We also mask tracksters that don't pass the PID cut
  for (unsigned int ts_idx_pt = 0; ts_idx_pt < tracksterCount; ts_idx_pt++) {
    tracksterMask_pt[ts_idx_pt] = !trackstersPassesPIDCut(inputTracksters[trackstersIndicesPt[ts_idx_pt]]);
  }

  for (unsigned int ts_seed_idx_pt = 0; ts_seed_idx_pt < tracksterCount; ts_seed_idx_pt++) {
    Trackster const& ts_seed = inputTracksters[trackstersIndicesPt[ts_seed_idx_pt]];
    if (ts_seed.raw_pt() <= seedThresholdPt_)
      break;  // Look only at seed tracksters passing threshold, take advantage of pt sorting for fast exit
    if (tracksterMask_pt[ts_seed_idx_pt])
      continue;  // Trackster does not pass PID cut

    outputSuperclusters.emplace_back(std::initializer_list<unsigned int>{trackstersIndicesPt[ts_seed_idx_pt]});
    resultTracksters.emplace_back(inputTracksters[trackstersIndicesPt[ts_seed_idx_pt]]);
    linkedTracksterIdToInputTracksterId.emplace_back(
        std::initializer_list<unsigned int>{trackstersIndicesPt[ts_seed_idx_pt]});

    for (unsigned int ts_cand_idx_pt = ts_seed_idx_pt + 1; ts_cand_idx_pt < tracksterCount; ts_cand_idx_pt++) {
      if (tracksterMask_pt[ts_cand_idx_pt])
        continue;  // Trackster is either already superclustered or did not pass PID cut

      Trackster const& ts_cand = inputTracksters[trackstersIndicesPt[ts_cand_idx_pt]];

      if (ts_cand.raw_energy() <= candidateEnergyThreshold_)
        continue;

      const bool passes_dphi = reco::MustacheKernel::inDynamicDPhiWindow(scDynamicDPhiParams_,
                                                                         ts_seed.barycenter().eta(),
                                                                         ts_seed.barycenter().phi(),
                                                                         ts_cand.raw_energy(),
                                                                         ts_cand.barycenter().eta(),
                                                                         ts_cand.barycenter().phi());

      if (passes_dphi && reco::MustacheKernel::inMustache(mustacheSCParams_,
                                                          ts_seed.barycenter().eta(),
                                                          ts_seed.barycenter().phi(),
                                                          ts_cand.raw_energy(),
                                                          ts_cand.barycenter().eta(),
                                                          ts_cand.barycenter().phi())) {
        outputSuperclusters.back().push_back(trackstersIndicesPt[ts_cand_idx_pt]);
        resultTracksters.back().mergeTracksters(ts_cand);
        linkedTracksterIdToInputTracksterId.back().push_back(trackstersIndicesPt[ts_cand_idx_pt]);
        tracksterMask_pt[ts_cand_idx_pt] = true;
      }
    }
  }
}

void TracksterLinkingbySuperClusteringMustache::fillPSetDescription(edm::ParameterSetDescription& desc) {
  TracksterLinkingAlgoBase::fillPSetDescription(desc);  // adds algo_verbosity
  desc.add<double>("seedThresholdPt", 1.)
      ->setComment("Minimum transverse energy of trackster to be considered as seed of a supercluster");
  desc.add<double>("candidateEnergyThreshold", 0.15)
      ->setComment("Minimum energy of trackster to be considered as candidate for superclustering");
  desc.add<bool>("filterByTracksterPID", true)->setComment("Filter tracksters before superclustering by PID score");
  desc.add<std::vector<int>>(
          "tracksterPIDCategoriesToFilter",
          {static_cast<int>(Trackster::ParticleType::photon), static_cast<int>(Trackster::ParticleType::electron)})
      ->setComment("List of PID particle types (ticl::Trackster::ParticleType enum) to consider for PID filtering");
  desc.add<double>("PIDThreshold", 0.8)->setComment("PID score threshold");
}
