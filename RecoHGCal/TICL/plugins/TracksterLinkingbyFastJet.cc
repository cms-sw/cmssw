#include "RecoHGCal/TICL/interface/TracksterLinkingAlgoBase.h"
#include "fastjet/ClusterSequence.hh"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoHGCal/TICL/plugins/TracksterLinkingbyFastJet.h"

using namespace ticl;

void TracksterLinkingbyFastJet::linkTracksters(
    const Inputs& input,
    std::vector<Trackster>& resultTracksters,
    std::vector<std::vector<unsigned int>>& linkedResultTracksters,
    std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) {
  // Create jets of tracksters using FastJet
  std::vector<fastjet::PseudoJet> fjInputs;
  for (size_t i = 0; i < input.tracksters.size(); ++i) {
    // Convert Trackster information to PseudoJet
    fastjet::PseudoJet pj(input.tracksters[i].barycenter().x(),
                          input.tracksters[i].barycenter().y(),
                          input.tracksters[i].barycenter().z(),
                          input.tracksters[i].raw_energy());
    pj.set_user_index(i);
    fjInputs.push_back(pj);
  }

  // Cluster tracksters into jets using FastJet
  fastjet::ClusterSequence sequence(fjInputs, fastjet::JetDefinition(algorithm_, radius_));
  auto jets = fastjet::sorted_by_pt(sequence.inclusive_jets(0));
  linkedTracksterIdToInputTracksterId.resize(jets.size());
  // Link tracksters based on which ones are components of the same jet
  for (unsigned int i = 0; i < jets.size(); ++i) {
    const auto& jet = jets[i];

    std::vector<unsigned int> linkedTracksters;
    Trackster outTrackster;
    if (!jet.constituents().empty()) {
      // Check if a trackster is a component of the current jet
      for (const auto& constituent : jet.constituents()) {
        auto tracksterIndex = constituent.user_index();
        linkedTracksterIdToInputTracksterId[i].push_back(tracksterIndex);
        outTrackster.mergeTracksters(input.tracksters[tracksterIndex]);
      }
      linkedTracksters.push_back(resultTracksters.size());
      resultTracksters.push_back(outTrackster);
      // Store the linked tracksters
      linkedResultTracksters.push_back(linkedTracksters);
    }
  }
}