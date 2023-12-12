#include "RecoHGCal/TICL/interface/TracksterLinkingAlgoBase.h"
#include "fastjet/ClusterSequence.hh"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoHGCal/TICL/plugins/TracksterLinkingbyFastJet.h"

using namespace ticl;

void TracksterLinkingbyFastJet::linkTracksters(const Inputs& input, std::vector<Trackster>& resultTracksters,
                    std::vector<std::vector<unsigned int>>& linkedResultTracksters,
                    std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) {
  // Create jets of tracksters using FastJet
  std::vector<fastjet::PseudoJet> fjInputs;
  for (const auto& trackster : input.tracksters) {
    // Convert Trackster information to PseudoJet
    fastjet::PseudoJet pj(trackster.barycenter().x(), trackster.barycenter().y(), trackster.barycenter().z(),
                          trackster.raw_energy());
    pj.set_user_index(&trackster - &input.tracksters[0]);
    fjInputs.push_back(pj);
  }
  
  // Cluster tracksters into jets using FastJet
  fastjet::ClusterSequence sequence(fjInputs, fastjet::JetDefinition(fastjet::antikt_algorithm, antikt_radius_));
  auto jets = fastjet::sorted_by_pt(sequence.inclusive_jets(0));
  linkedTracksterIdToInputTracksterId.resize(jets.size());
  // Link tracksters based on which ones are components of the same jet
  for (unsigned int i = 0; i < jets.size(); ++i) {
    const auto& jet = jets[i];
    
    std::vector<unsigned int> linkedTracksters;
    // Check if a trackster is a component of the current jet
    for (const auto& constituent : jet.constituents()) {
        auto tracksterIndex = constituent.user_index();
        linkedTracksters.push_back(resultTracksters.size());
        linkedTracksterIdToInputTracksterId[i].push_back(tracksterIndex);
        resultTracksters.push_back(input.tracksters[tracksterIndex]);
    }
    // Store the linked tracksters
    linkedResultTracksters.push_back(linkedTracksters);
  }
}



void TracksterLinkingbyFastJet::fillPSetDescription(edm::ParameterSetDescription &iDesc) {
  iDesc.add<int>("algo_verbosity", 0);
  iDesc.add<double>("antikt_radius", 0.4);
}
