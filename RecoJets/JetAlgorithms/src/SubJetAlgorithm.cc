#include <memory>

#include "RecoJets/JetAlgorithms/interface/SubJetAlgorithm.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "fastjet/ClusterSequenceArea.hh"

using namespace std;
using namespace edm;

void SubJetAlgorithm::set_zcut(double z) { zcut_ = z; }

void SubJetAlgorithm::set_rcut_factor(double r) { rcut_factor_ = r; }

//  Run the algorithm
//  ------------------
void SubJetAlgorithm::run(const vector<fastjet::PseudoJet>& cell_particles, vector<CompoundPseudoJet>& hardjetsOutput) {
  //for actual jet clustering, either the pruned or the original version is used.
  //For the pruned version, a new jet definition using the PrunedRecombPlugin is required:
  fastjet::FastPrunePlugin PRplugin(*fjJetDefinition_, *fjJetDefinition_, zcut_, rcut_factor_);
  fastjet::JetDefinition pjetdef(&PRplugin);

  // cluster the jets with the jet definition jetDef:
  // run algorithm
  std::shared_ptr<fastjet::ClusterSequence> fjClusterSeq;
  if (!doAreaFastjet_) {
    fjClusterSeq = std::make_shared<fastjet::ClusterSequence>(cell_particles, pjetdef);
  } else if (voronoiRfact_ <= 0) {
    fjClusterSeq = std::shared_ptr<fastjet::ClusterSequence>(
        new fastjet::ClusterSequenceActiveArea(cell_particles, pjetdef, *fjActiveArea_));
  } else {
    fjClusterSeq = std::shared_ptr<fastjet::ClusterSequence>(
        new fastjet::ClusterSequenceVoronoiArea(cell_particles, pjetdef, fastjet::VoronoiAreaSpec(voronoiRfact_)));
  }

  vector<fastjet::PseudoJet> inclusiveJets = fjClusterSeq->inclusive_jets(ptMin_);

  // These will store the indices of each subjet that
  // are present in each jet
  vector<vector<int> > indices(inclusiveJets.size());
  // Loop over inclusive jets, attempt to find substructure
  vector<fastjet::PseudoJet>::iterator jetIt = inclusiveJets.begin();
  for (; jetIt != inclusiveJets.end(); ++jetIt) {
    //decompose into requested number of subjets:
    vector<fastjet::PseudoJet> subjets = fjClusterSeq->exclusive_subjets(*jetIt, nSubjets_);
    //create the subjets objects to put into the "output" objects
    vector<CompoundPseudoSubJet> subjetsOutput;
    std::vector<fastjet::PseudoJet>::const_iterator itSubJetBegin = subjets.begin(), itSubJet = itSubJetBegin,
                                                    itSubJetEnd = subjets.end();
    for (; itSubJet != itSubJetEnd; ++itSubJet) {
      // Get the transient subjet constituents from fastjet
      vector<fastjet::PseudoJet> subjetFastjetConstituents = fjClusterSeq->constituents(*itSubJet);
      // Get the indices of the constituents:
      vector<int> constituents;
      vector<fastjet::PseudoJet>::const_iterator fastSubIt = subjetFastjetConstituents.begin(),
                                                 transConstEnd = subjetFastjetConstituents.end();
      for (; fastSubIt != transConstEnd; ++fastSubIt) {
        if (fastSubIt->user_index() >= 0) {
          constituents.push_back(fastSubIt->user_index());
        }
      }

      double subJetArea =
          (doAreaFastjet_) ? dynamic_cast<fastjet::ClusterSequenceActiveArea&>(*fjClusterSeq).area(*itSubJet) : 0.0;

      // Make a CompoundPseudoSubJet object to hold this subjet and the indices of its constituents
      subjetsOutput.push_back(CompoundPseudoSubJet(*itSubJet, subJetArea, constituents));
    }

    double fatJetArea =
        (doAreaFastjet_) ? dynamic_cast<fastjet::ClusterSequenceActiveArea&>(*fjClusterSeq).area(*jetIt) : 0.0;

    // Make a CompoundPseudoJet object to hold this hard jet, and the subjets that make it up
    hardjetsOutput.push_back(CompoundPseudoJet(*jetIt, fatJetArea, subjetsOutput));
  }
}
