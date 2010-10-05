/*
 * RecoTauPiZeroStripPlugin
 *
 * Merges PFGammas in a PFJet into Candidate piZeros defined as
 * strips in eta-phi.
 *
 * Author: Michail Bachtis (University of Wisconsin)
 *
 * Code modifications: Evan Friis (UC Davis)
 *
 * $Id $
 */

#include "RecoTauTag/RecoTau/interface/RecoTauPiZeroPlugins.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

#include <algorithm>

namespace reco { namespace tau {

class RecoTauPiZeroStripPlugin : public RecoTauPiZeroBuilderPlugin {
  public:
    explicit RecoTauPiZeroStripPlugin(const edm::ParameterSet& pset);
    ~RecoTauPiZeroStripPlugin() {}
    // Return type is auto_ptr<PiZeroVector>
    return_type operator()(const reco::PFJet& jet) const;

  private:
    std::vector<int> inputPdgIds_; //type of candidates to clusterize
    double etaAssociationDistance_;//eta Clustering Association Distance
    double phiAssociationDistance_;//phi Clustering Association Distance

    AddFourMomenta p4Builder_;
};

RecoTauPiZeroStripPlugin::RecoTauPiZeroStripPlugin(
    const edm::ParameterSet& pset):RecoTauPiZeroBuilderPlugin(pset) {
  inputPdgIds_ = pset.getParameter<std::vector<int> >(
      "stripCandidatesParticleIds");
  etaAssociationDistance_ = pset.getParameter<double>(
      "stripEtaAssociationDistance");
  phiAssociationDistance_ = pset.getParameter<double>(
      "stripPhiAssociationDistance");
}

RecoTauPiZeroStripPlugin::return_type RecoTauPiZeroStripPlugin::operator()(
    const reco::PFJet& jet) const {
  // Get list of gamma candidates
  typedef std::vector<reco::PFCandidatePtr> PFCandPtrs;
  typedef PFCandPtrs::iterator PFCandIter;
  PiZeroVector output;

  PFCandPtrs candsVector = tau::pfCandidates(jet, inputPdgIds_);

  // Convert to stl::list to allow fast deletions
  typedef std::list<reco::PFCandidatePtr> PFCandPtrList;
  typedef std::list<reco::PFCandidatePtr>::iterator PFCandPtrListIter;
  PFCandPtrList cands;
  cands.insert(cands.end(), candsVector.begin(), candsVector.end());

  while (cands.size() > 0) {
    // Seed this new strip, and delete it from future strips
    PFCandidatePtr seed = cands.front();
    cands.pop_front();

    // Add a new candidate to our collection using this seed
    output.push_back(new RecoTauPiZero(*seed, name()));
    RecoTauPiZero& strip = output.back();
    strip.addDaughter(seed);

    // Find all other objects in the strip
    PFCandPtrListIter stripCand = cands.begin();
    while(stripCand != cands.end()) {
      if( fabs(strip.eta() - (*stripCand)->eta()) < etaAssociationDistance_
          && fabs(deltaPhi(strip, **stripCand)) < phiAssociationDistance_ ) {
        // Add candidate to strip
        strip.addDaughter(*stripCand);
        // Update the strips four momenta
        p4Builder_.set(strip);
        // Delete this candidate from future strips and move on to
        // the next potential candidate
        stripCand = cands.erase(stripCand);
      } else {
        // This candidate isn't compatabile - just move to the next candidate
        ++stripCand;
      }
    }
  }
  return output.release();
}
}} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauPiZeroBuilderPluginFactory,
    reco::tau::RecoTauPiZeroStripPlugin, "RecoTauPiZeroStripPlugin");
