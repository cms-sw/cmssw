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
#include <algorithm>
#include <memory>

#include "RecoTauTag/RecoTau/interface/RecoTauPiZeroPlugins.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"

namespace reco { namespace tau {

class RecoTauPiZeroStripPlugin : public RecoTauPiZeroBuilderPlugin {
  public:
    explicit RecoTauPiZeroStripPlugin(const edm::ParameterSet& pset);
    virtual ~RecoTauPiZeroStripPlugin() {}
    // Return type is auto_ptr<PiZeroVector>
    return_type operator()(const reco::PFJet& jet) const;
    // Hook to update PV information
    virtual void beginEvent();

  private:
    // PV needed for quality cuts
    edm::InputTag pvSrc_;
    RecoTauQualityCuts qcuts_;

    std::vector<int> inputPdgIds_; //type of candidates to clusterize
    double etaAssociationDistance_;//eta Clustering Association Distance
    double phiAssociationDistance_;//phi Clustering Association Distance

    AddFourMomenta p4Builder_;
};

RecoTauPiZeroStripPlugin::RecoTauPiZeroStripPlugin(
    const edm::ParameterSet& pset):RecoTauPiZeroBuilderPlugin(pset),
    qcuts_(pset.getParameter<edm::ParameterSet>("qualityCuts"))
{
  pvSrc_ = pset.getParameter<edm::InputTag>("primaryVertexSrc");
  inputPdgIds_ = pset.getParameter<std::vector<int> >(
      "stripCandidatesParticleIds");
  etaAssociationDistance_ = pset.getParameter<double>(
      "stripEtaAssociationDistance");
  phiAssociationDistance_ = pset.getParameter<double>(
      "stripPhiAssociationDistance");
}

// Update the primary vertex
void RecoTauPiZeroStripPlugin::beginEvent() {
  edm::Handle<reco::VertexCollection> pvHandle;
  evt()->getByLabel(pvSrc_, pvHandle);
  if (pvHandle->size()) {
    qcuts_.setPV(reco::VertexRef(pvHandle, 0));
  }
}

RecoTauPiZeroStripPlugin::return_type RecoTauPiZeroStripPlugin::operator()(
    const reco::PFJet& jet) const {
  // Get list of gamma candidates
  typedef std::vector<reco::PFCandidatePtr> PFCandPtrs;
  typedef PFCandPtrs::iterator PFCandIter;
  PiZeroVector output;

  // Get the candidates passing our quality cuts
  //PFCandPtrs candsVector = qcuts_.filterRefs(pfCandidates(jet, inputPdgIds_));
  PFCandPtrs candsVector = qcuts_.filterRefs(pfGammas(jet));

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
    std::auto_ptr<RecoTauPiZero> strip(new RecoTauPiZero(
            *seed, RecoTauPiZero::kStrips));
    strip->addDaughter(seed);

    // Find all other objects in the strip
    PFCandPtrListIter stripCand = cands.begin();
    while(stripCand != cands.end()) {
      if( fabs(strip->eta() - (*stripCand)->eta()) < etaAssociationDistance_
          && fabs(deltaPhi(*strip, **stripCand)) < phiAssociationDistance_ ) {
        // Add candidate to strip
        strip->addDaughter(*stripCand);
        // Update the strips four momenta
        p4Builder_.set(*strip);
        // Delete this candidate from future strips and move on to
        // the next potential candidate
        stripCand = cands.erase(stripCand);
      } else {
        // This candidate isn't compatabile - just move to the next candidate
        ++stripCand;
      }
    }
    output.push_back(strip);
  }
  return output.release();
}
}} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauPiZeroBuilderPluginFactory,
    reco::tau::RecoTauPiZeroStripPlugin, "RecoTauPiZeroStripPlugin");
