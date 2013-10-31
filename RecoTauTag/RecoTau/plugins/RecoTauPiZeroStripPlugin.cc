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
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"
#include "RecoTauTag/RecoTau/interface/CombinatoricGenerator.h"

namespace reco { namespace tau {

namespace {
// Apply a hypothesis on the mass of the strips.
math::XYZTLorentzVector applyMassConstraint(
    const math::XYZTLorentzVector& vec,double mass) {
  double factor = sqrt(vec.energy()*vec.energy()-mass*mass)/vec.P();
  return math::XYZTLorentzVector(
      vec.px()*factor,vec.py()*factor,vec.pz()*factor,vec.energy());
}
}


class RecoTauPiZeroStripPlugin : public RecoTauPiZeroBuilderPlugin {
  public:
    explicit RecoTauPiZeroStripPlugin(const edm::ParameterSet& pset);
    virtual ~RecoTauPiZeroStripPlugin() {}
    // Return type is auto_ptr<PiZeroVector>
    return_type operator()(const reco::PFJet& jet) const;
    // Hook to update PV information
    virtual void beginEvent();

  private:
    RecoTauQualityCuts qcuts_;
    RecoTauVertexAssociator vertexAssociator_;

    std::vector<int> inputPdgIds_; //type of candidates to clusterize
    double etaAssociationDistance_;//eta Clustering Association Distance
    double phiAssociationDistance_;//phi Clustering Association Distance

    // Parameters for build strip combinations
    bool combineStrips_;
    int maxStrips_;
    double combinatoricStripMassHypo_;

    AddFourMomenta p4Builder_;
};

RecoTauPiZeroStripPlugin::RecoTauPiZeroStripPlugin(
    const edm::ParameterSet& pset):RecoTauPiZeroBuilderPlugin(pset),
    qcuts_(pset.getParameterSet(
          "qualityCuts").getParameterSet("signalQualityCuts")),
    vertexAssociator_(pset.getParameter<edm::ParameterSet>("qualityCuts")) {
  inputPdgIds_ = pset.getParameter<std::vector<int> >(
      "stripCandidatesParticleIds");
  etaAssociationDistance_ = pset.getParameter<double>(
      "stripEtaAssociationDistance");
  phiAssociationDistance_ = pset.getParameter<double>(
      "stripPhiAssociationDistance");
  combineStrips_ = pset.getParameter<bool>("makeCombinatoricStrips");
  if (combineStrips_) {
    maxStrips_ = pset.getParameter<int>("maxInputStrips");
    combinatoricStripMassHypo_ =
      pset.getParameter<double>("stripMassWhenCombining");
  }
}

// Update the primary vertex
void RecoTauPiZeroStripPlugin::beginEvent() {
  vertexAssociator_.setEvent(*evt());
}

RecoTauPiZeroStripPlugin::return_type RecoTauPiZeroStripPlugin::operator()(
    const reco::PFJet& jet) const {
  // Get list of gamma candidates
  typedef std::vector<reco::PFCandidatePtr> PFCandPtrs;
  typedef PFCandPtrs::iterator PFCandIter;
  PiZeroVector output;

  // Get the candidates passing our quality cuts
  qcuts_.setPV(vertexAssociator_.associatedVertex(jet));
  PFCandPtrs candsVector = qcuts_.filterCandRefs(pfCandidates(jet, inputPdgIds_));
  //PFCandPtrs candsVector = qcuts_.filterCandRefs(pfGammas(jet));

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
    // Update the vertex
    if (strip->daughterPtr(0).isNonnull())
      strip->setVertex(strip->daughterPtr(0)->vertex());
    output.push_back(strip);
  }

  // Check if we want to combine our strips
  if (combineStrips_ && output.size() > 1) {
    PiZeroVector stripCombinations;
    // Sort the output by descending pt
    output.sort(output.begin(), output.end(),
        boost::bind(&RecoTauPiZero::pt, _1) >
        boost::bind(&RecoTauPiZero::pt, _2));
    // Get the end of interesting set of strips to try and combine
    PiZeroVector::const_iterator end_iter = takeNElements(
        output.begin(), output.end(), maxStrips_);

    // Look at all the combinations
    for (PiZeroVector::const_iterator first = output.begin();
        first != end_iter-1; ++first) {
      for (PiZeroVector::const_iterator second = first+1;
          second != end_iter; ++second) {
        Candidate::LorentzVector firstP4 = first->p4();
        Candidate::LorentzVector secondP4 = second->p4();
        // If we assume a certain mass for each strip apply it here.
        firstP4 = applyMassConstraint(firstP4, combinatoricStripMassHypo_);
        secondP4 = applyMassConstraint(secondP4, combinatoricStripMassHypo_);
        Candidate::LorentzVector totalP4 = firstP4 + secondP4;
        // Make our new combined strip
        std::auto_ptr<RecoTauPiZero> combinedStrips(
            new RecoTauPiZero(0, totalP4,
              Candidate::Point(0, 0, 0),
              //111, 10001, true, RecoTauPiZero::kCombinatoricStrips));
              111, 10001, true, RecoTauPiZero::kUndefined));

        // Now loop over the strip members
        BOOST_FOREACH(const RecoTauPiZero::daughters::value_type& gamma,
            first->daughterPtrVector()) {
          combinedStrips->addDaughter(gamma);
        }
        BOOST_FOREACH(const RecoTauPiZero::daughters::value_type& gamma,
            second->daughterPtrVector()) {
          combinedStrips->addDaughter(gamma);
        }
        // Update the vertex
        if (combinedStrips->daughterPtr(0).isNonnull())
          combinedStrips->setVertex(combinedStrips->daughterPtr(0)->vertex());
        // Add to our collection of combined strips
        stripCombinations.push_back(combinedStrips);
      }
    }
    // When done doing all the combinations, add the combined strips to the
    // output.
    output.transfer(output.end(), stripCombinations);
  }

  return output.release();
}
}} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauPiZeroBuilderPluginFactory,
    reco::tau::RecoTauPiZeroStripPlugin, "RecoTauPiZeroStripPlugin");
