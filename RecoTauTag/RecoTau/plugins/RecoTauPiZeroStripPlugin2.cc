/*
 * RecoTauPiZeroStripPlugin2
 *
 * Merges PFGammas in a PFJet into Candidate piZeros defined as
 * strips in eta-phi.
 *
 * Author: Michail Bachtis (University of Wisconsin)
 *
 * Code modifications: Evan Friis (UC Davis),
 *                     Christian Veelken (LLR)
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

//-------------------------------------------------------------------------------
// CV: the following headers are needed only for debug print-out
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
//-------------------------------------------------------------------------------

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

class RecoTauPiZeroStripPlugin2 : public RecoTauPiZeroBuilderPlugin 
{
 public:
  explicit RecoTauPiZeroStripPlugin2(const edm::ParameterSet&);
  virtual ~RecoTauPiZeroStripPlugin2();
  // Return type is auto_ptr<PiZeroVector>
  return_type operator()(const reco::PFJet&) const;
  // Hook to update PV information
  virtual void beginEvent();
  
 private:
  typedef std::vector<reco::PFCandidatePtr> PFCandPtrs;
  void addCandsToStrip(RecoTauPiZero&, PFCandPtrs&, const std::vector<bool>&, std::set<size_t>&, bool&) const;

  RecoTauVertexAssociator vertexAssociator_;

  RecoTauQualityCuts* qcuts_;
  bool applyElecTrackQcuts_;
  double minGammaEtStripSeed_;
  double minGammaEtStripAdd_;

  double minStripEt_;

  std::vector<int> inputPdgIds_;  // type of candidates to clusterize
  double etaAssociationDistance_; // size of strip clustering window in eta direction
  double phiAssociationDistance_; // size of strip clustering window in phi direction

  bool updateStripAfterEachDaughter_;
  int maxStripBuildIterations_;

  // Parameters for build strip combinations
  bool combineStrips_;
  int maxStrips_;
  double combinatoricStripMassHypo_;

  AddFourMomenta p4Builder_;
};

RecoTauPiZeroStripPlugin2::RecoTauPiZeroStripPlugin2(const edm::ParameterSet& pset)
  : RecoTauPiZeroBuilderPlugin(pset),
    vertexAssociator_(pset.getParameter<edm::ParameterSet>("qualityCuts")),
    qcuts_(0)
{
  LogTrace("RecoTauPiZeroStripPlugin2") << "<RecoTauPiZeroStripPlugin2::RecoTauPiZeroStripPlugin2>:" ;

  minGammaEtStripSeed_ = pset.getParameter<double>("minGammaEtStripSeed");
  LogTrace("RecoTauPiZeroStripPlugin2") << " minGammaEtStripSeed = " << minGammaEtStripSeed_ ;
  minGammaEtStripAdd_ = pset.getParameter<double>("minGammaEtStripAdd");
  LogTrace("RecoTauPiZeroStripPlugin2") << " minGammaEtStripAdd = " << minGammaEtStripAdd_ ;

  minStripEt_ = pset.getParameter<double>("minStripEt");
  LogTrace("RecoTauPiZeroStripPlugin2") << " minStripEt = " << minStripEt_ ;
  
  edm::ParameterSet qcuts_pset = pset.getParameterSet("qualityCuts").getParameterSet("signalQualityCuts");
//-------------------------------------------------------------------------------
// CV: disable track quality cuts for PFElectronsPFElectron
//       (treat PFElectrons like PFGammas for the purpose of building eta-phi strips)
  applyElecTrackQcuts_ = pset.getParameter<bool>("applyElecTrackQcuts");
  if ( !applyElecTrackQcuts_ ) {
    qcuts_pset.addParameter<double>("minTrackPt", std::min(minGammaEtStripSeed_, minGammaEtStripAdd_));
    qcuts_pset.addParameter<double>("maxTrackChi2", 1.e+9);
    qcuts_pset.addParameter<double>("maxTransverseImpactParameter", 1.e+9);
    qcuts_pset.addParameter<double>("maxDeltaZ", 1.);
    qcuts_pset.addParameter<double>("minTrackVertexWeight", -1.);
    qcuts_pset.addParameter<unsigned>("minTrackPixelHits", 0);
    qcuts_pset.addParameter<unsigned>("minTrackHits", 0);
  }
//-------------------------------------------------------------------------------
  qcuts_pset.addParameter<double>("minGammaEt", std::min(minGammaEtStripSeed_, minGammaEtStripAdd_));
  qcuts_ = new RecoTauQualityCuts(qcuts_pset);

  inputPdgIds_ = pset.getParameter<std::vector<int> >("stripCandidatesParticleIds");
  etaAssociationDistance_ = pset.getParameter<double>("stripEtaAssociationDistance");
  phiAssociationDistance_ = pset.getParameter<double>("stripPhiAssociationDistance");

  updateStripAfterEachDaughter_ = pset.getParameter<bool>("updateStripAfterEachDaughter");
  maxStripBuildIterations_ = pset.getParameter<int>("maxStripBuildIterations");

  combineStrips_ = pset.getParameter<bool>("makeCombinatoricStrips");
  if ( combineStrips_ ) {
    maxStrips_ = pset.getParameter<int>("maxInputStrips");
    combinatoricStripMassHypo_ = pset.getParameter<double>("stripMassWhenCombining");
  }
}
  
RecoTauPiZeroStripPlugin2::~RecoTauPiZeroStripPlugin2()
{
  delete qcuts_;
}

// Update the primary vertex
void RecoTauPiZeroStripPlugin2::beginEvent() 
{
  vertexAssociator_.setEvent(*evt());
}

void RecoTauPiZeroStripPlugin2::addCandsToStrip(RecoTauPiZero& strip, PFCandPtrs& cands, const std::vector<bool>& candFlags, 
						std::set<size_t>& candIdsCurrentStrip, bool& isCandAdded) const
{
  size_t numCands = cands.size();
  for ( size_t candId = 0; candId < numCands; ++candId ) {
    if ( (!candFlags[candId]) && candIdsCurrentStrip.find(candId) == candIdsCurrentStrip.end() ) { // do not include same cand twice
      reco::PFCandidatePtr cand = cands[candId];
      if ( fabs(strip.eta() - cand->eta()) < etaAssociationDistance_ && // check if cand is within eta-phi window centered on strip 
	   fabs(strip.phi() - cand->phi()) < phiAssociationDistance_ ) {
	LogTrace("RecoTauPiZeroStripPlugin2") << "--> candId = " << candId << " has been added." ;
	strip.addDaughter(cand);
	if ( updateStripAfterEachDaughter_ ) p4Builder_.set(strip);
	isCandAdded = true;
	candIdsCurrentStrip.insert(candId);
      }
    }
  }
}

void markCandsInStrip(std::vector<bool>& candFlags, const std::set<size_t>& candIds)
{
  for ( std::set<size_t>::const_iterator candId = candIds.begin();
	candId != candIds.end(); ++candId ) {
    candFlags[*candId] = true;
  }
}

namespace {
  const reco::TrackBaseRef getTrack(const PFCandidate& cand) 
  {
    if      ( cand.trackRef().isNonnull()    ) return reco::TrackBaseRef(cand.trackRef());
    else if ( cand.gsfTrackRef().isNonnull() ) return reco::TrackBaseRef(cand.gsfTrackRef());
    else return reco::TrackBaseRef();
  }
}

RecoTauPiZeroStripPlugin2::return_type RecoTauPiZeroStripPlugin2::operator()(const reco::PFJet& jet) const 
{
  PiZeroVector output;

  // Get the candidates passing our quality cuts
  qcuts_->setPV(vertexAssociator_.associatedVertex(jet));
  PFCandPtrs candsVector = qcuts_->filterCandRefs(pfCandidates(jet, inputPdgIds_));

  // Convert to stl::list to allow fast deletions
  PFCandPtrs seedCands;
  PFCandPtrs addCands;
  int idx = 0;
  for ( PFCandPtrs::iterator cand = candsVector.begin();
	cand != candsVector.end(); ++cand ) {
    LogTrace("RecoTauPiZeroStripPlugin2") << "PFGamma (" << idx << "): Et = " << (*cand)->et() << "," 
    	        << " eta = " << (*cand)->eta() << ", phi = " << (*cand)->phi(); 
    if ( (*cand)->et() > minGammaEtStripSeed_ ) {
      LogTrace("RecoTauPiZeroStripPlugin2") << " --> assigning seedCandId = " << seedCands.size() ;
#ifdef EDM_ML_DEBUG
        const reco::TrackBaseRef candTrack = getTrack(*cand);
        if ( candTrack.isNonnull() ) {
	LogTrace("RecoTauPiZeroStripPlugin2") << "has Track: pt = " << candTrack->pt() << " +/- " << candTrack->ptError() << "," 
		    << " eta = " << candTrack->eta() << ", phi = " << candTrack->phi() << ","
		    << " charge = " << candTrack->charge() ;
	LogTrace("RecoTauPiZeroStripPlugin2") << " chi2 = " << candTrack->normalizedChi2() ;
	LogTrace("RecoTauPiZeroStripPlugin2") << " dIP = " << std::abs(candTrack->dxy(vertexAssociator_.associatedVertex(jet)->position())) ;
	LogTrace("RecoTauPiZeroStripPlugin2") << " dZ = " << std::abs(candTrack->dz(vertexAssociator_.associatedVertex(jet)->position())) ;
	LogTrace("RecoTauPiZeroStripPlugin2") << " vtxAssocWeight = " << vertexAssociator_.associatedVertex(jet)->trackWeight(candTrack) ;
	LogTrace("RecoTauPiZeroStripPlugin2") << " numPxlHits = " << candTrack->hitPattern().numberOfValidPixelHits() ;
        LogTrace("RecoTauPiZeroStripPlugin2") << " numTrkHits = " << candTrack->hitPattern().numberOfValidHits() ;
	}
#endif
      LogTrace("RecoTauPiZeroStripPlugin2") << "ECAL Et: calibrated = " << (*cand)->ecalEnergy()*sin((*cand)->theta()) << "," 
      	  << " raw = " << (*cand)->rawEcalEnergy()*sin((*cand)->theta()) ;
      LogTrace("RecoTauPiZeroStripPlugin2") << "HCAL Et: calibrated = " << (*cand)->hcalEnergy()*sin((*cand)->theta()) << "," 
      	  << " raw = " << (*cand)->rawHcalEnergy()*sin((*cand)->theta()) ;
      seedCands.push_back(*cand);
    } else if ( (*cand)->et() > minGammaEtStripAdd_  ) {
      LogTrace("RecoTauPiZeroStripPlugin2") << " --> assigning addCandId = " << addCands.size();
      addCands.push_back(*cand);
    }
    LogTrace("RecoTauPiZeroStripPlugin2") ;
    ++idx;
  }

  std::vector<bool> seedCandFlags(seedCands.size()); // true/false: seedCand is already/not yet included in strip
  std::vector<bool> addCandFlags(addCands.size());   // true/false: addCand  is already/not yet included in strip

  std::set<size_t> seedCandIdsCurrentStrip;
  std::set<size_t> addCandIdsCurrentStrip;

  size_t idxSeed = 0;
  while ( idxSeed < seedCands.size() ) {
    LogTrace("RecoTauPiZeroStripPlugin2") << "idxSeed = " << idxSeed ;

    seedCandIdsCurrentStrip.clear();
    addCandIdsCurrentStrip.clear();

    std::auto_ptr<RecoTauPiZero> strip(new RecoTauPiZero(*seedCands[idxSeed], RecoTauPiZero::kStrips));
    strip->addDaughter(seedCands[idxSeed]);
    seedCandIdsCurrentStrip.insert(idxSeed);

    bool isCandAdded;
    int stripBuildIteration = 0;
    do {
      isCandAdded = false;

      LogTrace("RecoTauPiZeroStripPlugin2") << " adding seedCands to strip..." ;
      addCandsToStrip(*strip, seedCands, seedCandFlags, seedCandIdsCurrentStrip, isCandAdded);
      LogTrace("RecoTauPiZeroStripPlugin2") << " adding addCands to strip..." ;
      addCandsToStrip(*strip, addCands,  addCandFlags,  addCandIdsCurrentStrip, isCandAdded);

      if ( !updateStripAfterEachDaughter_ ) p4Builder_.set(*strip);

      ++stripBuildIteration;
    } while ( isCandAdded && (stripBuildIteration < maxStripBuildIterations_ || maxStripBuildIterations_ == -1) );

    if ( strip->et() > minStripEt_ ) { // strip passed Et cuts, add it to the event
      LogTrace("RecoTauPiZeroStripPlugin2") << "Strip: Et = " << strip->et() << "," 
      	  << " eta = " << strip->eta() << ", phi = " << strip->phi() 
      	  << " --> building it !!" ;

      // Update the vertex
      if ( strip->daughterPtr(0).isNonnull() ) strip->setVertex(strip->daughterPtr(0)->vertex());
      output.push_back(strip);

      // Mark daughters as being part of this strip
      markCandsInStrip(seedCandFlags, seedCandIdsCurrentStrip);
      markCandsInStrip(addCandFlags,  addCandIdsCurrentStrip);
    } else { // strip failed Et cuts, just skip it
      LogTrace("RecoTauPiZeroStripPlugin2") << "Strip: Et = " << strip->et() << "," 
    	  << " eta = " << strip->eta() << ", phi = " << strip->phi() 
      	  << " --> discarding it !!" ;
    }

    ++idxSeed;
    while ( idxSeed < seedCands.size() && seedCandFlags[idxSeed] ) {
      ++idxSeed; // fast-forward to next seed cand not yet included in any strip
    }
  }

  // Check if we want to combine our strips
  if ( combineStrips_ && output.size() > 1 ) {
    PiZeroVector stripCombinations;
    // Sort the output by descending pt
    output.sort(output.begin(), output.end(),
        boost::bind(&RecoTauPiZero::pt, _1) >
        boost::bind(&RecoTauPiZero::pt, _2));
    // Get the end of interesting set of strips to try and combine
    PiZeroVector::const_iterator end_iter = takeNElements(
        output.begin(), output.end(), maxStrips_);

    // Look at all the combinations
    for ( PiZeroVector::const_iterator first = output.begin();
	  first != end_iter-1; ++first ) {
      for ( PiZeroVector::const_iterator second = first+1;
	    second != end_iter; ++second ) {
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
        if ( combinedStrips->daughterPtr(0).isNonnull() )
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
    reco::tau::RecoTauPiZeroStripPlugin2, "RecoTauPiZeroStripPlugin2");
