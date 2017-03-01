/*
 * RecoTauPiZeroStripPlugin3
 *
 * Merges PFGammas in a PFJet into Candidate piZeros defined as
 * strips in eta-phi.
 *
 * Author: Michail Bachtis (University of Wisconsin)
 *
 * Code modifications: Evan Friis (UC Davis),
 *                     Christian Veelken (LLR)
 *
 */

#include <algorithm>
#include <memory>

#include "boost/bind.hpp"

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

#include "TString.h"
#include "TFormula.h"

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

class RecoTauPiZeroStripPlugin3 : public RecoTauPiZeroBuilderPlugin 
{
 public:
  explicit RecoTauPiZeroStripPlugin3(const edm::ParameterSet&, edm::ConsumesCollector &&iC);
  virtual ~RecoTauPiZeroStripPlugin3();
  // Return type is auto_ptr<PiZeroVector>
  return_type operator()(const reco::PFJet&) const override;
  // Hook to update PV information
  virtual void beginEvent() override;
  
 private:
  typedef std::vector<reco::PFCandidatePtr> PFCandPtrs;
  void addCandsToStrip(RecoTauPiZero&, PFCandPtrs&, const std::vector<bool>&, std::set<size_t>&, bool&) const;

  RecoTauVertexAssociator vertexAssociator_;

  std::unique_ptr<RecoTauQualityCuts> qcuts_;
  bool applyElecTrackQcuts_;
  double minGammaEtStripSeed_;
  double minGammaEtStripAdd_;

  double minStripEt_;

  std::vector<int> inputPdgIds_;  // type of candidates to clusterize
  std::unique_ptr<const TFormula> etaAssociationDistance_; // size of strip clustering window in eta direction
  std::unique_ptr<const TFormula> phiAssociationDistance_; // size of strip clustering window in phi direction

  bool updateStripAfterEachDaughter_;
  int maxStripBuildIterations_;

  // Parameters for build strip combinations
  bool combineStrips_;
  int maxStrips_;
  double combinatoricStripMassHypo_;

  AddFourMomenta p4Builder_;

  int verbosity_;
};

namespace
{
  std::unique_ptr<TFormula> makeFunction(const std::string& functionName, const edm::ParameterSet& pset)
  {
    TString formula = pset.getParameter<std::string>("function");
    formula = formula.ReplaceAll("pT", "x");
    std::unique_ptr<TFormula> function(new TFormula(functionName.data(), formula.Data()));
    int numParameter = function->GetNpar();
    for ( int idxParameter = 0; idxParameter < numParameter; ++idxParameter ) {
      std::string parameterName = Form("par%i", idxParameter);
      double parameter = pset.getParameter<double>(parameterName);
      function->SetParameter(idxParameter, parameter);
    }
    return function;
  }
}

RecoTauPiZeroStripPlugin3::RecoTauPiZeroStripPlugin3(const edm::ParameterSet& pset, edm::ConsumesCollector &&iC)
  : RecoTauPiZeroBuilderPlugin(pset, std::move(iC)),
    vertexAssociator_(pset.getParameter<edm::ParameterSet>("qualityCuts"), std::move(iC)), qcuts_(nullptr), etaAssociationDistance_(nullptr), phiAssociationDistance_(nullptr)
{
  minGammaEtStripSeed_ = pset.getParameter<double>("minGammaEtStripSeed");
  minGammaEtStripAdd_ = pset.getParameter<double>("minGammaEtStripAdd");

  minStripEt_ = pset.getParameter<double>("minStripEt");
  
  edm::ParameterSet qcuts_pset = pset.getParameterSet("qualityCuts").getParameterSet("signalQualityCuts");
//-------------------------------------------------------------------------------
// CV: disable track quality cuts for PFElectronsPFElectron
//       (treat PFElectrons like PFGammas for the purpose of building eta-phi strips)
  applyElecTrackQcuts_ = pset.getParameter<bool>("applyElecTrackQcuts");
  if ( !applyElecTrackQcuts_ ) {
    qcuts_pset.addParameter<double>("minTrackPt", std::min(minGammaEtStripSeed_, minGammaEtStripAdd_));
    qcuts_pset.addParameter<double>("maxTrackChi2", 1.e+9);
    qcuts_pset.addParameter<double>("maxTransverseImpactParameter", 1.e+9);
    qcuts_pset.addParameter<double>("maxDeltaZ", 1.e+9);
    qcuts_pset.addParameter<double>("minTrackVertexWeight", -1.);
    qcuts_pset.addParameter<unsigned>("minTrackPixelHits", 0);
    qcuts_pset.addParameter<unsigned>("minTrackHits", 0);
  }
//-------------------------------------------------------------------------------
  qcuts_pset.addParameter<double>("minGammaEt", std::min(minGammaEtStripSeed_, minGammaEtStripAdd_));
  //qcuts_ = new RecoTauQualityCuts(qcuts_pset);
  //std::unique_ptr<RecoTauQualityCuts> qcuts_(new RecoTauQualityCuts(qcuts_pset));

  qcuts_.reset(new RecoTauQualityCuts(qcuts_pset));

  inputPdgIds_ = pset.getParameter<std::vector<int> >("stripCandidatesParticleIds");
  edm::ParameterSet stripSize_eta_pset = pset.getParameterSet("stripEtaAssociationDistance");
  etaAssociationDistance_ = makeFunction("etaAssociationDistance", stripSize_eta_pset);
  edm::ParameterSet stripSize_phi_pset = pset.getParameterSet("stripPhiAssociationDistance");
  phiAssociationDistance_ = makeFunction("phiAssociationDistance", stripSize_phi_pset);

  updateStripAfterEachDaughter_ = pset.getParameter<bool>("updateStripAfterEachDaughter");
  maxStripBuildIterations_ = pset.getParameter<int>("maxStripBuildIterations");

  combineStrips_ = pset.getParameter<bool>("makeCombinatoricStrips");
  if ( combineStrips_ ) {
    maxStrips_ = pset.getParameter<int>("maxInputStrips");
    combinatoricStripMassHypo_ = pset.getParameter<double>("stripMassWhenCombining");
  }

  verbosity_ = ( pset.exists("verbosity") ) ?
    pset.getParameter<int>("verbosity") : 0;
}
RecoTauPiZeroStripPlugin3::~RecoTauPiZeroStripPlugin3()
{
} 

// Update the primary vertex
void RecoTauPiZeroStripPlugin3::beginEvent() 
{
  vertexAssociator_.setEvent(*evt());
}

void RecoTauPiZeroStripPlugin3::addCandsToStrip(RecoTauPiZero& strip, PFCandPtrs& cands, const std::vector<bool>& candFlags, 
						std::set<size_t>& candIdsCurrentStrip, bool& isCandAdded) const
{
  if ( verbosity_ >= 1 ) {
    edm::LogPrint("RecoTauPiZeroStripPlugin3") << "<RecoTauPiZeroStripPlugin3::addCandsToStrip>:" ;
  }
  size_t numCands = cands.size();
  for ( size_t candId = 0; candId < numCands; ++candId ) {
    if ( (!candFlags[candId]) && candIdsCurrentStrip.find(candId) == candIdsCurrentStrip.end() ) { // do not include same cand twice
      reco::PFCandidatePtr cand = cands[candId];
      double etaAssociationDistance_value = etaAssociationDistance_->Eval(strip.pt()) + etaAssociationDistance_->Eval(cand->pt());
      double phiAssociationDistance_value = phiAssociationDistance_->Eval(strip.pt()) + phiAssociationDistance_->Eval(cand->pt());
      if ( fabs(strip.eta() - cand->eta()) < etaAssociationDistance_value && // check if cand is within eta-phi window centered on strip 
	   reco::deltaPhi(strip.phi(), cand->phi()) < phiAssociationDistance_value ) {
	if ( verbosity_ >= 2 ) {
	  edm::LogPrint("RecoTauPiZeroStripPlugin3") << "--> adding PFCand #" << candId << " (" << cand.id() << ":" << cand.key() << "): Et = " << cand->et() << ", eta = " << cand->eta() << ", phi = " << cand->phi() ;
	}
	strip.addDaughter(cand);
	if ( updateStripAfterEachDaughter_ ) p4Builder_.set(strip);
	isCandAdded = true;
	candIdsCurrentStrip.insert(candId);
      }
    }
  }
}

namespace
{
  void markCandsInStrip(std::vector<bool>& candFlags, const std::set<size_t>& candIds)
  {
    for ( std::set<size_t>::const_iterator candId = candIds.begin();
	  candId != candIds.end(); ++candId ) {
      candFlags[*candId] = true;
    }
  }

  inline const reco::TrackBaseRef getTrack(const PFCandidate& cand)
  {
    if      ( cand.trackRef().isNonnull()    ) return reco::TrackBaseRef(cand.trackRef());
    else if ( cand.gsfTrackRef().isNonnull() ) return reco::TrackBaseRef(cand.gsfTrackRef());
    else return reco::TrackBaseRef();
  }
}

RecoTauPiZeroStripPlugin3::return_type RecoTauPiZeroStripPlugin3::operator()(const reco::PFJet& jet) const 
{
  if ( verbosity_ >= 1 ) {
    edm::LogPrint("RecoTauPiZeroStripPlugin3") << "<RecoTauPiZeroStripPlugin3::operator()>:" ;
    edm::LogPrint("RecoTauPiZeroStripPlugin3") << " minGammaEtStripSeed = " << minGammaEtStripSeed_ ;
    edm::LogPrint("RecoTauPiZeroStripPlugin3") << " minGammaEtStripAdd = " << minGammaEtStripAdd_ ;
    edm::LogPrint("RecoTauPiZeroStripPlugin3") << " minStripEt = " << minStripEt_ ;
  }

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
    if ( verbosity_ >= 1 ) {
      edm::LogPrint("RecoTauPiZeroStripPlugin3") << "PFGamma #" << idx << " (" << cand->id() << ":" << cand->key() << "): Et = " << (*cand)->et() << ", eta = " << (*cand)->eta() << ", phi = " << (*cand)->phi() ;
    } 
    if ( (*cand)->et() > minGammaEtStripSeed_ ) {
      if ( verbosity_ >= 2 ) {
	edm::LogPrint("RecoTauPiZeroStripPlugin3") << "--> assigning seedCandId = " << seedCands.size() ;
        const reco::TrackBaseRef candTrack = getTrack(*cand);
        if ( candTrack.isNonnull() ) {
	  edm::LogPrint("RecoTauPiZeroStripPlugin3") << "track: Pt = " << candTrack->pt() << " eta = " << candTrack->eta() << ", phi = " << candTrack->phi() << ", charge = " << candTrack->charge() ;
	  edm::LogPrint("RecoTauPiZeroStripPlugin3") << " (dZ = " << candTrack->dz(vertexAssociator_.associatedVertex(jet)->position()) << ", dXY = " << candTrack->dxy(vertexAssociator_.associatedVertex(jet)->position()) << "," 
		    << " numHits = " << candTrack->hitPattern().numberOfValidTrackerHits() << ", numPxlHits = " << candTrack->hitPattern().numberOfValidPixelHits() << "," 
		    << " chi2 = " << candTrack->normalizedChi2() << ", dPt/Pt = " << (candTrack->ptError()/candTrack->pt()) << ")" ;
	}
	edm::LogPrint("RecoTauPiZeroStripPlugin3") << "ECAL Et: calibrated = " << (*cand)->ecalEnergy()*sin((*cand)->theta()) << "," 
		  << " raw = " << (*cand)->rawEcalEnergy()*sin((*cand)->theta()) ;
	edm::LogPrint("RecoTauPiZeroStripPlugin3") << "HCAL Et: calibrated = " << (*cand)->hcalEnergy()*sin((*cand)->theta()) << "," 
		  << " raw = " << (*cand)->rawHcalEnergy()*sin((*cand)->theta()) ;
      }
      seedCands.push_back(*cand);
    } else if ( (*cand)->et() > minGammaEtStripAdd_  ) {
      if ( verbosity_ >= 2 ) {
	edm::LogPrint("RecoTauPiZeroStripPlugin3") << "--> assigning addCandId = " << addCands.size() ;
      }
      addCands.push_back(*cand);
    }
    ++idx;
  }

  std::vector<bool> seedCandFlags(seedCands.size()); // true/false: seedCand is already/not yet included in strip
  std::vector<bool> addCandFlags(addCands.size());   // true/false: addCand  is already/not yet included in strip

  std::set<size_t> seedCandIdsCurrentStrip;
  std::set<size_t> addCandIdsCurrentStrip;

  size_t idxSeed = 0;
  while ( idxSeed < seedCands.size() ) {
    if ( verbosity_ >= 2 ) edm::LogPrint("RecoTauPiZeroStripPlugin3") << "processing seed #" << idxSeed ;

    seedCandIdsCurrentStrip.clear();
    addCandIdsCurrentStrip.clear();

    std::auto_ptr<RecoTauPiZero> strip(new RecoTauPiZero(*seedCands[idxSeed], RecoTauPiZero::kStrips));
    strip->addDaughter(seedCands[idxSeed]);
    seedCandIdsCurrentStrip.insert(idxSeed);

    bool isCandAdded;
    int stripBuildIteration = 0;
    do {
      isCandAdded = false;

      //if ( verbosity_ >= 2 ) edm::LogPrint("RecoTauPiZeroStripPlugin3") << " adding seedCands to strip..." ;
      addCandsToStrip(*strip, seedCands, seedCandFlags, seedCandIdsCurrentStrip, isCandAdded);
      //if ( verbosity_ >= 2 ) edm::LogPrint("RecoTauPiZeroStripPlugin3") << " adding addCands to strip..." ;
      addCandsToStrip(*strip, addCands,  addCandFlags,  addCandIdsCurrentStrip, isCandAdded);

      if ( !updateStripAfterEachDaughter_ ) p4Builder_.set(*strip);

      ++stripBuildIteration;
    } while ( isCandAdded && (stripBuildIteration < maxStripBuildIterations_ || maxStripBuildIterations_ == -1) );

    if ( strip->et() > minStripEt_ ) { // strip passed Et cuts, add it to the event
      if ( verbosity_ >= 2 ) edm::LogPrint("RecoTauPiZeroStripPlugin3") << "Building strip: Et = " << strip->et() << ", eta = " << strip->eta() << ", phi = " << strip->phi() ;

      // Update the vertex
      if ( strip->daughterPtr(0).isNonnull() ) strip->setVertex(strip->daughterPtr(0)->vertex());
      output.push_back(strip);

      // Mark daughters as being part of this strip
      markCandsInStrip(seedCandFlags, seedCandIdsCurrentStrip);
      markCandsInStrip(addCandFlags,  addCandIdsCurrentStrip);
    } else { // strip failed Et cuts, just skip it
      if ( verbosity_ >= 2 ) edm::LogPrint("RecoTauPiZeroStripPlugin3") << "Discarding strip: Et = " << strip->et() << ", eta = " << strip->eta() << ", phi = " << strip->phi() ;
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
        for ( auto const& gamma : first->daughterPtrVector()) {
          combinedStrips->addDaughter(gamma);
        }
        for ( auto const& gamma : second->daughterPtrVector()) {
          combinedStrips->addDaughter(gamma);
        }
        // Update the vertex
        if ( combinedStrips->daughterPtr(0).isNonnull() ) {
          combinedStrips->setVertex(combinedStrips->daughterPtr(0)->vertex());
	}

        // Add to our collection of combined strips
        stripCombinations.push_back(combinedStrips);
      }
    }
    // When done doing all the combinations, add the combined strips to the
    // output.
    output.transfer(output.end(), stripCombinations);
  }

  // Compute correction to account for spread of photon energy in eta and phi
  // in case charged pions make nuclear interactions or photons convert within the tracking detector
  for ( PiZeroVector::iterator strip = output.begin();
	  strip != output.end(); ++strip ) {
    double bendCorrEta = 0.;
    double bendCorrPhi = 0.;
    double energySum   = 0.;
    for (auto const& gamma : strip->daughterPtrVector()) {
      bendCorrEta += (gamma->energy()*etaAssociationDistance_->Eval(gamma->pt()));
      bendCorrPhi += (gamma->energy()*phiAssociationDistance_->Eval(gamma->pt()));
      energySum += gamma->energy();
    }
    if ( energySum > 1.e-2 ) {
      bendCorrEta /= energySum;
      bendCorrPhi /= energySum;
    }
    //std::cout << "stripPt = " << strip->pt() << ": bendCorrEta = " << bendCorrEta << ", bendCorrPhi = " << bendCorrPhi << std::endl;
    strip->setBendCorrEta(bendCorrEta);
    strip->setBendCorrPhi(bendCorrPhi);
  }
  
  return output.release();
}
}} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauPiZeroBuilderPluginFactory,
    reco::tau::RecoTauPiZeroStripPlugin3, "RecoTauPiZeroStripPlugin3");
