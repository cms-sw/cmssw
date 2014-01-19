/*
 * PFRecoTauChargedHadronFromPFCandidatePlugin
 *
 * Build PFRecoTauChargedHadron objects
 * using charged PFCandidates and/or PFNeutralHadrons as input
 *
 * Author: Christian Veelken, LLR
 *
 */

#include "RecoTauTag/RecoTau/interface/PFRecoTauChargedHadronPlugins.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"
#include "RecoTauTag/RecoTau/interface/pfRecoTauChargedHadronAuxFunctions.h"

#include <memory>

namespace reco 
{ 

namespace tau
{ 

class PFRecoTauChargedHadronFromPFCandidatePlugin : public PFRecoTauChargedHadronBuilderPlugin 
{
 public:
  explicit PFRecoTauChargedHadronFromPFCandidatePlugin(const edm::ParameterSet&, edm::ConsumesCollector &&iC);
  virtual ~PFRecoTauChargedHadronFromPFCandidatePlugin();
  // Return type is auto_ptr<ChargedHadronVector>
  return_type operator()(const reco::PFJet&) const;
  // Hook to update PV information
  virtual void beginEvent();
  
 private:
  typedef std::vector<reco::PFCandidatePtr> PFCandPtrs;

  RecoTauVertexAssociator vertexAssociator_;

  RecoTauQualityCuts* qcuts_;

  std::vector<int> inputPdgIds_;  // type of candidates to clusterize

  double dRmergeNeutralHadronWrtChargedHadron_;
  double dRmergeNeutralHadronWrtNeutralHadron_;
  double dRmergeNeutralHadronWrtElectron_;
  double dRmergeNeutralHadronWrtOther_;
  int minBlockElementMatchesNeutralHadron_;
  int maxUnmatchedBlockElementsNeutralHadron_;
  double dRmergePhotonWrtChargedHadron_;
  double dRmergePhotonWrtNeutralHadron_;
  double dRmergePhotonWrtElectron_;
  double dRmergePhotonWrtOther_;
  int minBlockElementMatchesPhoton_;
  int maxUnmatchedBlockElementsPhoton_;
  double minMergeNeutralHadronEt_;
  double minMergeGammaEt_;
  double minMergeChargedHadronPt_;

  int verbosity_;
};

  PFRecoTauChargedHadronFromPFCandidatePlugin::PFRecoTauChargedHadronFromPFCandidatePlugin(const edm::ParameterSet& pset, edm::ConsumesCollector &&iC)
    : PFRecoTauChargedHadronBuilderPlugin(pset,std::move(iC)),
      vertexAssociator_(pset.getParameter<edm::ParameterSet>("qualityCuts"),std::move(iC)),
    qcuts_(0)
{
  edm::ParameterSet qcuts_pset = pset.getParameterSet("qualityCuts").getParameterSet("signalQualityCuts");
  qcuts_ = new RecoTauQualityCuts(qcuts_pset);

  inputPdgIds_ = pset.getParameter<std::vector<int> >("chargedHadronCandidatesParticleIds");

  dRmergeNeutralHadronWrtChargedHadron_ = pset.getParameter<double>("dRmergeNeutralHadronWrtChargedHadron");
  dRmergeNeutralHadronWrtNeutralHadron_ = pset.getParameter<double>("dRmergeNeutralHadronWrtNeutralHadron");
  dRmergeNeutralHadronWrtElectron_ = pset.getParameter<double>("dRmergeNeutralHadronWrtElectron");
  dRmergeNeutralHadronWrtOther_ = pset.getParameter<double>("dRmergeNeutralHadronWrtOther");
  minBlockElementMatchesNeutralHadron_ = pset.getParameter<int>("minBlockElementMatchesNeutralHadron");
  maxUnmatchedBlockElementsNeutralHadron_ = pset.getParameter<int>("maxUnmatchedBlockElementsNeutralHadron");
  dRmergePhotonWrtChargedHadron_ = pset.getParameter<double>("dRmergePhotonWrtChargedHadron");
  dRmergePhotonWrtNeutralHadron_ = pset.getParameter<double>("dRmergePhotonWrtNeutralHadron");
  dRmergePhotonWrtElectron_ = pset.getParameter<double>("dRmergePhotonWrtElectron");
  dRmergePhotonWrtOther_ = pset.getParameter<double>("dRmergePhotonWrtOther");
  minBlockElementMatchesPhoton_ = pset.getParameter<int>("minBlockElementMatchesPhoton");
  maxUnmatchedBlockElementsPhoton_ = pset.getParameter<int>("maxUnmatchedBlockElementsPhoton");
  minMergeNeutralHadronEt_ = pset.getParameter<double>("minMergeNeutralHadronEt");
  minMergeGammaEt_ = pset.getParameter<double>("minMergeGammaEt"); 
  minMergeChargedHadronPt_ = pset.getParameter<double>("minMergeChargedHadronPt"); 

  verbosity_ = ( pset.exists("verbosity") ) ?
    pset.getParameter<int>("verbosity") : 0;
}
  
PFRecoTauChargedHadronFromPFCandidatePlugin::~PFRecoTauChargedHadronFromPFCandidatePlugin()
{
  delete qcuts_;
}

// Update the primary vertex
void PFRecoTauChargedHadronFromPFCandidatePlugin::beginEvent() 
{
  vertexAssociator_.setEvent(*evt());
}

namespace
{
  std::string getPFCandidateType(reco::PFCandidate::ParticleType pfCandidateType)
  {
    if      ( pfCandidateType == reco::PFCandidate::X         ) return "undefined";
    else if ( pfCandidateType == reco::PFCandidate::h         ) return "PFChargedHadron";
    else if ( pfCandidateType == reco::PFCandidate::e         ) return "PFElectron";
    else if ( pfCandidateType == reco::PFCandidate::mu        ) return "PFMuon";
    else if ( pfCandidateType == reco::PFCandidate::gamma     ) return "PFGamma";
    else if ( pfCandidateType == reco::PFCandidate::h0        ) return "PFNeutralHadron";
    else if ( pfCandidateType == reco::PFCandidate::h_HF      ) return "HF_had";
    else if ( pfCandidateType == reco::PFCandidate::egamma_HF ) return "HF_em";
    else assert(0);
  }

  bool isMatchedByBlockElement(const reco::PFCandidate& pfCandidate1, const reco::PFCandidate& pfCandidate2, int minMatches1, int minMatches2, int maxUnmatchedBlockElements1plus2)
  {
    reco::PFCandidate::ElementsInBlocks blockElements1 = pfCandidate1.elementsInBlocks();
    int numBlocks1 = blockElements1.size();
    reco::PFCandidate::ElementsInBlocks blockElements2 = pfCandidate2.elementsInBlocks();
    int numBlocks2 = blockElements2.size();
    int numBlocks_matched = 0;
    for ( reco::PFCandidate::ElementsInBlocks::const_iterator blockElement1 = blockElements1.begin();
	  blockElement1 != blockElements1.end(); ++blockElement1 ) {
      bool isMatched = false;
      for ( reco::PFCandidate::ElementsInBlocks::const_iterator blockElement2 = blockElements2.begin();
	    blockElement2 != blockElements2.end(); ++blockElement2 ) {
	if ( blockElement1->first.id()  == blockElement2->first.id()  && 
	     blockElement1->first.key() == blockElement2->first.key() && 
	     blockElement1->second      == blockElement2->second      ) {
	  isMatched = true;
	}
      }
      if ( isMatched ) ++numBlocks_matched;
    }
    assert(numBlocks_matched <= numBlocks1);
    assert(numBlocks_matched <= numBlocks2);
    if ( numBlocks_matched >= minMatches1 && numBlocks_matched >= minMatches2 &&
	 ((numBlocks1 - numBlocks_matched) + (numBlocks2 - numBlocks_matched)) <= maxUnmatchedBlockElements1plus2 ) {
      return true;
    } else {
      return false;
    }
  }
}

PFRecoTauChargedHadronFromPFCandidatePlugin::return_type PFRecoTauChargedHadronFromPFCandidatePlugin::operator()(const reco::PFJet& jet) const 
{
  if ( verbosity_ ) {
    std::cout << "<PFRecoTauChargedHadronFromPFCandidatePlugin::operator()>:" << std::endl;
    std::cout << " pluginName = " << name() << std::endl;
  }

  ChargedHadronVector output;

  // Get the candidates passing our quality cuts
  qcuts_->setPV(vertexAssociator_.associatedVertex(jet));
  PFCandPtrs candsVector = qcuts_->filterCandRefs(pfCandidates(jet, inputPdgIds_));

  for ( PFCandPtrs::iterator cand = candsVector.begin();
	cand != candsVector.end(); ++cand ) {
    if ( verbosity_ ) {
      std::cout << "processing PFCandidate: Pt = " << (*cand)->pt() << ", eta = " << (*cand)->eta() << ", phi = " << (*cand)->phi() 
		<< " (type = " << getPFCandidateType((*cand)->particleId()) << ", charge = " << (*cand)->charge() << ")" << std::endl;
    }
    
    PFRecoTauChargedHadron::PFRecoTauChargedHadronAlgorithm algo = PFRecoTauChargedHadron::kUndefined;
    if ( fabs((*cand)->charge()) > 0.5 ) algo = PFRecoTauChargedHadron::kChargedPFCandidate;
    else algo = PFRecoTauChargedHadron::kPFNeutralHadron;
    std::auto_ptr<PFRecoTauChargedHadron> chargedHadron(new PFRecoTauChargedHadron(**cand, algo));
    if ( (*cand)->trackRef().isNonnull() ) chargedHadron->track_ = edm::refToPtr((*cand)->trackRef());
    else if ( (*cand)->muonRef().isNonnull() && (*cand)->muonRef()->innerTrack().isNonnull()  ) chargedHadron->track_ = edm::refToPtr((*cand)->muonRef()->innerTrack());
    else if ( (*cand)->muonRef().isNonnull() && (*cand)->muonRef()->globalTrack().isNonnull() ) chargedHadron->track_ = edm::refToPtr((*cand)->muonRef()->globalTrack());
    else if ( (*cand)->muonRef().isNonnull() && (*cand)->muonRef()->outerTrack().isNonnull()  ) chargedHadron->track_ = edm::refToPtr((*cand)->muonRef()->outerTrack());
    else if ( (*cand)->gsfTrackRef().isNonnull() ) chargedHadron->track_ = edm::refToPtr((*cand)->gsfTrackRef());
    chargedHadron->chargedPFCandidate_ = (*cand);
    chargedHadron->addDaughter(*cand);

    chargedHadron->positionAtECALEntrance_ = (*cand)->positionAtECALEntrance();

    reco::PFCandidate::ParticleType chargedPFCandidateType = chargedHadron->chargedPFCandidate_->particleId();

    if ( chargedHadron->pt() > minMergeChargedHadronPt_ ) {
      std::vector<reco::PFCandidatePtr> jetConstituents = jet.getPFConstituents();
      for ( std::vector<reco::PFCandidatePtr>::const_iterator jetConstituent = jetConstituents.begin();
	    jetConstituent != jetConstituents.end(); ++jetConstituent ) {
	// CV: take care of not double-counting energy in case "charged" PFCandidate is in fact a PFNeutralHadron
	if ( (*jetConstituent) == chargedHadron->chargedPFCandidate_ ) continue;
	
	reco::PFCandidate::ParticleType jetConstituentType = (*jetConstituent)->particleId();
	if ( !(jetConstituentType == reco::PFCandidate::h0 || jetConstituentType == reco::PFCandidate::gamma) ) continue;

	double dR = deltaR((*jetConstituent)->positionAtECALEntrance(), chargedHadron->positionAtECALEntrance_);
	double dRmerge = -1.;      
	int minBlockElementMatches = 1000;
	int maxUnmatchedBlockElements = 0;
	double minMergeEt = 1.e+6;
	if ( jetConstituentType == reco::PFCandidate::h0 ) {
	  if      ( chargedPFCandidateType == reco::PFCandidate::h  ) dRmerge = dRmergeNeutralHadronWrtChargedHadron_;
	  else if ( chargedPFCandidateType == reco::PFCandidate::h0 ) dRmerge = dRmergeNeutralHadronWrtNeutralHadron_;
	  else if ( chargedPFCandidateType == reco::PFCandidate::e  ) dRmerge = dRmergeNeutralHadronWrtElectron_;
	  else                                                        dRmerge = dRmergeNeutralHadronWrtOther_;
	  minBlockElementMatches = minBlockElementMatchesNeutralHadron_;
	  maxUnmatchedBlockElements = maxUnmatchedBlockElementsNeutralHadron_;
	  minMergeEt = minMergeNeutralHadronEt_;
	} else if ( jetConstituentType == reco::PFCandidate::gamma ) {
	  if      ( chargedPFCandidateType == reco::PFCandidate::h  ) dRmerge = dRmergePhotonWrtChargedHadron_;
	  else if ( chargedPFCandidateType == reco::PFCandidate::h0 ) dRmerge = dRmergePhotonWrtNeutralHadron_;
	  else if ( chargedPFCandidateType == reco::PFCandidate::e  ) dRmerge = dRmergePhotonWrtElectron_;
	  else                                                        dRmerge = dRmergePhotonWrtOther_;
	  minBlockElementMatches = minBlockElementMatchesPhoton_;
	  maxUnmatchedBlockElements = maxUnmatchedBlockElementsPhoton_;
	  minMergeEt = minMergeGammaEt_;
	}
	if ( (*jetConstituent)->et() > minMergeEt && 
	     (dR < dRmerge || isMatchedByBlockElement(**jetConstituent, *chargedHadron->chargedPFCandidate_, minBlockElementMatches, minBlockElementMatches, maxUnmatchedBlockElements)) ) {
	  chargedHadron->neutralPFCandidates_.push_back(*jetConstituent);
	  chargedHadron->addDaughter(*jetConstituent);
	}
      }
    }

    setChargedHadronP4(*chargedHadron);

    if ( verbosity_ ) {
      chargedHadron->print(std::cout);
    }
    // Update the vertex
    if ( chargedHadron->daughterPtr(0).isNonnull() ) chargedHadron->setVertex(chargedHadron->daughterPtr(0)->vertex());
    output.push_back(chargedHadron);
  }

  return output.release();
}

}} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(PFRecoTauChargedHadronBuilderPluginFactory, reco::tau::PFRecoTauChargedHadronFromPFCandidatePlugin, "PFRecoTauChargedHadronFromPFCandidatePlugin");
