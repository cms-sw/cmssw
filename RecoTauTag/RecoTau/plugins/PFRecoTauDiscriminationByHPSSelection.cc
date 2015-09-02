#include <boost/foreach.hpp>

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadronFwd.h"

namespace {
  // Apply a hypothesis on the mass of the strips.
  math::XYZTLorentzVector applyMassConstraint(const math::XYZTLorentzVector& vec, double mass) 
  {
    double factor = sqrt(vec.energy()*vec.energy() - mass*mass)/vec.P();
    return math::XYZTLorentzVector(vec.px()*factor, vec.py()*factor, vec.pz()*factor, vec.energy());
  }
}

class PFRecoTauDiscriminationByHPSSelection : public PFTauDiscriminationProducerBase 
{
 public:
  explicit PFRecoTauDiscriminationByHPSSelection(const edm::ParameterSet&);
  ~PFRecoTauDiscriminationByHPSSelection();
  double discriminate(const reco::PFTauRef&) const override;

 private:
  typedef StringObjectFunction<reco::PFTau> TauFunc;

  struct DecayModeCuts 
  {
    DecayModeCuts()
      : maxMass_(0)
    {}
    ~DecayModeCuts() {} // CV: maxMass object gets deleted by PFRecoTauDiscriminationByHPSSelection destructor
    unsigned nTracksMin_;
    unsigned nChargedPFCandsMin_;
    double minMass_;
    TauFunc* maxMass_;
    double minPi0Mass_;
    double maxPi0Mass_;
    double assumeStripMass_;
  };

  typedef std::pair<unsigned int, unsigned int> IntPair;
  typedef std::pair<double, double> DoublePair;
  typedef std::map<IntPair, DecayModeCuts> DecayModeCutMap;
  
  DecayModeCutMap decayModeCuts_;
  double matchingCone_;
  double minPt_;

  bool requireTauChargedHadronsToBeChargedPFCands_;
  
  int minPixelHits_;

  int verbosity_;
};

PFRecoTauDiscriminationByHPSSelection::PFRecoTauDiscriminationByHPSSelection(const edm::ParameterSet& pset)
  : PFTauDiscriminationProducerBase(pset)
{
  // Get the matchign cut
  matchingCone_ = pset.getParameter<double>("matchingCone");
  minPt_ = pset.getParameter<double>("minTauPt");
  // Get the mass cuts for each decay mode
  typedef std::vector<edm::ParameterSet> VPSet;
  const VPSet& decayModes = pset.getParameter<VPSet>("decayModes");
  BOOST_FOREACH( const edm::ParameterSet &decayMode, decayModes ) {
    // The mass window(s)
    DecayModeCuts cuts;
    if ( decayMode.exists("nTracksMin") ) {
      cuts.nTracksMin_ = decayMode.getParameter<unsigned>("nTracksMin");
    } else {
      cuts.nTracksMin_ = 0;
    }
    if ( decayMode.exists("nChargedPFCandsMin") ) {
      cuts.nChargedPFCandsMin_ = decayMode.getParameter<unsigned>("nChargedPFCandsMin");
    } else {
      cuts.nChargedPFCandsMin_ = 0;
    }
    cuts.minMass_ = decayMode.getParameter<double>("minMass");
    cuts.maxMass_ = new TauFunc(decayMode.getParameter<std::string>("maxMass"));
    if ( decayMode.exists("minPi0Mass") ) {
      cuts.minPi0Mass_ = decayMode.getParameter<double>("minPi0Mass");
      cuts.maxPi0Mass_ = decayMode.getParameter<double>("maxPi0Mass");
    } else {
      cuts.minPi0Mass_ = -1.e3;
      cuts.maxPi0Mass_ = 1.e9;
    }
    if ( decayMode.exists("assumeStripMass") ) {
      cuts.assumeStripMass_ = decayMode.getParameter<double>("assumeStripMass");
    } else {
      cuts.assumeStripMass_ = -1.0;
    }
    decayModeCuts_.insert(std::make_pair(
            // The decay mode as a key
            std::make_pair(
                decayMode.getParameter<uint32_t>("nCharged"),
                decayMode.getParameter<uint32_t>("nPiZeros")),
            cuts
          ));
  }
  requireTauChargedHadronsToBeChargedPFCands_ = pset.getParameter<bool>("requireTauChargedHadronsToBeChargedPFCands");
  minPixelHits_ = pset.getParameter<int>("minPixelHits");
  verbosity_ = pset.exists("verbosity") ?
    pset.getParameter<int>("verbosity") : 0;
  

}

PFRecoTauDiscriminationByHPSSelection::~PFRecoTauDiscriminationByHPSSelection()
{
  for ( DecayModeCutMap::iterator it = decayModeCuts_.begin();
	it != decayModeCuts_.end(); ++it ) {
    delete it->second.maxMass_;
  }
}

double
PFRecoTauDiscriminationByHPSSelection::discriminate(const reco::PFTauRef& tau) const
{
  if ( verbosity_ ) {
    edm::LogPrint("PFTauByHPSSelect") << "<PFRecoTauDiscriminationByHPSSelection::discriminate>:" ;
    edm::LogPrint("PFTauByHPSSelect") << " nCharged = " << tau->signalTauChargedHadronCandidates().size() ;
    edm::LogPrint("PFTauByHPSSelect") << " nPiZeros = " << tau->signalPiZeroCandidates().size() ;
  }

  // Check if we pass the min pt
  if ( tau->pt() < minPt_ ) {
    if ( verbosity_ ) {
      edm::LogPrint("PFTauByHPSSelect") << " fails minPt cut." ;
    }
    return 0.0;
  }

  // See if we select this decay mode
  DecayModeCutMap::const_iterator massWindowIter =
      decayModeCuts_.find(std::make_pair(tau->signalTauChargedHadronCandidates().size(),
                                         tau->signalPiZeroCandidates().size()));
  // Check if decay mode is supported
  if ( massWindowIter == decayModeCuts_.end() ) {
    if ( verbosity_ ) {
      edm::LogPrint("PFTauByHPSSelect") << " fails mass-window definition requirement." ;
    }
    return 0.0;
  }
  const DecayModeCuts& massWindow = massWindowIter->second;

  if ( massWindow.nTracksMin_ > 0 ) {
    unsigned int nTracks = 0;
    const std::vector<reco::PFRecoTauChargedHadron>& chargedHadrons = tau->signalTauChargedHadronCandidates();
    for ( std::vector<reco::PFRecoTauChargedHadron>::const_iterator chargedHadron = chargedHadrons.begin();
	  chargedHadron != chargedHadrons.end(); ++chargedHadron ) {
      if ( chargedHadron->algoIs(reco::PFRecoTauChargedHadron::kChargedPFCandidate) || chargedHadron->algoIs(reco::PFRecoTauChargedHadron::kTrack) ) {
	++nTracks;
      }
    }
    if ( verbosity_ ) {
      edm::LogPrint("PFTauByHPSSelect") << "nTracks = " << nTracks << " (min = " << massWindow.nTracksMin_ << ")" ;
    }
    if ( nTracks < massWindow.nTracksMin_ ) {
      if ( verbosity_ ) {
	edm::LogPrint("PFTauByHPSSelect") << " fails nTracks requirement for mass-window." ;
      }
      return 0.0;
    }
  }
  if ( massWindow.nChargedPFCandsMin_ > 0 ) {
    unsigned int nChargedPFCands = 0;
    const std::vector<reco::PFRecoTauChargedHadron>& chargedHadrons = tau->signalTauChargedHadronCandidates();
    for ( std::vector<reco::PFRecoTauChargedHadron>::const_iterator chargedHadron = chargedHadrons.begin();
	  chargedHadron != chargedHadrons.end(); ++chargedHadron ) {
      if ( chargedHadron->algoIs(reco::PFRecoTauChargedHadron::kChargedPFCandidate) ) {
	++nChargedPFCands;
      }
    }
    if ( verbosity_ ) {
      edm::LogPrint("PFTauByHPSSelect") << "nChargedPFCands = " << nChargedPFCands << " (min = " << massWindow.nChargedPFCandsMin_ << ")" ;
    }
    if ( nChargedPFCands < massWindow.nChargedPFCandsMin_ ) {
      if ( verbosity_ ) {
	edm::LogPrint("PFTauByHPSSelect") << " fails nChargedPFCands requirement for mass-window." ;
      }
      return 0.0;
    }
  }

  math::XYZTLorentzVector tauP4 = tau->p4();
  if ( verbosity_ ) {
    edm::LogPrint("PFTauByHPSSelect") << "tau: Pt = " << tauP4.pt() << ", eta = " << tauP4.eta() << ", phi = " << tauP4.phi() << ", mass = " << tauP4.mass() ;
  }
  // Find the total pizero p4
  reco::Candidate::LorentzVector stripsP4;
  BOOST_FOREACH(const reco::RecoTauPiZero& cand, 
      tau->signalPiZeroCandidates()){
    math::XYZTLorentzVector candP4 = cand.p4();
    stripsP4 += candP4;
  }

  // Apply strip mass assumption corrections
  if (massWindow.assumeStripMass_ >= 0) {
    BOOST_FOREACH(const reco::RecoTauPiZero& cand, 
        tau->signalPiZeroCandidates()){
      math::XYZTLorentzVector uncorrected = cand.p4();
      math::XYZTLorentzVector corrected = 
        applyMassConstraint(uncorrected, massWindow.assumeStripMass_);
      math::XYZTLorentzVector correction = corrected - uncorrected;
      tauP4 += correction;
      stripsP4 += correction;
    }
  }
  if ( verbosity_ ) {
    edm::LogPrint("PFTauByHPSSelect") << "strips: Pt = " << stripsP4.pt() << ", eta = " << stripsP4.eta() << ", phi = " << stripsP4.phi() << ", mass = " << stripsP4.mass() ;
  }

  // Check if tau fails mass cut
  double maxMass_value = (*massWindow.maxMass_)(*tau);
  if ( !((tauP4.M() - tau->bendCorrMass()) < maxMass_value && (tauP4.M() + tau->bendCorrMass()) > massWindow.minMass_) ) {
    if ( verbosity_ ) {
      edm::LogPrint("PFTauByHPSSelect") << " fails tau mass-window cut." ;
    }
    return 0.0;
  }

  // Check if it fails the pi0 IM cut
  if ( stripsP4.M() > massWindow.maxPi0Mass_ ||
       stripsP4.M() < massWindow.minPi0Mass_ ) {
    if ( verbosity_ ) {
      edm::LogPrint("PFTauByHPSSelect") << " fails strip mass-window cut." ;
    }
    return 0.0;
  }

  // Check if tau passes matching cone cut
  //edm::LogPrint("PFTauByHPSSelect") << "dR(tau, jet) = " << deltaR(tauP4, tau->jetRef()->p4()) ;
  if ( deltaR(tauP4, tau->jetRef()->p4()) > matchingCone_ ) {
    if ( verbosity_ ) {
      edm::LogPrint("PFTauByHPSSelect") << " fails matching-cone cut." ;
    }
    return 0.0;
  }
  
  // Check if tau passes cone cut
  double cone_size = tau->signalConeSize();
  // Check if any charged objects fail the signal cone cut
  for (auto const& cand : tau->signalTauChargedHadronCandidates()) {
    if ( verbosity_ ) {
      edm::LogPrint("PFTauByHPSSelect") << "dR(tau, signalPFChargedHadr) = " << deltaR(cand.p4(), tauP4) ;
    }
    if ( deltaR(cand.p4(), tauP4) > cone_size ) {
      if ( verbosity_ ) {
	edm::LogPrint("PFTauByHPSSelect") << " fails signal-cone cut for charged hadron(s)." ;
      }
      return 0.0;
    }
  }
  // Now check the pizeros
  for (auto const& cand : tau->signalPiZeroCandidates()) {
    double dEta = std::max(0., fabs(cand.eta() - tauP4.eta()) - cand.bendCorrEta());
    double dPhi = std::max(0., std::abs(reco::deltaPhi(cand.phi(), tauP4.phi())) - cand.bendCorrPhi());
    double dR2 = dEta*dEta + dPhi*dPhi;
    if ( verbosity_ ) {
      edm::LogPrint("PFTauByHPSSelect") << "dR2(tau, signalPiZero) = " << dR2 ;
    }
    if ( dR2 > cone_size*cone_size ) {
      if ( verbosity_ ) {
	edm::LogPrint("PFTauByHPSSelect") << " fails signal-cone cut for strip(s)." ;
      }
      return 0.0;
    }
  }

  if ( requireTauChargedHadronsToBeChargedPFCands_ ) {
    BOOST_FOREACH(const reco::PFRecoTauChargedHadron& cand, tau->signalTauChargedHadronCandidates()) {
      if ( verbosity_ ) {
	std::string algo_string;
	if      ( cand.algo() == reco::PFRecoTauChargedHadron::kChargedPFCandidate ) algo_string = "ChargedPFCandidate";
	else if ( cand.algo() == reco::PFRecoTauChargedHadron::kTrack              ) algo_string = "Track";
	else if ( cand.algo() == reco::PFRecoTauChargedHadron::kPFNeutralHadron    ) algo_string = "PFNeutralHadron";
	else                                                                         algo_string = "Undefined";
	edm::LogPrint("PFTauByHPSSelect") << "algo(signalPFChargedHadr) = " << algo_string ;
      }
      if ( !(cand.algo() == reco::PFRecoTauChargedHadron::kChargedPFCandidate) ) {
	if ( verbosity_ ) {
	  edm::LogPrint("PFTauByHPSSelect") << " fails cut on PFRecoTauChargedHadron algo." ;
	}
	return 0.0;
      }
    }
  }

  if ( minPixelHits_ > 0 ) {
    int numPixelHits = 0;
    const std::vector<reco::PFCandidatePtr>& chargedHadrCands = tau->signalPFChargedHadrCands();
    for ( std::vector<reco::PFCandidatePtr>::const_iterator chargedHadrCand = chargedHadrCands.begin();
	  chargedHadrCand != chargedHadrCands.end(); ++chargedHadrCand ) {
      const reco::Track* track = 0;
      if ( (*chargedHadrCand)->trackRef().isNonnull() ) track = (*chargedHadrCand)->trackRef().get();
      else if ( (*chargedHadrCand)->gsfTrackRef().isNonnull() ) track = (*chargedHadrCand)->gsfTrackRef().get();
      if ( track ) {
	numPixelHits += track->hitPattern().numberOfValidPixelHits();
      }
    }
    if ( !(numPixelHits >= minPixelHits_) ) {
      if ( verbosity_ ) {
	edm::LogPrint("PFTauByHPSSelect") << " fails cut on sum of pixel hits." ;
      }
      return 0.0;
    }
  }

  // Otherwise, we pass!
  if ( verbosity_ ) {
    edm::LogPrint("PFTauByHPSSelect") << " passes all cuts." ;
  }
  return 1.0;
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByHPSSelection);
