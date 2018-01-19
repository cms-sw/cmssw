
/** \class PFRecoBaseTauDiscriminationAgainstMuonSimple
 *
 * Compute tau Id. discriminator against muons for MiniAOD.
 * 
 * \author Michal Bluj, NCBJ Warsaw
 * based on PFRecoTauDiscriminationAgainstMuon2 by Christian Veelken
 *
 * Note: it is not granted that information on muon track is/will be always 
 * accesible with MiniAOD, if not one can consider to veto muons which are 
 * not only Trk by also STA or RPC muons, i.e. have many (>=2) good muon segments 
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <vector>
#include <string>
#include <iostream>

namespace {

class PFRecoBaseTauDiscriminationAgainstMuonSimple final : public PFBaseTauDiscriminationProducerBase
{
 public:
  explicit PFRecoBaseTauDiscriminationAgainstMuonSimple(const edm::ParameterSet& cfg)
    : PFBaseTauDiscriminationProducerBase(cfg),
      moduleLabel_(cfg.getParameter<std::string>("@module_label"))
  {   
    hop_ = cfg.getParameter<double>("HoPMin");
    doCaloMuonVeto_ = cfg.getParameter<bool>("doCaloMuonVeto");
    srcPatMuons_ = cfg.getParameter<edm::InputTag>("srcPatMuons");
    Muons_token = consumes<pat::MuonCollection>(srcPatMuons_);
    dRmuonMatch_ = cfg.getParameter<double>("dRmuonMatch");
    dRmuonMatchLimitedToJetArea_ = cfg.getParameter<bool>("dRmuonMatchLimitedToJetArea");
    minPtMatchedMuon_ = cfg.getParameter<double>("minPtMatchedMuon");
    maxNumberOfMatches_ = cfg.getParameter<int>("maxNumberOfMatches");
    maxNumberOfHitsLast2Stations_ = cfg.getParameter<int>("maxNumberOfHitsLast2Stations");
    typedef std::vector<int> vint;
    maskMatchesDT_  = cfg.getParameter<vint>("maskMatchesDT");
    maskMatchesCSC_ = cfg.getParameter<vint>("maskMatchesCSC");
    maskMatchesRPC_ = cfg.getParameter<vint>("maskMatchesRPC");
    maskHitsDT_     = cfg.getParameter<vint>("maskHitsDT");
    maskHitsCSC_    = cfg.getParameter<vint>("maskHitsCSC");
    maskHitsRPC_    = cfg.getParameter<vint>("maskHitsRPC");
    maxNumberOfSTAMuons_ = cfg.getParameter<int>("maxNumberOfSTAMuons");
    maxNumberOfRPCMuons_ = cfg.getParameter<int>("maxNumberOfRPCMuons");

    numWarnings_ = 0;
    maxWarnings_ = 3;
    verbosity_ = cfg.exists("verbosity") ? cfg.getParameter<int>("verbosity") : 0;
   }
  ~PFRecoBaseTauDiscriminationAgainstMuonSimple() override {} 

  void beginEvent(const edm::Event&, const edm::EventSetup&) override;

  double discriminate(const reco::PFBaseTauRef&) const override;

 private:  
  std::string moduleLabel_;
  int discriminatorOption_;
  double hop_;
  bool doCaloMuonVeto_;
  edm::InputTag srcPatMuons_;
  edm::Handle<pat::MuonCollection> muons_;
  edm::EDGetTokenT<pat::MuonCollection> Muons_token;
  double dRmuonMatch_;
  bool dRmuonMatchLimitedToJetArea_;
  double minPtMatchedMuon_;
  int maxNumberOfMatches_;
  std::vector<int> maskMatchesDT_;
  std::vector<int> maskMatchesCSC_;
  std::vector<int> maskMatchesRPC_;
  int maxNumberOfHitsLast2Stations_;
  std::vector<int> maskHitsDT_;
  std::vector<int> maskHitsCSC_;
  std::vector<int> maskHitsRPC_;
  int maxNumberOfSTAMuons_, maxNumberOfRPCMuons_;

  mutable int numWarnings_;
  int maxWarnings_;
  int verbosity_;
};

void PFRecoBaseTauDiscriminationAgainstMuonSimple::beginEvent(const edm::Event& evt, const edm::EventSetup& es) 
{
  evt.getByToken(Muons_token, muons_);
}

namespace
{
  /* MB: not used in current implementation, but keep it if needed in future
  const reco::Track* getTrack(const reco::Candidate& cand) {
    const pat::PackedCandidate* pCand = dynamic_cast<const pat::PackedCandidate*>(&cand);
    if ( pCand != nullptr && pCand->hasTrackDetails() )
      return &pCand->pseudoTrack();

    return nullptr;
  }
  */
  void countHits(const pat::Muon& muon, std::vector<int>& numHitsDT, std::vector<int>& numHitsCSC, std::vector<int>& numHitsRPC)
  {
    if ( muon.outerTrack().isNonnull() ) {
      const reco::HitPattern &muonHitPattern = muon.outerTrack()->hitPattern();
      for (int iHit = 0; iHit < muonHitPattern.numberOfAllHits(reco::HitPattern::TRACK_HITS); ++iHit) {
          uint32_t hit = muonHitPattern.getHitPattern(reco::HitPattern::TRACK_HITS, iHit);
	if ( hit == 0 ) break;	    
	if ( muonHitPattern.muonHitFilter(hit) && (muonHitPattern.getHitType(hit) == TrackingRecHit::valid || muonHitPattern.getHitType(hit) == TrackingRecHit::bad) ) {
	  int muonStation = muonHitPattern.getMuonStation(hit) - 1; // CV: map into range 0..3
	  if ( muonStation >= 0 && muonStation < 4 ) {
	    if      ( muonHitPattern.muonDTHitFilter(hit)  ) ++numHitsDT[muonStation];
	    else if ( muonHitPattern.muonCSCHitFilter(hit) ) ++numHitsCSC[muonStation];
	    else if ( muonHitPattern.muonRPCHitFilter(hit) ) ++numHitsRPC[muonStation];
	  }
	}
      }
    }
  }

  void countMatches(const pat::Muon& muon, std::vector<int>& numMatchesDT, std::vector<int>& numMatchesCSC, std::vector<int>& numMatchesRPC)
  {
    const std::vector<reco::MuonChamberMatch>& muonSegments = muon.matches();
    for ( std::vector<reco::MuonChamberMatch>::const_iterator muonSegment = muonSegments.begin();
	  muonSegment != muonSegments.end(); ++muonSegment ) {
      if ( muonSegment->segmentMatches.empty() ) continue;
      int muonDetector = muonSegment->detector();
      int muonStation = muonSegment->station() - 1;
      assert(muonStation >= 0 && muonStation <= 3);
      if      ( muonDetector == MuonSubdetId::DT  ) ++numMatchesDT[muonStation];
      else if ( muonDetector == MuonSubdetId::CSC ) ++numMatchesCSC[muonStation];
      else if ( muonDetector == MuonSubdetId::RPC ) ++numMatchesRPC[muonStation];      
    }
  }

  std::string format_vint(const std::vector<int>& vi)
  {
    std::ostringstream os;    
    os << "{ ";
    unsigned numEntries = vi.size();
    for ( unsigned iEntry = 0; iEntry < numEntries; ++iEntry ) {
      os << vi[iEntry];
      if ( iEntry < (numEntries - 1) ) os << ", ";
    }
    os << " }";
    return os.str();
  }
  
}

double PFRecoBaseTauDiscriminationAgainstMuonSimple::discriminate(const reco::PFBaseTauRef& pfTau) const
{
  if ( verbosity_ ) {
    edm::LogPrint("PFBaseTauAgainstMuonSimple") << "<PFRecoBaseTauDiscriminationAgainstMuonSimple::discriminate>:" ;
    edm::LogPrint("PFBaseTauAgainstMuonSimple") << " moduleLabel = " << moduleLabel_ ;
    edm::LogPrint("PFBaseTauAgainstMuonSimple") << "tau #" << pfTau.key() << ": Pt = " << pfTau->pt() << ", eta = " << pfTau->eta() << ", phi = " << pfTau->phi() << ", decay mode = " << pfTau->decayMode() ;   
  }

  //if (pfTau->decayMode() >= 5) return true; //MB: accept all multi-prongs??

  const reco::CandidatePtr& pfLeadChargedHadron = pfTau->leadPFChargedHadrCand();
  bool passesCaloMuonVeto = true;
  if ( pfLeadChargedHadron.isNonnull() ) {
    const pat::PackedCandidate* pCand = dynamic_cast<const pat::PackedCandidate*>(pfLeadChargedHadron.get());
    if ( pCand != nullptr ) {
      double rawCaloEnergyFraction = pCand->rawCaloFraction();
      //if ( !(rawCaloEnergyFraction > 0.) ) rawCaloEnergyFraction = 99; //MB: hack against cases when rawCaloEnergyFraction is not stored; it makes performance of the H/P cut rather poor
      if ( verbosity_ ) {
	edm::LogPrint("PFBaseTauAgainstMuonSimple") << "decayMode = " << pfTau->decayMode() << ", rawCaloEnergy(ECAL+HCAL)Fraction = " << rawCaloEnergyFraction << ", leadPFChargedHadronP = " << pCand->p() << ", leadPFChargedHadron pdgId = " << pCand->pdgId();
      }
      if ( pfTau->decayMode() == 0 && rawCaloEnergyFraction < hop_ ) passesCaloMuonVeto = false;
    }
  }
  if ( doCaloMuonVeto_ && !passesCaloMuonVeto ) {
    if ( verbosity_ ) edm::LogPrint("PFBaseTauAgainstMuonSimple") << "--> CaloMuonVeto failed, returning 0";
    return 0.;
  }

  //iterate over muons to find ones to use for discrimination
  std::vector<const pat::Muon*> muonsToCheck;
  size_t numMuons = muons_->size();
  for ( size_t idxMuon = 0; idxMuon < numMuons; ++idxMuon ) {
    bool matched = false;
    const pat::MuonRef muon(muons_, idxMuon);
    if ( verbosity_ ) edm::LogPrint("PFBaseTauAgainstMuonSimple") << "muon #" << muon.key() << ": Pt = " << muon->pt() << ", eta = " << muon->eta() << ", phi = " << muon->phi() ;
    //firsty check if the muon corrsponds with the leading tau particle
    for ( size_t iCand = 0; iCand < muon->numberOfSourceCandidatePtrs(); ++iCand) {
      const reco::CandidatePtr& srcCand = muon->sourceCandidatePtr(iCand);
      if (srcCand.isNonnull() && pfLeadChargedHadron.isNonnull() &&
	  srcCand.id()==pfLeadChargedHadron.id() &&
	  srcCand.key()==pfLeadChargedHadron.key() ) {
	muonsToCheck.push_back(muon.get());
	matched = true;
	if ( verbosity_ ) edm::LogPrint("PFBaseTauAgainstMuonSimple") << " tau has muonRef." ;
	break;
      }
    }
    if (matched) continue;
    //check dR matching
    if ( !(muon->pt() > minPtMatchedMuon_) ) {
      if ( verbosity_ ){ edm::LogPrint("PFBaseTauAgainstMuonSimple") << " fails Pt cut --> skipping it." ;}
      continue;
    }
    double dR = deltaR(muon->p4(), pfTau->p4());
    double dRmatch = dRmuonMatch_;
    if ( dRmuonMatchLimitedToJetArea_ ) {
      double jetArea = 0.;
      if ( pfTau->jetRef().isNonnull() ) jetArea = pfTau->jetRef()->jetArea();
      if ( jetArea > 0. ) {
	dRmatch = TMath::Min(dRmatch, TMath::Sqrt(jetArea/TMath::Pi()));
      } else {
	if ( numWarnings_ < maxWarnings_ ) {
	  edm::LogInfo("PFRecoBaseTauDiscriminationAgainstMuonSimple::discriminate") 
	    << "Jet associated to Tau: Pt = " << pfTau->pt() << ", eta = " << pfTau->eta() << ", phi = " << pfTau->phi() << " has area = " << jetArea << " !!" ;
	  ++numWarnings_;
	}
	dRmatch = pfTau->signalConeSize();
      }
      dRmatch = TMath::Max(dRmatch,pfTau->signalConeSize());
      if ( dR < dRmatch ) {
	if ( verbosity_ ) edm::LogPrint("PFBaseTauAgainstMuonSimple") << " overlaps with tau, dR = " << dR ;
	muonsToCheck.push_back(muon.get());
      }
    }
  }
  //now examine selected muons
  bool pass = true;
  std::vector<int> numMatchesDT(4);
  std::vector<int> numMatchesCSC(4);
  std::vector<int> numMatchesRPC(4);
  std::vector<int> numHitsDT(4);
  std::vector<int> numHitsCSC(4);
  std::vector<int> numHitsRPC(4);
  //MB: clear counters of matched segments and hits globally as in the AgainstMuon2 discriminant, but note that they will sum for all matched muons. However it is not likely that there is more than one matched muon.
  for ( int iStation = 0; iStation < 4; ++iStation ) {
    numMatchesDT[iStation]  = 0;
    numMatchesCSC[iStation] = 0;
    numMatchesRPC[iStation] = 0;
    numHitsDT[iStation]  = 0;
    numHitsCSC[iStation] = 0;
    numHitsRPC[iStation] = 0;
  } 
  int numSTAMuons=0, numRPCMuons=0;
  if ( verbosity_ && !muonsToCheck.empty() ) edm::LogPrint("PFBaseTauAgainstMuonSimple") << "Muons to check (" << muonsToCheck.size() << "):";
  size_t iMu = 0;
  for (const auto &mu: muonsToCheck) {
    if ( mu->isStandAloneMuon() ) numSTAMuons++;
    if ( mu->muonID("RPCMuLoose") ) numRPCMuons++;
    countMatches(*mu, numMatchesDT, numMatchesCSC, numMatchesRPC);
    int numStationsWithMatches = 0;
    for ( int iStation = 0; iStation < 4; ++iStation ) {
      if ( numMatchesDT[iStation]  > 0 && !maskMatchesDT_[iStation]  ) ++numStationsWithMatches;
      if ( numMatchesCSC[iStation] > 0 && !maskMatchesCSC_[iStation] ) ++numStationsWithMatches;
      if ( numMatchesRPC[iStation] > 0 && !maskMatchesRPC_[iStation] ) ++numStationsWithMatches;
    }
    countHits(*mu, numHitsDT, numHitsCSC, numHitsRPC);
    int numLast2StationsWithHits = 0;
    for ( int iStation = 2; iStation < 4; ++iStation ) {
      if ( numHitsDT[iStation]  > 0 && !maskHitsDT_[iStation]  ) ++numLast2StationsWithHits;
      if ( numHitsCSC[iStation] > 0 && !maskHitsCSC_[iStation] ) ++numLast2StationsWithHits;
      if ( numHitsRPC[iStation] > 0 && !maskHitsRPC_[iStation] ) ++numLast2StationsWithHits;
    }
    if ( verbosity_ )
      edm::LogPrint("PFBaseTauAgainstMuonSimple") 
	<< "\t" << iMu << ": Pt = " << mu->pt() << ", eta = " << mu->eta() << ", phi = " << mu->phi() 
	<< "\n\t" 
	<< "   isSTA: "<<mu->isStandAloneMuon()
	<< ", isRPCLoose: "<<mu->muonID("RPCMuLoose")
	<< "\n\t   numMatchesDT  = " << format_vint(numMatchesDT)
	<< "\n\t   numMatchesCSC = " << format_vint(numMatchesCSC)
	<< "\n\t   numMatchesRPC = " << format_vint(numMatchesRPC)
	<< "\n\t   --> numStationsWithMatches = " << numStationsWithMatches
	<< "\n\t   numHitsDT  = " << format_vint(numHitsDT)
	<< "\n\t   numHitsCSC = " << format_vint(numHitsCSC)
	<< "\n\t   numHitsRPC = " << format_vint(numHitsRPC)
	<< "\n\t   --> numLast2StationsWithHits = " << numLast2StationsWithHits ;
    if ( maxNumberOfMatches_ >= 0 && numStationsWithMatches > maxNumberOfMatches_ ) {
      pass = false;
      break;
    }
    if ( maxNumberOfHitsLast2Stations_ >= 0 && numLast2StationsWithHits > maxNumberOfHitsLast2Stations_ ) {
      pass = false;
      break;
    }
    if ( maxNumberOfSTAMuons_ >= 0 && numSTAMuons > maxNumberOfSTAMuons_ ) {
      pass = false;
      break;
    }
    if ( maxNumberOfRPCMuons_ >= 0 && numRPCMuons > maxNumberOfRPCMuons_ ) {
      pass = false;
      break;
    }
    iMu++;
  }

  double discriminatorValue = pass ? 1.: 0.;
  if ( verbosity_ ) edm::LogPrint("PFBaseTauAgainstMuonSimple") << "--> returning discriminatorValue = " << discriminatorValue ;

  return discriminatorValue;
} 

}

DEFINE_FWK_MODULE(PFRecoBaseTauDiscriminationAgainstMuonSimple);
