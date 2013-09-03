
/** \class PFRecoTauDiscriminationAgainstMuon2
 *
 * Compute tau Id. discriminator against muons.
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.10 $
 *
 * $Id: PFRecoTauDiscriminationAgainstMuon2.cc,v 1.10 2013/04/08 11:45:27 jez Exp $
 *
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <vector>
#include <string>
#include <iostream>

class PFRecoTauDiscriminationAgainstMuon2 : public PFTauDiscriminationProducerBase 
{
  enum { kLoose, kMedium, kTight, kCustom };
 public:
  explicit PFRecoTauDiscriminationAgainstMuon2(const edm::ParameterSet& cfg)
    : PFTauDiscriminationProducerBase(cfg),
      moduleLabel_(cfg.getParameter<std::string>("@module_label"))
  {   
    std::string discriminatorOption_string  = cfg.getParameter<std::string>("discriminatorOption");  
    if      ( discriminatorOption_string == "loose"  ) discriminatorOption_ = kLoose;
    else if ( discriminatorOption_string == "medium" ) discriminatorOption_ = kMedium;
    else if ( discriminatorOption_string == "tight"  ) discriminatorOption_ = kTight;
    else if ( discriminatorOption_string == "custom" ) discriminatorOption_ = kCustom;
    else throw edm::Exception(edm::errors::UnimplementedFeature) 
      << " Invalid Configuration parameter 'discriminatorOption' = " << discriminatorOption_string << " !!\n";
    hop_ = cfg.getParameter<double>("HoPMin"); 
    maxNumberOfMatches_ = cfg.exists("maxNumberOfMatches") ? cfg.getParameter<int>("maxNumberOfMatches"): 0;
    doCaloMuonVeto_ = cfg.exists("doCaloMuonVeto") ? cfg.getParameter<bool>("doCaloMuonVeto"): false;
    maxNumberOfHitsLast2Stations_ = cfg.exists("maxNumberOfHitsLast2Stations") ? cfg.getParameter<int>("maxNumberOfHitsLast2Stations"): 0;
    if ( cfg.exists("srcMuons") ) {
      srcMuons_ = cfg.getParameter<edm::InputTag>("srcMuons");
      dRmuonMatch_ = cfg.getParameter<double>("dRmuonMatch");
    }
    verbosity_ = cfg.exists("verbosity") ? cfg.getParameter<int>("verbosity") : 0;
   }
  ~PFRecoTauDiscriminationAgainstMuon2() {} 

  void beginEvent(const edm::Event&, const edm::EventSetup&);

  double discriminate(const reco::PFTauRef&);

 private:  
  std::string moduleLabel_;
  int discriminatorOption_;
  double hop_;
  int maxNumberOfMatches_;
  bool doCaloMuonVeto_;
  int maxNumberOfHitsLast2Stations_;
  edm::InputTag srcMuons_;
  edm::Handle<reco::MuonCollection> muons_;
  double dRmuonMatch_;
  int verbosity_;
};

void PFRecoTauDiscriminationAgainstMuon2::beginEvent(const edm::Event& evt, const edm::EventSetup& es) 
{
  if ( srcMuons_.label() != "" ) {
    evt.getByLabel(srcMuons_, muons_);
  }
}

namespace
{
  void countHits(const reco::Muon& muon, std::vector<int>& numHitsDT, std::vector<int>& numHitsCSC, std::vector<int>& numHitsRPC)
  {
    if ( muon.outerTrack().isNonnull() ) {
      const reco::HitPattern& muonHitPattern = muon.outerTrack()->hitPattern();
      for ( int iHit = 0; iHit < muonHitPattern.numberOfHits(); ++iHit ) {
	uint32_t hit = muonHitPattern.getHitPattern(iHit);
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

double PFRecoTauDiscriminationAgainstMuon2::discriminate(const reco::PFTauRef& pfTau)
{
  if ( verbosity_ ) {
    std::cout << "<PFRecoTauDiscriminationAgainstMuon2::discriminate>:" << std::endl;
    std::cout << " moduleLabel = " << moduleLabel_ << std::endl;
    std::cout << "tau #" << pfTau.key() << ": Pt = " << pfTau->pt() << ", eta = " << pfTau->eta() << ", phi = " << pfTau->phi() << std::endl;   
  }

  int numMatches = 0;

  std::vector<int> numHitsDT(4);
  std::vector<int> numHitsCSC(4);
  std::vector<int> numHitsRPC(4);
  for ( int iStation = 0; iStation < 4; ++iStation ) {
    numHitsDT[iStation]  = 0;
    numHitsCSC[iStation] = 0;
    numHitsRPC[iStation] = 0;
  }

  const reco::PFCandidateRef& pfLeadChargedHadron = pfTau->leadPFChargedHadrCand();
  if ( pfLeadChargedHadron.isNonnull() ) {
    reco::MuonRef muonRef = pfLeadChargedHadron->muonRef();      
    if ( muonRef.isNonnull() ) {
      if ( verbosity_ ) std::cout << " has muonRef." << std::endl;
      numMatches = muonRef->numberOfMatches(reco::Muon::NoArbitration);
      countHits(*muonRef, numHitsDT, numHitsCSC, numHitsRPC);
    }
  }
  
  if ( srcMuons_.label() != "" ) {
    size_t numMuons = muons_->size();
    for ( size_t idxMuon = 0; idxMuon < numMuons; ++idxMuon ) {
      reco::MuonRef muon(muons_, idxMuon);
      if ( verbosity_ ) std::cout << "muon #" << muon.key() << ": Pt = " << muon->pt() << ", eta = " << muon->eta() << ", phi = " << muon->phi() << std::endl;
      if ( pfLeadChargedHadron.isNonnull() && pfLeadChargedHadron->muonRef().isNonnull() && muon == pfLeadChargedHadron->muonRef() ) {	
	if ( verbosity_ ) std::cout << " matches muonRef of tau --> skipping it." << std::endl;
	continue;
      }
      double dR = deltaR(muon->p4(), pfTau->p4());
      if ( dR < dRmuonMatch_ ) {
	if ( verbosity_ ) std::cout << " overlaps with tau, dR = " << dR << std::endl;
	numMatches += muon->numberOfMatches(reco::Muon::NoArbitration);
	countHits(*muon, numHitsDT, numHitsCSC, numHitsRPC);
      }
    }
  }

  if ( verbosity_ ) {
    std::cout << "numMatches = " << numMatches << std::endl;
    std::cout << "numHitsDT  = " << format_vint(numHitsDT)  << std::endl;
    std::cout << "numHitsCSC = " << format_vint(numHitsCSC) << std::endl;
    std::cout << "numHitsRPC = " << format_vint(numHitsRPC) << std::endl;
  }
  
  int numLast2StationsWithHits = 0;
  for ( int iStation = 2; iStation < 4; ++iStation ) {
    if ( numHitsDT[iStation]  > 0 ) ++numLast2StationsWithHits;
    if ( numHitsCSC[iStation] > 0 ) ++numLast2StationsWithHits;
    if ( numHitsRPC[iStation] > 0 ) ++numLast2StationsWithHits;
  }


  bool passesCaloMuonVeto = true;
  if ( pfLeadChargedHadron.isNonnull() ) {
    double energyECALplusHCAL = pfLeadChargedHadron->ecalEnergy() + pfLeadChargedHadron->hcalEnergy();    
    if ( verbosity_ ) {
      if ( pfLeadChargedHadron->trackRef().isNonnull() ) std::cout << "decayMode = " << pfTau->decayMode() << ", energy(ECAL+HCAL) = " << energyECALplusHCAL << ", leadPFChargedHadronP = " << pfLeadChargedHadron->trackRef()->p() << std::endl;
      else if ( pfLeadChargedHadron->gsfTrackRef().isNonnull() ) 
std::cout << "decayMode = " << pfTau->decayMode() << ", energy(ECAL+HCAL) = " << energyECALplusHCAL << ", leadPFChargedHadronP = " << pfLeadChargedHadron->gsfTrackRef()->p() << std::endl;

    }
    const reco::Track* leadTrack = 0;
    if ( pfLeadChargedHadron->trackRef().isNonnull() ) leadTrack = pfLeadChargedHadron->trackRef().get();
    else if ( pfLeadChargedHadron->gsfTrackRef().isNonnull() ) leadTrack = pfLeadChargedHadron->gsfTrackRef().get();
    if ( pfTau->decayMode() == 0 && leadTrack && energyECALplusHCAL < (hop_*leadTrack->p()) ) passesCaloMuonVeto = false;
  }
  
  double discriminatorValue = 0.;
  if      ( discriminatorOption_ == kLoose  && numMatches <= maxNumberOfMatches_                                                                                    ) discriminatorValue = 1.;
  else if ( discriminatorOption_ == kMedium && numMatches <= maxNumberOfMatches_ && numLast2StationsWithHits <= maxNumberOfHitsLast2Stations_                       ) discriminatorValue = 1.;
  else if ( discriminatorOption_ == kTight  && numMatches <= maxNumberOfMatches_ && numLast2StationsWithHits <= maxNumberOfHitsLast2Stations_ && passesCaloMuonVeto ) discriminatorValue = 1.;
  else if ( discriminatorOption_ == kCustom ) {
    bool pass = true;
    if ( maxNumberOfMatches_ >= 0 && numMatches > maxNumberOfMatches_ ) pass = false;
    if ( maxNumberOfHitsLast2Stations_ >= 0 && numLast2StationsWithHits > maxNumberOfHitsLast2Stations_ ) pass = false;
    if ( doCaloMuonVeto_ && !passesCaloMuonVeto ) pass = false;
    discriminatorValue = pass ? 1.: 0.;
  }
  if ( verbosity_ ) std::cout << "--> returning discriminatorValue = " << discriminatorValue << std::endl;

  return discriminatorValue;
} 

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstMuon2);
