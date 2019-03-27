
/** \class PFRecoTauDiscriminationAgainstMuon2
 *
 * Compute tau Id. discriminator against muons.
 * 
 * \author Christian Veelken, LLR
 *
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonChamberMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "RecoTauTag/RecoTau/interface/RecoTauMuonTools.h"

#include <vector>
#include <string>
#include <iostream>

using reco::tau::format_vint;

namespace {

class PFRecoTauDiscriminationAgainstMuon2 final : public PFTauDiscriminationProducerBase
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
    maxNumberOfMatches_ = cfg.getParameter<int>("maxNumberOfMatches");
    doCaloMuonVeto_     = cfg.getParameter<bool>("doCaloMuonVeto");
    maxNumberOfHitsLast2Stations_ = cfg.getParameter<int>("maxNumberOfHitsLast2Stations");
      srcMuons_ = cfg.getParameter<edm::InputTag>("srcMuons");
      Muons_token = consumes<reco::MuonCollection>(srcMuons_);
      dRmuonMatch_ = cfg.getParameter<double>("dRmuonMatch");
      dRmuonMatchLimitedToJetArea_ = cfg.getParameter<bool>("dRmuonMatchLimitedToJetArea");
      minPtMatchedMuon_ = cfg.getParameter<double>("minPtMatchedMuon");
    typedef std::vector<int> vint;
    maskMatchesDT_  = cfg.getParameter<vint>("maskMatchesDT");
    maskMatchesCSC_ = cfg.getParameter<vint>("maskMatchesCSC");
    maskMatchesRPC_ = cfg.getParameter<vint>("maskMatchesRPC");
    maskHitsDT_     = cfg.getParameter<vint>("maskHitsDT");
    maskHitsCSC_    = cfg.getParameter<vint>("maskHitsCSC");
    maskHitsRPC_    = cfg.getParameter<vint>("maskHitsRPC");
    numWarnings_ = 0;
    maxWarnings_ = 3;
    verbosity_ = cfg.getParameter<int>("verbosity");
   }
  ~PFRecoTauDiscriminationAgainstMuon2() override {} 

  void beginEvent(const edm::Event&, const edm::EventSetup&) override;

  double discriminate(const reco::PFTauRef&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:  
  std::string moduleLabel_;
  int discriminatorOption_;
  double hop_;
  int maxNumberOfMatches_;
  bool doCaloMuonVeto_;
  int maxNumberOfHitsLast2Stations_;
  edm::InputTag srcMuons_;
  edm::Handle<reco::MuonCollection> muons_;
  edm::EDGetTokenT<reco::MuonCollection> Muons_token;
  double dRmuonMatch_;
  bool dRmuonMatchLimitedToJetArea_;
  double minPtMatchedMuon_;
  std::vector<int> maskMatchesDT_;
  std::vector<int> maskMatchesCSC_;
  std::vector<int> maskMatchesRPC_;
  std::vector<int> maskHitsDT_;
  std::vector<int> maskHitsCSC_;
  std::vector<int> maskHitsRPC_;
  mutable int numWarnings_;
  int maxWarnings_;
  int verbosity_;
};

void PFRecoTauDiscriminationAgainstMuon2::beginEvent(const edm::Event& evt, const edm::EventSetup& es) 
{
  if ( !srcMuons_.label().empty() ) {
    evt.getByToken(Muons_token, muons_);
  }
}

double PFRecoTauDiscriminationAgainstMuon2::discriminate(const reco::PFTauRef& pfTau) const
{
  if ( verbosity_ ) {
    edm::LogPrint("PFTauAgainstMuon2") << "<PFRecoTauDiscriminationAgainstMuon2::discriminate>:" ;
    edm::LogPrint("PFTauAgainstMuon2") << " moduleLabel = " << moduleLabel_ ;
    edm::LogPrint("PFTauAgainstMuon2") << "tau #" << pfTau.key() << ": Pt = " << pfTau->pt() << ", eta = " << pfTau->eta() << ", phi = " << pfTau->phi() ;   
  }

  std::vector<int> numMatchesDT(4);
  std::vector<int> numMatchesCSC(4);
  std::vector<int> numMatchesRPC(4);
  std::vector<int> numHitsDT(4);
  std::vector<int> numHitsCSC(4);
  std::vector<int> numHitsRPC(4);
  for ( int iStation = 0; iStation < 4; ++iStation ) {
    numMatchesDT[iStation]  = 0;
    numMatchesCSC[iStation] = 0;
    numMatchesRPC[iStation] = 0;    
    numHitsDT[iStation]     = 0;
    numHitsCSC[iStation]    = 0;
    numHitsRPC[iStation]    = 0;
  } 

  const reco::PFCandidatePtr& pfLeadChargedHadron = pfTau->leadPFChargedHadrCand();
  if ( pfLeadChargedHadron.isNonnull() ) {
    reco::MuonRef muonRef = pfLeadChargedHadron->muonRef();
    if ( muonRef.isNonnull() ) {
      if ( verbosity_ ) edm::LogPrint("PFTauAgainstMuon2") << " has muonRef." ;
      reco::tau::countMatches(*muonRef, numMatchesDT, numMatchesCSC, numMatchesRPC);
      reco::tau::countHits(*muonRef, numHitsDT, numHitsCSC, numHitsRPC);
    }
  }
  
  if ( !srcMuons_.label().empty() ) {
    size_t numMuons = muons_->size();
    for ( size_t idxMuon = 0; idxMuon < numMuons; ++idxMuon ) {
      reco::MuonRef muon(muons_, idxMuon);
      if ( verbosity_ ) edm::LogPrint("PFTauAgainstMuon2") << "muon #" << muon.key() << ": Pt = " << muon->pt() << ", eta = " << muon->eta() << ", phi = " << muon->phi() ;
      if ( !(muon->pt() > minPtMatchedMuon_) ) {
	if ( verbosity_ ){ edm::LogPrint("PFTauAgainstMuon2") << " fails Pt cut --> skipping it." ;}
	continue;
       }
      if ( pfLeadChargedHadron.isNonnull()) {
	reco::MuonRef muonRef = pfLeadChargedHadron->muonRef();
	if (muonRef.isNonnull() && muon == pfLeadChargedHadron->muonRef() ) {
	  if ( verbosity_ ) { edm::LogPrint("PFTauAgainstMuon2") << " matches muonRef of tau --> skipping it."; }
	  continue;
	}
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
	    edm::LogInfo("PFRecoTauDiscriminationAgainstMuon2::discriminate") 
	      << "Jet associated to Tau: Pt = " << pfTau->pt() << ", eta = " << pfTau->eta() << ", phi = " << pfTau->phi() << " has area = " << jetArea << " !!" ;
	    ++numWarnings_;
	  }
	  dRmatch = 0.1;
	}
      }
      if ( dR < dRmatch ) {
	if ( verbosity_ ) edm::LogPrint("PFTauAgainstMuon2") << " overlaps with tau, dR = " << dR ;
	reco::tau::countMatches(*muon, numMatchesDT, numMatchesCSC, numMatchesRPC);
	reco::tau::countHits(*muon, numHitsDT, numHitsCSC, numHitsRPC);
      }
    }
  }

  int numStationsWithMatches = 0;
  for ( int iStation = 0; iStation < 4; ++iStation ) {
    if ( numMatchesDT[iStation]  > 0 && !maskMatchesDT_[iStation]  ) ++numStationsWithMatches;
    if ( numMatchesCSC[iStation] > 0 && !maskMatchesCSC_[iStation] ) ++numStationsWithMatches;
    if ( numMatchesRPC[iStation] > 0 && !maskMatchesRPC_[iStation] ) ++numStationsWithMatches;
  }

  int numLast2StationsWithHits = 0;
  for ( int iStation = 2; iStation < 4; ++iStation ) {
    if ( numHitsDT[iStation]  > 0 && !maskHitsDT_[iStation]  ) ++numLast2StationsWithHits;
    if ( numHitsCSC[iStation] > 0 && !maskHitsCSC_[iStation] ) ++numLast2StationsWithHits;
    if ( numHitsRPC[iStation] > 0 && !maskHitsRPC_[iStation] ) ++numLast2StationsWithHits;
  }

  if ( verbosity_ ) {
    edm::LogPrint("PFTauAgainstMuon2") << "numMatchesDT  = " << format_vint(numMatchesDT)  ;
    edm::LogPrint("PFTauAgainstMuon2") << "numMatchesCSC = " << format_vint(numMatchesCSC) ;
    edm::LogPrint("PFTauAgainstMuon2") << "numMatchesRPC = " << format_vint(numMatchesRPC) ;
    edm::LogPrint("PFTauAgainstMuon2") << " --> numStationsWithMatches = " << numStationsWithMatches ;
    edm::LogPrint("PFTauAgainstMuon2") << "numHitsDT  = " << format_vint(numHitsDT)  ;
    edm::LogPrint("PFTauAgainstMuon2") << "numHitsCSC = " << format_vint(numHitsCSC) ;
    edm::LogPrint("PFTauAgainstMuon2") << "numHitsRPC = " << format_vint(numHitsRPC) ;
    edm::LogPrint("PFTauAgainstMuon2") << " --> numLast2StationsWithHits = " << numLast2StationsWithHits ;
  }
  
  bool passesCaloMuonVeto = true;
  if ( pfLeadChargedHadron.isNonnull() ) {
    double energyECALplusHCAL = pfLeadChargedHadron->ecalEnergy() + pfLeadChargedHadron->hcalEnergy();
    if ( verbosity_ ) {
      if ( pfLeadChargedHadron->trackRef().isNonnull() ) {
	edm::LogPrint("PFTauAgainstMuon2") << "decayMode = " << pfTau->decayMode() << ", energy(ECAL+HCAL) = " << energyECALplusHCAL << ", leadPFChargedHadronP = " << pfLeadChargedHadron->trackRef()->p() ;
      } else if ( pfLeadChargedHadron->gsfTrackRef().isNonnull() ) {
	edm::LogPrint("PFTauAgainstMuon2") << "decayMode = " << pfTau->decayMode() << ", energy(ECAL+HCAL) = " << energyECALplusHCAL << ", leadPFChargedHadronP = " << pfLeadChargedHadron->gsfTrackRef()->p() ;
      }
    }
    const reco::Track* leadTrack = nullptr;
    if ( pfLeadChargedHadron->trackRef().isNonnull() )
      leadTrack = pfLeadChargedHadron->trackRef().get();
    else if ( pfLeadChargedHadron->gsfTrackRef().isNonnull() )
      leadTrack = pfLeadChargedHadron->gsfTrackRef().get();
    if ( pfTau->decayMode() == 0 && leadTrack && energyECALplusHCAL < (hop_*leadTrack->p()) )
      passesCaloMuonVeto = false;
  }
  
  double discriminatorValue = 0.;
  if      ( discriminatorOption_ == kLoose  && numStationsWithMatches <= maxNumberOfMatches_                                                                                    ) discriminatorValue = 1.;
  else if ( discriminatorOption_ == kMedium && numStationsWithMatches <= maxNumberOfMatches_ && numLast2StationsWithHits <= maxNumberOfHitsLast2Stations_                       ) discriminatorValue = 1.;
  else if ( discriminatorOption_ == kTight  && numStationsWithMatches <= maxNumberOfMatches_ && numLast2StationsWithHits <= maxNumberOfHitsLast2Stations_ && passesCaloMuonVeto ) discriminatorValue = 1.;
  else if ( discriminatorOption_ == kCustom ) {
    bool pass = true;
    if ( maxNumberOfMatches_ >= 0 && numStationsWithMatches > maxNumberOfMatches_ ) pass = false;
    if ( maxNumberOfHitsLast2Stations_ >= 0 && numLast2StationsWithHits > maxNumberOfHitsLast2Stations_ ) pass = false;
    if ( doCaloMuonVeto_ && !passesCaloMuonVeto ) pass = false;
    discriminatorValue = pass ? 1.: 0.;
  }
  if ( verbosity_ ) edm::LogPrint("PFTauAgainstMuon2") << "--> returning discriminatorValue = " << discriminatorValue ;

  return discriminatorValue;
} 

}

void
PFRecoTauDiscriminationAgainstMuon2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfRecoTauDiscriminationAgainstMuon2
  edm::ParameterSetDescription desc;
  desc.add<std::vector<int>>("maskHitsRPC", {
    0,
    0,
    0,
    0,
  });
  desc.add<int>("maxNumberOfHitsLast2Stations", 0);
  desc.add<std::vector<int>>("maskMatchesRPC", {
    0,
    0,
    0,
    0,
  });
  desc.add<std::vector<int>>("maskMatchesCSC", {
    1,
    0,
    0,
    0,
  });
  desc.add<std::vector<int>>("maskHitsCSC", {
    0,
    0,
    0,
    0,
  });
  desc.add<edm::InputTag>("PFTauProducer", edm::InputTag("pfRecoTauProducer"));
  desc.add<int>("verbosity", 0);
  desc.add<std::vector<int>>("maskMatchesDT", {
    0,
    0,
    0,
    0,
  });
  desc.add<double>("minPtMatchedMuon", 5.0);
  desc.add<bool>("dRmuonMatchLimitedToJetArea", false);
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("BooleanOperator", "and");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("cut"); //, 0.5);
      psd1.add<edm::InputTag>("Producer"); //, edm::InputTag("pfRecoTauDiscriminationByLeadingTrackFinding"));
      psd0.addOptional<edm::ParameterSetDescription>("leadTrack", psd1); // optional with default? 
    }
    // Prediscriminants can be
    // Prediscriminants = noPrediscriminants,
    // as in RecoTauTag/Configuration/python/HPSPFTaus_cff.py
    // 
    // and the definition is:
    // RecoTauTag/RecoTau/python/TauDiscriminatorTools.py
    // # Require no prediscriminants
    // noPrediscriminants = cms.PSet(
    //       BooleanOperator = cms.string("and"),
    //       )
    // -- so this is the minimum required definition
    // otherwise it inserts the leadTrack with Producer = InpuTag(...) and does not find the corresponding output during the run
    desc.add<edm::ParameterSetDescription>("Prediscriminants", psd0);
  }
  desc.add<std::vector<int>>("maskHitsDT", {
    0,
    0,
    0,
    0,
  });
  desc.add<double>("HoPMin", 0.2);
  desc.add<int>("maxNumberOfMatches", 0);
  desc.add<std::string>("discriminatorOption", "loose");
  desc.add<double>("dRmuonMatch", 0.3);
  desc.add<edm::InputTag>("srcMuons", edm::InputTag("muons"));
  desc.add<bool>("doCaloMuonVeto", false);
  descriptions.add("pfRecoTauDiscriminationAgainstMuon2", desc);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstMuon2);
