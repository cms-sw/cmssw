#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonChamberMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TauReco/interface/PFTau.h"

#include "RecoTauTag/RecoTau/interface/RecoTauMuonTools.h"

#include <vector>
#include <string>
#include <iostream>
#include <atomic>

struct PFRecoTauDiscriminationAgainstMuonConfigSet {
    enum { kLoose, kMedium, kTight, kCustom };
    
    PFRecoTauDiscriminationAgainstMuonConfigSet(int dOpt, double hop, int mNOM, bool doCMV, int mNHL2S) : discriminatorOption(dOpt), hop(hop), maxNumberOfMatches(mNOM), doCaloMuonVeto(doCMV), maxNumberOfHitsLast2Stations(mNHL2S){}
    
    int discriminatorOption;
    double hop;
    int maxNumberOfMatches;
    bool doCaloMuonVeto;
    int maxNumberOfHitsLast2Stations;
};

struct PFRecoTauDiscriminationAgainstMuon2Helper {   
    
    double energyECALplusHCAL_;
    const reco::PFCandidatePtr& pfLeadChargedHadron_;
    const reco::Track* leadTrack_ = nullptr;
    int numStationsWithMatches_ = 0;
    int numLast2StationsWithHits_ = 0;
    
    PFRecoTauDiscriminationAgainstMuon2Helper(
    const bool&,
    const std::string&,
    const bool,
    const double&,
    const double&,
    const bool&,
    std::atomic<unsigned int>&,
    const unsigned int&,
    const std::vector<int>&,
    const std::vector<int>&,
    const std::vector<int>&,
    const std::vector<int>&,
    const std::vector<int>&, 
    const std::vector<int>&, const edm::Handle<reco::MuonCollection>&, const reco::PFTauRef&);
    bool eval(const PFRecoTauDiscriminationAgainstMuonConfigSet&, const reco::PFTauRef&) const;
};