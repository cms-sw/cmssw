#include "DataFormats/PatCandidates/interface/Photon.h"

using pat::Photon;

/// default constructor
Photon::Photon() :
    PATObject<PhotonType>(PhotonType()),
    trackIso_(0), caloIso_(0), photonID_(-1.0) 
{
    // no common constructor, so initialize the candidate manually
    setCharge(0);
    setP4(reco::Particle::LorentzVector(0, 0, 0, 0));
    setVertex(reco::Particle::Point(0, 0, 0));
}


/// constructor from PhotonType
Photon::Photon(const PhotonType & aPhoton) :
    PATObject<PhotonType>(aPhoton),
    trackIso_(0), caloIso_(0), photonID_(-1.0) 
{
}


/// constructor from ref to PhotonType
Photon::Photon(const edm::Ref<std::vector<PhotonType> > & aPhotonRef) :
    PATObject<PhotonType>(aPhotonRef),
    trackIso_(0), caloIso_(0), photonID_(-1.0) 
{
}


/// destructor
Photon::~Photon() {
}


/// return the match to the generated lepton
const reco::Particle * Photon::genPhoton() const {
    return (genPhoton_.size() > 0 ? &genPhoton_.front() : 0);
}

/// method to set the generated lepton
void Photon::setGenPhoton(const reco::Particle & gl) {
    genPhoton_.clear();
    genPhoton_.push_back(gl);
}


