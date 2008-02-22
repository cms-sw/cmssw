//
// $Id: Photon.cc,v 1.2 2008/01/23 15:53:15 lowette Exp $
//

#include "DataFormats/PatCandidates/interface/Photon.h"


using pat::Photon;


/// default constructor
Photon::Photon() :
    PATObject<PhotonType>(PhotonType(0, reco::Particle::LorentzVector(0, 0, 0, 0), reco::Particle::Point(0, 0, 0), 0, 0, 0)),
    trackIso_(0), caloIso_(0), photonID_(-1.0) 
{
}


/// constructor from PhotonType
Photon::Photon(const PhotonType & aPhoton) :
    PATObject<PhotonType>(aPhoton),
    trackIso_(0), caloIso_(0), photonID_(-1.0) 
{
}


/// constructor from ref to PhotonType
Photon::Photon(const edm::RefToBase<PhotonType> & aPhotonRef) :
    PATObject<PhotonType>(aPhotonRef),
    trackIso_(0), caloIso_(0), photonID_(-1.0) 
{
}


/// destructor
Photon::~Photon() {
}


/// return the match to the generated photon
const reco::Particle * Photon::genPhoton() const {
  return (genPhoton_.size() > 0 ? &genPhoton_.front() : 0);
}


/// method to set the generated photon
void Photon::setGenPhoton(const reco::Particle & gp) {
  genPhoton_.clear();
  genPhoton_.push_back(gp);
}
