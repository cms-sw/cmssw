//
// $Id: Electron.cc,v 1.25 2011/03/31 10:13:26 namapane Exp $
//

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <limits>

using namespace pat;

/// default constructor
Electron::Electron() :
    Lepton<reco::GsfElectron>(),
    embeddedGsfElectronCore_(false),
    embeddedGsfTrack_(false),
    embeddedSuperCluster_(false),
    embeddedTrack_(false),
    embeddedPFCandidate_(false),
    ecalDrivenMomentum_(Candidate::LorentzVector(0.,0.,0.,0.)),
    cachedDB_(false),
    dB_(0.0),
    edB_(0.0)
{
  initImpactParameters();
}

/// constructor from reco::GsfElectron
Electron::Electron(const reco::GsfElectron & anElectron) :
    Lepton<reco::GsfElectron>(anElectron),
    embeddedGsfElectronCore_(false),
    embeddedGsfTrack_(false),
    embeddedSuperCluster_(false),
    embeddedTrack_(false),
    embeddedPFCandidate_(false),
    ecalDrivenMomentum_(anElectron.p4()),
    cachedDB_(false),
    dB_(0.0),
    edB_(0.0)
{
  initImpactParameters();
}

/// constructor from ref to reco::GsfElectron
Electron::Electron(const edm::RefToBase<reco::GsfElectron> & anElectronRef) :
    Lepton<reco::GsfElectron>(anElectronRef),
    embeddedGsfElectronCore_(false),
    embeddedGsfTrack_(false),
    embeddedSuperCluster_(false),
    embeddedTrack_(false),
    embeddedPFCandidate_(false),
    ecalDrivenMomentum_(anElectronRef->p4()),
    cachedDB_(false),
    dB_(0.0),
    edB_(0.0)
{
  initImpactParameters();
}

/// constructor from Ptr to reco::GsfElectron
Electron::Electron(const edm::Ptr<reco::GsfElectron> & anElectronRef) :
    Lepton<reco::GsfElectron>(anElectronRef),
    embeddedGsfElectronCore_(false),
    embeddedGsfTrack_(false),
    embeddedSuperCluster_(false),
    embeddedTrack_(false),
    embeddedPFCandidate_(false),
    ecalDrivenMomentum_(anElectronRef->p4()),
    cachedDB_(false),
    dB_(0.0),
    edB_(0.0)
{
  initImpactParameters();
}

/// destructor
Electron::~Electron() {
}

std::ostream& 
reco::operator<<(std::ostream& out, const pat::Electron& obj) 
{
  if(!out) return out;
  
  out << "\tpat::Electron: ";
  out << std::setiosflags(std::ios::right);
  out << std::setiosflags(std::ios::fixed);
  out << std::setprecision(3);
  out << " E/pT/eta/phi " 
      << obj.energy()<<"/"
      << obj.pt()<<"/"
      << obj.eta()<<"/"
      << obj.phi();
  return out; 
}

// initialize impact parameter container vars
void Electron::initImpactParameters() {
  for (int i_ = 0; i_<5; ++i_){
    ip_.push_back(0.0);
    eip_.push_back(0.0);
    cachedIP_.push_back(false);
  }
}


/// override the reco::GsfElectron::gsfTrack method, to access the internal storage of the supercluster
reco::GsfTrackRef Electron::gsfTrack() const {
  if (embeddedGsfTrack_) {
    return reco::GsfTrackRef(&gsfTrack_, 0);
  } else {
    return reco::GsfElectron::gsfTrack();
  }
}

/// override the virtual reco::GsfElectron::core method, so that the embedded core can be used by GsfElectron client methods
reco::GsfElectronCoreRef Electron::core() const {
  if (embeddedGsfElectronCore_) {
    return reco::GsfElectronCoreRef(&gsfElectronCore_, 0);
  } else {
    return reco::GsfElectron::core();
  }
}


/// override the reco::GsfElectron::superCluster method, to access the internal storage of the supercluster
reco::SuperClusterRef Electron::superCluster() const {
  if (embeddedSuperCluster_) {
    return reco::SuperClusterRef(&superCluster_, 0);
  } else {
    return reco::GsfElectron::superCluster();
  }
}


/// override the reco::GsfElectron::track method, to access the internal storage of the track
reco::TrackRef Electron::track() const {
  if (embeddedTrack_) {
    return reco::TrackRef(&track_, 0);
  } else {
    return reco::GsfElectron::track();
  }
}

/// method to store the electron's gsfElectronCore internally
void Electron::embedGsfElectronCore() {
  gsfElectronCore_.clear();
  if (reco::GsfElectron::core().isNonnull()) {
      gsfElectronCore_.push_back(*reco::GsfElectron::core());
      embeddedGsfElectronCore_ = true;
  }
}

/// method to store the electron's gsfTrack internally
void Electron::embedGsfTrack() {
  gsfTrack_.clear();
  if (reco::GsfElectron::gsfTrack().isNonnull()) {
      gsfTrack_.push_back(*reco::GsfElectron::gsfTrack());
      embeddedGsfTrack_ = true;
  }
}


/// method to store the electron's supercluster internally
void Electron::embedSuperCluster() {
  superCluster_.clear();
  if (reco::GsfElectron::superCluster().isNonnull()) {
      superCluster_.push_back(*reco::GsfElectron::superCluster());
      embeddedSuperCluster_ = true;
  }
}


/// method to store the electron's track internally
void Electron::embedTrack() {
  track_.clear();
  if (reco::GsfElectron::track().isNonnull()) {
      track_.push_back(*reco::GsfElectron::track());
      embeddedTrack_ = true;
  }
}

// method to retrieve a lepton ID (or throw)
float Electron::electronID(const std::string & name) const {
    for (std::vector<IdPair>::const_iterator it = electronIDs_.begin(), ed = electronIDs_.end(); it != ed; ++it) {
        if (it->first == name) return it->second;
    }
    cms::Exception ex("Key not found");
    ex << "pat::Electron: the ID " << name << " can't be found in this pat::Electron.\n";
    ex << "The available IDs are: ";
    for (std::vector<IdPair>::const_iterator it = electronIDs_.begin(), ed = electronIDs_.end(); it != ed; ++it) {
        ex << "'" << it->first << "' ";
    }
    ex << ".\n";
    throw ex;
}
// check if an ID is there
bool Electron::isElectronIDAvailable(const std::string & name) const {
    for (std::vector<IdPair>::const_iterator it = electronIDs_.begin(), ed = electronIDs_.end(); it != ed; ++it) {
        if (it->first == name) return true;
    }
    return false;
}


/// reference to the source PFCandidates
reco::PFCandidateRef Electron::pfCandidateRef() const {
  if (embeddedPFCandidate_) {
    return reco::PFCandidateRef(&pfCandidate_, 0);
  } else {
    return pfCandidateRef_;
  }
}
/// embed the IsolatedPFCandidate pointed to by pfCandidateRef_
void Electron::embedPFCandidate() {
  pfCandidate_.clear();
  if ( pfCandidateRef_.isAvailable() && pfCandidateRef_.isNonnull()) {
    pfCandidate_.push_back( *pfCandidateRef_ );
    embeddedPFCandidate_ = true;
  }
}

/// reference to the parent PF candidate for use in TopProjector
reco::CandidatePtr Electron::sourceCandidatePtr( size_type i ) const {
  if (embeddedPFCandidate_) {
    return reco::CandidatePtr( pfCandidateRef_.id(), pfCandidateRef_.get(), pfCandidateRef_.key() );
  } else {
    return reco::CandidatePtr();
  }
}



/// dB gives the impact parameter wrt the beamline.
/// If this is not cached it is not meaningful, since
/// it relies on the distance to the beamline. 
double Electron::dB(IpType type_) const {
  // preserve old functionality exactly
  if (type_ == None){
    if ( cachedDB_ ) {
      return dB_;
    } else {
      return std::numeric_limits<double>::max();
    }
  }
  // more IP types (new)
  else if ( cachedIP_[type_] ) {
    return ip_[type_];
  } else {
    return std::numeric_limits<double>::max();
  }
}

/// edB gives the uncertainty on the impact parameter wrt the beamline.
/// If this is not cached it is not meaningful, since
/// it relies on the distance to the beamline. 
double Electron::edB(IpType type_) const {
  // preserve old functionality exactly
  if (type_ == None) {
    if ( cachedDB_ ) {
      return edB_;
    } else {
      return std::numeric_limits<double>::max();
    }
  }
  // more IP types (new)
  else if ( cachedIP_[type_] ) {
    return eip_[type_];
  } else {
    return std::numeric_limits<double>::max();
  }

}

void Electron::setDB(double dB, double edB, IpType type){
  if (type == None) { // Preserve  old functionality exactly
    dB_ = dB; edB_ = edB;
    cachedDB_ = true;
  } else {
    ip_[type] = dB; 
    eip_[type] = edB; 
    cachedIP_[type] = true;
  }
}
 
